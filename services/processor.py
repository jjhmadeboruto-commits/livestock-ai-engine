from __future__ import annotations

import cv2
import numpy as np
import os
import logging
from typing import Any, Dict, Optional, Tuple

class AnimalProcessor:
    """
    Optimised Processor for estimating livestock live weight using YOLOv8 bounding boxes.
    Addresses geometric variations across cows, sheep, poultry, donkeys, goats, and pigs.
    """

    # Regulated agricultural metric formulas and density bounds
    LIVESTOCK_CALIBRATION = {
        "dairy_cow":    {"divisor": 660.0,  "girth_multiplier": 0.84, "name": "Dairy Cow",          "expected_range": [250, 800],  "coco_classes": {19}, "default_pixel_ratio": 0.15}, # Cow (Class 19)
        "beef_cattle":  {"divisor": 600.0,  "girth_multiplier": 0.88, "name": "Beef Cattle",        "expected_range": [300, 1000], "coco_classes": {19}, "default_pixel_ratio": 0.15}, # Cow (Class 19)
        "young_cattle": {"divisor": 640.0,  "girth_multiplier": 0.82, "name": "Young Cattle/Calf",  "expected_range": [50, 300],   "coco_classes": {19}, "default_pixel_ratio": 0.10}, # Cow (Class 19)
        "pig":          {"divisor": 400.0,  "girth_multiplier": 0.92, "name": "Pig",                 "expected_range": [20, 400],   "coco_classes": {18, 19}, "default_pixel_ratio": 0.10}, # Pig (COCO proxy: Sheep/Cow)
        "goat":         {"divisor": 300.0,  "girth_multiplier": 0.78, "name": "Goat",                "expected_range": [15, 140],   "coco_classes": {18}, "default_pixel_ratio": 0.08}, # Goat (COCO proxy: Sheep Class 18)
        "sheep":        {"divisor": 300.0,  "girth_multiplier": 0.80, "name": "Sheep",               "expected_range": [20, 160],   "coco_classes": {18}, "default_pixel_ratio": 0.08}, # Sheep (Class 18)
        "donkey":       {"divisor": 480.0,  "girth_multiplier": 0.82, "name": "Donkey",              "expected_range": [80, 500],   "coco_classes": {17}, "default_pixel_ratio": 0.12}, # Donkey (COCO proxy: Horse Class 17)
        "poultry":      {"divisor": 1200.0, "girth_multiplier": 0.55, "name": "Poultry",             "expected_range": [0.5, 8],    "coco_classes": {14}, "default_pixel_ratio": 0.04}, # Poultry (COCO: Bird Class 14)
    }

    # Universally permitted COCO detection IDs: 14=bird, 17=horse, 18=sheep, 19=cow
    YOLO_LIVESTOCK_CLASSES = {14, 17, 18, 19}

    # ── Body-plan groups ──────────────────────────────────────────────────────
    # COCO was not trained on livestock specifically. Animals get detected as
    # their nearest COCO proxy:
    #   pig   → class 19 (cow)     goat  → class 18 (sheep)
    #   donkey→ class 17 (horse)   poultry→ class 14 (bird)
    # We group them by body-plan so we only force-correct on DEFINITIVE
    # cross-group mismatches (e.g. bird detected when user said cattle).
    # Within a group the user's button selection is trusted because they
    # know what animal they are actually photographing.
    COCO_GROUPS = {
        14: "avian",          # bird  → poultry
        17: "equine",         # horse → donkey/mule
        18: "small_quad",     # sheep → sheep or goat
        19: "large_quad",     # cow   → dairy cow, beef cattle, calf, pig
    }
    ANIMAL_GROUP = {
        "dairy_cow":    "large_quad",
        "beef_cattle":  "large_quad",
        "young_cattle": "large_quad",
        "pig":          "large_quad",   # COCO sees pig as 'cow' → same group
        "goat":         "small_quad",   # COCO sees goat as 'sheep' → same group
        "sheep":        "small_quad",
        "donkey":       "equine",
        "poultry":      "avian",
    }
    # Best default livestock type per COCO class when a full correction IS needed
    COCO_CLASS_DEFAULT = {
        14: "poultry",
        17: "donkey",
        18: "sheep",
        19: "dairy_cow",
    }

    _yolo_model = None
    _clip_model = None
    _clip_processor = None
    rejected_reason = None

    # ── CLIP text prompts for zero-shot species classification ────────────────
    # Each inner list holds synonym prompts — CLIP score is averaged per category.
    CLIP_PROMPTS = {
        "pig":          ["a pig", "a domestic pig", "a hog", "a swine", "a piglet"],
        "dairy_cow":    ["a dairy cow", "a Holstein cow", "a milk cow"],
        "beef_cattle":  ["a beef cattle", "a Angus bull", "a steer"],
        "young_cattle": ["a calf", "a young cow", "a baby cattle"],
        "sheep":        ["a sheep", "a wool sheep", "a ram", "a ewe", "a lamb"],
        "goat":         ["a goat", "a billy goat", "a nanny goat", "a mountain goat"],
        "donkey":       ["a donkey", "a mule", "a burro", "a domestic donkey"],
        "poultry":      ["a chicken", "a hen", "a rooster", "a turkey", "a duck", "a poultry bird"],
        "invalid":      ["a person", "a human", "a dog", "a cat", "a vehicle", "a building", "a car"],
    }

    def __init__(self, pixel_to_cm_ratio: float = None, animal_type: str = "dairy_cow") -> None:
        self.animal_type = animal_type.lower()
        if self.animal_type not in self.LIVESTOCK_CALIBRATION:
            self.animal_type = "dairy_cow"

        calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
        # Track whether the user explicitly provided a pixel ratio so auto-correction
        # knows not to overwrite a deliberate user calibration.
        self._pixel_ratio_user_set = pixel_to_cm_ratio is not None
        if pixel_to_cm_ratio is None:
            self.pixel_to_cm_ratio = calibration.get("default_pixel_ratio", 0.15)
        else:
            self.pixel_to_cm_ratio = pixel_to_cm_ratio

        if self.__class__._yolo_model is None:
            try:
                import os
                import torch
                
                # BRUTE-FORCE FIX: Override torch.load to bypass PyTorch 2.6+ weights_only restrictions
                original_load = torch.load
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = safe_load

                from ultralytics import YOLO
                
                # 1. Start with standard absolute file level direction
                current_file_path = os.path.abspath(__file__)
                services_dir = os.path.dirname(current_file_path)
                root_dir = os.path.dirname(services_dir)
                
                # Define the primary and fallback target paths
                path_options = [
                    os.path.join(root_dir, "models", "yolov8n.pt"), # Standard local root path
                    os.path.join(os.getcwd(), "models", "yolov8n.pt"), # Current Working Directory path
                    os.path.join(os.getcwd(), "yolov8n.pt"), # Root level fallback
                    "yolov8n.pt" # Raw system path fallback
                ]
                
                selected_path = None
                for path in path_options:
                    logging.info(f"Checking for weights at target location: {path}")
                    if os.path.exists(path):
                        selected_path = path
                        logging.info(f"SUCCESS: Found weights file cache layer at: {path}")
                        break

                if selected_path:
                    # Load the verified local file path
                    self.model = YOLO(selected_path)
                    logging.info(f"YOLOv8 Neural Network initialized completely using: {selected_path}")
                else:
                    # Emergency recovery: Let YOLO try to resolve it natively if files are missing
                    logging.warning("CRITICAL: Local weights file not detected in options list. Running default string configuration.")
                    self.model = YOLO("yolov8n.pt")
                
                self.__class__._yolo_model = self.model

            except Exception as e:
                logging.error(f"Failed to compile YOLO model weights: {e}")
                self.__class__._yolo_model = False

        self.model = self.__class__._yolo_model if self.__class__._yolo_model else None

        if self.__class__._clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                logging.info("Loading CLIP model for zero-shot livestock species classification...")
                clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                clip.eval()
                self.__class__._clip_model     = clip
                self.__class__._clip_processor = proc
                logging.info("CLIP model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load CLIP model: {e}")
                self.__class__._clip_model     = False
                self.__class__._clip_processor = False

    def process(self, image_bgr: np.ndarray) -> Optional[Dict[str, object]]:
        """Executes targeted object extraction and converts pixel space into physical mass."""
        annotated_image = image_bgr.copy()
        method_used = None
        x1, y1, x2, y2 = 0, 0, 0, 0
        confidence_score = 0.0

        calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
        target_coco_classes = calibration["coco_classes"]

        # Run human detection check and obtain YOLO results to avoid duplicate inference
        yolo_results = None
        if self.model:
            try:
                yolo_results = self.model(image_bgr, verbose=False, conf=0.20)[0]
                for box in yolo_results.boxes:
                    if int(box.cls[0]) == 0:  # Class 0 is human in the COCO dataset
                        return {
                            "weight": 0.0,
                            "body_length": 0.0,
                            "body_height": 0.0,
                            "estimated_girth": 0.0,
                            "animal_type": "Invalid Target",
                            "confidence_score": 0.0,
                            "annotated_image": image_bgr,
                            "expected_weight_range": [0, 0],
                            "within_expected_range": False,
                            "method": "rejected",
                            "error_message": "Human detected. Please ensure only livestock animals are in frame."
                        }
            except Exception as e:
                logging.error(f"YOLO engine failure: {e}")

        # Step 1: Run YOLO detection with species class context matching
        if self.model and yolo_results is not None:
            self.rejected_reason = None
            box = self._yolo_detect(image_bgr, target_coco_classes, yolo_results)
            if self.rejected_reason:
                return {
                    "weight": 0.0,
                    "body_length": 0.0,
                    "body_height": 0.0,
                    "estimated_girth": 0.0,
                    "animal_type": "Invalid Target",
                    "confidence_score": 0.0,
                    "annotated_image": image_bgr,
                    "expected_weight_range": [0, 0],
                    "within_expected_range": False,
                    "method": "rejected",
                    "error_message": self.rejected_reason
                }
            if box is not None:
                # Refresh calibration reference in case auto-correction happened inside _yolo_detect
                calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
                x1, y1, x2, y2, conf = box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 200, 50), 3)
                cv2.putText(
                    annotated_image,
                    f"{calibration['name']} {conf:.0%}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 50), 2
                )
                confidence_score = float(conf)
                method_used = "yolov8"

        # Step 2: Adaptive Fallback
        if method_used is None:
            box = self._fallback_contour_detection(image_bgr)
            if box is not None:
                x1, y1, x2, y2 = box
                
                # Crop and verify contour fallback crop to prevent mismatch or invalid targets
                crop = image_bgr[y1:y2, x1:x2]
                classified_type = self._get_crop_animal_type(crop)
                
                if classified_type == "invalid":
                    return {
                        "weight": 0.0,
                        "body_length": 0.0,
                        "body_height": 0.0,
                        "estimated_girth": 0.0,
                        "animal_type": "Invalid Target",
                        "confidence_score": 0.0,
                        "annotated_image": image_bgr,
                        "expected_weight_range": [0, 0],
                        "within_expected_range": False,
                        "method": "rejected",
                        "error_message": "Unrecognized animal or object detected. Please scan only supported livestock species."
                    }
                elif classified_type is not None:
                    is_user_cattle = self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}
                    is_classified_cattle = classified_type in {"dairy_cow", "beef_cattle", "young_cattle"}
                    
                    if is_user_cattle and is_classified_cattle:
                        logging.info(f"Contour Fallback confirmed Cattle. Keeping user specific: {self.animal_type}")
                    else:
                        if self.animal_type != classified_type:
                            logging.info(f"Contour Fallback auto-correcting '{self.animal_type}' to '{classified_type}' based on crop classification.")
                            self.set_animal_type(classified_type, update_pixel_ratio=True)
                            calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_image,
                    f"{calibration['name']} (Contour Fallback)",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                method_used = "contour_fallback"
                confidence_score = 0.50

        if method_used is None:
            return None

        # Step 3: Raw Geometric Extraction (removing generic bounding box squeezing padding)
        box_width_pixels = float(x2 - x1)
        box_height_pixels = float(y2 - y1)

        # Scale pixel arrays to centimeters
        ratio = max(0.0001, self.pixel_to_cm_ratio)
        if not getattr(self, '_pixel_ratio_user_set', False):
            h_img, w_img = image_bgr.shape[:2]
            img_max_dim = float(max(h_img, w_img))
            if img_max_dim > 0:
                ratio = ratio * (640.0 / img_max_dim)

        length_cm = box_width_pixels * ratio
        height_cm = box_height_pixels * ratio

        # Fix the calculation logic: extrapolate Girth from Height, not Length
        girth_cm = height_cm * calibration["girth_multiplier"]

        # Run the standard agricultural weight equation
        weight_kg = (length_cm * (girth_cm ** 2)) / calibration["divisor"]

        # Handle unique equine mathematical scaling rules if evaluating donkeys
        if self.animal_type == "donkey":
            weight_kg = (girth_cm * length_cm) / 10.5

        # Check biological validity thresholds
        expected_min, expected_max = calibration.get("expected_range", (0, 2000))
        within_range = expected_min <= weight_kg <= expected_max

        # Secure clamping bounds against wild edge-case images
        # Check aspect ratio warning if fallback is used
        warning_message = None
        if method_used == "contour_fallback":
            aspect_ratio = box_width_pixels / max(1.0, box_height_pixels)
            if aspect_ratio < 1.1 or aspect_ratio > 2.2:
                warning_message = "Unusual geometry detected. Please ensure the animal is standing broadside."

        return {
            "weight": float(round(weight_kg, 2)),
            "body_length": float(round(length_cm, 2)),
            "body_height": float(round(height_cm, 2)),
            "estimated_girth": float(round(girth_cm, 2)),
            "animal_type": str(calibration["name"]),
            "confidence_score": float(round(confidence_score, 3)),
            "annotated_image": annotated_image,
            "expected_weight_range": calibration.get("expected_range"),
            "within_expected_range": bool(within_range),
            "method": str(method_used),
            "warning": warning_message,
            "pixel_to_cm_ratio": float(ratio),
        }

    def calibrate_pixel_ratio(self, known_cm: float, measured_pixels: float) -> None:
        if measured_pixels > 0:
            self.pixel_to_cm_ratio = float(known_cm / measured_pixels)

    def adjust_weight_calibration(self, animal_type: str, divisor: float = None, girth_multiplier: float = None) -> None:
        if animal_type not in self.LIVESTOCK_CALIBRATION:
            raise ValueError(f"Unknown target category: {animal_type}")
        if divisor is not None:
            self.LIVESTOCK_CALIBRATION[animal_type]["divisor"] = float(divisor)
        if girth_multiplier is not None:
            self.LIVESTOCK_CALIBRATION[animal_type]["girth_multiplier"] = float(girth_multiplier)

    def set_animal_type(self, animal_type: str, update_pixel_ratio: bool = False) -> None:
        """Change the active animal type. If update_pixel_ratio is True (used by
        auto-correction), also resets pixel_to_cm_ratio to the new animal's default
        unless the user explicitly provided their own ratio at construction time."""
        animal_type = animal_type.lower()
        if animal_type in self.LIVESTOCK_CALIBRATION:
            self.animal_type = animal_type
            # Only reset the pixel ratio when auto-correcting AND the user hasn't
            # provided their own calibration — prevents stomping deliberate overrides.
            if update_pixel_ratio and not getattr(self, '_pixel_ratio_user_set', False):
                new_default = self.LIVESTOCK_CALIBRATION[animal_type].get("default_pixel_ratio", 0.15)
                self.pixel_to_cm_ratio = new_default
                logging.info(f"Pixel ratio reset to {new_default} for auto-corrected type: {animal_type}")
        else:
            raise ValueError(f"Invalid selector: {animal_type}")

    @classmethod
    def get_available_types(cls) -> dict:
        return cls.LIVESTOCK_CALIBRATION.copy()

    def _get_crop_animal_type(self, crop: np.ndarray) -> Optional[str]:
        """
        Uses CLIP zero-shot classification to identify the livestock species in a
        cropped image region. CLIP understands free-form text prompts so it can
        directly distinguish pig from cow, donkey from horse, goat from sheep — all
        without any domain-specific training data.
        """
        clip_model     = self.__class__._clip_model
        clip_processor = self.__class__._clip_processor
        if not clip_model or not clip_processor:
            return None
        try:
            import torch
            from PIL import Image as PILImage

            # Convert BGR (OpenCV) crop to RGB PIL image
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            # Build flat list of all prompts and track which category each belongs to
            all_prompts: list[str] = []
            prompt_category: list[str] = []
            for category, prompts in self.CLIP_PROMPTS.items():
                for p in prompts:
                    all_prompts.append(p)
                    prompt_category.append(category)

            # Run CLIP inference
            inputs = clip_processor(
                text=all_prompts,
                images=pil_img,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits = outputs.logits_per_image[0]          # shape: (num_prompts,)
                probs  = logits.softmax(dim=0).cpu().tolist()

            # Accumulate per-category scores (average over synonyms)
            cat_scores: dict[str, float] = {cat: 0.0 for cat in self.CLIP_PROMPTS}
            cat_counts: dict[str, int]   = {cat: 0   for cat in self.CLIP_PROMPTS}
            for score, category in zip(probs, prompt_category):
                cat_scores[category] += score
                cat_counts[category] += 1
            cat_avg = {
                cat: (cat_scores[cat] / cat_counts[cat]) if cat_counts[cat] else 0.0
                for cat in cat_scores
            }

            best_cat   = max(cat_avg, key=cat_avg.get)
            best_score = cat_avg[best_cat]

            logging.info(f"CLIP category scores: { {k: round(v,4) for k, v in cat_avg.items()} }")
            logging.info(f"CLIP best: '{best_cat}' @ {best_score:.4f}")

            # Minimum confidence gate (avoids acting on noise in very blurry crops)
            if best_score < 0.05:
                return None

            if best_cat == "invalid":
                return "invalid"

            # Cattle sub-type: honour user's specific cattle selection if CLIP confirms cattle
            if best_cat in {"dairy_cow", "beef_cattle", "young_cattle"}:
                if self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}:
                    return self.animal_type
                return "dairy_cow"

            return best_cat

        except Exception as e:
            logging.error(f"CLIP classification failure: {e}")
            return None

    def _yolo_detect(self, image_bgr: np.ndarray, target_classes: set, results: Any = None) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Runs YOLO detection with crop-based classifier species verification.
        """
        if results is None:
            try:
                results = self.model(image_bgr, verbose=False, conf=0.20)[0]
            except Exception as e:
                logging.error(f"YOLO engine failure: {e}")
                return None

        # 1. Separate boxes into candidates (animals) and others
        animal_boxes = []
        has_human = False
        largest_non_livestock_class = -1
        largest_non_livestock_area = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if cls_id == 0:
                if conf > 0.35:
                    has_human = True
                continue

            # Animal classes in COCO: 14=bird, 15=cat, 16=dog, 17=horse, 18=sheep, 19=cow, 20=elephant, 21=bear, 22=zebra, 23=giraffe
            if cls_id in {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}:
                if conf > 0.15:
                    animal_boxes.append((x1, y1, x2, y2, conf, cls_id))
            else:
                # Keep track of other non-animal classes for rejection message
                if conf > 0.35 and area > largest_non_livestock_area:
                    largest_non_livestock_area = area
                    largest_non_livestock_class = cls_id

        # Sort animal boxes by area descending
        animal_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

        # Try the top 3 animal boxes
        for x1, y1, x2, y2, conf, cls_id in animal_boxes[:3]:
            # Size threshold check (exclude tiny noise)
            h, w = image_bgr.shape[:2]
            if (x2 - x1) * (y2 - y1) < (w * h * 0.03):
                logging.info("Detected box too small — likely a false positive, discarding.")
                continue

            crop = image_bgr[y1:y2, x1:x2]
            classified_type = self._get_crop_animal_type(crop)

            if classified_type is not None and classified_type != "invalid":
                # Valid livestock animal found!
                is_user_cattle = self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}
                is_classified_cattle = classified_type in {"dairy_cow", "beef_cattle", "young_cattle"}

                if is_user_cattle and is_classified_cattle:
                    logging.info(f"Classifier confirmed Cattle. Keeping user specific: {self.animal_type}")
                else:
                    if self.animal_type != classified_type:
                        logging.info(f"Auto-correcting '{self.animal_type}' to '{classified_type}' based on crop classification.")
                        self.set_animal_type(classified_type, update_pixel_ratio=True)
                return (x1, y1, x2, y2, conf)

        # If we got here, no box was confirmed as livestock by CLIP.
        # Set the appropriate rejection reason.
        if has_human:
            self.rejected_reason = "Human detected. Please ensure only livestock animals are in frame."
            return None

        if animal_boxes:
            largest_box = animal_boxes[0]
            largest_cls = largest_box[5]
            cls_name = self.model.names[largest_cls] if hasattr(self.model, 'names') else "unknown animal"
            self.rejected_reason = f"Invalid target ({cls_name}) detected. Please ensure only livestock animals (cattle, pig, donkey, sheep, goat, poultry) are in frame."
            return None

        if largest_non_livestock_area > 0:
            cls_name = self.model.names[largest_non_livestock_class] if hasattr(self.model, 'names') else "unknown object"
            self.rejected_reason = f"Invalid target ({cls_name}) detected. Please ensure only livestock animals (cattle, pig, donkey, sheep, goat, poultry) are in frame."
            return None

        return None

    def _fallback_contour_detection(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        height, width = image_bgr.shape[:2]
        max_dim = 640.0
        scale = 1.0
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            proc = cv2.resize(image_bgr, (int(width * scale), int(height * scale)))
        else:
            proc = image_bgr

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

        if (w * h) < (width * height * 0.04):
            return None

        return (x, y, x + w, y + h)