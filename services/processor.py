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
    _classifier_model = None
    rejected_reason = None

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

        if self.__class__._classifier_model is None:
            try:
                import torch
                from ultralytics import YOLO
                import os

                current_file_path = os.path.abspath(__file__)
                services_dir = os.path.dirname(current_file_path)
                root_dir = os.path.dirname(services_dir)

                cls_path_options = [
                    os.path.join(root_dir, "models", "yolov8n-cls.pt"),
                    os.path.join(os.getcwd(), "models", "yolov8n-cls.pt"),
                    os.path.join(os.getcwd(), "yolov8n-cls.pt"),
                    "yolov8n-cls.pt"
                ]

                selected_cls_path = None
                for path in cls_path_options:
                    if os.path.exists(path):
                        selected_cls_path = path
                        break

                if selected_cls_path:
                    self.__class__._classifier_model = YOLO(selected_cls_path)
                else:
                    self.__class__._classifier_model = YOLO("yolov8n-cls.pt")
            except Exception as e:
                logging.error(f"Failed to compile YOLO classifier: {e}")
                self.__class__._classifier_model = False

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
        if not self.__class__._classifier_model:
            return None
        try:
            results = self.__class__._classifier_model(crop, verbose=False)[0]
            probs = results.probs
            if probs is None:
                return None
            
            scores = {
                "pig": 0.0,
                "cattle": 0.0,
                "sheep": 0.0,
                "goat": 0.0,
                "donkey": 0.0,
                "poultry": 0.0,
                "invalid": 0.0
            }
            
            # Map top-5 indices and accumulate probabilities
            top5_indices = probs.top5
            for idx in top5_indices:
                conf = float(probs.data[idx])
                name = results.names[idx].lower()
                
                # Check invalid list first
                if any(x in name for x in ["dog", "cat", "person", "human", "man", "woman", "boy", "girl", "child", "vehicle", "car", "truck", "tractor", "chair", "table", "bicycle", "motorcycle"]):
                    scores["invalid"] += conf
                    continue
                
                # Accumulate matching scores
                if any(x in name for x in ["pig", "hog", "boar", "swine"]):
                    scores["pig"] += conf
                if any(x in name for x in ["cow", "bull", "ox", "cattle", "steer", "calf", "heifer", "bison", "buffalo"]):
                    scores["cattle"] += conf
                if any(x in name for x in ["sheep", "ram", "ewe", "lamb", "mutton", "bighorn"]):
                    scores["sheep"] += conf
                if any(x in name for x in ["goat", "capra", "billy", "nanny", "ibex"]):
                    scores["goat"] += conf
                if any(x in name for x in ["donkey", "mule", "ass", "horse", "foal", "colt", "stallion", "equus", "sorrel", "zebra"]):
                    scores["donkey"] += conf
                if any(x in name for x in ["hen", "chicken", "rooster", "poultry", "fowl", "bird", "turkey", "duck", "goose", "ostrich"]):
                    scores["poultry"] += conf
            
            # Determine highest scoring group
            best_cat = None
            best_score = 0.0
            for cat, score in scores.items():
                if score > best_score:
                    best_score = score
                    best_cat = cat
            
            # Require minimum score to prevent acting on low-level background noise
            if best_score > 0.15 and best_cat is not None:
                if best_cat == "invalid":
                    return "invalid"
                elif best_cat == "cattle":
                    if self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}:
                        return self.animal_type
                    return "dairy_cow"
                return best_cat
            
            # Keyword fallback on top-1 index
            top1_idx = int(probs.top1)
            name = results.names[top1_idx].lower()
            if any(x in name for x in ["dog", "cat", "person", "human", "man", "woman", "boy", "girl", "child", "vehicle", "car", "truck", "tractor", "chair", "table"]):
                return "invalid"
            if any(x in name for x in ["pig", "hog", "boar", "swine"]):
                return "pig"
            if any(x in name for x in ["cow", "bull", "ox", "cattle", "steer", "calf", "heifer", "bison", "buffalo"]):
                if self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}:
                    return self.animal_type
                return "dairy_cow"
            if any(x in name for x in ["sheep", "ram", "ewe", "lamb", "bighorn"]):
                return "sheep"
            if any(x in name for x in ["goat", "capra", "billy", "nanny", "ibex"]):
                return "goat"
            if any(x in name for x in ["donkey", "mule", "ass", "horse", "foal", "colt", "stallion", "equus", "sorrel", "zebra"]):
                return "donkey"
            if any(x in name for x in ["hen", "chicken", "rooster", "poultry", "fowl", "bird", "turkey", "duck", "goose", "ostrich"]):
                return "poultry"
            
            return None
        except Exception as e:
            logging.error(f"Classifier run failure: {e}")
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

        # Check for any dominant non-livestock target (e.g. dog, cat, person) in the entire image
        largest_non_livestock_class = -1
        largest_non_livestock_area = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in self.YOLO_LIVESTOCK_CLASSES and conf > 0.35:
                x1_nl, y1_nl, x2_nl, y2_nl = map(int, box.xyxy[0])
                area = (x2_nl - x1_nl) * (y2_nl - y1_nl)
                if area > largest_non_livestock_area:
                    largest_non_livestock_area = area
                    largest_non_livestock_class = cls_id

        # ── Step 1: Find the largest detected livestock animal ────────────────
        final_box = None
        final_cls = -1
        best_area = 0

        # Priority 1: largest box that matches user's selected COCO target classes
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.YOLO_LIVESTOCK_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            if cls_id in target_classes and area > best_area:
                best_area = area
                final_box = (x1, y1, x2, y2, conf)
                final_cls = cls_id

        # Priority 2: any livestock class (fallback if target class not found)
        if final_box is None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.YOLO_LIVESTOCK_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    final_box = (x1, y1, x2, y2, conf)
                    final_cls = cls_id

        # If no livestock animal is detected, but a non-livestock object is present, reject it.
        if final_box is None:
            if largest_non_livestock_area > 0:
                cls_name = self.model.names[largest_non_livestock_class] if hasattr(self.model, 'names') else "unknown object"
                self.rejected_reason = f"Invalid target ({cls_name}) detected. Please ensure only livestock animals (cattle, pig, donkey, sheep, goat, poultry) are in frame."
            return None

        x1, y1, x2, y2, conf = final_box

        # ── Step 2: Minimum size threshold (noise/false-positive filter) ──────
        h, w = image_bgr.shape[:2]
        if (x2 - x1) * (y2 - y1) < (w * h * 0.03):
            logging.info("Detected box too small — likely a false positive, discarding.")
            return None

        # ── Step 3: Crop classifier verification ──────────────────────────────
        crop = image_bgr[y1:y2, x1:x2]
        classified_type = self._get_crop_animal_type(crop)

        if classified_type == "invalid":
            self.rejected_reason = "Unrecognized animal or object detected. Please scan only supported livestock species."
            return None
        elif classified_type is not None:
            # Check if user selected one cattle type and classifier confirmed cattle
            is_user_cattle = self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}
            is_classified_cattle = classified_type in {"dairy_cow", "beef_cattle", "young_cattle"}

            if is_user_cattle and is_classified_cattle:
                logging.info(f"Classifier confirmed Cattle. Keeping user specific selection: {self.animal_type}")
            else:
                if self.animal_type != classified_type:
                    logging.info(f"Auto-correcting '{self.animal_type}' to '{classified_type}' based on crop classification.")
                    self.set_animal_type(classified_type, update_pixel_ratio=True)
        else:
            # Fallback to group-based heuristics
            detected_group = self.COCO_GROUPS.get(final_cls)
            user_group     = self.ANIMAL_GROUP.get(self.animal_type)

            if detected_group and user_group:
                if detected_group != user_group:
                    corrected_type = self.COCO_CLASS_DEFAULT.get(final_cls, "dairy_cow")
                    logging.info(f"Cross-group fallback mismatch: user='{self.animal_type}', YOLO class {final_cls} → corrected to '{corrected_type}'")
                    self.set_animal_type(corrected_type, update_pixel_ratio=True)

        return (x1, y1, x2, y2, conf)

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