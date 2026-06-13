from __future__ import annotations

import cv2
import numpy as np
import os
import json
import logging
from typing import Any, Dict, Optional, Tuple

class AnimalProcessor:
    """
    Optimised Processor for estimating livestock live weight using YOLOv8 bounding boxes.
    Addresses geometric variations across cows, sheep, poultry, donkeys, goats, and pigs.

    v2.0 — Adds:
      • Google Gemini 1.5 Flash enrichment layer (breed, body condition score, health notes)
      • Body Condition Score (BCS) weight correction factor
      • Hard species weight caps (cattle ≤ 800 kg, etc.)
      • Smarter pixel-to-cm ratio calibration (distance-aware scaling)
      • Photo-taking guidance returned per request
      • AI attribution in every result
    """

    # Regulated agricultural metric formulas and density bounds
    LIVESTOCK_CALIBRATION = {
        "dairy_cow":    {"divisor": 660.0,  "girth_multiplier": 0.84, "name": "Dairy Cow",          "expected_range": [250, 800],  "coco_classes": {19}, "default_pixel_ratio": 0.15},
        "beef_cattle":  {"divisor": 600.0,  "girth_multiplier": 0.88, "name": "Beef Cattle",        "expected_range": [300, 800],  "coco_classes": {19}, "default_pixel_ratio": 0.15},
        "young_cattle": {"divisor": 640.0,  "girth_multiplier": 0.82, "name": "Young Cattle/Calf",  "expected_range": [50,  300],  "coco_classes": {19}, "default_pixel_ratio": 0.10},
        "pig":          {"divisor": 400.0,  "girth_multiplier": 0.92, "name": "Pig",                 "expected_range": [5,   350],  "coco_classes": {18, 19}, "default_pixel_ratio": 0.10},
        "goat":         {"divisor": 300.0,  "girth_multiplier": 0.78, "name": "Goat",                "expected_range": [8,   120],  "coco_classes": {18}, "default_pixel_ratio": 0.08},
        "sheep":        {"divisor": 300.0,  "girth_multiplier": 0.80, "name": "Sheep",               "expected_range": [10,  160],  "coco_classes": {18}, "default_pixel_ratio": 0.08},
        "donkey":       {"divisor": 480.0,  "girth_multiplier": 0.82, "name": "Donkey",              "expected_range": [80,  450],  "coco_classes": {17}, "default_pixel_ratio": 0.12},
        "poultry":      {"divisor": 1200.0, "girth_multiplier": 0.55, "name": "Poultry",             "expected_range": [0.5, 12],   "coco_classes": {14}, "default_pixel_ratio": 0.04},
    }

    # Hard weight caps — the AI will NEVER output above these regardless of formula output.
    # Based on world-record and biologically-plausible maximums.
    WEIGHT_HARD_CAPS = {
        "dairy_cow":    800,   # Largest dairy breeds (Holstein) ~700 kg
        "beef_cattle":  800,   # Largest beef breeds (Charolais) ~800 kg
        "young_cattle": 300,   # Calves up to ~300 kg before becoming adult
        "pig":          350,   # Commercial hogs 100–250 kg typical
        "goat":         120,   # Largest breeds (Boer) ~120 kg
        "sheep":        160,   # Largest breeds (Suffolk) ~160 kg
        "donkey":       450,   # Largest donkeys ~450 kg
        "poultry":      12,    # Turkey max ~12 kg, chickens ~4 kg
    }

    # Body Condition Score → weight correction multiplier
    # BCS is a standardised veterinary scoring system (1–5 or thin/fair/good/excellent).
    # A thin animal has less muscle mass than its frame suggests; an excellent-condition
    # animal is well-muscled and heavier than a thin animal of the same frame dimensions.
    BCS_CORRECTION = {
        "thin":      0.88,   # Animal is underweight for its frame
        "fair":      1.00,   # Baseline — formula is accurate at "fair" condition
        "good":      1.10,   # Well-muscled, moderate fat cover
        "excellent": 1.18,   # Peak condition, maximum healthy muscle + fat
        "unknown":   1.00,   # No Gemini enrichment — no correction applied
    }

    # Universally permitted COCO detection IDs: 14=bird, 17=horse, 18=sheep, 19=cow
    YOLO_LIVESTOCK_CLASSES = {14, 17, 18, 19}

    COCO_GROUPS = {
        14: "avian",
        17: "equine",
        18: "small_quad",
        19: "large_quad",
    }
    ANIMAL_GROUP = {
        "dairy_cow":    "large_quad",
        "beef_cattle":  "large_quad",
        "young_cattle": "large_quad",
        "pig":          "large_quad",
        "goat":         "small_quad",
        "sheep":        "small_quad",
        "donkey":       "equine",
        "poultry":      "avian",
    }
    COCO_CLASS_DEFAULT = {
        14: "poultry",
        17: "donkey",
        18: "sheep",
        19: "dairy_cow",
    }

    # Photo-taking guidance — returned in every API response so the frontend
    # can display it as a toast or modal to help users get accurate readings.
    PHOTO_GUIDANCE = {
        "steps": [
            "📐  Stand exactly 2–3 metres (6–10 feet) away from the animal.",
            "🐄  Capture a FULL SIDE PROFILE — head to tail, all four legs visible.",
            "📷  Hold the camera at the animal's mid-body height (not above or below).",
            "☀️  Use good, even lighting — avoid harsh shadows across the body.",
            "🔲  Place an A4 sheet of paper (29.7 × 21 cm) flat on the ground beside the animal.",
            "✋  Keep the animal standing still — a moving animal blurs measurements.",
            "🚫  Remove other animals, people, or vehicles from the background.",
        ],
        "reference_objects": [
            {"name": "A4 Paper (recommended)", "width_cm": 21.0,  "height_cm": 29.7,  "note": "Place flat on ground beside animal"},
            {"name": "Credit Card",             "width_cm": 8.56,  "height_cm": 5.40,  "note": "Tape to fence rail beside animal"},
            {"name": "Standard Ruler",          "width_cm": 30.0,  "height_cm": 3.0,   "note": "Hold vertically against animal's shoulder"},
            {"name": "Fence Rail (typical)",    "width_cm": None,  "height_cm": 120.0, "note": "Standard cattle fence rail is ~1.2 m high"},
        ],
        "distance_guide": "2–3 metres (6–10 feet) gives the best balance of full-body visibility and pixel density.",
        "accuracy_table": [
            {"method": "No reference object, random distance", "typical_error": "±20–30%"},
            {"method": "Correct distance (2–3 m), no reference",  "typical_error": "±10–15%"},
            {"method": "Reference object in frame",               "typical_error": "±5–10%"},
            {"method": "Reference object + correct distance",     "typical_error": "±3–8%"},
        ],
    }

    _yolo_model = None
    _clip_model = None
    _clip_processor = None
    rejected_reason = None

    # ── CLIP text prompts for zero-shot species classification ────────────────
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
        self._pixel_ratio_user_set = pixel_to_cm_ratio is not None
        if pixel_to_cm_ratio is None:
            self.pixel_to_cm_ratio = calibration.get("default_pixel_ratio", 0.15)
        else:
            self.pixel_to_cm_ratio = pixel_to_cm_ratio

        if self.__class__._yolo_model is None:
            try:
                import torch

                original_load = torch.load
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = safe_load

                from ultralytics import YOLO

                current_file_path = os.path.abspath(__file__)
                services_dir = os.path.dirname(current_file_path)
                root_dir = os.path.dirname(services_dir)

                path_options = [
                    os.path.join(root_dir, "models", "yolov8n.pt"),
                    os.path.join(os.getcwd(), "models", "yolov8n.pt"),
                    os.path.join(os.getcwd(), "yolov8n.pt"),
                    "yolov8n.pt"
                ]

                selected_path = None
                for path in path_options:
                    logging.info(f"Checking for weights at target location: {path}")
                    if os.path.exists(path):
                        selected_path = path
                        logging.info(f"SUCCESS: Found weights file at: {path}")
                        break

                if selected_path:
                    self.model = YOLO(selected_path)
                    logging.info(f"YOLOv8 initialized using: {selected_path}")
                else:
                    logging.warning("Local weights not found. Trying default string config.")
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

    # ─────────────────────────────────────────────────────────────────────────
    # Gemini Enrichment Layer
    # ─────────────────────────────────────────────────────────────────────────

    def _gemini_enrich(self, image_bgr: np.ndarray) -> dict:
        """
        Calls Google Gemini 1.5 Flash to enrich a successful scan with:
          - breed: specific breed name (e.g. "Holstein Friesian", "Boer Goat")
          - body_condition: thin / fair / good / excellent
          - body_condition_score: numeric BCS 1–5
          - health_notes: one-sentence veterinary observation
          - gemini_weight_range: Gemini's own sanity estimate for cross-check

        The GEMINI_API_KEY env-var must be set in Hugging Face Space Secrets.
        If the key is absent or the call fails, returns an empty dict (graceful).
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            logging.info("GEMINI_API_KEY not set — skipping Gemini enrichment.")
            return {}

        try:
            import google.generativeai as genai
            from PIL import Image as PILImage

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Convert BGR → RGB → PIL
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            prompt = """You are a professional livestock veterinarian and body-condition scorer.
Analyze this livestock image and return ONLY a valid JSON object with these exact keys:

{
  "breed": "<specific breed name or Mixed/Unknown>",
  "body_condition": "<exactly one of: thin, fair, good, excellent>",
  "body_condition_score": <integer 1 to 5>,
  "health_notes": "<one short sentence about visible health indicators>",
  "estimated_weight_range_kg": [<min_integer>, <max_integer>],
  "photo_quality_note": "<one short sentence on photo quality and positioning>"
}

Body condition scoring guide:
  1 = Emaciated (thin)
  2 = Very thin (thin)
  3 = Moderate (fair)
  4 = Good (good)
  5 = Excellent/Obese (excellent)

Be specific about the breed. Do NOT add any text outside the JSON object."""

            response = model.generate_content([pil_img, prompt])
            raw = response.text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            logging.info(f"Gemini enrichment success: {data}")
            return data

        except json.JSONDecodeError as e:
            logging.warning(f"Gemini returned non-JSON response: {e}")
            return {}
        except Exception as e:
            logging.warning(f"Gemini enrichment failed (non-critical): {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Main Processing Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, image_bgr: np.ndarray) -> Optional[Dict[str, object]]:
        """Executes targeted object extraction and converts pixel space into physical mass."""
        annotated_image = image_bgr.copy()
        method_used = None
        x1, y1, x2, y2 = 0, 0, 0, 0
        confidence_score = 0.0

        calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
        target_coco_classes = calibration["coco_classes"]

        # Run human detection check and obtain YOLO results
        yolo_results = None
        if self.model:
            try:
                yolo_results = self.model(image_bgr, verbose=False, conf=0.20)[0]
                for box in yolo_results.boxes:
                    if int(box.cls[0]) == 0:  # Class 0 = human
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

        # Step 1: YOLO detection with CLIP species verification
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

        # Step 2: Contour fallback
        if method_used is None:
            box = self._fallback_contour_detection(image_bgr)
            if box is not None:
                x1, y1, x2, y2 = box
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
                        logging.info(f"Contour fallback confirmed Cattle. Keeping: {self.animal_type}")
                    else:
                        if self.animal_type != classified_type:
                            logging.info(f"Contour fallback auto-correcting '{self.animal_type}' → '{classified_type}'")
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

        # Step 3: Geometric extraction
        box_width_pixels  = float(x2 - x1)
        box_height_pixels = float(y2 - y1)

        # Distance-aware pixel-to-cm ratio
        ratio = max(0.0001, self.pixel_to_cm_ratio)
        if not getattr(self, '_pixel_ratio_user_set', False):
            h_img, w_img = image_bgr.shape[:2]
            img_max_dim = float(max(h_img, w_img))
            if img_max_dim > 0:
                ratio = ratio * (640.0 / img_max_dim)

        length_cm = box_width_pixels  * ratio
        height_cm = box_height_pixels * ratio
        girth_cm  = height_cm * calibration["girth_multiplier"]

        # Agricultural girth formula (Schoorl formula)
        weight_kg = (length_cm * (girth_cm ** 2)) / calibration["divisor"]

        # Equine formula for donkeys
        if self.animal_type == "donkey":
            weight_kg = (girth_cm * length_cm) / 10.5

        # ── Gemini Enrichment ──────────────────────────────────────────────
        # Crop the bounding box for Gemini so it focuses on the animal only
        crop_for_gemini = image_bgr[max(0, y1):min(image_bgr.shape[0], y2),
                                    max(0, x1):min(image_bgr.shape[1], x2)]
        gemini_data = self._gemini_enrich(crop_for_gemini)

        # ── BCS Weight Correction ──────────────────────────────────────────
        body_condition_raw = gemini_data.get("body_condition", "unknown") or "unknown"
        body_condition     = body_condition_raw.lower().strip()
        if body_condition not in self.BCS_CORRECTION:
            body_condition = "unknown"

        bcs_factor  = self.BCS_CORRECTION[body_condition]
        weight_kg   = weight_kg * bcs_factor

        # ── Cross-check with Gemini weight range ───────────────────────────
        gemini_range = gemini_data.get("estimated_weight_range_kg")
        gemini_cross_check = None
        if gemini_range and isinstance(gemini_range, (list, tuple)) and len(gemini_range) == 2:
            g_min, g_max = float(gemini_range[0]), float(gemini_range[1])
            gemini_mid   = (g_min + g_max) / 2.0
            # Blend: 70% formula + 30% Gemini midpoint (Gemini is rough, formula is geometry-based)
            weight_kg    = weight_kg * 0.70 + gemini_mid * 0.30
            gemini_cross_check = {"min": g_min, "max": g_max, "midpoint": round(gemini_mid, 1)}

        # ── Hard Species Weight Cap (never exceed biological maximum) ──────
        hard_cap  = self.WEIGHT_HARD_CAPS.get(self.animal_type, 800)
        min_floor = calibration["expected_range"][0]
        weight_kg = max(min_floor * 0.5, min(weight_kg, hard_cap))

        # ── AI Attribution ─────────────────────────────────────────────────
        gemini_used = bool(gemini_data)
        ai_attribution = {
            "detection_model":      "YOLOv8n (Ultralytics)",
            "classification_model": "CLIP ViT-B/32 (OpenAI)",
            "enrichment_model":     "Google Gemini 1.5 Flash" if gemini_used else None,
            "weight_formula":       "Schoorl Girth Formula (agricultural standard)",
            "bcs_correction":       f"Body Condition Score adjustment: {body_condition} (×{bcs_factor})" if gemini_used else "No BCS correction (Gemini not available)",
            "gemini_cross_check":   gemini_cross_check,
        }

        # ── Biological range check ─────────────────────────────────────────
        expected_min, expected_max = calibration.get("expected_range", (0, 2000))
        within_range = expected_min <= weight_kg <= expected_max

        # ── Aspect ratio warning for contour fallback ──────────────────────
        warning_message = None
        if method_used == "contour_fallback":
            aspect_ratio = box_width_pixels / max(1.0, box_height_pixels)
            if aspect_ratio < 1.1 or aspect_ratio > 2.2:
                warning_message = "Unusual geometry detected. Please ensure the animal is standing broadside."

        return {
            # Core weight result
            "weight":                  float(round(weight_kg, 2)),
            "body_length":             float(round(length_cm, 2)),
            "body_height":             float(round(height_cm, 2)),
            "estimated_girth":         float(round(girth_cm, 2)),
            "animal_type":             str(calibration["name"]),
            "confidence_score":        float(round(confidence_score, 3)),
            "annotated_image":         annotated_image,
            "expected_weight_range":   calibration.get("expected_range"),
            "within_expected_range":   bool(within_range),
            "method":                  str(method_used),
            "warning":                 warning_message,
            "pixel_to_cm_ratio":       float(ratio),
            # Gemini enrichment fields
            "breed":                   gemini_data.get("breed", "Unknown"),
            "body_condition":          body_condition if gemini_used else "Not assessed",
            "body_condition_score":    gemini_data.get("body_condition_score"),
            "health_notes":            gemini_data.get("health_notes", ""),
            "photo_quality_note":      gemini_data.get("photo_quality_note", ""),
            "gemini_cross_check":      gemini_cross_check,
            # AI attribution
            "ai_attribution":          ai_attribution,
            # Photo guidance (always returned so frontend can show toasts)
            "photo_guidance":          self.PHOTO_GUIDANCE,
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
        cropped image region.
        """
        clip_model     = self.__class__._clip_model
        clip_processor = self.__class__._clip_processor
        if not clip_model or not clip_processor:
            return None
        try:
            import torch
            from PIL import Image as PILImage

            rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            all_prompts: list[str] = []
            prompt_category: list[str] = []
            for category, prompts in self.CLIP_PROMPTS.items():
                for p in prompts:
                    all_prompts.append(p)
                    prompt_category.append(category)

            inputs = clip_processor(
                text=all_prompts,
                images=pil_img,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits  = outputs.logits_per_image[0]
                probs   = logits.softmax(dim=0).cpu().tolist()

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

            logging.info(f"CLIP scores: { {k: round(v,4) for k, v in cat_avg.items()} }")
            logging.info(f"CLIP best: '{best_cat}' @ {best_score:.4f}")

            if best_score < 0.05:
                return None

            if best_cat == "invalid":
                return "invalid"

            if best_cat in {"dairy_cow", "beef_cattle", "young_cattle"}:
                if self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}:
                    return self.animal_type
                return "dairy_cow"

            return best_cat

        except Exception as e:
            logging.error(f"CLIP classification failure: {e}")
            return None

    def _yolo_detect(self, image_bgr: np.ndarray, target_classes: set, results: Any = None) -> Optional[Tuple[int, int, int, int, float]]:
        """Runs YOLO detection with crop-based classifier species verification."""
        if results is None:
            try:
                results = self.model(image_bgr, verbose=False, conf=0.20)[0]
            except Exception as e:
                logging.error(f"YOLO engine failure: {e}")
                return None

        animal_boxes = []
        has_human    = False
        largest_non_livestock_class = -1
        largest_non_livestock_area  = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area   = (x2 - x1) * (y2 - y1)

            if cls_id == 0:
                if conf > 0.35:
                    has_human = True
                continue

            if cls_id in {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}:
                if conf > 0.15:
                    animal_boxes.append((x1, y1, x2, y2, conf, cls_id))
            else:
                if conf > 0.35 and area > largest_non_livestock_area:
                    largest_non_livestock_area  = area
                    largest_non_livestock_class = cls_id

        animal_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

        for x1, y1, x2, y2, conf, cls_id in animal_boxes[:3]:
            h, w = image_bgr.shape[:2]
            if (x2 - x1) * (y2 - y1) < (w * h * 0.03):
                logging.info("Detected box too small — likely a false positive, discarding.")
                continue

            crop = image_bgr[y1:y2, x1:x2]
            classified_type = self._get_crop_animal_type(crop)

            if classified_type is not None and classified_type != "invalid":
                is_user_cattle       = self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}
                is_classified_cattle = classified_type in {"dairy_cow", "beef_cattle", "young_cattle"}

                if is_user_cattle and is_classified_cattle:
                    logging.info(f"Classifier confirmed Cattle. Keeping: {self.animal_type}")
                else:
                    if self.animal_type != classified_type:
                        logging.info(f"Auto-correcting '{self.animal_type}' → '{classified_type}'")
                        self.set_animal_type(classified_type, update_pixel_ratio=True)
                return (x1, y1, x2, y2, conf)

        if has_human:
            self.rejected_reason = "Human detected. Please ensure only livestock animals are in frame."
            return None

        if animal_boxes:
            largest_cls  = animal_boxes[0][5]
            cls_name     = self.model.names[largest_cls] if hasattr(self.model, 'names') else "unknown animal"
            self.rejected_reason = (
                f"Invalid target ({cls_name}) detected. "
                "Please ensure only livestock animals (cattle, pig, donkey, sheep, goat, poultry) are in frame."
            )
            return None

        if largest_non_livestock_area > 0:
            cls_name = self.model.names[largest_non_livestock_class] if hasattr(self.model, 'names') else "unknown object"
            self.rejected_reason = (
                f"Invalid target ({cls_name}) detected. "
                "Please ensure only livestock animals (cattle, pig, donkey, sheep, goat, poultry) are in frame."
            )
            return None

        return None

    def _fallback_contour_detection(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        height, width = image_bgr.shape[:2]
        max_dim = 640.0
        scale   = 1.0
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            proc  = cv2.resize(image_bgr, (int(width * scale), int(height * scale)))
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