from __future__ import annotations

import base64
import cv2
import json
import logging
import numpy as np
import os
import ssl
import urllib.request
from typing import Any, Dict, Optional, Tuple


class AnimalProcessor:
    """
    Livestock weight estimation using YOLOv8 bounding boxes + CLIP species
    classification + Google Gemini 1.5 Flash enrichment.

    v2.0 additions:
      * Direct REST call to Gemini 1.5 Flash (no SDK needed, pure stdlib)
      * Body Condition Score (BCS) weight correction factor
      * Hard species weight caps (cattle max 800 kg, etc.)
      * Photo-taking guidance returned in every response
      * AI attribution in every result
    """

    # ── Calibration constants ─────────────────────────────────────────────────
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

    # Hard weight caps — never exceed biologically plausible maximums
    WEIGHT_HARD_CAPS = {
        "dairy_cow":    800,
        "beef_cattle":  800,
        "young_cattle": 300,
        "pig":          350,
        "goat":         120,
        "sheep":        160,
        "donkey":       450,
        "poultry":      12,
    }

    # Body Condition Score correction multipliers
    BCS_CORRECTION = {
        "thin":      0.88,
        "fair":      1.00,
        "good":      1.10,
        "excellent": 1.18,
        "unknown":   1.00,
    }

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

    # Photo guidance returned in every successful scan response
    PHOTO_GUIDANCE = {
        "steps": [
            "Stand exactly 2-3 metres (6-10 feet) away from the animal.",
            "Capture a FULL SIDE PROFILE - head to tail, all four legs visible.",
            "Hold the camera at the animal's mid-body height (not above or below).",
            "Use good, even lighting - avoid harsh shadows across the body.",
            "Place an A4 sheet of paper (29.7 x 21 cm) flat on the ground beside the animal.",
            "Keep the animal standing still - a moving animal blurs measurements.",
            "Remove other animals, people, or vehicles from the background.",
        ],
        "reference_objects": [
            {"name": "A4 Paper (recommended)", "width_cm": 21.0,  "height_cm": 29.7,  "note": "Place flat on ground beside animal"},
            {"name": "Credit Card",             "width_cm": 8.56,  "height_cm": 5.40,  "note": "Tape to fence rail beside animal"},
            {"name": "Standard Ruler",          "width_cm": 30.0,  "height_cm": 3.0,   "note": "Hold vertically against animal's shoulder"},
            {"name": "Fence Rail (typical)",    "width_cm": None,  "height_cm": 120.0, "note": "Standard cattle fence rail is ~1.2 m high"},
        ],
        "distance_guide": "2-3 metres (6-10 feet) gives the best balance of full-body visibility and pixel density.",
        "accuracy_table": [
            {"method": "No reference object, random distance",    "typical_error": "+-20-30%"},
            {"method": "Correct distance (2-3 m), no reference",  "typical_error": "+-10-15%"},
            {"method": "Reference object in frame",               "typical_error": "+-5-10%"},
            {"method": "Reference object + correct distance",     "typical_error": "+-3-8%"},
        ],
    }

    # CLIP text prompts for zero-shot species classification
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

    _yolo_model = None
    _clip_model = None
    _clip_processor = None
    rejected_reason = None

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
                root_dir = os.path.dirname(os.path.dirname(current_file_path))

                path_options = [
                    os.path.join(root_dir, "models", "yolov8n.pt"),
                    os.path.join(os.getcwd(), "models", "yolov8n.pt"),
                    os.path.join(os.getcwd(), "yolov8n.pt"),
                    "yolov8n.pt"
                ]

                selected_path = None
                for path in path_options:
                    if os.path.exists(path):
                        selected_path = path
                        logging.info(f"Found YOLO weights at: {path}")
                        break

                self.model = YOLO(selected_path or "yolov8n.pt")
                logging.info(f"YOLOv8 initialized using: {selected_path or 'yolov8n.pt'}")
                self.__class__._yolo_model = self.model

            except Exception as e:
                logging.error(f"Failed to load YOLO model: {e}")
                self.__class__._yolo_model = False

        self.model = self.__class__._yolo_model if self.__class__._yolo_model else None

        if self.__class__._clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                logging.info("Loading CLIP model...")
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
    # Gemini Enrichment Layer (pure stdlib — no SDK dependency)
    # ─────────────────────────────────────────────────────────────────────────

    def _gemini_enrich(self, image_bgr: np.ndarray) -> dict:
        """
        Calls Google Gemini 1.5 Flash via direct REST API to enrich a scan with:
          - breed, body_condition (thin/fair/good/excellent), body_condition_score,
            health_notes, estimated_weight_range_kg, photo_quality_note.

        Uses only stdlib (base64, urllib, json, ssl) — zero extra packages.
        GEMINI_API_KEY must be set as a Hugging Face Space Secret.
        Returns {} gracefully on any failure so the main pipeline always continues.
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            logging.info("GEMINI_API_KEY not set — skipping Gemini enrichment.")
            return {}

        try:
            # Encode crop as JPEG base64 using cv2 only (PIL not required here)
            success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logging.warning("Gemini: cv2.imencode failed — skipping.")
                return {}
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            prompt = (
                "You are a professional livestock veterinarian and body-condition scorer.\n"
                "Analyze this livestock image and return ONLY a valid JSON object with these exact keys:\n\n"
                "{\n"
                '  "breed": "<specific breed name or Mixed/Unknown>",\n'
                '  "body_condition": "<exactly one of: thin, fair, good, excellent>",\n'
                '  "body_condition_score": <integer 1 to 5>,\n'
                '  "health_notes": "<one short sentence about visible health indicators>",\n'
                '  "estimated_weight_range_kg": [<min_integer>, <max_integer>],\n'
                '  "photo_quality_note": "<one short sentence on photo quality and positioning>"\n'
                "}\n\n"
                "Body condition scoring guide:\n"
                "  1 = Emaciated (thin)\n"
                "  2 = Very thin (thin)\n"
                "  3 = Moderate (fair)\n"
                "  4 = Good (good)\n"
                "  5 = Excellent/Obese (excellent)\n\n"
                "Be specific about the breed. Return ONLY the JSON object, no other text."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                    "maxOutputTokens": 300
                }
            }

            url = (
                "https://generativelanguage.googleapis.com/v1/models/"
                f"gemini-2.0-flash-lite:generateContent?key={api_key}"
            )
            req_body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_body, method="POST")
            req.add_header("Content-Type", "application/json")

            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, context=ctx, timeout=25) as resp:
                response = json.loads(resp.read().decode("utf-8"))

            # Extract the generated text
            raw = response["candidates"][0]["content"]["parts"][0]["text"].strip()

            # Strip markdown code fences if Gemini wraps the JSON
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            logging.info(f"Gemini enrichment success: breed={data.get('breed')}, "
                         f"condition={data.get('body_condition')}, "
                         f"range={data.get('estimated_weight_range_kg')}")
            return data

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logging.warning(f"Gemini HTTP {e.code}: {err_body[:400]}")
            return {}
        except json.JSONDecodeError as e:
            logging.warning(f"Gemini returned non-JSON: {e}")
            return {}
        except Exception as e:
            logging.warning(f"Gemini enrichment failed ({type(e).__name__}): {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Main Processing Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, image_bgr: np.ndarray) -> Optional[Dict[str, object]]:
        """Executes targeted object extraction and converts pixel space into physical mass."""
        annotated_image  = image_bgr.copy()
        method_used      = None
        x1, y1, x2, y2  = 0, 0, 0, 0
        confidence_score = 0.0

        calibration          = self.LIVESTOCK_CALIBRATION[self.animal_type]
        target_coco_classes  = calibration["coco_classes"]

        # ── Human detection + YOLO inference ──────────────────────────────────
        yolo_results = None
        if self.model:
            try:
                yolo_results = self.model(image_bgr, verbose=False, conf=0.20)[0]
                for box in yolo_results.boxes:
                    if int(box.cls[0]) == 0:  # class 0 = person
                        return {
                            "weight": 0.0, "body_length": 0.0, "body_height": 0.0,
                            "estimated_girth": 0.0, "animal_type": "Invalid Target",
                            "confidence_score": 0.0, "annotated_image": image_bgr,
                            "expected_weight_range": [0, 0], "within_expected_range": False,
                            "method": "rejected",
                            "error_message": "Human detected. Please ensure only livestock animals are in frame."
                        }
            except Exception as e:
                logging.error(f"YOLO inference error: {e}")

        # ── Step 1: YOLO detection with CLIP species verification ──────────────
        if self.model and yolo_results is not None:
            self.rejected_reason = None
            box = self._yolo_detect(image_bgr, target_coco_classes, yolo_results)
            if self.rejected_reason:
                return {
                    "weight": 0.0, "body_length": 0.0, "body_height": 0.0,
                    "estimated_girth": 0.0, "animal_type": "Invalid Target",
                    "confidence_score": 0.0, "annotated_image": image_bgr,
                    "expected_weight_range": [0, 0], "within_expected_range": False,
                    "method": "rejected", "error_message": self.rejected_reason
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

        # ── Step 2: Contour fallback ───────────────────────────────────────────
        if method_used is None:
            box = self._fallback_contour_detection(image_bgr)
            if box is not None:
                x1, y1, x2, y2 = box
                crop = image_bgr[y1:y2, x1:x2]
                classified_type = self._get_crop_animal_type(crop)

                if classified_type == "invalid":
                    return {
                        "weight": 0.0, "body_length": 0.0, "body_height": 0.0,
                        "estimated_girth": 0.0, "animal_type": "Invalid Target",
                        "confidence_score": 0.0, "annotated_image": image_bgr,
                        "expected_weight_range": [0, 0], "within_expected_range": False,
                        "method": "rejected",
                        "error_message": "Unrecognized animal detected. Please scan only supported livestock species."
                    }
                elif classified_type is not None:
                    is_user_cattle       = self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}
                    is_classified_cattle = classified_type  in {"dairy_cow", "beef_cattle", "young_cattle"}
                    if not (is_user_cattle and is_classified_cattle):
                        if self.animal_type != classified_type:
                            logging.info(f"Contour fallback: auto-correcting '{self.animal_type}' -> '{classified_type}'")
                            self.set_animal_type(classified_type, update_pixel_ratio=True)
                            calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_image,
                    f"{calibration['name']} (Contour Fallback)",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                method_used      = "contour_fallback"
                confidence_score = 0.50

        if method_used is None:
            return None

        # ── Step 3: Geometric extraction ──────────────────────────────────────
        box_width_pixels  = float(x2 - x1)
        box_height_pixels = float(y2 - y1)

        ratio = max(0.0001, self.pixel_to_cm_ratio)
        if not getattr(self, '_pixel_ratio_user_set', False):
            h_img, w_img = image_bgr.shape[:2]
            img_max_dim  = float(max(h_img, w_img))
            if img_max_dim > 0:
                ratio = ratio * (640.0 / img_max_dim)

        length_cm = box_width_pixels  * ratio
        height_cm = box_height_pixels * ratio
        girth_cm  = height_cm * calibration["girth_multiplier"]

        # Schoorl girth formula (agricultural standard)
        weight_kg = (length_cm * (girth_cm ** 2)) / calibration["divisor"]

        # Equine formula for donkeys
        if self.animal_type == "donkey":
            weight_kg = (girth_cm * length_cm) / 10.5

        # ── Step 4: Gemini Enrichment ──────────────────────────────────────────
        crop_y1 = max(0, y1)
        crop_y2 = min(image_bgr.shape[0], y2)
        crop_x1 = max(0, x1)
        crop_x2 = min(image_bgr.shape[1], x2)
        crop_for_gemini = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        gemini_data = self._gemini_enrich(crop_for_gemini)

        # ── Step 5: BCS Weight Correction ─────────────────────────────────────
        body_condition_raw = (gemini_data.get("body_condition") or "unknown").lower().strip()
        body_condition     = body_condition_raw if body_condition_raw in self.BCS_CORRECTION else "unknown"
        bcs_factor         = self.BCS_CORRECTION[body_condition]
        weight_kg          = weight_kg * bcs_factor

        # ── Step 6: Gemini weight cross-check blend ────────────────────────────
        gemini_range = gemini_data.get("estimated_weight_range_kg")
        gemini_cross_check = None
        if gemini_range and isinstance(gemini_range, (list, tuple)) and len(gemini_range) == 2:
            g_min, g_max = float(gemini_range[0]), float(gemini_range[1])
            gemini_mid   = (g_min + g_max) / 2.0
            # 70% formula (geometry-based) + 30% Gemini midpoint (vision-based sanity check)
            weight_kg    = weight_kg * 0.70 + gemini_mid * 0.30
            gemini_cross_check = {"min": g_min, "max": g_max, "midpoint": round(gemini_mid, 1)}

        # ── Step 7: Hard species weight cap ───────────────────────────────────
        hard_cap  = self.WEIGHT_HARD_CAPS.get(self.animal_type, 800)
        min_floor = calibration["expected_range"][0] * 0.5
        weight_kg = max(min_floor, min(weight_kg, hard_cap))

        # ── AI attribution ─────────────────────────────────────────────────────
        gemini_used    = bool(gemini_data)
        ai_attribution = {
            "detection_model":      "YOLOv8n (Ultralytics)",
            "classification_model": "CLIP ViT-B/32 (OpenAI)",
            "enrichment_model":     "Google Gemini 2.0 Flash Lite" if gemini_used else None,
            "weight_formula":       "Schoorl Girth Formula (agricultural standard)",
            "bcs_correction":       (
                f"BCS adjustment: {body_condition} (x{bcs_factor})"
                if gemini_used else "No BCS correction (Gemini key not configured)"
            ),
            "gemini_cross_check": gemini_cross_check,
        }

        # ── Biological range check ─────────────────────────────────────────────
        expected_min, expected_max = calibration.get("expected_range", (0, 2000))
        within_range = expected_min <= weight_kg <= expected_max

        # ── Aspect ratio warning for contour fallback ──────────────────────────
        warning_message = None
        if method_used == "contour_fallback":
            aspect_ratio = box_width_pixels / max(1.0, box_height_pixels)
            if aspect_ratio < 1.1 or aspect_ratio > 2.2:
                warning_message = "Unusual geometry detected. Please ensure the animal is standing broadside."

        return {
            "weight":                 float(round(weight_kg, 2)),
            "body_length":            float(round(length_cm, 2)),
            "body_height":            float(round(height_cm, 2)),
            "estimated_girth":        float(round(girth_cm, 2)),
            "animal_type":            str(calibration["name"]),
            "confidence_score":       float(round(confidence_score, 3)),
            "annotated_image":        annotated_image,
            "expected_weight_range":  calibration.get("expected_range"),
            "within_expected_range":  bool(within_range),
            "method":                 str(method_used),
            "warning":                warning_message,
            "pixel_to_cm_ratio":      float(ratio),
            # Gemini enrichment
            "breed":                  gemini_data.get("breed", "Unknown"),
            "body_condition":         body_condition if gemini_used else "Not assessed",
            "body_condition_score":   gemini_data.get("body_condition_score"),
            "health_notes":           gemini_data.get("health_notes", ""),
            "photo_quality_note":     gemini_data.get("photo_quality_note", ""),
            "gemini_cross_check":     gemini_cross_check,
            # AI attribution
            "ai_attribution":         ai_attribution,
            # Photo guidance (for frontend toasts)
            "photo_guidance":         self.PHOTO_GUIDANCE,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration helpers
    # ─────────────────────────────────────────────────────────────────────────

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
        """Change the active animal type. Optionally resets pixel ratio to species default."""
        animal_type = animal_type.lower()
        if animal_type in self.LIVESTOCK_CALIBRATION:
            self.animal_type = animal_type
            if update_pixel_ratio and not getattr(self, '_pixel_ratio_user_set', False):
                new_default = self.LIVESTOCK_CALIBRATION[animal_type].get("default_pixel_ratio", 0.15)
                self.pixel_to_cm_ratio = new_default
                logging.info(f"Pixel ratio reset to {new_default} for '{animal_type}'")
        else:
            raise ValueError(f"Invalid selector: {animal_type}")

    @classmethod
    def get_available_types(cls) -> dict:
        return cls.LIVESTOCK_CALIBRATION.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # CLIP Zero-Shot Species Classifier
    # ─────────────────────────────────────────────────────────────────────────

    def _get_crop_animal_type(self, crop: np.ndarray) -> Optional[str]:
        """CLIP zero-shot classification on a cropped bounding box."""
        clip_model     = self.__class__._clip_model
        clip_processor = self.__class__._clip_processor
        if not clip_model or not clip_processor:
            return None
        try:
            import torch
            from PIL import Image as PILImage

            rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            all_prompts:    list[str] = []
            prompt_category: list[str] = []
            for category, prompts in self.CLIP_PROMPTS.items():
                for p in prompts:
                    all_prompts.append(p)
                    prompt_category.append(category)

            inputs = clip_processor(text=all_prompts, images=pil_img, return_tensors="pt", padding=True)
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

    # ─────────────────────────────────────────────────────────────────────────
    # YOLO Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _yolo_detect(self, image_bgr: np.ndarray, target_classes: set, results: Any = None) -> Optional[Tuple[int, int, int, int, float]]:
        """Runs YOLO detection with CLIP crop-based species verification."""
        if results is None:
            try:
                results = self.model(image_bgr, verbose=False, conf=0.20)[0]
            except Exception as e:
                logging.error(f"YOLO inference failure: {e}")
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

            # COCO animal classes
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
                logging.info("Bounding box too small — discarding.")
                continue

            crop = image_bgr[y1:y2, x1:x2]
            classified_type = self._get_crop_animal_type(crop)

            if classified_type is not None and classified_type != "invalid":
                is_user_cattle       = self.animal_type in {"dairy_cow", "beef_cattle", "young_cattle"}
                is_classified_cattle = classified_type  in {"dairy_cow", "beef_cattle", "young_cattle"}

                if is_user_cattle and is_classified_cattle:
                    logging.info(f"CLIP confirmed Cattle. Keeping: {self.animal_type}")
                elif self.animal_type != classified_type:
                    logging.info(f"Auto-correcting '{self.animal_type}' -> '{classified_type}'")
                    self.set_animal_type(classified_type, update_pixel_ratio=True)

                return (x1, y1, x2, y2, conf)

        # Rejection reason
        if has_human:
            self.rejected_reason = "Human detected. Please ensure only livestock animals are in frame."
        elif animal_boxes:
            cls_name = self.model.names.get(animal_boxes[0][5], "unknown animal") if hasattr(self.model, 'names') else "unknown animal"
            self.rejected_reason = (
                f"Invalid target ({cls_name}) detected. "
                "Please ensure only livestock animals (cattle, pig, donkey, sheep, goat, poultry) are in frame."
            )
        elif largest_non_livestock_area > 0:
            cls_name = self.model.names.get(largest_non_livestock_class, "unknown object") if hasattr(self.model, 'names') else "unknown object"
            self.rejected_reason = (
                f"Invalid target ({cls_name}) detected. "
                "Please ensure only livestock animals are in frame."
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Contour Fallback Detection
    # ─────────────────────────────────────────────────────────────────────────

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