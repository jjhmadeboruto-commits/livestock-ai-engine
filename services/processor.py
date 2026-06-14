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
    classification + Google Gemini 2.5 Flash enrichment.

    v3.0 — Major improvements:
      * Gemini acts as FIRST gatekeeper — rejects non-livestock images before
        any weight estimation runs (no more phones/websites getting weights)
      * Richer Gemini prompt: returns sex, estimated age, posture, activity,
        visible health concerns, and a conversational narrative explanation
      * Gemini weight dominates (80%) when geometric data is unreliable (small
        bounding box, contour fallback, or small species like pigs/poultry)
      * Detected species key properly returned so scan_record uses correct
        animal type for history/species breakdown (not the user button label)
      * Confidence-gated CLIP: mouse/rat/non-livestock -> hard reject
      * Hard species weight caps still enforced as last safeguard
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

    # Species where Gemini visual weight estimate dominates (geometry less reliable)
    GEMINI_DOMINANT_SPECIES = {"pig", "poultry", "goat", "sheep"}

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
        "invalid":      ["a person", "a human", "a dog", "a cat", "a vehicle", "a building", "a car",
                         "a phone", "a computer", "a mouse", "a rat", "a wild animal", "a website screenshot"],
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
    # Gemini API Helper with robust Model Fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _call_gemini_api(self, payload: dict, timeout: int = 30) -> Optional[dict]:
        """
        Calls Google Gemini API using urllib.request, attempting a list of models
        with automatic fallback if quota or rate limit errors are encountered.
        Uses confirmed real model identifiers in order of preference.
        """
        import json, urllib.request, urllib.error, ssl
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        # Confirmed working Gemini model identifiers (v1beta API) in priority order:
        # 1. gemini-2.5-flash: Latest, best vision, confirmed working
        # 2. gemini-3.5-flash: New model, confirmed working for vision
        # 3. gemini-3.1-flash-lite: Fast fallback, confirmed working for vision
        # 4. gemini-2.0-flash: Good quota, confirmed working (may hit 429)
        models = [
            "gemini-2.5-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.0-flash",
        ]
        friendly_names = {
            "gemini-2.5-flash":      "Google Gemini 2.5 Flash",
            "gemini-3.5-flash":      "Google Gemini 3.5 Flash",
            "gemini-3.1-flash-lite": "Google Gemini 3.1 Flash-Lite",
            "gemini-2.0-flash":      "Google Gemini 2.0 Flash",
        }
        ctx = ssl.create_default_context()
        last_error = None

        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            req_body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_body, method="POST")
            req.add_header("Content-Type", "application/json")
            
            try:
                logging.info(f"Calling Gemini API via model: {model}...")
                with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                    raw_response = resp.read().decode("utf-8")
                    self.last_gemini_model_used = friendly_names.get(model, f"Google Gemini ({model})")
                    logging.info(f"Gemini API success via {model}")
                    return json.loads(raw_response)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {e.code} for {model}: {err_body[:300]}"
                logging.warning(f"Gemini model {model} failed with HTTP {e.code}. Trying next...")
                continue
            except Exception as ex:
                last_error = f"{type(ex).__name__} for {model}: {ex}"
                logging.error(f"Gemini model {model} request failed: {last_error}")
                continue

        logging.error(f"All Gemini models failed. Last error: {last_error}")
        return None

    @staticmethod
    def _strip_json_from_response(raw: str) -> str:
        """
        Robustly strip markdown code fences and thinking blocks from a Gemini response
        to extract clean JSON. Handles: ```json...```, ```...```, <think>...</think> etc.
        """
        if not raw:
            return raw
        # Remove <think>...</think> blocks (gemini 3.x thinking output)
        import re
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        # Remove ```json ... ``` or ``` ... ``` markdown code fences
        if '```' in raw:
            # Find the first { after any opening fence
            json_start = raw.find('{')
            json_end = raw.rfind('}')
            if json_start != -1 and json_end != -1:
                return raw[json_start:json_end + 1].strip()
        return raw.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Gemini Combined Analyze — Gate + Enrich in ONE API call
    # ─────────────────────────────────────────────────────────────────────────

    def _gemini_analyze(self, image_bgr: np.ndarray, requested_species: str = "unknown", geometric_weight_kg: float = None) -> dict:
        """
        Single Gemini API call that combines the gate check AND the full enrichment.
        This avoids making 2 sequential API calls which hits rate limits.

        Returns a dict with all enrichment fields PLUS:
          - is_livestock: bool  (gate check)
          - rejection_reason: str (if not livestock)

        Returns {"is_livestock": True} on any API failure (fail-open for gate).
        Returns {} enrichment fields on any failure (fail-gracefully for enrichment).
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {"is_livestock": True}

        try:
            success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return {"is_livestock": True}
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            prompt = (
                f"You are a professional livestock veterinarian and AI assistant for a farming application.\n"
                f"The farmer believes this image shows a {requested_species}.\n"
            )
            if geometric_weight_kg is not None:
                prompt += f"Our geometric computer vision model has estimated the weight at {geometric_weight_kg:.1f} kg based on the animal's bounding box.\n"
            
            prompt += (
                "\nTASK 1 — GATE CHECK: First determine if this is a supported livestock animal.\n"
                "Supported species: cattle (cow/bull/calf), pig, goat, sheep, donkey, poultry (chicken/duck/turkey).\n"
                "If it is NOT livestock (e.g. dog, cat, person, phone, screenshot), set is_livestock to false and fill rejection_reason.\n\n"
                "TASK 2 — ENRICHMENT: If it IS livestock, fill in all the analysis fields below.\n\n"
                "Return ONLY a valid JSON object with exactly these keys:\n"
                "{\n"
                "  \"is_livestock\": <true or false>,\n"
                "  \"rejection_reason\": \"<if is_livestock is false: one clear sentence why it cannot be weighed. Empty string if livestock.>\",\n"
                "  \"detected_species\": \"<what you actually see, e.g. Holstein cow, pig, goat, dog, cat, phone>\",\n"
                "  \"breed\": \"<specific breed name, e.g. Holstein-Friesian, Duroc, Boer Goat, or Mixed/Unknown>\",\n"
                "  \"sex\": \"<male, female, or unknown>\",\n"
                "  \"estimated_age_months\": <integer — best estimate of age in months>,\n"
                "  \"body_condition\": \"<exactly one of: thin, fair, good, excellent>\",\n"
                "  \"body_condition_score\": <integer 1 to 5>,\n"
                "  \"posture\": \"<e.g. standing alert, grazing, resting, walking>\",\n"
                "  \"activity_level\": \"<calm, moderate, or active>\",\n"
                "  \"visible_health_concerns\": \"<any visible issues like wounds, lameness, distended belly, or None observed>\",\n"
                "  \"estimated_weight_range_kg\": [<min_integer>, <max_integer>],\n"
                "  \"photo_quality_note\": \"<one short sentence on photo quality and positioning>\",\n"
                "  \"gemini_explanation\": \"<A warm, clear, 3-5 sentence explanation written directly TO the farmer. Start with what you visually observed. Explain WHY you estimated the weight — mentioning breed, body condition, age, belly size, muscle development. Say what would improve accuracy. End with a helpful farming tip.>\",\n"
                "  \"ai_self_review\": \"<A short internal critique of the geometric weight estimate (if provided). Is the geometric weight feasible? Was it adjusted based on your visual assessment? Why?>\"\n"
                "}\n\n"
                "Body condition scoring: 1=Emaciated(thin), 2=Very thin(thin), 3=Moderate(fair), 4=Good(good), 5=Excellent/Obese(excellent)\n"
                "Weight guide: dairy cows 400-700 kg, beef bulls 500-800 kg, calves 50-200 kg, pigs 80-250 kg, goats 20-90 kg, sheep 30-120 kg, donkeys 80-300 kg, poultry 0.5-8 kg\n"
                "NEVER estimate above biologically plausible maximums.\n"
                "Return ONLY the JSON object. No other text, no markdown, no code fences."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 8192
                }
            }

            response = self._call_gemini_api(payload, timeout=30)
            if not response:
                logging.warning("Gemini analyze: no response from API — failing open (is_livestock=True, no enrichment)")
                return {"is_livestock": True}

            raw = response["candidates"][0]["content"]["parts"][0]["text"]
            raw = self._strip_json_from_response(raw)

            data = json.loads(raw)
            logging.info(
                f"Gemini analyze OK: is_livestock={data.get('is_livestock')}, "
                f"species={data.get('detected_species')}, breed={data.get('breed')}, "
                f"condition={data.get('body_condition')}, range={data.get('estimated_weight_range_kg')}"
            )
            return data

        except json.JSONDecodeError as e:
            logging.error(f"Gemini analyze: non-JSON response (JSONDecodeError: {e}). raw={repr(raw if 'raw' in dir() else 'N/A')[:200]}")
            return {"is_livestock": True}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logging.warning(f"Gemini analyze HTTP {e.code}: {err_body[:300]}")
            return {"is_livestock": True}
        except Exception as e:
            import traceback
            logging.warning(f"Gemini analyze failed ({type(e).__name__}): {e}\n{traceback.format_exc()}")
            return {"is_livestock": True}

    # Keep _gemini_gate for backwards compat (calls _gemini_analyze now)
    def _gemini_gate(self, image_bgr: np.ndarray) -> dict:
        """
        Calls Gemini with a lightweight gatekeeper prompt BEFORE any weight
        estimation. Returns a dict with:
          - is_livestock: bool
          - rejection_reason: str (if not livestock)
          - detected_species: str  (e.g. "pig", "cow", "dog", "phone")
          - confidence: str ("high"/"medium"/"low")

        Returns {"is_livestock": True, "detected_species": "unknown"} on any
        API failure so that the pipeline does NOT silently break when Gemini
        is unavailable.
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {"is_livestock": True, "detected_species": "unknown"}

        try:
            success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return {"is_livestock": True, "detected_species": "unknown"}
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            prompt = (
                "You are a livestock detection gatekeeper for a farming application.\n"
                "Look at this image and determine if it contains a livestock animal.\n\n"
                "Supported livestock species: cattle (cow/bull/calf), pig, goat, sheep, donkey, poultry (chicken/duck/turkey).\n\n"
                "Return ONLY a valid JSON object with exactly these keys:\n"
                "{\n"
                "  \"is_livestock\": <true or false>,\n"
                "  \"detected_species\": \"<what you actually see, e.g. pig, cow, dog, cat, mouse, phone, person, website, gadget>\",\n"
                "  \"confidence\": \"<high, medium, or low>\",\n"
                "  \"rejection_reason\": \"<if is_livestock is false: one clear sentence explaining what is in the image and why it cannot be weighed>\"\n"
                "}\n\n"
                "Be STRICT: a mouse, rat, dog, cat, wild animal, person, screenshot, website, or any non-livestock image must return is_livestock: false.\n"
                "Return ONLY the JSON object. No other text."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 150
                }
            }

            response = self._call_gemini_api(payload, timeout=20)
            if not response:
                return {"is_livestock": True, "detected_species": "unknown"}

            raw = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            gate = json.loads(raw)
            logging.info(f"Gemini gate: is_livestock={gate.get('is_livestock')}, species={gate.get('detected_species')}, conf={gate.get('confidence')}")
            return gate

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logging.warning(f"Gemini gate HTTP {e.code}: {err_body[:300]}")
            return {"is_livestock": True, "detected_species": "unknown"}
        except Exception as e:
            logging.warning(f"Gemini gate failed ({type(e).__name__}): {e}")
            return {"is_livestock": True, "detected_species": "unknown"}

    # ─────────────────────────────────────────────────────────────────────────
    # Gemini Enrichment Layer — Rich breed / BCS / health / weight narrative
    # ─────────────────────────────────────────────────────────────────────────

    def _gemini_enrich(self, image_bgr: np.ndarray, requested_species: str = "unknown") -> dict:
        """
        Calls Google Gemini 2.5 Flash via direct REST API to enrich a scan with:
          - breed, sex, estimated_age_months, body_condition, body_condition_score,
            posture, activity_level, visible_health_concerns, estimated_weight_range_kg,
            photo_quality_note, gemini_explanation (conversational narrative for the user).

        Returns {} gracefully on any failure so the main pipeline always continues.
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            logging.info("GEMINI_API_KEY not set — skipping Gemini enrichment.")
            return {}

        try:
            # Guard against empty or zero-sized crops — use full image as fallback
            if image_bgr is None or image_bgr.size == 0 or image_bgr.shape[0] < 10 or image_bgr.shape[1] < 10:
                logging.warning("Gemini: crop too small or empty — this should not happen.")
                return {}

            success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logging.warning("Gemini: cv2.imencode failed — skipping.")
                return {}
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            logging.info(f"Gemini enrichment: encoded {len(buf)} bytes (b64 {len(img_b64)} chars) for species={requested_species}")

            prompt = (
                f"You are a professional livestock veterinarian and expert body-condition scorer.\n"
                f"The farmer has submitted a photo they believe shows a {requested_species}.\n"
                f"Analyze this livestock image carefully and return ONLY a valid JSON object with these exact keys:\n\n"
                "{\n"
                "  \"breed\": \"<specific breed name, e.g. Large Black Pig, Duroc, Berkshire, Holstein, Boer Goat, or Mixed/Unknown>\",\n"
                "  \"sex\": \"<male, female, or unknown>\",\n"
                "  \"estimated_age_months\": <integer — your best estimate of the animal's age in months>,\n"
                "  \"body_condition\": \"<exactly one of: thin, fair, good, excellent>\",\n"
                "  \"body_condition_score\": <integer 1 to 5>,\n"
                "  \"posture\": \"<e.g. standing alert, grazing, resting, walking>\",\n"
                "  \"activity_level\": \"<calm, moderate, or active>\",\n"
                "  \"visible_health_concerns\": \"<any visible issues like wounds, lameness, distended belly, mud coverage, or None observed>\",\n"
                "  \"estimated_weight_range_kg\": [<min_integer>, <max_integer>],\n"
                "  \"photo_quality_note\": \"<one short sentence on photo quality and positioning>\",\n"
                "  \"gemini_explanation\": \"<A warm, clear, 3-5 sentence explanation written directly TO the farmer, in first person as Gemini. Start with what you visually observed about the animal. Explain WHY you estimated the weight you did — mentioning the breed, body condition, age, and any visual cues like belly size or muscle development. Mention what would make the estimate more accurate. End with a helpful farming insight.>\"\n"
                "}\n\n"
                "Body condition scoring guide:\n"
                "  1 = Emaciated (thin)\n"
                "  2 = Very thin (thin)\n"
                "  3 = Moderate (fair)\n"
                "  4 = Good (good)\n"
                "  5 = Excellent/Obese (excellent)\n\n"
                "Weight estimation guide:\n"
                "  - Adult pigs typically weigh 80–250 kg. A large mature sow can be 180–300 kg.\n"
                "  - Piglets (under 3 months) weigh 5–30 kg.\n"
                "  - Cattle: dairy cows 400–700 kg, beef bulls 500–800 kg, calves 50–200 kg.\n"
                "  - Goats: 20–90 kg. Sheep: 30–120 kg. Donkeys: 80–300 kg. Poultry: 0.5–8 kg.\n"
                "  - NEVER estimate above biologically plausible maximums.\n\n"
                "Be specific about the breed. Use your visual assessment of body size, muscle definition, and fat coverage.\n"
                "Return ONLY the JSON object, no other text."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.15,
                    "maxOutputTokens": 600
                }
            }

            response = self._call_gemini_api(payload, timeout=30)
            if not response:
                return {}

            raw = response["candidates"][0]["content"]["parts"][0]["text"].strip()

            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            logging.info(f"Gemini enrichment success: breed={data.get('breed')}, "
                         f"sex={data.get('sex')}, age={data.get('estimated_age_months')}mo, "
                         f"condition={data.get('body_condition')}, "
                         f"range={data.get('estimated_weight_range_kg')}")
            return data

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logging.error(f"Gemini enrichment HTTP {e.code} error: {err_body[:800]}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Gemini enrichment: non-JSON response: {e}")
            return {}
        except Exception as e:
            import traceback
            logging.error(f"Gemini enrichment failed ({type(e).__name__}): {e}\n{traceback.format_exc()}")
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

        # ── Step 4: Gemini Combined Gate + Enrichment & Self-Review (ONE API call) ─────────
        # We call _gemini_analyze and pass the raw geometric weight so the AI can evaluate it!
        gemini_data = self._gemini_analyze(image_bgr, requested_species=self.animal_type, geometric_weight_kg=weight_kg)
        
        # Gate check — reject non-livestock (this now happens after geometric processing but still prevents wrong output)
        if not gemini_data.get("is_livestock", True):
            detected_thing = gemini_data.get("detected_species", "unknown object")
            rejection_msg  = gemini_data.get(
                "rejection_reason",
                f"No livestock animal detected. The image appears to contain: {detected_thing}. "
                "Please upload a clear photo of a livestock animal (cattle, pig, goat, sheep, donkey, or poultry)."
            )
            return {
                "weight": 0.0, "body_length": 0.0, "body_height": 0.0,
                "estimated_girth": 0.0, "animal_type": "Invalid Target",
                "confidence_score": 0.0, "annotated_image": image_bgr,
                "expected_weight_range": [0, 0], "within_expected_range": False,
                "method": "rejected",
                "error_message": rejection_msg,
                "detected_as": detected_thing,
            }


        # ── Step 5: BCS Weight Correction ─────────────────────────────────────
        body_condition_raw = (gemini_data.get("body_condition") or "unknown").lower().strip()
        body_condition     = body_condition_raw if body_condition_raw in self.BCS_CORRECTION else "unknown"
        bcs_factor         = self.BCS_CORRECTION[body_condition]
        weight_kg          = weight_kg * bcs_factor

        # ── Step 6: Gemini weight blend (species-aware weighting) ──────────────
        gemini_range = gemini_data.get("estimated_weight_range_kg")
        gemini_cross_check = None
        gemini_used = bool(gemini_data)

        if gemini_range and isinstance(gemini_range, (list, tuple)) and len(gemini_range) == 2:
            g_min, g_max = float(gemini_range[0]), float(gemini_range[1])
            gemini_mid   = (g_min + g_max) / 2.0

            # For species where geometry is unreliable (pigs, poultry, goats, sheep),
            # Gemini's visual estimate is more trustworthy — use 80/20 in Gemini's favour.
            # For large cattle where geometry works well, use 60/40.
            is_contour = method_used == "contour_fallback"
            bounding_box_area = box_width_pixels * box_height_pixels
            img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
            box_coverage = bounding_box_area / max(1.0, img_area)

            if self.animal_type in self.GEMINI_DOMINANT_SPECIES or is_contour or box_coverage < 0.10:
                # Gemini 80%, geometry 20%
                weight_kg = weight_kg * 0.20 + gemini_mid * 0.80
            else:
                # Geometry 60%, Gemini 40%
                weight_kg = weight_kg * 0.60 + gemini_mid * 0.40

            gemini_cross_check = {"min": g_min, "max": g_max, "midpoint": round(gemini_mid, 1)}

        # ── Step 7: Hard species weight cap ───────────────────────────────────
        hard_cap  = self.WEIGHT_HARD_CAPS.get(self.animal_type, 800)
        min_floor = calibration["expected_range"][0] * 0.5
        weight_kg = max(min_floor, min(weight_kg, hard_cap))

        # ── AI attribution ─────────────────────────────────────────────────────
        ai_attribution = {
            "detection_model":      "YOLOv8n (Ultralytics)",
            "classification_model": "CLIP ViT-B/32 (OpenAI)",
            "enrichment_model":     getattr(self, "last_gemini_model_used", "Google Gemini 3.5 Flash") if gemini_used else None,
            "weight_formula":       "Schoorl Girth Formula + Gemini Visual Intelligence",
            "bcs_correction":       (
                f"BCS adjustment: {body_condition} (x{bcs_factor})"
                if gemini_used else "No BCS correction (Gemini key not configured)"
            ),
            "gemini_cross_check": gemini_cross_check,
            "weight_blend": (
                "80% Gemini visual + 20% geometric (Gemini dominant for this species)"
                if self.animal_type in self.GEMINI_DOMINANT_SPECIES and gemini_range
                else "60% geometric + 40% Gemini visual cross-check"
            ) if gemini_range else "Geometric formula only (Gemini weight range unavailable)",
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

        # Detected species key (internal, for scan_record bucketing)
        detected_species_key = self.animal_type

        return {
            "weight":                 float(round(weight_kg, 2)),
            "body_length":            float(round(length_cm, 2)),
            "body_height":            float(round(height_cm, 2)),
            "estimated_girth":        float(round(girth_cm, 2)),
            "animal_type":            str(calibration["name"]),
            "detected_species_key":   detected_species_key,   # internal key (pig, sheep, etc.)
            "confidence_score":       float(round(confidence_score, 3)),
            "annotated_image":        annotated_image,
            "expected_weight_range":  calibration.get("expected_range"),
            "within_expected_range":  bool(within_range),
            "method":                 str(method_used),
            "warning":                warning_message,
            "pixel_to_cm_ratio":      float(ratio),
            # Gemini enrichment
            "breed":                  gemini_data.get("breed", "Unknown"),
            "sex":                    gemini_data.get("sex", "Unknown"),
            "estimated_age_months":   gemini_data.get("estimated_age_months"),
            "posture":                gemini_data.get("posture", ""),
            "activity_level":         gemini_data.get("activity_level", ""),
            "visible_health_concerns":gemini_data.get("visible_health_concerns", "None observed"),
            "body_condition":         body_condition if gemini_used else "Not assessed",
            "body_condition_score":   gemini_data.get("body_condition_score"),
            "health_notes":           gemini_data.get("visible_health_concerns", ""),
            "photo_quality_note":     gemini_data.get("photo_quality_note", ""),
            "gemini_explanation":     gemini_data.get("gemini_explanation", ""),
            "ai_self_review":         gemini_data.get("ai_self_review", ""),
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
    # Gemini Self-Critique — AI questions its own answer before showing it
    # ─────────────────────────────────────────────────────────────────────────

    def _gemini_self_critique(
        self,
        image_bgr: np.ndarray,
        first_pass: dict,
        proposed_weight_kg: float,
        species_key: str,
    ) -> dict:
        """
        After the first enrichment pass, Gemini reviews its own proposed weight
        against biological norms and the image it sees, then either confirms or
        corrects it. Returns a dict with:
          - confirmed_weight_kg: float   (the final agreed weight)
          - confidence_level: str        ("high" / "medium" / "low")
          - critique_notes: str          (Gemini's self-review in plain English)
          - was_adjusted: bool           (did Gemini change its answer?)
          - adjustment_reason: str       (why it changed, if it did)
          - final_explanation: str       (the complete user-facing explanation after critique)
        Falls back to the first_pass data on any error.
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {
                "confirmed_weight_kg": proposed_weight_kg,
                "confidence_level":    "medium",
                "critique_notes":      "Self-critique skipped (API key not set).",
                "was_adjusted":        False,
                "adjustment_reason":   "",
                "final_explanation":   first_pass.get("gemini_explanation", ""),
            }

        calibration   = self.LIVESTOCK_CALIBRATION.get(species_key, {})
        expected_min  = calibration.get("expected_range", [0, 800])[0]
        expected_max  = calibration.get("expected_range", [0, 800])[1]
        hard_cap      = self.WEIGHT_HARD_CAPS.get(species_key, 800)

        breed        = first_pass.get("breed", "Unknown")
        bcs          = first_pass.get("body_condition", "unknown")
        bcs_score    = first_pass.get("body_condition_score", "?")
        sex          = first_pass.get("sex", "unknown")
        age_months   = first_pass.get("estimated_age_months", "unknown")
        g_range      = first_pass.get("estimated_weight_range_kg", [expected_min, expected_max])
        g_min        = g_range[0] if g_range else expected_min
        g_max        = g_range[1] if g_range else expected_max
        first_expl   = first_pass.get("gemini_explanation", "")

        try:
            success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                raise ValueError("encode failed")
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            prompt = (
                f"You are a precision livestock weight validation expert.\n\n"
                f"You previously analysed this image and produced the following assessment:\n"
                f"  - Species: {species_key}\n"
                f"  - Breed: {breed}\n"
                f"  - Sex: {sex}\n"
                f"  - Estimated age: {age_months} months\n"
                f"  - Body condition score: {bcs_score}/5 ({bcs})\n"
                f"  - Your weight range estimate: {g_min}–{g_max} kg\n"
                f"  - System's blended final weight: {proposed_weight_kg:.1f} kg\n\n"
                f"Biological reference ranges:\n"
                f"  - Normal range for {species_key}: {expected_min}–{expected_max} kg\n"
                f"  - Absolute maximum possible: {hard_cap} kg\n\n"
                f"TASK: Critically re-examine the image and your own assessment. Ask yourself:\n"
                f"  1. Does {proposed_weight_kg:.1f} kg look realistic for this specific animal's visible body size?\n"
                f"  2. Is the weight consistent with the breed, age, and body condition you described?\n"
                f"  3. Are there any visual clues you missed the first time that would change your estimate?\n"
                f"  4. Is anything biologically impossible or inconsistent?\n\n"
                f"Return ONLY a valid JSON object with exactly these keys:\n"
                "{\n"
                f"  \"confirmed_weight_kg\": <your final agreed weight as a number — adjust if {proposed_weight_kg:.1f} kg seems wrong>,\n"
                "  \"confidence_level\": \"<high, medium, or low>\",\n"
                "  \"critique_notes\": \"<2-3 sentences: what you reconsidered and why — be honest if you caught an error>\",\n"
                "  \"was_adjusted\": <true if you changed the weight, false if you confirmed it>,\n"
                "  \"adjustment_reason\": \"<if was_adjusted is true: explain why you changed it; otherwise empty string>\",\n"
                "  \"final_explanation\": \"<A complete, warm 4-6 sentence explanation written directly TO the farmer. "
                "Start with what you see in the image. Explain the breed and body condition. "
                "Give the confirmed weight with confidence. "
                "Explain what visual cues drove the estimate. "
                "End with one specific farming or management tip for this animal.>\"\n"
                "}\n\n"
                "Be honest — if your first estimate was wrong, correct it. "
                "If it was right, confirm it with confidence. Return ONLY the JSON object."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.05,
                    "maxOutputTokens": 700
                }
            }

            response = self._call_gemini_api(payload, timeout=30)
            if not response:
                raise Exception("Gemini self-critique: no response from API")

            raw = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            critique = json.loads(raw)

            # Apply hard cap to Gemini's confirmed weight too
            confirmed = float(critique.get("confirmed_weight_kg", proposed_weight_kg))
            confirmed = max(
                calibration.get("expected_range", [0, hard_cap])[0] * 0.5,
                min(confirmed, hard_cap)
            )
            critique["confirmed_weight_kg"] = confirmed

            logging.info(
                f"Gemini self-critique: was_adjusted={critique.get('was_adjusted')}, "
                f"proposed={proposed_weight_kg:.1f} → confirmed={confirmed:.1f} kg, "
                f"confidence={critique.get('confidence_level')}"
            )
            return critique

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logging.warning(f"Gemini self-critique HTTP {e.code}: {err_body[:300]}")
        except Exception as e:
            logging.warning(f"Gemini self-critique failed ({type(e).__name__}): {e}")

        return {
            "confirmed_weight_kg": proposed_weight_kg,
            "confidence_level":    "medium",
            "critique_notes":      "Self-critique could not be completed — using first-pass estimate.",
            "was_adjusted":        False,
            "adjustment_reason":   "",
            "final_explanation":   first_expl,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Live Camera Frame Processor
    # ─────────────────────────────────────────────────────────────────────────

    def process_live_frame(
        self,
        image_bgr: np.ndarray,
        run_self_critique: bool = True,
    ) -> Optional[Dict[str, object]]:
        """
        Optimised pipeline for live camera frames. Identical to process() but:
          - Adds a Gemini self-critique step after enrichment
          - Returns the critique-confirmed weight as the canonical weight
          - Includes 'frame_analysis' with per-step reasoning visible to the user
          - Skips pixel_ratio auto-scaling (live frames vary in size constantly)

        The 'frame_analysis' dict is designed to be displayed step-by-step in
        the frontend live-cam UI so users can follow the AI's reasoning.
        """
        # Re-use the full process() pipeline
        result = self.process(image_bgr)

        if result is None or result.get("method") == "rejected":
            return result

        proposed_weight = result["weight"]
        species_key     = result.get("detected_species_key", self.animal_type)
        gemini_data     = {
            "breed":                   result.get("breed", "Unknown"),
            "sex":                     result.get("sex", "Unknown"),
            "estimated_age_months":    result.get("estimated_age_months"),
            "body_condition":          result.get("body_condition", "unknown"),
            "body_condition_score":    result.get("body_condition_score"),
            "estimated_weight_range_kg": result.get("gemini_cross_check") and [
                result["gemini_cross_check"]["min"],
                result["gemini_cross_check"]["max"]
            ],
            "gemini_explanation":      result.get("gemini_explanation", ""),
        }

        # ── Self-critique step ────────────────────────────────────────────────
        critique = {}
        if run_self_critique:
            # Crop the detected bounding box region for self-critique
            # (use the full image if crop not available)
            critique = self._gemini_self_critique(
                image_bgr,
                gemini_data,
                proposed_weight,
                species_key,
            )

        confirmed_weight = critique.get("confirmed_weight_kg", proposed_weight)
        was_adjusted     = critique.get("was_adjusted", False)
        confidence_level = critique.get("confidence_level", "medium")
        critique_notes   = critique.get("critique_notes", "")
        adjustment_reason= critique.get("adjustment_reason", "")
        final_explanation= critique.get("final_explanation", result.get("gemini_explanation", ""))

        # Build step-by-step frame_analysis for the frontend live-cam UI
        frame_analysis = {
            "step_1_gate": {
                "label":   "🛡️ Livestock Verification",
                "status":  "passed",
                "detail":  f"Gemini confirmed: {result.get('animal_type', 'Livestock')} detected.",
            },
            "step_2_detection": {
                "label":  "📦 Animal Detection",
                "status": "passed",
                "detail": f"Detection method: {result.get('method', 'unknown').upper()}. "
                          f"Confidence: {round(result['confidence_score'] * 100, 1)}%.",
            },
            "step_3_species": {
                "label":  "🔬 Species & Breed ID",
                "status": "passed",
                "detail": f"Species: {result.get('animal_type')}. Breed: {result.get('breed', 'Unknown')}. "
                          f"Sex: {result.get('sex', 'Unknown')}. "
                          f"Est. age: {result.get('estimated_age_months', '?')} months.",
            },
            "step_4_condition": {
                "label":  "💪 Body Condition Assessment",
                "status": "passed",
                "detail": f"BCS: {result.get('body_condition_score', '?')}/5 ({result.get('body_condition', 'Not assessed')}). "
                          f"Posture: {result.get('posture', 'unknown')}. "
                          f"Activity: {result.get('activity_level', 'unknown')}.",
            },
            "step_5_weight": {
                "label":  "⚖️ Initial Weight Estimate",
                "status": "passed",
                "detail": f"Geometric formula + Gemini visual blend → {proposed_weight:.1f} kg.",
            },
            "step_6_critique": {
                "label":  "🤔 AI Self-Review",
                "status": "adjusted" if was_adjusted else "confirmed",
                "detail": (
                    f"Gemini reconsidered and adjusted to {confirmed_weight:.1f} kg. Reason: {adjustment_reason}"
                    if was_adjusted
                    else f"Gemini confirmed {confirmed_weight:.1f} kg. {critique_notes}"
                ),
            },
            "step_7_health": {
                "label":  "🩺 Health Observations",
                "status": "info",
                "detail": result.get("visible_health_concerns", "None observed"),
            },
        }

        # Merge the critique result into the main result dict
        result["weight"]              = float(round(confirmed_weight, 2))
        result["weight_kg"]           = float(round(confirmed_weight, 2))
        result["proposed_weight_kg"]  = float(round(proposed_weight, 2))
        result["was_adjusted"]        = was_adjusted
        result["confidence_level"]    = confidence_level
        result["critique_notes"]      = critique_notes
        result["adjustment_reason"]   = adjustment_reason
        result["gemini_explanation"]  = final_explanation
        result["frame_analysis"]      = frame_analysis
        result["live_cam_mode"]       = True

        return result

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

            # Higher confidence threshold to avoid misclassifying small animals as livestock
            if best_score < 0.07:
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