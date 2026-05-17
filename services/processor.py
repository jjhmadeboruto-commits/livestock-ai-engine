from __future__ import annotations

import cv2
import numpy as np
import os
import logging
from typing import Any, Dict, Optional, Tuple


class AnimalProcessor:
    """Processor for estimating livestock weight using YOLOv8 bounding box detection."""

    LIVESTOCK_CALIBRATION = {
        "dairy_cow":    {"divisor": 850.0,    "girth_multiplier": 0.95, "name": "Dairy Cow",         "expected_range": [250, 700]},
        "beef_cattle":  {"divisor": 600.0,    "girth_multiplier": 1.22, "name": "Beef Cattle",        "expected_range": [300, 800]},
        "young_cattle": {"divisor": 800.0,    "girth_multiplier": 1.15, "name": "Young Cattle/Calf",  "expected_range": [80, 350]},

        "goat":         {"divisor": 8400.0,   "girth_multiplier": 1.45, "name": "Goat",               "expected_range": [20, 120]},
        "sheep":        {"divisor": 6000.0,   "girth_multiplier": 1.35, "name": "Sheep",              "expected_range": [30, 140]},
        "donkey":       {"divisor": 1200.0,   "girth_multiplier": 1.25, "name": "Donkey",             "expected_range": [120, 450]},
        "pig":          {"divisor": 1500.0,   "girth_multiplier": 1.50, "name": "Pig",                "expected_range": [30, 350]},
        "poultry":      {"divisor": 180000.0, "girth_multiplier": 0.60, "name": "Poultry",            "expected_range": [0.5, 12]},
    }

    # YOLO class IDs that correspond to livestock in the COCO dataset
    # 16=bird(poultry), 17=cat, 18=dog, 19=horse, 20=sheep, 21=cow, 22=elephant, 23=bear
    YOLO_LIVESTOCK_CLASSES = {16, 19, 20, 21}  # bird, horse, sheep, cow

    # Class-level cached YOLO model
    _yolo_model = None

    def __init__(self, pixel_to_cm_ratio: float = 0.15, animal_type: str = "dairy_cow") -> None:
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
        self.animal_type = animal_type.lower()
        if self.animal_type not in self.LIVESTOCK_CALIBRATION:
            self.animal_type = "dairy_cow"

        # Lazy-load YOLO once and cache at class level
        if self.__class__._yolo_model is None:
            try:
                from ultralytics import YOLO
                import os
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                default_model_path = os.path.join(BASE_DIR, "..", "models", "yolov8n.pt")
                model_path = os.environ.get("MODEL_PATH", os.path.abspath(default_model_path))
                self.__class__._yolo_model = YOLO(model_path)
                logging.info(f"YOLO loaded from {model_path}")
            except Exception as e:
                self.__class__._yolo_model = False  # mark as unavailable so we don't retry
                logging.error(f"[AnimalProcessor] YOLOv8 unavailable: {e}")

        self.model = self.__class__._yolo_model if self.__class__._yolo_model else None

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def process(self, image_bgr: np.ndarray) -> Optional[Dict[str, object]]:
        """Estimate animal weight from a BGR image.

        Tries YOLOv8 bounding box detection first for all species.
        Falls back to OpenCV contour detection if YOLO finds nothing.

        Returns a result dict or None if detection failed entirely.
        """
        annotated_image = image_bgr.copy()
        method_used = None
        body_length_px = 0.0
        body_height_px = 0.0
        confidence_score = 0.0

        # ── Step 1: Try YOLOv8 for ALL species ──
        if self.model:
            box = self._yolo_detect(image_bgr)
            if box is not None:
                x1, y1, x2, y2, conf = box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 200, 50), 3)
                cv2.putText(
                    annotated_image,
                    f"{self.LIVESTOCK_CALIBRATION[self.animal_type]['name']} {conf:.0%}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 50), 2
                )
                body_length_px = (x2 - x1) * 0.80
                body_height_px = (y2 - y1) * 0.85
                confidence_score = float(conf)
                method_used = "yolov8"

        # ── Step 2: Contour fallback if YOLO found nothing ──
        if method_used is None:
            box = self._fallback_contour_detection(image_bgr)
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_image,
                    f"{self.LIVESTOCK_CALIBRATION[self.animal_type]['name']} (Contour)",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                body_length_px = (x2 - x1) * 0.75
                body_height_px = (y2 - y1) * 0.80
                method_used = "contour_fallback"
                confidence_score = 0.65

        if method_used is None:
            return None

        # ── Step 3: Convert pixels → cm → weight ──
        ratio = max(0.001, self.pixel_to_cm_ratio)
        length_cm = body_length_px * ratio
        height_cm = body_height_px * ratio

        calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
        girth_cm = length_cm * calibration["girth_multiplier"]
        weight_kg = (length_cm * (girth_cm ** 2)) / calibration["divisor"]

        expected_min, expected_max = calibration.get("expected_range", (None, None))
        if expected_min is not None and expected_max is not None:
            if weight_kg > expected_max:
                weight_kg = expected_max
            elif weight_kg < expected_min:
                weight_kg = expected_min

        within_range = True
        if expected_min is not None and expected_max is not None:
            within_range = expected_min <= weight_kg <= expected_max

        return {
            "weight":                float(round(weight_kg, 2)),
            "body_length":           float(round(length_cm, 2)),
            "body_height":           float(round(height_cm, 2)),
            "estimated_girth":       float(round(girth_cm, 2)),
            "animal_type":           str(calibration["name"]),
            "confidence_score":      float(round(confidence_score, 3)),
            "annotated_image":       annotated_image,
            "expected_weight_range": calibration.get("expected_range"),
            "within_expected_range": bool(within_range),
            "method":                str(method_used),
        }

    def calibrate_pixel_ratio(self, known_cm: float, measured_pixels: float) -> None:
        if measured_pixels > 0:
            self.pixel_to_cm_ratio = known_cm / measured_pixels

    def adjust_weight_calibration(self, animal_type: str, divisor: float = None, girth_multiplier: float = None) -> None:
        if animal_type not in self.LIVESTOCK_CALIBRATION:
            raise ValueError(f"Unknown animal type: {animal_type}")
        if divisor is not None:
            self.LIVESTOCK_CALIBRATION[animal_type]["divisor"] = divisor
        if girth_multiplier is not None:
            self.LIVESTOCK_CALIBRATION[animal_type]["girth_multiplier"] = girth_multiplier

    def set_animal_type(self, animal_type: str) -> None:
        animal_type = animal_type.lower()
        if animal_type in self.LIVESTOCK_CALIBRATION:
            self.animal_type = animal_type
        else:
            raise ValueError(f"Unknown animal type: {animal_type}. Available: {list(self.LIVESTOCK_CALIBRATION.keys())}")

    @classmethod
    def get_available_types(cls) -> dict:
        return cls.LIVESTOCK_CALIBRATION.copy()

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _yolo_detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Run YOLOv8 detection and return the best animal bounding box (x1,y1,x2,y2,conf)."""
        try:
            results = self.model(image_bgr, verbose=False, conf=0.25)[0]
        except Exception as e:
            logging.error(f"[AnimalProcessor] YOLO inference error: {e}")
            return None

        best_box = None
        best_conf = 0.0
        best_area = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            # Only accept known livestock YOLO classes — avoids false positives (trees, cars, fences)
            is_animal_class = cls_id in self.YOLO_LIVESTOCK_CLASSES
            is_any_object = False

            if is_animal_class and conf > best_conf and area > best_area:
                best_conf = conf
                best_area = area
                best_box = (x1, y1, x2, y2, conf)

        if best_box is None:
            return None

        # Reject tiny boxes (likely noise)
        x1, y1, x2, y2, conf = best_box
        h, w = image_bgr.shape[:2]
        if (x2 - x1) * (y2 - y1) < w * h * 0.05:
            return None

        return best_box

    def _fallback_contour_detection(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return bounding box (x1,y1,x2,y2) of the largest foreground contour."""
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

        if (w * h) < (width * height * 0.05):
            return None

        return (x, y, x + w, y + h)