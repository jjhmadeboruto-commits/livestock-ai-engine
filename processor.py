from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]


class AnimalProcessor:
    """Estimate livestock weight from YOLOv8 bounding boxes (with contour fallback)."""

    # Species-specific calibration: weight_kg = (length_cm * girth_cm^2) / divisor
    LIVESTOCK_CALIBRATION = {
        "dairy_cow":    {"divisor": 660.0,    "girth_multiplier": 1.20, "name": "Dairy Cow",         "expected_range": [250, 700]},
        "beef_cattle":  {"divisor": 600.0,    "girth_multiplier": 1.22, "name": "Beef Cattle",        "expected_range": [300, 800]},
        "young_cattle": {"divisor": 800.0,    "girth_multiplier": 1.15, "name": "Young Cattle/Calf",  "expected_range": [80, 350]},
        "goat":         {"divisor": 8400.0,   "girth_multiplier": 1.45, "name": "Goat",               "expected_range": [20, 120]},
        "sheep":        {"divisor": 6000.0,   "girth_multiplier": 1.35, "name": "Sheep",              "expected_range": [30, 140]},
        "donkey":       {"divisor": 1200.0,   "girth_multiplier": 1.25, "name": "Donkey",             "expected_range": [120, 450]},
        "pig":          {"divisor": 1500.0,   "girth_multiplier": 1.50, "name": "Pig",                "expected_range": [30, 350]},
        "poultry":      {"divisor": 180000.0, "girth_multiplier": 0.60, "name": "Poultry",            "expected_range": [0.5, 12]},
    }

    _yolo_model = None
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_LENGTH_SCALE = 0.75
    YOLO_HEIGHT_SCALE = 0.80
    CONTOUR_LENGTH_SCALE = 0.75
    CONTOUR_HEIGHT_SCALE = 0.80
    MIN_BODY_PX = 30

    def __init__(self, pixel_to_cm_ratio: float = 0.264, animal_type: str = "dairy_cow") -> None:
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
        self.animal_type = animal_type.lower()
        if self.animal_type not in self.LIVESTOCK_CALIBRATION:
            self.animal_type = "dairy_cow"

    @classmethod
    def is_yolo_available(cls) -> bool:
        return YOLO is not None

    @classmethod
    def _get_yolo_model(cls):
        if cls._yolo_model is None:
            if YOLO is None:
                raise ImportError("ultralytics is not installed")
            cls._yolo_model = YOLO("yolov8n.pt")
        return cls._yolo_model

    def process(self, image_bgr: np.ndarray) -> Optional[Dict[str, object]]:
        """Detect animal via YOLOv8, fall back to contours, then estimate weight."""
        annotated_image = image_bgr.copy()
        method_used = None
        body_length_px = 0.0
        body_height_px = 0.0
        confidence_score = 0.0

        box = self._detect_with_yolo(image_bgr)
        if box is not None:
            x1, y1, x2, y2, confidence_score = box
            body_length_px = (x2 - x1) * self.YOLO_LENGTH_SCALE
            body_height_px = (y2 - y1) * self.YOLO_HEIGHT_SCALE
            method_used = "yolo"
            label = f"{self.LIVESTOCK_CALIBRATION[self.animal_type]['name']} (YOLO)"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_image, label, (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        if method_used is None:
            contour_box = self._fallback_contour_detection(image_bgr)
            if contour_box is not None:
                x1, y1, x2, y2 = contour_box
                body_length_px = (x2 - x1) * self.CONTOUR_LENGTH_SCALE
                body_height_px = (y2 - y1) * self.CONTOUR_HEIGHT_SCALE
                method_used = "contour_fallback"
                confidence_score = 0.6
                label = f"{self.LIVESTOCK_CALIBRATION[self.animal_type]['name']} (Contour)"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_image, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

        if method_used is None:
            return None

        if body_length_px < self.MIN_BODY_PX or body_height_px < self.MIN_BODY_PX:
            return None

        ratio = self._get_pixel_to_cm_ratio()
        length_cm = body_length_px * ratio
        height_cm = body_height_px * ratio

        calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
        girth_cm = length_cm * calibration["girth_multiplier"]
        weight_kg = (length_cm * (girth_cm ** 2)) / calibration["divisor"]

        expected_min, expected_max = calibration.get("expected_range", (None, None))
        within_range = True
        if expected_min is not None and expected_max is not None:
            within_range = expected_min <= weight_kg <= expected_max

        return {
            "weight": round(weight_kg, 2),
            "body_length": round(length_cm, 2),
            "body_height": round(height_cm, 2),
            "estimated_girth": round(girth_cm, 2),
            "animal_type": calibration["name"],
            "confidence_score": round(confidence_score, 3),
            "annotated_image": annotated_image,
            "expected_weight_range": calibration.get("expected_range"),
            "within_expected_range": within_range,
            "method": method_used,
        }

    def _detect_with_yolo(
        self, image_bgr: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """Return (x1, y1, x2, y2, confidence) for the largest confident detection."""
        try:
            model = self._get_yolo_model()
        except ImportError:
            return None

        results = model(image_bgr, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        best_box = None
        best_area = 0
        best_conf = 0.0

        for box in results[0].boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < self.YOLO_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
                best_conf = conf

        if best_box is None:
            return None
        x1, y1, x2, y2 = best_box
        return x1, y1, x2, y2, best_conf

    def calibrate_pixel_ratio(self, known_cm: float, measured_pixels: float) -> None:
        if measured_pixels > 0:
            self.pixel_to_cm_ratio = known_cm / measured_pixels

    def adjust_weight_calibration(
        self, animal_type: str, divisor: float = None, girth_multiplier: float = None,
    ) -> None:
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
            raise ValueError(
                f"Unknown animal type: {animal_type}. "
                f"Available: {list(self.LIVESTOCK_CALIBRATION.keys())}"
            )

    @classmethod
    def get_available_types(cls) -> dict:
        return cls.LIVESTOCK_CALIBRATION.copy()

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

        if (w * h) < (width * height * 0.05):
            return None

        return x, y, x + w, y + h

    def _get_pixel_to_cm_ratio(self) -> float:
        return max(0.001, self.pixel_to_cm_ratio)
