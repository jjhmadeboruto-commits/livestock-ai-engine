from __future__ import annotations

import cv2
import importlib
import numpy as np
from typing import Any, Dict, Optional, Tuple


class AnimalProcessor:
    """Processor for estimating livestock weight from pose landmarks."""

    # Species-specific calibration based on agricultural morphometric data.
    # Formula: weight_kg = (length_cm * girth_cm^2) / divisor
    # where girth_cm = length_cm * girth_multiplier
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

    # Scale factor to convert shoulder→hip distance (MediaPipe) to full body length.
    # Shoulder-to-hip is roughly 45% of total body length for most livestock.
    BODY_LENGTH_SCALE = 2.2

    # Class-level cached MediaPipe objects to avoid re-loading models per request.
    _mp_pose = None
    _drawing_utils = None
    _pose_instance = None

    def __init__(self, pixel_to_cm_ratio: float = 0.264, animal_type: str = "dairy_cow") -> None:
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
        self.animal_type = animal_type.lower()
        if self.animal_type not in self.LIVESTOCK_CALIBRATION:
            self.animal_type = "dairy_cow"

        # Lazy-load MediaPipe once and cache at class level.
        if not (self.__class__._mp_pose and self.__class__._pose_instance):
            mp = None
            mp_solutions = None
            try:
                mp = importlib.import_module('mediapipe')
                mp_solutions = getattr(mp, 'solutions', None)
            except Exception:
                pass

            if mp_solutions is None:
                try:
                    mp_solutions = importlib.import_module('mediapipe.python.solutions')
                except Exception:
                    pass

            if mp_solutions is None:
                try:
                    mp_python = importlib.import_module('mediapipe.python')
                    mp_solutions = getattr(mp_python, 'solutions', None)
                except Exception as err:
                    raise ImportError('Could not import MediaPipe solutions.') from err

            self.__class__._mp_pose = mp_solutions.pose
            self.__class__._drawing_utils = mp_solutions.drawing_utils
            self.__class__._pose_instance = self.__class__._mp_pose.Pose(static_image_mode=True)

        self.mp_pose = self.__class__._mp_pose
        self.pose = self.__class__._pose_instance
        self.drawing_utils = self.__class__._drawing_utils

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def process(self, image_bgr: np.ndarray) -> Optional[Dict[str, object]]:
        """Estimate animal weight from a BGR image.

        Tries MediaPipe pose detection first for all species.
        Falls back to OpenCV contour detection if pose landmarks are not found.

        Returns a result dict or None if detection failed entirely.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        annotated_image = image_bgr.copy()
        method_used = None
        body_length_px = 0.0
        body_height_px = 0.0
        confidence_score = 0.0

        # ── Step 1: Try MediaPipe for ALL species ──
        if self.pose is not None:
            results_mp = self.pose.process(image_rgb)
            if results_mp.pose_landmarks:
                image_height, image_width = image_bgr.shape[:2]
                landmarks = results_mp.pose_landmarks.landmark
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip      = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                left_heel     = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]

                if (self._is_landmark_valid(left_shoulder)
                        and self._is_landmark_valid(left_hip)
                        and self._is_landmark_valid(left_heel)):

                    shoulder_px = self._normalized_to_pixel(left_shoulder, image_width, image_height)
                    hip_px      = self._normalized_to_pixel(left_hip,      image_width, image_height)
                    heel_px     = self._normalized_to_pixel(left_heel,     image_width, image_height)

                    # shoulder→hip is ~45% of body length, scale up to get full length
                    body_length_px = max(1.0, self._euclidean_distance(shoulder_px, hip_px)) * self.BODY_LENGTH_SCALE
                    body_height_px = max(1.0, self._euclidean_distance(hip_px, heel_px))

                    # Sanity check — reject near-zero detections
                    if body_length_px >= 30 and body_height_px >= 30:
                        self._draw_pose(annotated_image, results_mp)
                        confidence_score = float(np.mean([
                            left_shoulder.visibility,
                            left_hip.visibility,
                            left_heel.visibility
                        ]))
                        method_used = 'mediapipe'

        # ── Step 2: Contour fallback if MediaPipe failed ──
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
                # Bounding box is always larger than the animal — trim with scale factors
                body_length_px = (x2 - x1) * 0.75
                body_height_px = (y2 - y1) * 0.80
                method_used = 'contour_fallback'
                confidence_score = 0.6

        if method_used is None:
            return None

        # ── Step 3: Convert pixels → cm → weight ──
        ratio = self._get_pixel_to_cm_ratio()
        length_cm = body_length_px * ratio
        height_cm = body_height_px * ratio

        calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
        girth_cm = length_cm * calibration["girth_multiplier"]
        weight_kg = (length_cm * (girth_cm ** 2)) / calibration["divisor"]

        expected_min, expected_max = calibration.get('expected_range', (None, None))
        within_range = True
        if expected_min is not None and expected_max is not None:
            within_range = expected_min <= weight_kg <= expected_max

        return {
            "weight":               round(weight_kg, 2),
            "body_length":          round(length_cm, 2),
            "body_height":          round(height_cm, 2),
            "estimated_girth":      round(girth_cm, 2),
            "animal_type":          calibration["name"],
            "confidence_score":     round(confidence_score, 3),
            "annotated_image":      annotated_image,
            "expected_weight_range": calibration.get('expected_range'),
            "within_expected_range": within_range,
            "method":               method_used,
        }

    def calibrate_pixel_ratio(self, known_cm: float, measured_pixels: float) -> None:
        """Set pixel_to_cm_ratio from a known reference measurement."""
        if measured_pixels > 0:
            self.pixel_to_cm_ratio = known_cm / measured_pixels

    def adjust_weight_calibration(self, animal_type: str, divisor: float = None, girth_multiplier: float = None) -> None:
        """Fine-tune weight calibration for a specific animal type."""
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

        # Scale back to original dimensions
        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

        # Reject if contour covers less than 5% of the image (probably noise)
        if (w * h) < (width * height * 0.05):
            return None

        return (x, y, x + w, y + h)

    def _normalized_to_pixel(self, landmark: Any, image_width: int, image_height: int) -> Tuple[int, int]:
        x_px = int(max(0.0, min(1.0, landmark.x)) * image_width)
        y_px = int(max(0.0, min(1.0, landmark.y)) * image_height)
        return x_px, y_px

    def _euclidean_distance(self, point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(point_a, dtype=np.float32) - np.array(point_b, dtype=np.float32)))

    def _get_pixel_to_cm_ratio(self) -> float:
        return max(0.001, self.pixel_to_cm_ratio)

    def _is_landmark_valid(self, landmark: Any) -> bool:
        return landmark.visibility > 0.4 and 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0

    def _draw_pose(self, image: np.ndarray, results: Any) -> None:
        self.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=self.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
