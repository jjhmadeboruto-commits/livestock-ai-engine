from __future__ import annotations

import cv2
import importlib
import numpy as np
from typing import Any, Dict, Optional, Tuple


class AnimalProcessor:
    """Processor for estimating livestock weight from pose landmarks."""

    # Calibration data for different livestock types based on agricultural data
    LIVESTOCK_CALIBRATION = {
        "dairy_cow": {
            "divisor": 11888.0,
            "girth_multiplier": 1.20,
            "name": "Dairy Cow (300-600 kg)",
            "expected_range": [250, 700]
        },
        "beef_cattle": {
            "divisor": 11888.0,
            "girth_multiplier": 1.22,
            "name": "Beef Cattle (350-700 kg)",
            "expected_range": [300, 800]
        },
        "young_cattle": {
            "divisor": 11888.0,
            "girth_multiplier": 1.15,
            "name": "Young Cattle/Calf (100-300 kg)",
            "expected_range": [80, 350]
        },
        "goat": {
            "divisor": 11888.0,
            "girth_multiplier": 1.45,
            "name": "Goat (30-100 kg)",
            "expected_range": [20, 120]
        },
        "sheep": {
            "divisor": 11888.0,
            "girth_multiplier": 1.35,
            "name": "Sheep (40-120 kg)",
            "expected_range": [30, 140]
        },
        "donkey": {
            "divisor": 11888.0,
            "girth_multiplier": 1.25,
            "name": "Donkey (150-400 kg)",
            "expected_range": [120, 450]
        },
        "poultry": {
            "divisor": 11888.0,
            "girth_multiplier": 0.60,
            "name": "Poultry (1-8 kg)",
            "expected_range": [0.5, 12]
        },
        "pig": {
            "divisor": 11888.0,
            "girth_multiplier": 1.50,
            "name": "Pig (50-300 kg)",
            "expected_range": [30, 350]
        }
    }

    # Class-level cached MediaPipe objects to avoid re-loading models per-request
    _mp_pose = None
    _drawing_utils = None
    _pose_instance = None
    _yolo_model = None

    def __init__(self, pixel_to_cm_ratio: float = 0.264, animal_type: str = "dairy_cow") -> None:
        """Initialize MediaPipe Pose and scaling configuration.
        
        Args:
            pixel_to_cm_ratio: Conversion factor from pixels to centimeters
            animal_type: Type of animal ('dairy_cow', 'beef_cattle', 'young_cattle', 'goat', 'sheep')
        """
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
        self.animal_type = animal_type.lower()
        if self.animal_type not in self.LIVESTOCK_CALIBRATION:
            self.animal_type = "dairy_cow"

        # Lazy import MediaPipe and cache heavy objects at the class level so
        # subsequent requests reuse the loaded model and drawing utilities.
        if not (self.__class__._mp_pose and self.__class__._pose_instance and self.__class__._drawing_utils):
            mp = None
            mp_solutions = None
            try:
                mp = importlib.import_module('mediapipe')
                mp_solutions = getattr(mp, 'solutions', None)
            except Exception:
                mp = None

            if mp_solutions is None:
                try:
                    mp_solutions = importlib.import_module('mediapipe.python.solutions')
                except Exception:
                    mp_solutions = None

            if mp_solutions is None:
                try:
                    mp_python = importlib.import_module('mediapipe.python')
                    mp_solutions = getattr(mp_python, 'solutions', None)
                except Exception as err:
                    raise ImportError(
                        'Could not import MediaPipe solutions from mediapipe or mediapipe.python'
                    ) from err

            # Cache the pose class and drawing utils and create a single Pose instance
            self.__class__._mp_pose = mp_solutions.pose
            self.__class__._drawing_utils = mp_solutions.drawing_utils
            # Creating a single Pose() instance avoids re-loading weights on every request.
            self.__class__._pose_instance = self.__class__._mp_pose.Pose(static_image_mode=True)

        # Instance-level references point to the cached class objects.
        self.mp_pose = self.__class__._mp_pose
        self.pose = self.__class__._pose_instance
        self.drawing_utils = self.__class__._drawing_utils

    def _get_yolo_model(self):
        """Lazily load YOLO model."""
        if self.__class__._yolo_model is None:
            try:
                from ultralytics import YOLO
                self.__class__._yolo_model = YOLO('yolov8n.pt')
            except ImportError:
                return None
        return self.__class__._yolo_model

    def process(self, image_bgr: np.ndarray) -> Optional[Dict[str, object]]:
        """Estimate animal weight from a BGR image and annotate detected landmarks.

        Args:
            image_bgr: The input image in BGR format.

        Returns:
            A dictionary with results if landmarks were detected, otherwise None.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Determine whether to prioritize MediaPipe or YOLO based on animal type
        # MediaPipe is generally better for cattle, goats, sheep
        yolo_preferred = self.animal_type in ["pig", "poultry"]
        
        results = None
        method_used = None
        body_length_px = 0
        body_height_px = 0
        annotated_image = image_bgr.copy()
        confidence_score = 0.0

        if not yolo_preferred and self.pose is not None:
            results_mp = self.pose.process(image_rgb)
            if results_mp.pose_landmarks:
                image_height, image_width = image_bgr.shape[:2]
                landmarks = results_mp.pose_landmarks.landmark
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]

                if self._is_landmark_valid(left_shoulder) and self._is_landmark_valid(left_hip) and self._is_landmark_valid(left_heel):
                    shoulder_px = self._normalized_to_pixel(left_shoulder, image_width, image_height)
                    hip_px = self._normalized_to_pixel(left_hip, image_width, image_height)
                    heel_px = self._normalized_to_pixel(left_heel, image_width, image_height)

                    body_length_px = self._euclidean_distance(shoulder_px, hip_px)
                    body_height_px = self._euclidean_distance(hip_px, heel_px)
                    self._draw_pose(annotated_image, results_mp)
                    confidence_score = float(np.mean([left_shoulder.visibility, left_hip.visibility, left_heel.visibility]))
                    method_used = 'mediapipe'

        # Fallback to YOLO if MediaPipe failed or YOLO is preferred
        if method_used is None:
            yolo_model = self._get_yolo_model()
            if yolo_model is not None:
                results_yolo = yolo_model(image_rgb, verbose=False)
                boxes = results_yolo[0].boxes if len(results_yolo) > 0 else None
                if boxes is not None and len(boxes) > 0:
                    # Find the largest bounding box (assuming it's the main animal)
                    largest_box = None
                    max_area = 0
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            largest_box = (x1, y1, x2, y2)
                            confidence_score = float(box.conf[0].cpu().numpy())
                    
                    if largest_box is not None:
                        x1, y1, x2, y2 = largest_box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_image, f"{self.LIVESTOCK_CALIBRATION[self.animal_type]['name']} (YOLO)", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Use bounding box dimensions as a proxy for body length and height
                        # We apply a slight scaling factor since bounding box includes head/legs
                        # whereas mediapipe shoulder-to-hip is shorter
                        scale_length = 0.85
                        scale_height = 0.90
                        body_length_px = (x2 - x1) * scale_length
                        body_height_px = (y2 - y1) * scale_height
                        method_used = 'yolo'

        if method_used is None:
            return None

        length_cm = body_length_px * self._get_pixel_to_cm_ratio()
        height_cm = body_height_px * self._get_pixel_to_cm_ratio()

        calibration = self.LIVESTOCK_CALIBRATION[self.animal_type]
        estimated_girth_cm = length_cm * calibration["girth_multiplier"]
        weight_kg = (length_cm * (estimated_girth_cm ** 2)) / calibration["divisor"]

        expected_min, expected_max = calibration.get('expected_range', (None, None))
        is_within_expected = True
        if expected_min is not None and expected_max is not None:
            is_within_expected = expected_min <= weight_kg <= expected_max

        return {
            "weight": round(weight_kg, 2),
            "body_length": round(length_cm, 2),
            "body_height": round(height_cm, 2),
            "estimated_girth": round(estimated_girth_cm, 2),
            "animal_type": self.LIVESTOCK_CALIBRATION[self.animal_type]["name"],
            "confidence_score": round(confidence_score, 3),
            "annotated_image": annotated_image,
            "expected_weight_range": calibration.get('expected_range'),
            "within_expected_range": is_within_expected,
            "method": method_used,
        }

    def _normalized_to_pixel(
        self, landmark: mp.framework.formats.landmark_pb2.NormalizedLandmark, image_width: int, image_height: int
    ) -> Tuple[int, int]:
        """Convert normalized landmark coordinates to pixel coordinates."""
        x_px = int(max(0.0, min(1.0, landmark.x)) * image_width)
        y_px = int(max(0.0, min(1.0, landmark.y)) * image_height)
        return x_px, y_px

    def _euclidean_distance(self, point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
        """Compute Euclidean distance between two pixel points."""
        return float(np.linalg.norm(np.array(point_a, dtype=np.float32) - np.array(point_b, dtype=np.float32)))

    def _get_pixel_to_cm_ratio(self) -> float:
        """Return the active pixel-to-centimeter conversion ratio."""
        return self.pixel_to_cm_ratio

    def set_animal_type(self, animal_type: str) -> None:
        """Set the animal type for weight calibration.
        
        Args:
            animal_type: One of 'dairy_cow', 'beef_cattle', 'young_cattle', 'goat', 'sheep'
        """
        animal_type = animal_type.lower()
        if animal_type in self.LIVESTOCK_CALIBRATION:
            self.animal_type = animal_type
        else:
            raise ValueError(f"Unknown animal type: {animal_type}. Available: {list(self.LIVESTOCK_CALIBRATION.keys())}")

    def calibrate_pixel_ratio(self, known_cm: float, measured_pixels: float) -> None:
        """Calibrate the pixel-to-cm ratio using a known measurement.
        
        Args:
            known_cm: Known measurement in centimeters (e.g., body part length)
            measured_pixels: Corresponding measurement in pixels from image
        """
        if measured_pixels > 0:
            self.pixel_to_cm_ratio = known_cm / measured_pixels

    def adjust_weight_calibration(self, animal_type: str, divisor: float = None, girth_multiplier: float = None) -> None:
        """Fine-tune weight calibration for a specific animal type.
        
        Args:
            animal_type: Type of animal to calibrate
            divisor: New divisor value for weight formula
            girth_multiplier: New girth multiplier value
        """
        if animal_type not in self.LIVESTOCK_CALIBRATION:
            raise ValueError(f"Unknown animal type: {animal_type}")
        
        if divisor is not None:
            self.LIVESTOCK_CALIBRATION[animal_type]["divisor"] = divisor
        if girth_multiplier is not None:
            self.LIVESTOCK_CALIBRATION[animal_type]["girth_multiplier"] = girth_multiplier

    @classmethod
    def get_available_types(cls) -> dict:
        """Get information about all available livestock types and their current calibration."""
        return cls.LIVESTOCK_CALIBRATION.copy()

    def _is_landmark_valid(self, landmark: mp.framework.formats.landmark_pb2.NormalizedLandmark) -> bool:
        """Check whether a landmark is sufficiently visible and within normalized bounds."""
        return landmark.visibility > 0.4 and 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0

    def _draw_pose(self, image: np.ndarray, results: Any) -> None:
        """Draw landmarks and pose connections onto the image."""
        self.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=self.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
