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
        "dairy_cow":    {"divisor": 660.0,  "girth_multiplier": 0.84, "name": "Dairy Cow",          "expected_range": [250, 800],  "coco_classes": {21}}, # Cow
        "beef_cattle":  {"divisor": 600.0,  "girth_multiplier": 0.88, "name": "Beef Cattle",        "expected_range": [300, 1000], "coco_classes": {21}}, # Cow
        "young_cattle": {"divisor": 640.0,  "girth_multiplier": 0.82, "name": "Young Cattle/Calf",  "expected_range": [50, 300],   "coco_classes": {21}}, # Cow
        "pig":          {"divisor": 400.0,  "girth_multiplier": 0.92, "name": "Pig",                 "expected_range": [20, 400],   "coco_classes": {22}}, # Pig (Class 22)
        "goat":         {"divisor": 300.0,  "girth_multiplier": 0.78, "name": "Goat",                "expected_range": [15, 140],   "coco_classes": {20, 21}}, # Proxy: Sheep/Cow
        "sheep":        {"divisor": 300.0,  "girth_multiplier": 0.80, "name": "Sheep",               "expected_range": [20, 160],   "coco_classes": {20}}, # Sheep
        "donkey":       {"divisor": 480.0,  "girth_multiplier": 0.82, "name": "Donkey",              "expected_range": [80, 500],   "coco_classes": {19}}, # Proxy: Horse
        "poultry":      {"divisor": 1200.0, "girth_multiplier": 0.55, "name": "Poultry",             "expected_range": [0.5, 8],    "coco_classes": {16}}, # Bird
    }

    # Universally permitted COCO detection IDs: 16=bird, 19=horse, 20=sheep, 21=cow, 22=pig
    YOLO_LIVESTOCK_CLASSES = {16, 19, 20, 21, 22}

    _yolo_model = None

    def __init__(self, pixel_to_cm_ratio: float = 0.15, animal_type: str = "dairy_cow") -> None:
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
        self.animal_type = animal_type.lower()
        if self.animal_type not in self.LIVESTOCK_CALIBRATION:
            self.animal_type = "dairy_cow"

        if self.__class__._yolo_model is None:
            try:
                import os
                from ultralytics import YOLO
                
                # Build an absolute path targeting the models folder
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                local_weights_path = os.path.join(base_dir, "models", "yolov8n.pt")
                
                logging.info(f"Targeting local model path: {local_weights_path}")
                
                # Load the static weights directly without initiating external web downloads
                self.model = YOLO(local_weights_path)
                self.__class__._yolo_model = self.model
                logging.info("YOLOv8 Neural Network initialized successfully from local asset cache!")
            except Exception as e:
                logging.error(f"Failed to load YOLO weights locally: {e}")
                self.__class__._yolo_model = False

        self.model = self.__class__._yolo_model if self.__class__._yolo_model else None

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
            box = self._yolo_detect(image_bgr, target_coco_classes, yolo_results)
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

    def set_animal_type(self, animal_type: str) -> None:
        animal_type = animal_type.lower()
        if animal_type in self.LIVESTOCK_CALIBRATION:
            self.animal_type = animal_type
        else:
            raise ValueError(f"Invalid selector: {animal_type}")

    @classmethod
    def get_available_types(cls) -> dict:
        return cls.LIVESTOCK_CALIBRATION.copy()

    def _yolo_detect(self, image_bgr: np.ndarray, target_classes: set, results: Any = None) -> Optional[Tuple[int, int, int, int, float]]:
        """Runs context-aware YOLO filter passes, prioritizing the largest animal by area, and auto-corrects mismatched animal selections."""
        if results is None:
            try:
                results = self.model(image_bgr, verbose=False, conf=0.20)[0]
            except Exception as e:
                logging.error(f"YOLO engine failure: {e}")
                return None

        class_mapping = {16: "poultry", 20: "sheep", 21: "dairy_cow", 22: "pig", 19: "donkey"}

        # 1. Identify the single largest animal/object detected to determine the main subject
        main_box = None
        main_area = 0
        main_conf = 0.0
        main_cls = -1

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > main_area:
                main_area = area
                main_conf = conf
                main_box = (x1, y1, x2, y2, conf)
                main_cls = cls_id

        # 2. Find other boxes matching the target class group
        best_box = None
        best_area = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if cls_id in target_classes:
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2, conf)

        # 3. Find other broad livestock boxes
        best_broad_box = None
        best_broad_area = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.YOLO_LIVESTOCK_CLASSES:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > best_broad_area:
                    best_broad_area = area
                    best_broad_box = (x1, y1, x2, y2, conf)

        # 4. Resolve the best candidate box
        final_box = None
        if main_box is not None and (main_cls in target_classes or (main_cls in class_mapping and main_cls not in target_classes and main_conf > 0.60)):
            if main_cls in class_mapping and main_cls not in target_classes and main_conf > 0.60:
                detected_type = class_mapping[main_cls]
                logging.info(f"Auto-correcting animal type override to: {detected_type} (Largest subject)")
                self.set_animal_type(detected_type)
            final_box = main_box
        elif best_box is not None:
            final_box = best_box
        elif best_broad_box is not None:
            final_box = best_broad_box

        # 5. Check if the final selected box passes the minimum size threshold (noise reduction)
        if final_box is not None:
            x1, y1, x2, y2, conf = final_box
            h, w = image_bgr.shape[:2]
            if (x2 - x1) * (y2 - y1) < (w * h * 0.03):
                return None
            return final_box

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