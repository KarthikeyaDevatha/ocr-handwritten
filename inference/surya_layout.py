"""
Surya-based Layout Detection module.
Replaces YOLO-based layout detection with Surya's document-specialized model.
Falls back to the existing YOLO/CV detector if Surya is not available.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
from dataclasses import dataclass

from inference.layout import Detection


# Class name mapping from Surya's label set to our internal labels
SURYA_CLASS_MAP = {
    "Text": "text_line",
    "TextInlineMath": "text_line",      # Treat as text with inline math
    "Title": "text_line",
    "SectionHeader": "text_line",
    "Caption": "text_line",
    "Footnote": "text_line",
    "PageHeader": "text_line",
    "PageFooter": "text_line",
    "Formula": "math_formula",
    "DisplayFormula": "math_formula",
    "Table": "table",
    "Figure": "figure",
    "Picture": "figure",
    "ListItem": "text_line",
    "Code": "text_line",
}

# Map our internal class names to numeric IDs
CLASS_NAME_TO_ID = {
    "text_line": 0,
    "math_formula": 1,
    "table": 2,
    "figure": 3,
}


class SuryaLayoutDetector:
    """
    Document layout detector using Surya.
    Detects text, math, tables, and figures with high accuracy.
    """

    def __init__(
        self,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        """
        Initialize Surya layout detector.

        Args:
            conf_threshold: Minimum confidence threshold for detections
            device: 'cpu' or 'cuda'
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.processor = None
        self._available = False

        try:
            from surya.detection import DetectionPredictor
            self.predictor = DetectionPredictor(device=device)
            self._available = True
            print("[OK] Surya layout detector loaded")
        except ImportError:
            print("[WARN] Surya not installed. Install with: pip install surya-ocr")
            print("       Falling back to YOLO/CV layout detection.")
        except Exception as e:
            print(f"[WARN] Surya initialization failed: {e}")
            print("       Falling back to YOLO/CV layout detection.")

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run layout detection on an image.

        Args:
            image: Input image (BGR numpy array)

        Returns:
            List of Detection objects
        """
        if not self._available:
            return []

        # Convert BGR to RGB PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        pil_image = Image.fromarray(rgb)

        try:
            # Run Surya detection
            predictions = self.predictor([pil_image])

            detections = []
            if predictions and len(predictions) > 0:
                page_result = predictions[0]

                # Surya returns bboxes with labels
                for bbox_result in page_result.bboxes:
                    confidence = getattr(bbox_result, 'confidence', 0.5)

                    if confidence < self.conf_threshold:
                        continue

                    # Get label and map to our class system
                    label = getattr(bbox_result, 'label', 'Text')
                    class_name = SURYA_CLASS_MAP.get(label, "text_line")
                    class_id = CLASS_NAME_TO_ID.get(class_name, 0)

                    # Get bounding box coordinates
                    bbox = bbox_result.bbox  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    # Clamp to image bounds
                    h, w = image.shape[:2]
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    # Skip tiny detections
                    if (x2 - x1) < 10 or (y2 - y1) < 5:
                        continue

                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidence)
                    ))

            return detections

        except Exception as e:
            print(f"Surya detection error: {e}")
            return []


class HybridLayoutDetector:
    """
    Hybrid layout detector that uses Surya as primary and falls back
    to YOLO or CV-based detection.
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolo/yolov8n-layout.onnx",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cpu",
    ):
        """
        Initialize hybrid detector — tries Surya first, then YOLO, then CV.

        Args:
            yolo_model_path: Path to fallback YOLO ONNX model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            input_size: YOLO input size
            device: 'cpu' or 'cuda'
        """
        self.conf_threshold = conf_threshold

        # Primary: Surya
        self.surya_detector = SuryaLayoutDetector(
            conf_threshold=conf_threshold,
            device=device,
        )

        # Fallback: existing YOLO + CV detector
        from inference.layout import LayoutDetector
        self.yolo_detector = LayoutDetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
            device=device,
        )

        if self.surya_detector.is_available:
            print("Layout strategy: PRIMARY=Surya, FALLBACK=YOLO/CV")
        else:
            print("Layout strategy: PRIMARY=YOLO/CV (Surya unavailable)")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect layout regions using best available model.

        Args:
            image: Input image (BGR numpy array)

        Returns:
            List of Detection objects
        """
        # Try Surya first
        if self.surya_detector.is_available:
            detections = self.surya_detector.detect(image)
            if detections:
                return detections
            print("Surya returned no detections — falling back to YOLO/CV")

        # Fallback to YOLO/CV
        return self.yolo_detector.detect(image)
