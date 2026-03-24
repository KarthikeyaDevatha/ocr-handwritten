"""
Mathpix API OCR Engine implementation.
Provides a drop-in replacement for both text and math OCR using the Mathpix API.
"""

import time
import base64
import requests
import cv2
import numpy as np
from PIL import Image

from inference.ocr_math import MathOCRResult
from inference.ocr_text import OCRResult


class MathpixEngine:
    """Core engine to communicate with Mathpix API."""
    
    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key
        self.url = "https://api.mathpix.com/v3/text"

    def _call_api(self, image) -> dict:
        """Send image to Mathpix."""
        if isinstance(image, np.ndarray):
            # The pipeline passes RGB images
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', image_bgr)
            else:
                _, buffer = cv2.imencode('.jpg', image)
            img_b64 = base64.b64encode(buffer).decode("utf-8")
        elif isinstance(image, Image.Image):
            import io
            buf = io.BytesIO()
            image.convert('RGB').save(buf, format='JPEG')
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            raise ValueError("Unsupported image type")
            
        headers = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "Content-type": "application/json"
        }
        data = {
            "src": f"data:image/jpeg;base64,{img_b64}",
            "formats": ["text"]
        }
        response = requests.post(self.url, headers=headers, json=data, timeout=15)
        return response.json()

    def recognize_math(self, image) -> MathOCRResult:
        start_time = time.time()
        try:
            res = self._call_api(image)
            text = res.get("text", "")
            inference_time = time.time() - start_time
            
            error = res.get("error", "")
            if error:
                return MathOCRResult(
                    latex=f"[Mathpix Error: {error}]", 
                    confidence=0.0, 
                    inference_time=inference_time, 
                    discarded=True
                )
                
            is_display = "$$" in text or "\\[" in text
            
            # Clean up Mathpix delimiters so pipeline can use standard LaTeX
            text = text.replace("\\[", "").replace("\\]", "")
            text = text.replace("\\(", "").replace("\\)", "")
            text = text.replace("$$", "").replace("$", "").strip()
            
            return MathOCRResult(
                latex=text,
                confidence=0.99,  # High confidence for API
                inference_time=inference_time,
                discarded=False,
                is_display_math=is_display
            )
        except Exception as e:
            return MathOCRResult(
                latex="", 
                confidence=0.0, 
                inference_time=time.time() - start_time, 
                discarded=True
            )

    def recognize_text(self, image) -> OCRResult:
        start_time = time.time()
        try:
            res = self._call_api(image)
            text = res.get("text", "")
            inference_time = time.time() - start_time
            
            error = res.get("error", "")
            if error:
                 return OCRResult(
                     text=f"[Mathpix Error: {error}]", 
                     confidence=0.0, 
                     inference_time=inference_time, 
                     rerouted=True
                 )
            
            # Remove accidental math delimiters
            text = text.replace("\\[", "").replace("\\]", "")
            text = text.replace("\\(", "").replace("\\)", "")
            text = text.replace("$$", "").replace("$", "").strip()
            
            return OCRResult(
                text=text,
                confidence=0.99,
                inference_time=inference_time,
                rerouted=False
            )
        except Exception as e:
            return OCRResult(
                text="", 
                confidence=0.0, 
                inference_time=time.time() - start_time, 
                rerouted=True
            )


class MathpixMathWrapper:
    """Wrapper to act as MathOCR."""
    def __init__(self, engine: MathpixEngine):
        self.engine = engine
        self.timeout = 15.0  # Dummy attr for pipeline checks
        self.confidence_threshold = 0.4
        
    def recognize(self, image) -> MathOCRResult:
        return self.engine.recognize_math(image)


class MathpixTextWrapper:
    """Wrapper to act as TextOCR."""
    def __init__(self, engine: MathpixEngine):
        self.engine = engine
        self.timeout = 15.0
        self.confidence_threshold = 0.4
        self.max_new_tokens = 200
        
    def recognize(self, image) -> OCRResult:
        return self.engine.recognize_text(image)
