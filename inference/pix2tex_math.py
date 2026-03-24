"""
Pix2Tex (LaTeX-OCR) Engine implementation.
Provides an open-source, local equivalent to Mathpix.
"""

import time
import cv2
import numpy as np
from PIL import Image

from inference.ocr_math import MathOCRResult


class Pix2TexEngine:
    """Core engine to load and communicate with local Pix2Tex."""
    
    def __init__(self):
        try:
            from pix2tex.cli import LatexOCR
            self.model = LatexOCR()
        except ImportError:
            print("Error: pix2tex is not installed. Run `pip install pix2tex`.")
            self.model = None

    def _prepare_image(self, image) -> Image.Image:
        """Pad and resize image to square as required by some math models or Pix2Tex optimal sizing."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed, but pipeline normally passes RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                # We assume RGB from pipeline
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
            
        return pil_image

    def recognize_math(self, image) -> MathOCRResult:
        start_time = time.time()
        
        if self.model is None:
             return MathOCRResult(
                latex="[Error: Pix2Tex not installed]",
                confidence=0.0,
                inference_time=time.time() - start_time,
                discarded=True
            )
            
        try:
            pil_image = self._prepare_image(image)
            
            # Predict using LaTeX OCR
            # Usually it returns a string of the LaTeX
            prediction = self.model(pil_image)
            latex_str = str(prediction).strip()
            
            inference_time = time.time() - start_time
            
            is_display = "\\displaystyle" in latex_str or "\\int" in latex_str or "\\sum" in latex_str
            
            return MathOCRResult(
                latex=latex_str,
                confidence=0.95,  # Pix2Tex doesn't natively expose conf easily, assume high if successful
                inference_time=inference_time,
                discarded=False,
                is_display_math=is_display
            )
            
        except Exception as e:
            return MathOCRResult(
                latex=f"[Pix2Tex Error: {str(e)}]", 
                confidence=0.0, 
                inference_time=time.time() - start_time, 
                discarded=True
            )


class Pix2TexWrapper:
    """Wrapper to act as MathOCR."""
    def __init__(self, engine: Pix2TexEngine):
        self.engine = engine
        self.timeout = 15.0
        self.confidence_threshold = 0.4
        self.max_new_tokens = 200
        
    def recognize(self, image) -> MathOCRResult:
        return self.engine.recognize_math(image)
