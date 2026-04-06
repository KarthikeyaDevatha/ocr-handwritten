"""
TexTeller Math OCR Engine implementation.
Provides an open-source, local alternative to Mathpix (previously pix2tex).
"""

import time
import cv2
import numpy as np
from PIL import Image
import torch

from inference.ocr_math import MathOCRResult


class Pix2TexEngine:
    """Core engine backed by TexTeller (drop-in replacement for pix2tex).
    
    Model loading is lazy — deferred to the first recognize_math() call —
    so import/init never crashes Streamlit at startup.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._img2latex = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_error: str = ""
        self.model = True  # truthy sentinel; callers check this

    def _ensure_loaded(self) -> bool:
        """Lazily load TexTeller on first use. Returns True if ready."""
        if self._model is not None:
            return True
        if self._load_error:
            return False  # already failed, don't retry
        try:
            from texteller import load_model, load_tokenizer, img2latex
            self._img2latex = img2latex
            self._model = load_model(use_onnx=False)
            self._tokenizer = load_tokenizer()
            return True
        except Exception as e:
            self._load_error = str(e)
            self.model = None
            print(f"TexTeller load failed: {e}")
            return False

    def _prepare_image(self, image) -> np.ndarray:
        """Return an RGB numpy array suitable for texteller."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                return image  # assume RGB from pipeline
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # PIL Image
        return np.array(image.convert("RGB"))

    def recognize_math(self, image) -> MathOCRResult:
        start_time = time.time()

        if not self._ensure_loaded():
            return MathOCRResult(
                latex=f"[TexTeller unavailable: {self._load_error}]",
                confidence=0.0,
                inference_time=time.time() - start_time,
                discarded=True
            )

        try:
            rgb_img = self._prepare_image(image)

            results = self._img2latex(
                self._model,
                self._tokenizer,
                [rgb_img],
                device=self._device,
            )
            latex_str = results[0].strip() if results else ""

            inference_time = time.time() - start_time

            is_display = any(kw in latex_str for kw in (
                "\\displaystyle", "\\int", "\\sum", "\\frac", "\\begin"
            ))

            return MathOCRResult(
                latex=latex_str,
                confidence=0.95,
                inference_time=inference_time,
                discarded=False,
                is_display_math=is_display
            )

        except Exception as e:
            return MathOCRResult(
                latex=f"[TexTeller Error: {str(e)}]",
                confidence=0.0,
                inference_time=time.time() - start_time,
                discarded=True
            )


class Pix2TexWrapper:
    """Wrapper to act as MathOCR — interface unchanged."""
    def __init__(self, engine: Pix2TexEngine):
        self.engine = engine
        self.timeout = 15.0
        self.confidence_threshold = 0.4
        self.max_new_tokens = 200

    def recognize(self, image) -> MathOCRResult:
        return self.engine.recognize_math(image)
