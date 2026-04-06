"""
UniMERNet-based Math OCR module.
Provides state-of-the-art mathematical expression recognition using the
UniMERNet (Universal Mathematical Expression Recognition Network) model.

Falls back to existing TrOCR-based math OCR or pix2tex if unavailable.

Architecture: Swin Transformer encoder + mBART decoder
Trained on: UniMER-1M dataset (1M+ image-LaTeX pairs)
"""

import os
import re
import time
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple
from dataclasses import dataclass

from inference.ocr_math import MathOCRResult


class UniMERNetEngine:
    """
    Math OCR engine using the UniMERNet model.
    Converts images of mathematical expressions to LaTeX.
    """

    # Pattern to determine if output looks like valid math
    MATH_PATTERN = re.compile(r'[\\\\_\^{}]')

    def __init__(
        self,
        model_name: str = "wanderkid/unimernet_base",
        device: str = "cpu",
        max_new_tokens: int = 512,
        confidence_threshold: float = 0.4,
    ):
        """
        Initialize UniMERNet engine.

        Args:
            model_name: HuggingFace model name or local path
            device: 'cpu' or 'cuda'
            max_new_tokens: Maximum tokens to generate
            confidence_threshold: Minimum confidence to accept result
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.confidence_threshold = confidence_threshold
        self._available = False
        self.model = None
        self.processor = None

        try:
            from unimernet.common.config import Config
            from unimernet.processors import load_processor
            import unimernet.tasks as tasks
            from torchvision import transforms as T
            import torch

            # Try to load UniMERNet
            self.torch = torch
            cfg_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "unimernet_base.yaml"
            )

            # If config doesn't exist, use HuggingFace-based loading
            if not os.path.exists(cfg_path):
                self._init_from_huggingface(model_name, device)
            else:
                self._init_from_config(cfg_path, device)

        except ImportError:
            print("[WARN] UniMERNet not installed. Trying alternative loading method...")
            self._init_from_transformers(model_name, device)

    def _init_from_huggingface(self, model_name: str, device: str):
        """Initialize from HuggingFace transformers directly."""
        try:
            from transformers import (
                VisionEncoderDecoderModel,
                AutoProcessor,
                AutoTokenizer,
                SwinModel,
            )
            import torch

            print(f"Loading UniMERNet from HuggingFace: {model_name}")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model.eval()

            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"

            self._available = True
            print(f"[OK] UniMERNet loaded on {self.device}")

        except Exception as e:
            print(f"[WARN] Could not load UniMERNet from HuggingFace: {e}")
            self._available = False

    def _init_from_transformers(self, model_name: str, device: str):
        """Try generic transformers-based loading as last resort."""
        try:
            from transformers import VisionEncoderDecoderModel, TrOCRProcessor
            import torch

            # Try loading as a VisionEncoderDecoder model
            print(f"Attempting generic VED loading for: {model_name}")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model.eval()

            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"

            self._available = True
            print(f"[OK] Math OCR model loaded on {self.device}")

        except Exception as e:
            print(f"[WARN] Could not load math model: {e}")
            self._available = False

    def _init_from_config(self, cfg_path: str, device: str):
        """Initialize from UniMERNet config file."""
        try:
            from unimernet.common.config import Config
            import unimernet.tasks as tasks
            import torch

            cfg = Config(cfg_path)
            task = tasks.setup_task(cfg)
            self.model = task.build_model(cfg)
            self.model.eval()

            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"

            self._available = True
            print(f"[OK] UniMERNet loaded from config on {self.device}")

        except Exception as e:
            print(f"[WARN] Could not load UniMERNet from config: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def _postprocess_latex(self, latex: str) -> str:
        """Clean up and fix common LaTeX issues."""
        latex = latex.strip()

        # Remove BOS/EOS tokens if present
        for token in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]"]:
            latex = latex.replace(token, "")

        # Remove $ delimiters if present
        if latex.startswith("$$") and latex.endswith("$$"):
            latex = latex[2:-2]
        elif latex.startswith("$") and latex.endswith("$"):
            latex = latex[1:-1]

        # Remove \[ \] delimiters
        latex = latex.replace("\\[", "").replace("\\]", "")
        latex = latex.replace("\\(", "").replace("\\)", "")

        latex = latex.strip()

        # Balance braces
        open_braces = latex.count('{')
        close_braces = latex.count('}')
        if open_braces > close_braces:
            latex += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            latex = '{' * (close_braces - open_braces) + latex

        # Fix spacing around common commands
        latex = re.sub(r'\\frac\s*{', r'\\frac{', latex)
        latex = re.sub(r'\\sqrt\s*{', r'\\sqrt{', latex)

        return latex

    def _is_display_math(self, latex: str) -> bool:
        """Determine if expression should be display math."""
        display_patterns = [
            r'\\frac', r'\\sum', r'\\prod', r'\\int', r'\\lim',
            r'\\begin\{', r'\\matrix', r'\\iint', r'\\iiint',
            r'\\oint', r'\\binom',
        ]
        for pattern in display_patterns:
            if re.search(pattern, latex):
                return True
        return False

    def _validate_math_output(self, latex: str) -> bool:
        """Validate that output looks like mathematical LaTeX."""
        if not latex or len(latex) < 2:
            return False
        if not self.MATH_PATTERN.search(latex):
            return False
        # Check brace balance
        if latex.count('{') != latex.count('}'):
            return False
        return True

    def recognize(self, image) -> MathOCRResult:
        """
        Recognize mathematical expression in image.

        Args:
            image: Input image (numpy array RGB or PIL Image)

        Returns:
            MathOCRResult with LaTeX and metadata
        """
        start_time = time.time()

        if not self._available:
            return MathOCRResult(
                latex="[UniMERNet model not loaded]",
                confidence=0.0,
                inference_time=time.time() - start_time,
                discarded=True,
            )

        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = image

        try:
            import torch

            # Process image
            if self.processor is not None:
                pixel_values = self.processor(
                    images=pil_image, return_tensors="pt"
                ).pixel_values
            else:
                # Fallback: manual preprocessing
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
                pixel_values = transform(pil_image).unsqueeze(0)

            if self.device == "cuda":
                pixel_values = pixel_values.cuda()

            # Generate LaTeX
            with torch.no_grad():
                if hasattr(self.model, "generate"):
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=5,
                        early_stopping=True,
                        no_repeat_ngram_size=5,
                    )
                else:
                    # UniMERNet-specific generate call
                    generated_ids = self.model.generate(
                        {"image": pixel_values}
                    )

            # Decode
            if self.processor is not None and hasattr(self.processor, "batch_decode"):
                latex = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            elif self.processor is not None and hasattr(self.processor, "tokenizer"):
                latex = self.processor.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
            else:
                latex = str(generated_ids)

            # Post-process
            latex = self._postprocess_latex(latex)

            inference_time = time.time() - start_time
            is_valid = self._validate_math_output(latex)
            is_display = self._is_display_math(latex)

            return MathOCRResult(
                latex=latex,
                confidence=0.95 if is_valid else 0.3,
                inference_time=inference_time,
                discarded=not is_valid,
                is_display_math=is_display,
            )

        except Exception as e:
            print(f"UniMERNet inference error: {e}")
            return MathOCRResult(
                latex="",
                confidence=0.0,
                inference_time=time.time() - start_time,
                discarded=True,
            )


class UniMERNetWrapper:
    """
    Wrapper to make UniMERNetEngine compatible with the pipeline's
    MathOCR interface (exposes .recognize() and required attributes).
    """

    def __init__(self, engine: UniMERNetEngine):
        self.engine = engine
        self.timeout = 10.0
        self.confidence_threshold = 0.4
        self.max_new_tokens = 512

    def recognize(self, image) -> MathOCRResult:
        return self.engine.recognize(image)

    def batch_recognize(self, images: List) -> List[MathOCRResult]:
        return [self.engine.recognize(img) for img in images]


class CascadingMathOCR:
    """
    Multi-model cascading math OCR.
    Tries UniMERNet first, falls back to pix2tex, then TrOCR-math.
    Uses confidence thresholds to decide when to cascade.
    """

    def __init__(
        self,
        device: str = "cpu",
        high_conf_threshold: float = 0.85,
        med_conf_threshold: float = 0.6,
    ):
        """
        Initialize cascading math OCR.

        Args:
            device: 'cpu' or 'cuda'
            high_conf_threshold: Accept if primary model exceeds this
            med_conf_threshold: Try fallback if below this
        """
        self.high_conf_threshold = high_conf_threshold
        self.med_conf_threshold = med_conf_threshold
        self.models = []

        # Try loading models in priority order
        # 1. UniMERNet (best accuracy)
        engine = UniMERNetEngine(device=device)
        if engine.is_available:
            self.models.append(("UniMERNet", UniMERNetWrapper(engine)))

        # 2. Pix2Tex (good backup)
        try:
            from inference.pix2tex_math import Pix2TexEngine, Pix2TexWrapper
            pix2tex = Pix2TexEngine()
            self.models.append(("Pix2Tex", Pix2TexWrapper(pix2tex)))
        except Exception:
            pass

        # 3. TrOCR-Math (last resort)
        try:
            from inference.ocr_math import create_math_ocr
            trocr_math = create_math_ocr(device=device)
            self.models.append(("TrOCR-Math", trocr_math))
        except Exception:
            pass

        model_names = [name for name, _ in self.models]
        print(f"CascadingMathOCR initialized with: {model_names}")

    def recognize(self, image) -> MathOCRResult:
        """
        Recognize math using cascading models.
        Returns the first high-confidence result, or the best across all models.
        """
        best_result = None

        for name, model in self.models:
            try:
                result = model.recognize(image)

                # If high confidence, accept immediately
                if result.confidence >= self.high_conf_threshold and not result.discarded:
                    return result

                # Track best result so far
                if best_result is None or (
                    result.confidence > best_result.confidence and not result.discarded
                ):
                    best_result = result

                # If medium confidence, try next model
                if result.confidence >= self.med_conf_threshold and not result.discarded:
                    continue  # Try next but keep this as candidate

            except Exception as e:
                print(f"  {name} failed: {e}")
                continue

        # Return best result found, or failure
        if best_result is not None:
            return best_result

        return MathOCRResult(
            latex="",
            confidence=0.0,
            inference_time=0.0,
            discarded=True,
        )
