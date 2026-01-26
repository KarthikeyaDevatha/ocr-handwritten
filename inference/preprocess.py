"""
Image preprocessing utilities for OCR pipeline.
Handles letterbox resize, grayscale conversion, and stride-32 padding.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        target_size: Target (height, width)
        color: Padding color
        auto: Minimum rectangle padding
        scale_fill: Stretch to fill (no padding)
        stride: Stride for padding alignment
    
    Returns:
        Tuple of (padded_image, scale_ratio, (pad_w, pad_h))
    """
    shape = image.shape[:2]  # current shape [height, width]
    
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    # Scale ratio (new / old)
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    
    # Compute new unpadded size
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (target_size[1], target_size[0])
        r = target_size[1] / shape[1], target_size[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color)
    
    return image, r, (int(dw), int(dh))


def stride32_pad(
    image: np.ndarray,
    stride: int = 32,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to be divisible by stride.
    
    Args:
        image: Input image
        stride: Stride value (default 32 for YOLO)
        color: Padding color
    
    Returns:
        Tuple of (padded_image, original_size)
    """
    h, w = image.shape[:2]
    original_size = (h, w)
    
    new_h = int(np.ceil(h / stride) * stride)
    new_w = int(np.ceil(w / stride) * stride)
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=color
        )
    
    return image, original_size


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_for_ocr(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image for OCR models (ImageNet normalization).
    
    Args:
        image: Input image (H, W, C) in range [0, 255]
        mean: Channel means
        std: Channel stds
    
    Returns:
        Normalized image in range suitable for model input
    """
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image


def deskew_image(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Detect and correct skew in document image.
    
    Args:
        image: Input grayscale image
        max_angle: Maximum skew angle to correct
    
    Returns:
        Deskewed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find coordinates of non-zero pixels
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) < 10:
        return image
    
    # Get rotation angle from minAreaRect
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Clamp to max angle
    if abs(angle) > max_angle:
        return image
    
    # Rotate
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if len(image.shape) == 3:
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    if len(image.shape) == 3:
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def binarize_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """Apply adaptive thresholding for binarization."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )


def preprocess_for_layout(
    image: Union[np.ndarray, Image.Image, str],
    target_size: int = 640,
    stride: int = 32
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for layout detection.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        target_size: Target size for layout model
        stride: Stride for padding
    
    Returns:
        Tuple of (preprocessed_image, metadata_dict)
    """
    # Load image
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    original_shape = img.shape[:2]
    
    # Deskew
    deskewed = deskew_image(img)
    
    # Letterbox resize
    resized, ratio, pad = letterbox_resize(
        deskewed, 
        target_size=(target_size, target_size),
        stride=stride
    )
    
    # Normalize for YOLO (0-1 range)
    normalized = resized.astype(np.float32) / 255.0
    
    # HWC to CHW
    normalized = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    normalized = np.expand_dims(normalized, 0)
    
    metadata = {
        "original_shape": original_shape,
        "ratio": ratio,
        "pad": pad,
        "resized_shape": resized.shape[:2]
    }
    
    return normalized, metadata


def preprocess_for_ocr(
    image: Union[np.ndarray, Image.Image, str],
    target_size: Optional[Tuple[int, int]] = None,
    grayscale: bool = False
) -> np.ndarray:
    """
    Preprocessing for TrOCR input.
    
    Args:
        image: Input image
        target_size: Optional resize target (H, W)
        grayscale: Whether to convert to grayscale
    
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            img = Image.fromarray(image).convert('RGB')
        else:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image.convert('RGB')
    
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    
    if grayscale:
        img = img.convert('L').convert('RGB')
    
    return np.array(img)


def crop_region(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 5
) -> np.ndarray:
    """
    Crop a region from image with optional padding.
    
    Args:
        image: Source image
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Padding pixels
    
    Returns:
        Cropped image region
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding and clamp
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]
