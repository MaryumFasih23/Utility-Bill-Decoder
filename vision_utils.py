"""
vision_utils.py - Vision preprocessing, quality inspection, and section cropping.
"""

from typing import Tuple, Dict, Any, Optional
import os
import io
from PIL import Image, ImageOps
import cv2
import numpy as np


def load_image(image_path: str) -> Image.Image:
    """Load an image using PIL, correcting EXIF orientation if needed."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = Image.open(image_path)
    # Correct orientation based on EXIF
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def inspect_image_quality(image_path: str) -> Dict[str, Any]:
    """Inspect image quality: resolution, aspect ratio, blur score, brightness."""
    img = load_image(image_path)
    width, height = img.size
    
    # Convert to grayscale OpenCV format for blurriness check
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    
    # Laplacian variance (blur detection)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Brightness (mean pixel intensity)
    brightness = float(np.mean(gray))
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / max(1, height), 3),
        "laplacian_variance": round(laplacian_var, 2),
        "is_blurry": laplacian_var < 80.0,
        "brightness": round(brightness, 2),
        "is_dark": brightness < 60.0
    }


def crop_image_region(
    img: Image.Image,
    box_rel: Tuple[float, float, float, float]
) -> Image.Image:
    """
    Crop region using relative coordinates (ymin, xmin, ymax, xmax) in [0, 1].
    """
    w, h = img.size
    ymin, xmin, ymax, xmax = box_rel
    left = int(max(0, min(1, xmin)) * w)
    top = int(max(0, min(1, ymin)) * h)
    right = int(max(0, min(1, xmax)) * w)
    bottom = int(max(0, min(1, ymax)) * h)
    return img.crop((left, top, right, bottom))


def image_to_bytes(img: Image.Image, format: str = "JPEG", quality: int = 95) -> bytes:
    """Convert PIL Image to bytes buffer."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    return buf.getvalue()


def optimize_image_for_llm(
    img: Image.Image,
    max_dimension: int = 2400
) -> Image.Image:
    """
    Resize image to ensure it does not exceed max_dimension while maintaining
    high resolution for crystal-clear text extraction.
    """
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def ensure_min_dimension(img: Optional[Image.Image], min_long_side: int = 1600,
                         max_scale: float = 3.0) -> Optional[Image.Image]:
    """Upscale small images/crops so the long side is at least min_long_side (bounded)."""
    if img is None:
        return None
    w, h = img.size
    long_side = max(w, h)
    if long_side >= min_long_side or long_side == 0:
        return img
    scale = min(max_scale, min_long_side / long_side)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def crop_box_1000(img: Image.Image, box: Any, pad: float = 0.03,
                  min_area_frac: float = 0.01) -> Optional[Image.Image]:
    """
    Crop using a [ymin, xmin, ymax, xmax] box normalized to 0-1000.
    Returns None for missing/degenerate boxes so callers fall back to the full image.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(v) / 1000.0 for v in box]
    except (TypeError, ValueError):
        return None
    if ymax <= ymin or xmax <= xmin:
        return None
    ymin, xmin = max(0.0, ymin - pad), max(0.0, xmin - pad)
    ymax, xmax = min(1.0, ymax + pad), min(1.0, xmax + pad)
    if (ymax - ymin) * (xmax - xmin) < min_area_frac:
        return None
    return crop_image_region(img, (ymin, xmin, ymax, xmax))
