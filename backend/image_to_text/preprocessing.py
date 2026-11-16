import io
import os
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ImageOps
import numpy as np

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
MIN_DIMENSION = 64
MAX_DIMENSION = 4096

class ImagePreprocessingError(Exception):
    """Custom exception raised during image validation or preprocessing."""
    pass

def validate_image_bytes(file_bytes: bytes, filename: str = "image.png") -> Dict[str, Any]:
    """
    Validates image file bytes against safety, size, and format constraints.
    Returns basic validation metadata.
    """
    if not file_bytes:
        raise ImagePreprocessingError("Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ImagePreprocessingError(
            f"File size {len(file_bytes) / (1024*1024):.1f}MB exceeds the maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB."
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img_format = (img.format or "").upper()
            if img_format not in SUPPORTED_FORMATS:
                ext = filename.split(".")[-1].upper() if "." in filename else ""
                if ext not in SUPPORTED_FORMATS:
                    raise ImagePreprocessingError(
                        f"Unsupported image format: '{img_format or ext}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}."
                    )

            width, height = img.size
            if width < MIN_DIMENSION or height < MIN_DIMENSION:
                raise ImagePreprocessingError(
                    f"Image resolution {width}x{height} is too small. Minimum required dimension is {MIN_DIMENSION}x{MIN_DIMENSION}."
                )

            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ImagePreprocessingError(
                    f"Image resolution {width}x{height} exceeds maximum allowed dimension of {MAX_DIMENSION}x{MAX_DIMENSION}."
                )

            return {
                "valid": True,
                "format": img_format or "PNG",
                "width": width,
                "height": height,
                "size_bytes": len(file_bytes),
                "mode": img.mode,
            }
    except ImagePreprocessingError:
        raise
    except Exception as e:
        raise ImagePreprocessingError(f"Corrupted or invalid image data: {str(e)}")

def preprocess_image(
    file_bytes: bytes,
    target_max_size: Optional[Tuple[int, int]] = (1024, 1024),
    strip_exif: bool = True,
) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses raw image bytes into a clean, normalized PIL Image and NumPy RGB array.
    
    Steps:
    1. Validate format, size, dimensions.
    2. Convert to RGB (handles RGBA, Palette, Grayscale).
    3. Auto-orient according to EXIF orientation tag.
    4. Strip all proprietary metadata/EXIF for research safety.
    5. Optionally downscale if larger than target_max_size while preserving aspect ratio.
    6. Convert to normalized NumPy array [0.0, 1.0].
    """
    validation_meta = validate_image_bytes(file_bytes)

    with Image.open(io.BytesIO(file_bytes)) as raw_img:
        # Correct orientation if EXIF tag exists
        try:
            oriented_img = ImageOps.exif_transpose(raw_img)
        except Exception:
            oriented_img = raw_img

        # Convert to RGB mode
        if oriented_img.mode != "RGB":
            rgb_img = oriented_img.convert("RGB")
        else:
            rgb_img = oriented_img.copy()

        # Resize if exceeding max size
        orig_w, orig_h = rgb_img.size
        if target_max_size is not None:
            max_w, max_h = target_max_size
            if orig_w > max_w or orig_h > max_h:
                rgb_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        proc_w, proc_h = rgb_img.size

        # Create numpy array normalized to [0, 1]
        np_arr = np.array(rgb_img, dtype=np.float32) / 255.0

        metadata = {
            "original_width": orig_w,
            "original_height": orig_h,
            "processed_width": proc_w,
            "processed_height": proc_h,
            "channels": 3,
            "mean_intensity": float(np.mean(np_arr)),
            "std_intensity": float(np.std(np_arr)),
            "format": validation_meta["format"],
            "exif_stripped": strip_exif,
        }

        return rgb_img, np_arr, metadata
