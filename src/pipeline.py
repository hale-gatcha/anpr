"""Stage 4 - Connects detection and OCR into one function.

The Streamlit app (app.py) only calls run_pipeline(). Keeping this
separate means the same code can be tested without the interface.
"""

from dataclasses import dataclass
from typing import Optional

from .detection import DetectionResult, detect_plate
from .ocr import OCRResult, read_plate

MIN_SIDE = 60   # pixels; smaller images cannot contain a readable plate


@dataclass
class PipelineResult:
    detection: DetectionResult
    ocr: Optional[OCRResult] = None   # None when no plate was detected


def check_image_size(image_bgr):
    """Raise ValueError if the image is too small to work with."""
    h, w = image_bgr.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise ValueError(
            f"The image is too small ({w}x{h} px). Use an image at least "
            f"{MIN_SIDE} px on its shortest side.")


def run_pipeline(image_bgr):
    """Detect the plate, then read it. Returns a PipelineResult."""
    check_image_size(image_bgr)
    detection = detect_plate(image_bgr)
    if not detection.found:
        return PipelineResult(detection=detection)
    return PipelineResult(detection=detection, ocr=read_plate(detection.plate_crop))
