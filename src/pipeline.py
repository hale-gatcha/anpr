"""Stage 4 - Connects detection and OCR into one function.

The Streamlit app (app.py) only calls run_pipeline(). Keeping this
separate means the same code can be tested without the interface.
"""

from dataclasses import dataclass, field
import math
from typing import Optional

from .detection import DetectionResult, detect_plate
from .ocr import OCRResult, read_plate

MIN_SIDE = 60   # pixels; smaller images cannot contain a readable plate


@dataclass
class PipelineResult:
    detection: DetectionResult
    ocr: Optional[OCRResult] = None   # None when no plate was detected
    alternatives: list = field(default_factory=list)
    winning_variant: str = ""


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
    candidates = detection.candidates or [{"score": 1.0, "crop": detection.plate_crop, "box": detection.box}]
    best_detection = candidates[0].get("score", 0.0)
    eligible = [c for c in candidates if c.get("score", 0.0) >= 0.65 * best_detection]
    readings = []
    for candidate in eligible:
        ocr_result = read_plate(candidate["crop"])
        if not ocr_result.ok:
            continue
        text_length = len(ocr_result.text.replace(" ", ""))
        if ocr_result.format_ok and text_length in (6, 7):
            format_match = 1.0
        elif ocr_result.text and ocr_result.text.isalnum() and text_length >= 5:
            format_match = 0.55
        elif ocr_result.format_ok:
            format_match = 0.45
        else:
            format_match = 0.3
        det_score = candidate.get("score", 0.0)
        ocr_confidence = (
            ocr_result.confidence
            if math.isfinite(ocr_result.confidence) else 0.0
        )
        combined = det_score * max(0.05, ocr_confidence / 100.0) * format_match
        readings.append((combined, candidate, ocr_result))
    if not readings:
        return PipelineResult(detection=detection, ocr=read_plate(detection.plate_crop))
    readings.sort(key=lambda x: x[0], reverse=True)
    score, candidate, ocr_result = readings[0]
    # Main detection fields always describe the selected OCR candidate.
    detection.box = candidate.get("box", detection.box)
    detection.plate_crop = candidate["crop"]
    detection.annotated = detection.annotated.copy()
    x,y,w,h=detection.box
    import cv2
    cv2.rectangle(detection.annotated,(x,y),(x+w,y+h),(0,255,0),3)
    alternatives = [{"text": r.text, "raw_text": r.raw_text, "score": s,
                     "detection_score": c.get("score",0.0), "confidence": r.confidence}
                    for s,c,r in readings[1:]]
    return PipelineResult(detection=detection, ocr=ocr_result, alternatives=alternatives,
                          winning_variant=ocr_result.winning_variant)
