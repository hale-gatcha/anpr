"""Stage 2 - License plate detection (classical contour method).

Idea: characters create many vertical edges close together. Closing
those edges with a wide kernel merges them into one solid rectangle.
Contours of that rectangle are then filtered by size and shape.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import preprocessing as pp
from .characters import find_char_row

# Philippine plates: cars are roughly 2:1 to 3.5:1 (wider than tall),
# motorcycles are closer to 1.2:1 - 2:1. We accept a broad range.
MIN_ASPECT = 1.5
MAX_ASPECT = 5.5
MIN_AREA_FRACTION = 0.002   # plate must cover at least 0.2% of the image
MAX_AREA_FRACTION = 0.60   # allow close-up photos where the plate fills much of the frame
MIN_RECTANGULARITY = 0.30   # contour area / bounding-box area
CLOSING_WIDTHS = (9, 15, 25)  # morphological closing widths tried (pixels)


@dataclass
class DetectionResult:
    found: bool
    box: tuple = None            # (x, y, w, h) on the resized image
    plate_crop: np.ndarray = None
    annotated: np.ndarray = None
    message: str = ""
    steps: dict = field(default_factory=dict)  # name -> image, for display


def _char_score(gray, box, cache):
    """0..1 evidence that the box holds a row of plate characters."""
    if box not in cache:
        x, y, w, h = box
        cache[box] = find_char_row(gray[y:y + h, x:x + w])[0]
    n = cache[box]
    return 1.0 if n >= 5 else 0.7 if n == 4 else 0.4 if n == 3 else 0.0


def _score(candidate, image_shape, gray, cache):
    """Higher is better. Combines shape, size, position and - most
    importantly - whether the region contains a row of characters."""
    x, y, w, h, rect = candidate
    img_h, img_w = image_shape[:2]
    aspect = w / h
    aspect_score = 1.0 - min(abs(aspect - 3.0) / 3.0, 1.0)
    cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
    position_score = 1.0 - min(abs(cx - 0.5) + abs(cy - 0.65), 1.0)
    area_frac = (w * h) / (img_w * img_h)
    size_score = min((area_frac ** 0.5) / 0.3, 1.0)   # reaches 1.0 at ~9% of the image
    char_score = _char_score(gray, (x, y, w, h), cache)
    return (0.25 * rect + 0.20 * aspect_score + 0.10 * position_score
            + 0.20 * size_score + 0.25 * char_score)


def _filter_candidates(contours, image_shape):
    """Keep contours whose size and shape look like a license plate."""
    img_area = image_shape[0] * image_shape[1]
    kept = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / h
        area_frac = (w * h) / img_area
        rect = cv2.contourArea(c) / float(w * h)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        if not (MIN_AREA_FRACTION <= area_frac <= MAX_AREA_FRACTION):
            continue
        if rect < MIN_RECTANGULARITY:
            continue
        kept.append((x, y, w, h, rect))
    return kept


def detect_plate(image_bgr):
    """Find the most plate-like region. Returns a DetectionResult."""
    steps = {}

    resized = pp.resize_to_width(image_bgr, 800)
    steps["1 Resized"] = resized

    gray = pp.to_grayscale(resized)
    steps["2 Grayscale"] = gray

    smooth = pp.denoise(gray)
    steps["3 Bilateral filter"] = smooth

    enhanced, applied = pp.enhance_contrast_if_needed(smooth)
    steps["4 CLAHE (applied)" if applied else "4 CLAHE (skipped)"] = enhanced

    edges = pp.detect_edges(enhanced)
    steps["5 Canny edges"] = edges

    # Close gaps between characters so the plate becomes one blob.
    # The right closing width depends on how far apart the characters are,
    # which changes with plate size and photo distance. A narrow kernel
    # keeps the plate separate from nearby logos; a wide one joins widely
    # spaced characters. We try several widths and keep the best candidate.
    best_score, best, best_closed, best_contours, best_width = -1.0, None, None, None, None
    n_candidates = 0
    char_cache = {}
    for width in CLOSING_WIDTHS:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, None, iterations=1)

        # RETR_LIST returns nested contours too. RETR_EXTERNAL would miss a
        # plate that sits inside the bigger outline of the car body.
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cand in _filter_candidates(contours, resized.shape):
            n_candidates += 1
            score = _score(cand, resized.shape, gray, char_cache)
            if score > best_score:
                best_score, best, best_closed = score, cand, closed
                best_contours, best_width = contours, width

    if best is None:
        steps["6 Morphological closing"] = closed
        return DetectionResult(
            found=False,
            annotated=resized,
            message="No plate-like region found. Try a clearer, closer image.",
            steps=steps,
        )

    steps[f"6 Morphological closing (width {best_width})"] = best_closed
    all_boxes = resized.copy()
    for c in best_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(all_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)
    steps["7 All contours"] = all_boxes

    x, y, w, h, _ = best
    candidates = [None] * n_candidates   # only the count is reported below

    # Small padding so characters at the border are not cut off
    pad = int(0.05 * h)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1 = min(x + w + pad, resized.shape[1])
    y1 = min(y + h + pad, resized.shape[0])

    crop = resized[y0:y1, x0:x1].copy()
    annotated = resized.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 3)
    steps["8 Detected plate"] = annotated

    return DetectionResult(
        found=True,
        box=(x0, y0, x1 - x0, y1 - y0),
        plate_crop=crop,
        annotated=annotated,
        message=f"Plate-like region found ({len(candidates)} candidate(s) after filtering).",
        steps=steps,
    )
