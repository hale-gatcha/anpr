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
    candidates: list = field(default_factory=list)  # optional scored plate crops


def _char_score(gray, box, cache):
    """0..1 evidence that the box holds a row of plate characters."""
    if box not in cache:
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        count = find_char_row(roi)[0]
        adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, max(3, (min(roi.shape)//4)|1), 5)
        count = max(count, find_char_row(adaptive)[0])
        cache[box] = count
    n = cache[box]
    return 1.0 if n >= 5 else 0.7 if n == 4 else 0.4 if n == 3 else 0.0


def _yellow_plate_score(image, box):
    """Use HSV color segmentation to recognize reflective yellow plate paint."""
    x, y, w, h = box
    roi = image[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = (
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 40)
        & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 150)
    )
    return float(yellow.mean())


def _score(candidate, image_shape, gray, image, cache):
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
    yellow_score = min(_yellow_plate_score(image, (x, y, w, h)) / 0.10, 1.0)
    return (0.25 * rect + 0.20 * aspect_score + 0.10 * position_score
            + 0.15 * size_score + 0.10 * char_score + 0.20 * yellow_score)


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

    edge_paths = pp.edge_maps(enhanced)
    steps.update({f"5 {name} edges": value for name, value in edge_paths.items()})

    # Close gaps between characters so the plate becomes one blob.
    # The right closing width depends on how far apart the characters are,
    # which changes with plate size and photo distance. A narrow kernel
    # keeps the plate separate from nearby logos; a wide one joins widely
    # spaced characters. We try several widths and keep the best candidate.
    ranked = []
    n_candidates = 0
    char_cache = {}
    widths = sorted(set(max(5, int(resized.shape[1] * q) | 1) for q in (0.012, 0.02, 0.032)))
    for edge_name, edges in edge_paths.items():
      for width in widths:
       for kh in (3, 5):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, kh))
        closed = cv2.dilate(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel), None, iterations=1)

        # RETR_LIST returns nested contours too. RETR_EXTERNAL would miss a
        # plate that sits inside the bigger outline of the car body.
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cand in _filter_candidates(contours, resized.shape):
            n_candidates += 1
            score = _score(cand, resized.shape, gray, resized, char_cache)
            ranked.append((score, cand, closed, edge_name, width, kh))

    # Deduplicate overlapping boxes; retain several distinct regions so OCR
    # can choose a full plate over a character-only sub-region.
    ranked.sort(key=lambda item: item[0], reverse=True)
    top_box = ranked[0][1][:4] if ranked else None
    top_yellow = _yellow_plate_score(resized, top_box) if top_box else 0.0
    top_x, top_y, top_w, top_h = top_box or (0, 0, 0, 0)
    yellow_scene = (
        top_yellow >= 0.08
        and top_x > 0.10 * resized.shape[1]
        and top_y > 0.10 * resized.shape[0]
    )
    selected = []
    for item in ranked:
        box = item[1][:4]
        if yellow_scene and _yellow_plate_score(resized, box) < 0.03:
            continue
        if any(
            min(box[2] * box[3], b[2] * b[3]) > 0
            and max(0, min(box[0]+box[2], b[0]+b[2])-max(box[0], b[0]))
            * max(0, min(box[1]+box[3], b[1]+b[3])-max(box[1], b[1]))
            / min(box[2] * box[3], b[2] * b[3]) > .65
            for b in [x[1][:4] for x in selected]
        ):
            continue
        selected.append(item)
        if len(selected) == 8: break
    best = selected[0][1] if selected else None

    if best is None:
        steps["6 Morphological closing"] = edge_paths["Canny"]
        return DetectionResult(
            found=False,
            annotated=resized,
            message="No plate-like region found. Try a clearer, closer image.",
            steps=steps,
        )

    steps[f"6 Morphological closing ({selected[0][3]}, width {selected[0][4]})"] = selected[0][2]
    all_boxes = resized.copy()
    for score, c, _, _, _, _ in selected:
        x, y, w, h, _ = c
        cv2.rectangle(all_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)
    steps["7 All contours"] = all_boxes

    x, y, w, h, _ = best
    crops = []
    for score, cand, _, _, _, _ in selected:
        cx, cy, cw, ch, _ = cand
        p = int(.05 * ch)
        xa, ya, xb, yb = max(0,cx-p), max(0,cy-p), min(resized.shape[1],cx+cw+p), min(resized.shape[0],cy+ch+p)
        crops.append({"score": score, "box": (xa,ya,xb-xa,yb-ya), "crop": resized[ya:yb,xa:xb].copy()})

    # Small padding so characters at the border are not cut off
    pad = int(0.05 * h)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1 = min(x + w + pad, resized.shape[1])
    y1 = min(y + h + pad, resized.shape[0])

    crop = resized[y0:y1, x0:x1].copy()
    annotated = resized.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 3)
    steps["8 Detected plate"] = annotated

    # Correct perspective when the contour has a four-corner approximation.
    contour = None
    # Candidate crops use an axis-aligned fallback; preserve compatibility.
    return DetectionResult(
        found=True,
        box=(x0, y0, x1 - x0, y1 - y0),
        plate_crop=crop,
        annotated=annotated,
        message=f"Plate-like region found ({n_candidates} candidate(s) scored; {len(crops)} retained).",
        steps=steps,
        candidates=crops,
    )
