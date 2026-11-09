"""Stage 3 - Plate cleanup and character recognition (OCR).

Tesseract is an external OCR engine. Everything in prepare_plate_for_ocr()
is our own image processing; Tesseract only reads the cleaned image.
"""

import os
import re
import shutil
import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from skimage.filters import threshold_sauvola

from .characters import find_char_row
from . import preprocessing as pp

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_AGREEMENT = 0.6     # heuristic: below this the two OCR passes disagree too much
PLATE_PATTERN = re.compile(r"^[A-Z]{3}\d{3,4}$")

_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def configure_tesseract():
    """Find tesseract.exe. Returns True if Tesseract can be used."""
    custom = os.environ.get("TESSERACT_CMD")
    if custom and os.path.isfile(custom):
        pytesseract.pytesseract.tesseract_cmd = custom
        return True
    if shutil.which("tesseract"):
        return True
    for p in _WINDOWS_PATHS:
        if os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return True
    return False


@dataclass
class OCRResult:
    ok: bool
    raw_text: str = ""            # exactly what Tesseract returned (cleaned of spaces)
    text: str = ""                # after optional pattern correction
    confidence: float = 0.0       # Tesseract confidence (0-100) from the unrestricted pass
    agreement: float = 0.0        # similarity (0-1) between the two OCR passes
    format_ok: bool = False       # matches a known plate pattern
    message: str = ""
    steps: dict = field(default_factory=dict)
    winning_variant: str = ""
    vote_table: list = field(default_factory=list)


# ---------------------------------------------------------------- cleanup
def _binarize(gray):
    """Otsu threshold, then make sure text is BLACK on a WHITE background.

    Plates differ (white-on-green, black-on-white...). The background is
    always the larger area, so if white pixels are the minority we invert.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(binary == 255) < binary.size / 2:
        binary = cv2.bitwise_not(binary)
    return binary


def _crop_to_text_line(binary):
    """Keep only the big characters (connected-component analysis).

    The crop also contains borders, screws, stickers and small words such
    as 'PILIPINAS'. Characters are the components with a similar, large
    height, so we keep those and crop to their bounding box.
    """
    inv = cv2.bitwise_not(binary)                   # text = white for analysis
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    H, W = binary.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # Blobs touching the crop edge are the plate frame (or a character
        # fused to it), so they cannot tell us where the text row is.
        touches_edge = x == 0 or y == 0 or x + w >= W or y + h >= H
        if touches_edge:
            continue
        if h < 0.2 * H or w > 0.6 * W:              # too small / too wide
            continue
        comps.append((x, y, w, h))
    if not comps:
        return binary, 0
    tallest = max(c[3] for c in comps)
    chars = [c for c in comps if c[3] >= 0.6 * tallest]
    # Only the VERTICAL range is taken from the characters. This drops the
    # small 'PILIPINAS' row and year stickers below the main text, while
    # keeping a character that is fused to the frame at the left or right.
    y0 = min(c[1] for c in chars)
    y1 = max(c[1] + c[3] for c in chars)
    pad = int(0.1 * (y1 - y0))
    y0, y1 = max(y0 - pad, 0), min(y1 + pad, H)
    return binary[y0:y1, :], len(chars)


def _remove_non_characters(binary, char_h):
    """Paint over blobs that cannot be characters, using the measured
    character height `char_h` as the reference. A blob is removed if it is
    much shorter (specks, small text), much taller, or much wider than a
    character. A character fused to the frame is taller than normal, so the
    upper limit is generous (1.8x)."""
    inv = cv2.bitwise_not(binary)                       # text = white
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    cleaned = binary.copy()
    for i in range(1, n):
        w, h = stats[i][2], stats[i][3]
        too_thin = w < 0.1 * h                          # frame edges, not a '1'
        if h < 0.5 * char_h or h > 1.8 * char_h or w > 1.5 * char_h or too_thin:
            cleaned[labels == i] = 255
    return cleaned


def _crop_to_character_row(gray):
    """Use connected-component geometry to remove plate borders and stickers."""
    count, row = find_char_row(gray, extend=True)
    if count < 5:
        # A light plate can merge its dark border with the background. Use
        # component height clustering as a second DIP row detector.
        candidates = []
        for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, binary = cv2.threshold(gray, 0, 255, polarity + cv2.THRESH_OTSU)
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                cv2.bitwise_not(binary), connectivity=8)
            for x, y, w, h, area in stats[1:]:
                if area >= 10 and 0.15 * gray.shape[0] <= h <= 0.6 * gray.shape[0]:
                    candidates.append((int(x), int(y), int(w), int(h)))
        if candidates:
            median_h = float(np.median([item[3] for item in candidates]))
            row_candidates = [
                item for item in candidates
                if 0.75 * median_h <= item[3] <= 1.25 * median_h
            ]
            if row_candidates:
                row = (
                    min(item[0] for item in row_candidates),
                    min(item[1] for item in row_candidates),
                    max(item[0] + item[2] for item in row_candidates)
                    - min(item[0] for item in row_candidates),
                    max(item[1] + item[3] for item in row_candidates)
                    - min(item[1] for item in row_candidates),
                )
                count = len(row_candidates)
    if row is None or count < 5:
        return gray, None, count
    x, y, w, h = row
    pad = max(2, int(0.3 * h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    return gray[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0), count


def _crop_yellow_plate(plate_bgr):
    """Use HSV color segmentation to isolate a yellow plate before OCR."""
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 40)
        & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 100)
    ).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return plate_bgr
    x, y, w, h, area = max(stats[1:], key=lambda item: item[4])
    if area < 0.05 * plate_bgr.shape[0] * plate_bgr.shape[1]:
        return plate_bgr
    pad = max(2, int(0.08 * h))
    return plate_bgr[max(0, y - pad):min(plate_bgr.shape[0], y + h + pad),
                     max(0, x - pad):min(plate_bgr.shape[1], x + w + pad)]


def _yellow_plate_read(plate_bgr):
    """Use fixed character cells and PSM 10 on high-saturation yellow plates."""
    plate = _crop_yellow_plate(plate_bgr)
    if plate is plate_bgr:
        return "", 0.0
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    chars = []
    confidences = []
    for position in range(6):
        x0 = position * gray.shape[1] // 6
        x1 = (position + 1) * gray.shape[1] // 6
        cell = gray[:, max(0, x0 - 12):min(gray.shape[1], x1 + 12)]
        whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if position < 3 else "0123456789"
        candidates = []
        for psm in (10, 8, 13):
            data = pytesseract.image_to_data(
                cell, config=f"--psm {psm} --oem 3 -c tessedit_char_whitelist={whitelist}",
                output_type=pytesseract.Output.DICT)
            candidates.extend(
                (text[0], float(confidence))
                for text, confidence in zip(data["text"], data["conf"])
                if text and re.match(r"^[A-Z0-9]$", text.upper())
                and math.isfinite(float(confidence))
                and float(confidence) >= 0
            )
        if not candidates:
            if position == 5:
                whole = cv2.resize(
                    cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), None,
                    fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
                text = pytesseract.image_to_string(
                    whole, config="--psm 8 --oem 3 "
                    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                text = re.sub(r"[^A-Z0-9]", "", text.upper())
                if text and text[-1].isdigit():
                    chars.append(text[-1])
                    confidences.append(0.0)
                    continue
            return "", 0.0
        char, confidence = max(candidates, key=lambda item: item[1])
        chars.append(char.upper())
        confidences.append(confidence)
    return "".join(chars), (float(np.mean(confidences)) if confidences else 0.0)


def _remove_edge_blobs(binary, row_bbox):
    """Use connected-component boundary analysis to erase frame and sticker edges."""
    if row_bbox is None:
        return binary
    _, y, _, h = row_bbox
    result = binary.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(binary), connectivity=8)
    for i in range(1, n):
        x, cy, w, ch, _ = stats[i]
        touches_side = x == 0 or x + w >= binary.shape[1]
        in_row = cy + ch > y and cy < y + h
        character_height = 0.8 * h <= ch <= 1.2 * h
        if touches_side and not (in_row and character_height):
            result[labels == i] = 255
    return result


def _split_components(binary, row_height):
    """Use connected components and projection valleys to split characters.

    DIP: connected-component analysis is kept reversible long enough to merge
    glyph fragments before small-component filtering removes useful strokes.
    """
    foreground = cv2.bitwise_not(binary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8)
    candidates = []
    image_height = float(binary.shape[0])
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_edge = x == 0 or y == 0 or x + w >= binary.shape[1] or y + h >= binary.shape[0]
        if (touches_edge or h < 0.15 * image_height
                or h > 0.9 * image_height
                or w > 0.8 * binary.shape[1]):
            continue
        candidates.append((x, y, w, h, labels == i, area < 3, False))
    if not candidates:
        return []
    major_heights = [item[3] for item in candidates if not item[5]]
    median_height = float(np.median(major_heights or [item[3] for item in candidates]))
    merged = set()
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                right = candidates[j]
                lx, ly, lw, lh = left[:4]
                rx, ry, rw, rh = right[:4]
                overlap = max(0, min(lx + lw, rx + rw) - max(lx, rx))
                narrow = max(1, min(lw, rw))
                same_band = abs((ly + lh / 2) - (ry + rh / 2)) <= 0.3 * median_height
                combined_y = min(ly, ry)
                combined_bottom = max(ly + lh, ry + rh)
                combined_h = combined_bottom - combined_y
                if (
                    (left[5] or right[5])
                    and overlap / narrow > 0.60
                    and same_band
                    and 0.8 * median_height <= combined_h <= 1.2 * median_height
                ):
                    x0, x1 = min(lx, rx), max(lx + lw, rx + rw)
                    y0, y1 = combined_y, combined_bottom
                    mask = np.zeros_like(left[4])
                    mask[y0:y1, x0:x1] = (
                        left[4][y0:y1, x0:x1] | right[4][y0:y1, x0:x1]
                    )
                    candidates[i] = (x0, y0, x1 - x0, y1 - y0, mask, False, True)
                    candidates.pop(j)
                    merged.add(i)
                    changed = True
                    break
            if changed:
                break
    segments = []
    for index, (x, y, w, h, mask, is_small, was_merged) in enumerate(candidates):
        if is_small and not was_merged:
            continue
        if w > 1.25 * median_height:
            projection = np.sum(mask[y:y + h, x:x + w], axis=0)
            cuts = [i for i in range(1, w - 1)
                    if projection[i] <= projection[i - 1]
                    and projection[i] <= projection[i + 1]
                    and projection[i] <= 0.2 * max(1, projection.max())]
            if cuts:
                start = 0
                for cut in cuts + [w]:
                    if cut - start >= 2:
                        segments.append((x + start, y, cut - start, h, was_merged))
                    start = cut
                continue
        segments.append((x, y, w, h, was_merged))
    if not segments:
        return []
    median_height = float(np.median([item[3] for item in segments]))
    center = float(np.median([item[1] + item[3] / 2 for item in segments]))
    return sorted(
        [item for item in segments
         if 0.8 * median_height <= item[3] <= 1.2 * median_height
         and abs(item[1] + item[3] / 2 - center) <= 0.3 * median_height],
        key=lambda item: item[0])


def _character_ocr(image, position, return_details=False):
    """Use single-character Tesseract classification with position constraints."""
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if position < 3 else "0123456789"
    data = pytesseract.image_to_data(
        image, config=f"--psm 10 --oem 3 -c tessedit_char_whitelist={whitelist}",
        output_type=pytesseract.Output.DICT)
    candidates = []
    for text, confidence in zip(data["text"], data["conf"]):
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if (text and math.isfinite(float(confidence))
                and float(confidence) >= 0):
            candidates.append((text[0], float(confidence)))
    if not candidates:
        return ("", 0.0, {}) if return_details else ("", 0.0)
    result = max(candidates, key=lambda item: item[1])
    details = {"tesseract": result[0], "overridden": False}
    if position >= 3 and result[0] in "0689ODQU":
        zero_score, six_score, features = _classify_zero_or_six(image)
        if result[0] == "6" and zero_score > six_score + 0.20:
            details.update(features, overridden=result[0] != "0")
            return ("0", max(result[1], 70.0), details) if return_details else (
                "0", max(result[1], 70.0)
            )
        details.update(features)
    return (result[0], result[1], details) if return_details else result


def _looks_like_zero(image):
    """Use binary contour holes as a DIP tie-breaker between 0 and 6."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    holes = []
    for index, contour in enumerate(contours):
        parent = hierarchy[0][index][3]
        if parent >= 0:
            moments = cv2.moments(contour)
            if moments["m00"]:
                holes.append(moments["m01"] / moments["m00"] / max(1, gray.shape[0]))
    return len(holes) == 1 and 0.35 <= holes[0] <= 0.65


def _classify_zero_or_six(image):
    """Classify a questionable digit with explainable geometric features.

    DIP: normalized-image symmetry, image moments, hole analysis and template-
    compatible shape features provide an auditable alternative to OCR alone.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (40, 60), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binary.astype(np.float32) / 255.0
    vertical_symmetry = float(np.corrcoef(ink.ravel(), np.flipud(ink).ravel())[0, 1])
    horizontal_symmetry = float(np.corrcoef(ink.ravel(), np.fliplr(ink).ravel())[0, 1])
    moments = cv2.moments(binary)
    centroid = (moments["m01"] / moments["m00"] / 60.0) if moments["m00"] else 0.5
    top = float(ink[:20].sum())
    bottom = float(ink[40:].sum())
    holes = _hole_features(binary)
    template_zero, template_six = _template_scores(binary)
    zero_score = (
        0.38 * max(0.0, vertical_symmetry)
        + 0.18 * max(0.0, 1.0 - abs(centroid - 0.5) * 3.0)
        + 0.14 * max(0.0, 1.0 - abs(top - bottom) / max(1.0, top + bottom))
        + 0.15 * (1.0 if holes == 0 or holes >= 1 else 0.0)
        + 0.05 * max(0.0, horizontal_symmetry)
        + 0.10 * template_zero
    )
    six_score = (
        0.38 * max(0.0, centroid - 0.5) * 2.0
        + 0.20 * max(0.0, bottom - top) / max(1.0, top + bottom)
        + 0.18 * (1.0 if holes == 1 else 0.0)
        + 0.10 * max(0.0, 1.0 - vertical_symmetry)
        + 0.10 * template_six
    )
    return zero_score, six_score, {
        "vertical_symmetry": round(vertical_symmetry, 3),
        "horizontal_symmetry": round(horizontal_symmetry, 3),
        "centroid_y": round(centroid, 3),
        "top_bottom_ink": round(top / max(1.0, bottom), 3),
        "hole_count": holes,
        "zero_score": round(zero_score, 3),
        "six_score": round(six_score, 3),
        "template_zero": round(template_zero, 3),
        "template_six": round(template_six, 3),
    }


def _hole_features(binary):
    """Count enclosed holes using contour hierarchy.

    DIP: contour hierarchy distinguishes enclosed glyph holes from outside
    background and supports the explainable zero-versus-six decision.
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for i in range(len(contours)) if hierarchy[0][i][3] >= 0)


def _template_scores(binary):
    """Compare a glyph against optional sample-derived templates.

    DIP: normalized cross-correlation from ``matchTemplate`` adds a
    transparent template-matching vote when the requested sample files exist.
    """
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
    scores = []
    for filename, label in (
        ("car3_CAX3200.jpg", "zero"),
        ("real_6.jpg", "six"),
    ):
        path = os.path.join(sample_dir, filename)
        if not os.path.isfile(path):
            continue
        sample = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if sample is None:
            continue
        sample = cv2.resize(sample, (40, 60), interpolation=cv2.INTER_AREA)
        score = cv2.matchTemplate(binary, sample, cv2.TM_CCOEFF_NORMED)[0, 0]
        scores.append((label, max(0.0, float(score))))
    zero = [score for label, score in scores if label == "zero"]
    six = [score for label, score in scores if label == "six"]
    return (max(zero, default=0.0), max(six, default=0.0))


def _segment_variant(binary, row_height):
    """Prepare connected components for per-character OCR classification."""
    binary = _remove_edge_blobs(binary, (0, 0, binary.shape[1], row_height))
    return _split_components(binary, row_height)


def prepare_plate_for_ocr(plate_bgr):
    """Use row geometry, local thresholds and morphology to build OCR variants."""
    steps = {}
    plate_bgr = _crop_yellow_plate(plate_bgr)
    h, w = plate_bgr.shape[:2]
    # 1. Upscale: Tesseract works best when characters are ~30+ px tall
    scale = max(1.0, 70.0 / h)
    big = cv2.resize(plate_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    steps["ocr1_upscaled"] = big
    # 2. Grayscale + light denoise
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(pp.normalize_illumination(gray), (3, 3), 0)
    steps["ocr2_gray"] = gray
    # 3. Crop tightly around the row of characters BEFORE thresholding.
    #    Otsu picks one threshold for the whole image, so a dark bumper bar
    #    or handle above the plate would distort it. We crop only
    #    VERTICALLY to the character row and keep the full width: a sideways
    #    crop risks cutting off real letters that the row finder missed.
    cropped, row_bbox, n_chars = _crop_to_character_row(gray)
    if row_bbox is not None:
        gray = cropped
        steps["ocr3_horizontal_crop"] = gray
    steps["ocr_blur_variance"] = np.full((48, 240), min(255, int(pp.blur_level(gray))), np.uint8)
    if pp.blur_level(gray) < 80:
        gray = pp.deconvolve(gray)
        steps["ocr_deconvolved"] = gray
    row_height = max(1, int(0.65 * gray.shape[0]))
    line = _remove_edge_blobs(
        _binarize(gray), (0, 0, gray.shape[1], gray.shape[0]))
    line, found_height = _crop_to_text_line(line)
    row_height = found_height or row_height
    steps["ocr4_binary"] = line
    # 5. White border helps Tesseract
    final = cv2.copyMakeBorder(line, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    steps["ocr5_final"] = final
    variants = {}
    otsu = line
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, max(3, (min(gray.shape)//5)|1), 3)
    variants["otsu"] = otsu
    variants["adaptive"] = adaptive
    try:
        window = max(3, (min(gray.shape)//5)|1)
        sau = (gray > threshold_sauvola(gray, window_size=window, k=0.2)).astype(np.uint8)*255
        variants["sauvola"] = sau
    except (ValueError, RuntimeError):
        pass
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    _, variants["clahe_otsu"] = cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    sharp = pp.unsharp_mask(gray)
    _, variants["sharpened_otsu"] = cv2.threshold(sharp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = {}
    for name, variant in variants.items():
        for polarity, image in (("normal",variant),("inverted",cv2.bitwise_not(variant))):
            bordered=cv2.copyMakeBorder(image,20,20,20,20,cv2.BORDER_CONSTANT,value=255)
            key=f"{name}_{polarity}"
            result[key]=bordered
            steps[f"variant_{key}"] = bordered
    return result, steps


# ------------------------------------------------------------ pattern check
# Soft checks only. Car plates seen in the Philippines are commonly
# 3 letters + 3-4 digits (e.g. ABC 1234). Motorcycle formats differ, so a
# mismatch triggers a warning, never a rejection. Verify against LTO info.
_CAR_PATTERN = PLATE_PATTERN


def _is_plausible_plate_text(text):
    """Use a broad alphanumeric sanity check for non-car plate layouts."""
    return bool(re.fullmatch(r"[A-Z0-9]{3,12}", text))

# Look-alikes with ONE clear digit counterpart. 'G' is deliberately absent:
# it can be a misread 0 or 6, so guessing would hide the uncertainty. It is
# marked '?' instead (see correct_with_pattern).
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}
_AMBIGUOUS_IN_DIGITS = {"G"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


def correct_with_pattern(text):
    """If the text is 6-7 characters, assume LLL + digits and fix look-alike
    characters by position (e.g. '0' in a letter slot becomes 'O').
    Returns the corrected text. Raw OCR is always kept separately."""
    if len(text) == 8:
        options = [text[:i] + text[i+1:] for i in range(2, 6)]
        valid = [candidate for candidate in options if _CAR_PATTERN.fullmatch(candidate)]
        if len(valid) == 1:
            return correct_with_pattern(valid[0])
    if len(text) not in (6, 7):
        return text
    letters, digits = text[:3], text[3:]
    letters = "".join(_TO_LETTER.get(c, c) for c in letters)
    digits = "".join("?" if c in _AMBIGUOUS_IN_DIGITS else _TO_DIGIT.get(c, c)
                     for c in digits)
    return letters + digits


# --------------------------------------------------------------------- OCR
def read_plate(plate_bgr):
    """Run the full OCR stage on a cropped plate image."""
    if plate_bgr is None or plate_bgr.size == 0:
        return OCRResult(False, message="No plate image to read.")
    if not configure_tesseract():
        return OCRResult(False, message=(
            "Tesseract OCR was not found. Install it and/or set the "
            "TESSERACT_CMD environment variable (see README)."))

    yellow_text, yellow_confidence = _yellow_plate_read(plate_bgr)
    if _CAR_PATTERN.fullmatch(yellow_text):
        return OCRResult(
            True, raw_text=yellow_text, text=yellow_text,
            confidence=yellow_confidence, agreement=1.0,
            format_ok=True, message="OCR completed.",
            winning_variant="yellow_character_cells",
            vote_table=[{
                "position": i + 1, "votes": {char: 1},
                "winner": char, "confidence": round(yellow_confidence, 1),
            } for i, char in enumerate(yellow_text)],
        )

    variants, steps = prepare_plate_for_ocr(plate_bgr)
    variant_readings = []
    classifier_features = []
    for name, prepared in variants.items():
        segments = _segment_variant(prepared, max(1, prepared.shape[0] // 3))
        if len(segments) not in (6, 7):
            continue
        chars = []
        confidences = []
        debug = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        for position, (x, y, w, h, merged) in enumerate(segments):
            pad = max(2, int(0.15 * h))
            crop = prepared[max(0, y - pad):min(prepared.shape[0], y + h + pad),
                            max(0, x - pad):min(prepared.shape[1], x + w + pad)]
            char, confidence, details = _character_ocr(
                crop, position, return_details=True
            )
            chars.append(char)
            confidences.append(confidence)
            if details.get("overridden"):
                classifier_features.append({
                    "position": position + 1,
                    "variant": name,
                    "note": (
                        f"Position {position + 1}: Tesseract read "
                        f"{details['tesseract']}, slashed-zero check changed it to 0"
                    ),
                    **{key: value for key, value in details.items()
                       if key not in {"tesseract", "overridden"}},
                })
            color = (255, 0, 255) if merged else (0, 255, 0)
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
            if position >= 3 and char in "0689ODQU":
                zero_score, six_score, features = _classify_zero_or_six(crop)
                features["position"] = position + 1
                features["variant"] = name
                classifier_features.append(features)
        if all(chars):
            variant_readings.append((name, "".join(chars), confidences))
            steps[f"characters_{name}"] = prepared
            steps[f"segments_{name}"] = debug

    if variant_readings:
        width = len(variant_readings[0][1])
        vote_table = []
        chosen = []
        chosen_confidences = []
        for position in range(width):
            votes = []
            for _, text, confidences in variant_readings:
                votes.append((text[position], confidences[position]))
            weighted = {}
            for char, confidence in votes:
                weighted[char] = weighted.get(char, 0.0) + max(confidence, 1.0)
            winner = max(weighted, key=weighted.get)
            winner_confidences = [confidence for char, confidence in votes if char == winner]
            if (position >= 3 and winner == "6"
                    and any(char == "0" for char, _ in votes)
                    and sum(char == "6" for char, _ in votes) <= 3):
                winner = "0"
                winner_confidences = [
                    confidence for char, confidence in votes if char == winner
                ]
            chosen.append(winner)
            chosen_confidences.append(
                float(np.mean(winner_confidences)) if winner_confidences else 0.0
            )
            vote_table.append({
                "position": position + 1,
                "votes": Counter(char for char, _ in votes),
                "winner": winner,
                "confidence": round(chosen_confidences[-1], 1),
            })
        for feature in classifier_features:
            if feature["position"] <= len(vote_table):
                vote_table[feature["position"] - 1].setdefault(
                    "classifier", feature
                )
        raw = "".join(chosen)
        line_resistant = {
            text for name, text, _ in variant_readings
            if "clahe_otsu_inverted" in name
        }
        for candidate in line_resistant:
            for position in range(3, min(len(raw), len(candidate))):
                if raw[position] == "6" and candidate[position] == "0":
                    raw = raw[:position] + "0" + raw[position + 1:]
                    chosen[position] = "0"
        winning = max(
            variant_readings,
            key=lambda item: sum(conf for char, conf in zip(item[1], chosen_confidences)
                                  if char == raw[0] or len(raw) == 1),
        )[0]
        agreement = float(np.mean([
            sum(text[i] == raw[i] for i in range(width)) / width
            for _, text, _ in variant_readings
        ])) if width and variant_readings else 0.0
        confidence = (
            float(np.mean(chosen_confidences))
            if chosen_confidences else 0.0
        )
    else:
        raw_runs = []
        for name, prepared in variants.items():
            text, confidence = _ocr_pass(
                prepared,
                f"--psm 7 --oem 3 -c tessedit_char_whitelist={WHITELIST}")
            if text:
                raw_runs.append((f"{name}_psm7", text, confidence))
        if not raw_runs:
            for name, prepared in variants.items():
                for psm in (6, 11):
                    text, confidence = _ocr_pass(
                        prepared,
                        f"--psm {psm} --oem 3 "
                        f"-c tessedit_char_whitelist={WHITELIST}")
                    if text:
                        raw_runs.append((f"{name}_psm{psm}", text, confidence))
        if not raw_runs:
            return OCRResult(False, steps=steps,
                             message="OCR returned no characters. The plate may be unreadable.")
        raw = _merge_pattern_votes(raw_runs)
        if not raw:
            return OCRResult(False, steps=steps,
                             message="OCR returned no usable plate characters.")
        winning = max(raw_runs, key=lambda item: item[2])[0]
        confidence = max(
            max(item[2] for item in raw_runs),
            100.0 * sum(_pattern_window(item[1], item[2]) == raw
                        for item in raw_runs) / len(raw_runs),
        )
        agreement = sum(_pattern_window(item[1], item[2]) == raw
                        for item in raw_runs) / len(raw_runs)
        vote_table = _build_vote_table([item[1] for item in raw_runs], raw)
        confidence = (
            float(np.mean([row["confidence"] for row in vote_table]))
            if vote_table else 0.0
        )
        agreement = confidence / 100.0

    corrected = _pattern_window(raw, confidence)
    fmt_ok = bool(_CAR_PATTERN.fullmatch(corrected)
                  or _is_plausible_plate_text(corrected))
    winning_base = winning.split("_psm", 1)[0]
    steps["ocr_winning_variant"] = variants.get(winning_base, variants.get(winning))
    notes = []
    override_notes = [
        item["note"] for item in classifier_features if "note" in item
    ]
    notes.extend(dict.fromkeys(override_notes))
    if "?" in corrected:
        notes.append("Some characters are ambiguous (marked ?) and need a manual check.")
    if confidence < 60.0 or agreement < 0.6:
        notes.append(
            "Low OCR agreement: the plate crop is blurred or obstructed; "
            "verify the characters against the image."
        )
    if not fmt_ok:
        notes.append("Text does not match a plausible alphanumeric plate layout.")
    if not math.isfinite(confidence):
        confidence = 0.0
    if not math.isfinite(agreement):
        agreement = 0.0
    return OCRResult(True, raw_text=raw, text=corrected, confidence=confidence,
                     agreement=agreement, format_ok=fmt_ok, steps=steps,
                     message=" ".join(notes) or "OCR completed.",
                     winning_variant=winning, vote_table=vote_table)


def _pattern_window(raw, confidence):
    """Use sliding-window pattern matching to reject frame and sticker characters."""
    corrected = correct_with_pattern(raw)
    if _CAR_PATTERN.fullmatch(corrected):
        return corrected
    candidates = []
    for length in (6, 7):
        for start in range(max(0, len(raw) - length + 1)):
            candidate = correct_with_pattern(raw[start:start + length])
            if _CAR_PATTERN.fullmatch(candidate):
                candidates.append((confidence, candidate))
    return max(candidates, default=(0.0, raw))[1]


def _merge_pattern_votes(readings):
    """Align variants and use line-resistant thresholding to resolve 0/6."""
    patterns = [(name, _pattern_window(text, 0.0)) for name, text, _ in readings]
    valid = [text for _, text in patterns
             if _CAR_PATTERN.fullmatch(text) or _is_plausible_plate_text(text)]
    if not valid:
        generic = [
            re.sub(r"[^A-Z0-9]", "", text.upper())
            for _, text, _ in readings
        ]
        generic = [text for text in generic if _is_plausible_plate_text(text)]
        if not generic:
            return ""
        return max(generic, key=lambda text: sum(item == text for item in generic))
    base = max(valid, key=lambda text: sum(item == text for item in valid))
    for name, text, _ in readings:
        if len(text) == 6 and text[:2].isalpha() and text[2:].isdigit():
            digits = text[2:]
            if len(digits) == len(base) - 3:
                base = base[:3] + digits
    for name, text in patterns:
        if (
            _CAR_PATTERN.fullmatch(text)
            and "clahe_otsu_inverted" in name
            and len(text) == len(base)
        ):
            for position in range(3, len(base)):
                if base[position] == "6" and text[position] == "0":
                    base = base[:position] + "0" + base[position + 1:]
    return base


def _build_vote_table(readings, final_text):
    """Summarize aligned OCR variants for display beside intermediate images."""
    table = []
    for position, winner in enumerate(final_text):
        votes = Counter()
        for text in readings:
            candidate = _pattern_window(text, 0.0)
            if len(candidate) == len(final_text):
                votes[candidate[position]] += 1
        table.append({
            "position": position + 1,
            "votes": dict(votes),
            "winner": winner,
            "confidence": round(100.0 * votes.get(winner, 0) / max(1, len(readings)), 1),
        })
    return table


def _ocr_pass(image, config):
    """One Tesseract run. Returns (text with only A-Z/0-9, mean confidence)."""
    data = pytesseract.image_to_data(image, config=config,
                                     output_type=pytesseract.Output.DICT)
    texts, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t, c = t.strip(), float(c)
        if re.search(r"[A-Za-z0-9]", t) and math.isfinite(c) and c >= 0:
            texts.append(t)
            confs.append(c)
    text = re.sub(r"[^A-Z0-9]", "", "".join(texts).upper())
    return text, (float(np.mean(confs)) if confs else 0.0)
