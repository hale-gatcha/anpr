"""Stage 3 - Plate cleanup and character recognition (OCR).

Tesseract is an external OCR engine. Everything in prepare_plate_for_ocr()
is our own image processing; Tesseract only reads the cleaned image.
"""

import os
import re
import shutil
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

from .characters import find_char_row

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_AGREEMENT = 0.6     # heuristic: below this the two OCR passes disagree too much

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


def prepare_plate_for_ocr(plate_bgr):
    """Return (image for Tesseract, dict of intermediate steps)."""
    steps = {}
    h, w = plate_bgr.shape[:2]
    # 1. Upscale: Tesseract works best when characters are ~30+ px tall
    scale = max(1.0, 200.0 / h)
    big = cv2.resize(plate_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    steps["ocr1_upscaled"] = big
    # 2. Grayscale + light denoise
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    steps["ocr2_gray"] = gray
    # 3. Crop tightly around the row of characters BEFORE thresholding.
    #    Otsu picks one threshold for the whole image, so a dark bumper bar
    #    or handle above the plate would distort it. We crop only
    #    VERTICALLY to the character row and keep the full width: a sideways
    #    crop risks cutting off real letters that the row finder missed.
    n_chars, row = find_char_row(gray)
    if row is not None:
        _, y, _, h = row
        py = int(0.1 * h)
        y0, y1 = max(y - py, 0), min(y + h + py, gray.shape[0])
        gray = gray[y0:y1, :]
        steps["ocr3_tight_crop"] = gray
        # 4. Otsu threshold with automatic polarity, on the tight crop only
        line = _binarize(gray)
        # 5. Erase blobs that are clearly not characters (logos, small
        #    text like 'NCR', dots, frame pieces), judged against the
        #    character height we measured.
        line = _remove_non_characters(line, h)
        steps["ocr4_binary"] = line
    else:
        # No clear character row found: fall back to thresholding the whole
        # crop and removing everything that is not character-sized.
        binary = _binarize(gray)
        steps["ocr3_binary"] = binary
        line, _ = _crop_to_text_line(binary)
        steps["ocr4_text_line"] = line
    # 5. White border helps Tesseract
    final = cv2.copyMakeBorder(line, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    steps["ocr5_final"] = final
    return final, steps


# ------------------------------------------------------------ pattern check
# Soft checks only. Car plates seen in the Philippines are commonly
# 3 letters + 3-4 digits (e.g. ABC 1234). Motorcycle formats differ, so a
# mismatch triggers a warning, never a rejection. Verify against LTO info.
_CAR_PATTERN = re.compile(r"^[A-Z]{3}\d{3,4}$")

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

    prepared, steps = prepare_plate_for_ocr(plate_bgr)

    # Pass A: restricted to A-Z / 0-9. This gives the TEXT.
    # (With a whitelist, Tesseract's own confidence value is not meaningful,
    #  so we do not use it.)
    raw, _ = _ocr_pass(prepared, f"--psm 7 --oem 3 -c tessedit_char_whitelist={WHITELIST}")
    if not raw:
        return OCRResult(False, steps=steps,
                         message="OCR returned no characters. The plate may be unreadable.")

    # Pass B: unrestricted. Used for a confidence value and an agreement check.
    other, confidence = _ocr_pass(prepared, "--psm 7 --oem 3")
    agreement = SequenceMatcher(None, raw, other).ratio() if other else 0.0

    corrected = correct_with_pattern(raw)
    fmt_ok = bool(_CAR_PATTERN.match(corrected))

    notes = []
    if len(raw) < 5:
        notes.append("Very few characters read: result is unreliable.")
    if agreement < MIN_AGREEMENT:
        notes.append(f"The two OCR passes disagree (agreement {agreement:.0%}): result may be unreliable.")
    if "?" in corrected:
        notes.append("Some characters are ambiguous (marked ?) and need a manual check.")
    elif not fmt_ok:
        notes.append("Text does not match the common 3-letter + 3-4-digit pattern.")
    return OCRResult(True, raw_text=raw, text=corrected, confidence=confidence,
                     agreement=agreement, format_ok=fmt_ok, steps=steps,
                     message=" ".join(notes) or "OCR completed.")


def _ocr_pass(image, config):
    """One Tesseract run. Returns (text with only A-Z/0-9, mean confidence)."""
    data = pytesseract.image_to_data(image, config=config,
                                     output_type=pytesseract.Output.DICT)
    texts, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t, c = t.strip(), float(c)
        if re.search(r"[A-Za-z0-9]", t) and c >= 0:
            texts.append(t)
            confs.append(c)
    text = re.sub(r"[^A-Z0-9]", "", "".join(texts).upper())
    return text, (float(np.mean(confs)) if confs else 0.0)
