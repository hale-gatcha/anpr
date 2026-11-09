"""Shared helper: find a row of similar-height characters in a gray image.

A license plate holds a short row of characters with almost the same
height, lined up horizontally. A window, a bumper or a body panel does
not. This is used in two places:
  * detection.py - to prefer regions that contain a character row
  * ocr.py       - to crop tightly around the characters before thresholding
"""

import cv2


def _components(mask):
    """All connected blobs as (x, y, w, h, touches_edge)."""
    H, W = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        edge = x == 0 or y == 0 or x + w >= W or y + h >= H
        out.append((int(x), int(y), int(w), int(h), edge))
    return out


def _row_in_mask(mask):
    """Largest group of character-like blobs with similar height and the
    same vertical position, inside one binary mask (foreground = 255).
    Returns (group, all_components)."""
    H, W = mask.shape
    comps = _components(mask)
    chars = []
    for x, y, w, h, edge in comps:
        if edge:
            continue                       # touches the edge: frame, not a character
        if not (0.25 * H <= h <= 0.9 * H):
            continue                       # too small or too tall for a character
        if not (0.1 <= w / h <= 1.2):
            continue                       # characters are roughly as wide as tall or narrower
        chars.append((x, y, w, h))

    best = []
    for ref in chars:
        group = [c for c in chars
                 if abs(c[3] - ref[3]) <= 0.2 * ref[3]
                 and abs((c[1] + c[3] / 2) - (ref[1] + ref[3] / 2)) <= 0.3 * ref[3]]
        if len(group) > len(best):
            best = group
    return best, comps


def _extend_row(x0, x1, y0, y1, comps):
    """Grow the row sideways to take in a neighbouring character that is
    fused to the plate frame (so it was excluded as an 'edge' blob).

    A blob is accepted if it is a similar height (up to 1.75x because it
    includes part of the frame), not too wide, overlaps the row vertically
    and sits within 0.8 character-heights of the current end of the row.
    """
    row_h = y1 - y0
    changed = True
    while changed:
        changed = False
        for x, y, w, h, edge in comps:
            if not edge:
                continue
            if not (0.7 * row_h <= h <= 1.75 * row_h) or w > 1.3 * row_h:
                continue
            overlap = min(y + h, y1) - max(y, y0)
            if overlap < 0.5 * row_h:
                continue
            if x + w <= x0 and x0 - (x + w) <= 0.8 * row_h:      # to the left
                x0, changed = x, True
            elif x >= x1 and x - x1 <= 0.8 * row_h:             # to the right
                x1, changed = x + w, True
    return x0, x1


def find_char_row(gray, extend=False):
    """Return (count, bbox) for the best character row in a gray image.

    bbox is (x, y, w, h) in the coordinates of `gray`, or None if fewer
    than 3 characters were found. Both polarities are tried because plates
    can be dark-on-light or light-on-dark. With extend=True the box also
    takes in neighbouring characters fused to the plate frame.
    """
    if gray is None or gray.size == 0:
        return 0, None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best, best_comps = [], []
    for mask in (binary, cv2.bitwise_not(binary)):
        group, comps = _row_in_mask(mask)
        if len(group) > len(best):
            best, best_comps = group, comps
    if len(best) < 3:
        return len(best), None
    x0 = min(c[0] for c in best)
    y0 = min(c[1] for c in best)
    x1 = max(c[0] + c[2] for c in best)
    y1 = max(c[1] + c[3] for c in best)
    if extend:
        x0, x1 = _extend_row(x0, x1, y0, y1, best_comps)
    return len(best), (x0, y0, x1 - x0, y1 - y0)
