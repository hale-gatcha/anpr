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
YELLOW_HUE_RANGE = (15, 35)    # OpenCV HSV hue scale
YELLOW_SATURATION_MIN = 80
YELLOW_VALUE_MIN = 100
GREEN_HUE_RANGE = (35, 95)
GREEN_SATURATION_MIN = 45
GREEN_VALUE_MIN = 55
COLOR_MIN_AREA_FRACTION = 0.001


@dataclass
class DetectionResult:
    found: bool
    box: tuple = None            # (x, y, w, h) on the resized image
    plate_crop: np.ndarray = None
    annotated: np.ndarray = None
    message: str = ""
    steps: dict = field(default_factory=dict)  # name -> image, for display
    candidates: list = field(default_factory=list)  # optional scored plate crops
    polygon: np.ndarray = None   # four corners on the resized working image
    polygon_original: np.ndarray = None  # four corners in input-image coordinates
    rectified_crop: np.ndarray = None
    de_sheared_crop: np.ndarray = None


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


def _plate_color_scores(roi):
    """Use HSV saturation and value to reward bright plate paint and penalize saturated red bodywork."""
    if roi is None or roi.size == 0:
        return 0.0, 1.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    neutral = ((v > 135) & (s < 100)).mean()
    green = ((v > 85) & (s > 45) & (h >= 30) & (h <= 100)).mean()
    yellow = ((v > 130) & (s > 70) & (h >= 12) & (h <= 40)).mean()
    red = ((s > 110) & (v > 75) & ((h <= 12) | (h >= 170))).mean()
    return float(max(neutral, green, yellow)), float(red)


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
    red_fraction = _plate_color_scores(image[y:y+h, x:x+w])[1]
    yellow_score = min(_yellow_plate_score(image, (x, y, w, h)) / 0.10, 1.0)
    return (0.25 * rect + 0.20 * aspect_score + 0.10 * position_score
            + 0.15 * size_score + 0.10 * char_score + 0.20 * yellow_score
            - 0.12 * red_fraction)


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


def _order_corners(points):
    """Order a quadrilateral clockwise using its centroid and image axes."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    v1, v2 = pts[1] - pts[0], pts[2] - pts[1]
    if v1[0] * v2[1] - v1[1] * v2[0] < 0:
        pts = pts[[0, 3, 2, 1]]
    return pts


def _bright_quadrilaterals(image):
    """Find bright, low-saturation plate regions using HSV thresholding, morphology and polygon approximation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    _, bright = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_sat = cv2.inRange(saturation, 0, 115)
    masks = [cv2.bitwise_and(bright, low_sat),
             cv2.bitwise_and(cv2.inRange(value, 155, 255), low_sat)]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    results = []
    img_area = image.shape[0] * image.shape[1]
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 0.005 * img_area <= area <= 0.60 * img_area:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(1.0, hull_area)
            if solidity < 0.85:
                continue
            perimeter = cv2.arcLength(hull, True)
            quad = None
            for fraction in (0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(hull, fraction * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4, 2)
                    break
            if quad is None:
                quad = cv2.boxPoints(cv2.minAreaRect(hull))
            quad = _order_corners(quad)
            rect = cv2.minAreaRect(quad)
            rw, rh = rect[1]
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if not 1.5 <= aspect <= 5.5:
                continue
            results.append((quad, solidity, area))
    return results


def _color_plate_proposals(image):
    """Use HSV thresholding, morphology and minAreaRect geometry to propose yellow and green plates."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    configs = (
        ("HSV yellow", YELLOW_HUE_RANGE, YELLOW_SATURATION_MIN, YELLOW_VALUE_MIN),
        ("HSV green", GREEN_HUE_RANGE, GREEN_SATURATION_MIN, GREEN_VALUE_MIN),
    )
    total = image.shape[0] * image.shape[1]
    results = []
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for name, hue, sat_min, value_min in configs:
        mask = cv2.inRange(hsv, np.array([hue[0], sat_min, value_min], np.uint8),
                          np.array([hue[1], 255, 255], np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not COLOR_MIN_AREA_FRACTION * total <= area <= .60 * total:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if not MIN_ASPECT <= aspect <= MAX_ASPECT:
                continue
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            polygon = None
            for epsilon in (.02, .03, .04):
                approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    polygon = approx.reshape(4, 2)
                    break
            if polygon is None:
                polygon = cv2.boxPoints(rect)
            x, y, w, h = cv2.boundingRect(np.asarray(polygon, np.int32))
            if w < 8 or h < 6:
                continue
            results.append((name, _order_corners(polygon),
                            area / max(1.0, cv2.contourArea(hull)), area))
    return results


def _hough_quadrilaterals(gray):
    """Propose fallback quadrilaterals from long Hough line segments and line intersections."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55,
                            minLineLength=max(35, gray.shape[1] // 12), maxLineGap=18)
    if lines is None:
        return []
    horizontal, vertical = [], []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if angle < 35 or angle > 145:
            horizontal.append((x1, y1, x2, y2, length))
        elif 55 < angle < 125:
            vertical.append((x1, y1, x2, y2, length))

    def intersection(a, b):
        x1, y1, x2, y2 = a[:4]; x3, y3, x4, y4 = b[:4]
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-5: return None
        d1, d2 = x1*y2-y1*x2, x3*y4-y3*x4
        return np.array([(d1*(x3-x4)-(x1-x2)*d2)/den,
                         (d1*(y3-y4)-(y1-y2)*d2)/den], np.float32)

    results = []
    # Pair long edge families. Limit to the strongest lines to keep this fallback inexpensive.
    for h0 in sorted(horizontal, key=lambda q: q[4], reverse=True)[:10]:
        for h1 in sorted(horizontal, key=lambda q: q[4], reverse=True)[:10]:
            if h0 is h1: continue
            for v0 in sorted(vertical, key=lambda q: q[4], reverse=True)[:10]:
                for v1 in sorted(vertical, key=lambda q: q[4], reverse=True)[:10]:
                    if v0 is v1: continue
                    pts = [intersection(h0,v0), intersection(h0,v1), intersection(h1,v1), intersection(h1,v0)]
                    if any(p is None for p in pts): continue
                    q = np.asarray(pts, np.float32)
                    if np.any(q[:,0] < 0) or np.any(q[:,0] >= gray.shape[1]) or np.any(q[:,1] < 0) or np.any(q[:,1] >= gray.shape[0]): continue
                    q = _order_corners(q)
                    area = cv2.contourArea(q)
                    if .005 * gray.size <= area <= .60 * gray.size:
                        results.append((q, .85, area))
        if len(results) >= 20: break
    return results


def _rectify_polygon(image, polygon):
    """Apply a homography to map the ordered plate quadrilateral onto a 440x200 canvas."""
    points = _order_corners(polygon)
    target = np.float32([[0, 0], [439, 0], [439, 199], [0, 199]])
    transform = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(image, transform, (440, 200),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _needs_perspective(polygon):
    """Use quadrilateral rectification only when edge slopes show meaningful perspective or skew."""
    p = _order_corners(polygon)
    top = abs(np.degrees(np.arctan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0])))
    bottom = abs(np.degrees(np.arctan2(p[2, 1] - p[3, 1], p[2, 0] - p[3, 0])))
    return max(top, bottom) >= 4.0 or abs(top - bottom) >= 5.0


def _deskew_by_projection_profile(image):
    """Fall back to deskewing by maximizing projection-profile contrast over rotation and shear."""
    original_size = (image.shape[1], image.shape[0])
    scale = min(1.0, 160.0 / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def contrast(mask):
        row = mask.sum(axis=1).astype(np.float32) / max(1, w)
        col = mask.sum(axis=0).astype(np.float32) / max(1, h)
        valleys = np.count_nonzero((col[1:-1] <= col[:-2]) & (col[1:-1] <= col[2:])
                                   & (col[1:-1] < .35 * max(1.0, col.max())))
        return float(np.var(row) + np.var(col) + valleys * max(1.0, col.max()) ** 2 / w)

    best_image, best_score = image, contrast(binary)
    center = (w / 2.0, h / 2.0)
    for angle in range(-35, 36, 2):
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        for shear in np.arange(-0.5, 0.5001, 0.05):
            matrix = rotation.copy()
            matrix[0, 1] += shear
            matrix[0, 2] -= shear * center[1]
            warped = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            warped_gray = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, warped_binary = cv2.threshold(warped_gray, 0, 255,
                                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            score = contrast(warped_binary)
            if score > best_score:
                best_image, best_score = warped, score
    if best_image.shape[1] != original_size[0] or best_image.shape[0] != original_size[1]:
        best_image = cv2.resize(best_image, original_size, interpolation=cv2.INTER_CUBIC)
    return best_image, best_score


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
            if _char_score(gray, cand[:4], char_cache) == 0.0:
                continue
            n_candidates += 1
            score = _score(cand, resized.shape, gray, resized, char_cache)
            ranked.append((score, cand, closed, edge_name, width, kh))

    # Merge the color-segmented quadrilateral and Hough fallback proposals
    # into the same ranking as the existing Canny/Sobel/morphology path.
    quad_by_box = {}
    path_by_box = {}
    fallback_by_box = {}
    fallback_attempts = 0
    proposals = [("White quadrilateral", polygon, solidity, area)
                 for polygon, solidity, area in _bright_quadrilaterals(resized)]
    proposals.extend(("Hough", polygon, solidity, area)
                     for polygon, solidity, area in _hough_quadrilaterals(gray))
    proposals.extend(_color_plate_proposals(resized))
    for path_name, polygon, solidity, poly_area in proposals:
            x, y, w, h = cv2.boundingRect(np.asarray(polygon, np.int32))
            if w < 8 or h < 6:
                continue
            rect = cv2.minAreaRect(np.asarray(polygon, np.float32))
            rw, rh = rect[1]
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            aspect_score = 1.0 - min(abs(aspect - 3.0) / 3.0, 1.0)
            roi = gray[y:y+h, x:x+w]
            rectified = _rectify_polygon(resized, polygon)
            rect_gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
            count0 = find_char_row(rect_gray)[0]
            count180 = find_char_row(cv2.rotate(rect_gray, cv2.ROTATE_180))[0]
            char_count = max(count0, count180)
            if char_count < 3:
                if fallback_attempts >= 2:
                    continue
                fallback_attempts += 1
                axis_crop = resized[y:y+h, x:x+w]
                fallback, _ = _deskew_by_projection_profile(axis_crop)
                fallback_gray = cv2.cvtColor(fallback, cv2.COLOR_BGR2GRAY)
                char_count = find_char_row(fallback_gray)[0]
                if char_count >= 3:
                    fallback_by_box[(x, y, w, h)] = fallback
                    quad_by_box.pop((x, y, w, h), None)
            # Rectified character rows provide a more relevant shape cue than the slanted axis box.
            char_score = (1.0 if char_count >= 5 else 0.7 if char_count == 4
                          else 0.4 if char_count == 3 else 0.0)
            area_frac = poly_area / float(gray.size)
            size_score = min((area_frac ** 0.5) / 0.3, 1.0)
            neutral_bright, red_fraction = _plate_color_scores(resized[y:y+h, x:x+w])
            color_match = neutral_bright
            if path_name == "HSV yellow":
                color_match = max(color_match, min(_yellow_plate_score(resized, (x,y,w,h)) / .10, 1.0))
            elif path_name == "HSV green":
                hsv_roi = cv2.cvtColor(resized[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                green_fraction = ((hsv_roi[:,:,0] >= GREEN_HUE_RANGE[0])
                                  & (hsv_roi[:,:,0] <= GREEN_HUE_RANGE[1])
                                  & (hsv_roi[:,:,1] >= GREEN_SATURATION_MIN)
                                  & (hsv_roi[:,:,2] >= GREEN_VALUE_MIN)).mean()
                color_match = max(color_match, float(green_fraction))
            # Character evidence dominates, while saturation penalizes red bodywork.
            if char_count < 3:
                continue
            score = (0.23 * solidity + 0.16 * aspect_score + 0.12 * size_score
                     + 0.34 * char_score + 0.15 * color_match
                     - 0.12 * red_fraction)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
            ranked.append((score, (x, y, w, h, solidity), mask,
                           path_name, 0, 0))
            quad_by_box[(x, y, w, h)] = np.asarray(polygon, np.float32)
            path_by_box[(x, y, w, h)] = path_name
            n_candidates += 1

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
        polygon = quad_by_box.get((x, y, w, h))
        if polygon is not None:
            cv2.polylines(all_boxes, [polygon.astype(np.int32)], True, (0, 0, 255), 2)
        else:
            cv2.rectangle(all_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)
    steps["7 All contours"] = all_boxes

    x, y, w, h, _ = best
    crops = []
    for score, cand, _, _, _, _ in selected:
        cx, cy, cw, ch, _ = cand
        p = int(.05 * ch)
        xa, ya, xb, yb = max(0,cx-p), max(0,cy-p), min(resized.shape[1],cx+cw+p), min(resized.shape[0],cy+ch+p)
        record = {"score": score, "box": (xa,ya,xb-xa,yb-ya), "crop": resized[ya:yb,xa:xb].copy()}
        polygon = quad_by_box.get((cx, cy, cw, ch))
        if polygon is not None:
            record["polygon"] = polygon.copy()
            record["color_path"] = path_by_box.get((cx, cy, cw, ch), "")
            if _needs_perspective(polygon):
                rectified = _rectify_polygon(resized, polygon)
                gray0 = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
                gray180 = cv2.rotate(gray0, cv2.ROTATE_180)
                if find_char_row(gray180)[0] > find_char_row(gray0)[0]:
                    rectified = cv2.rotate(rectified, cv2.ROTATE_180)
                record["axis_crop"] = record["crop"]
                record["crop"] = rectified
                record["rectified_crop"] = rectified
        elif (cx, cy, cw, ch) in fallback_by_box:
            record["crop"] = fallback_by_box[(cx, cy, cw, ch)]
            record["de_sheared_crop"] = record["crop"]
        crops.append(record)

    # Small padding so characters at the border are not cut off
    pad = int(0.05 * h)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1 = min(x + w + pad, resized.shape[1])
    y1 = min(y + h + pad, resized.shape[0])

    best_polygon = quad_by_box.get((x, y, w, h))
    crop = resized[y0:y1, x0:x1].copy()
    if best_polygon is not None and _needs_perspective(best_polygon):
        crop = _rectify_polygon(resized, best_polygon)
        gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if find_char_row(cv2.rotate(gray0, cv2.ROTATE_180))[0] > find_char_row(gray0)[0]:
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        steps["9 Rectified plate"] = crop
    annotated = resized.copy()
    if best_polygon is not None:
        cv2.polylines(annotated, [best_polygon.astype(np.int32)], True, (0, 255, 0), 3)
    else:
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
        polygon=best_polygon,
        polygon_original=(best_polygon * np.array([
            image_bgr.shape[1] / resized.shape[1], image_bgr.shape[0] / resized.shape[0]
        ], np.float32)) if best_polygon is not None else None,
        rectified_crop=crop if best_polygon is not None else None,
    )
