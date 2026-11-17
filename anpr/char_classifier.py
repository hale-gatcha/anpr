"""Optional HOG/SVM character classification for explainable plate OCR."""

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "char_clf.joblib"
_MODEL = None
_MODEL_CHECKED = False


def normalize_glyph(image):
    """Use thresholding and aspect-preserving resize to normalize a glyph to 32x48."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return np.full((48, 32), 255, np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Keep the majority background white and the glyph black regardless of
    # whether a source crop was supplied in normal or inverted polarity.
    if np.count_nonzero(binary == 0) > binary.size / 2:
        binary = cv2.bitwise_not(binary)
    points = cv2.findNonZero(255 - binary)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        binary = binary[y:y+h, x:x+w]
    scale = min(28 / max(1, binary.shape[1]), 44 / max(1, binary.shape[0]))
    resized = cv2.resize(binary, (max(1, round(binary.shape[1] * scale)),
                                  max(1, round(binary.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((48, 32), 255, np.uint8)
    y0, x0 = (48 - resized.shape[0]) // 2, (32 - resized.shape[1]) // 2
    canvas[y0:y0+resized.shape[0], x0:x0+resized.shape[1]] = resized
    return canvas


def hog_features(image):
    """Combine 9-bin HOG gradients with symmetry, centroid, aspect, holes and stroke-width features."""
    glyph = normalize_glyph(image)
    descriptor = cv2.HOGDescriptor((32, 48), (16, 16), (8, 8), (8, 8), 9)
    hog = descriptor.compute(glyph).reshape(-1).astype(np.float32)
    ink = (glyph < 128).astype(np.uint8)
    top_bottom = float(1.0 - np.mean(np.abs(ink.astype(float) - np.flipud(ink).astype(float))))
    left_right = float(1.0 - np.mean(np.abs(ink.astype(float) - np.fliplr(ink).astype(float))))
    ys, xs = np.where(ink > 0)
    if len(xs):
        centroid_y = float(np.mean(ys) / 47.0)
        aspect = float((xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1))
    else:
        centroid_y, aspect = .5, 0.0
    contours, hierarchy = cv2.findContours(ink * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            if hierarchy[0][index][3] >= 0:
                moments = cv2.moments(contour)
                if moments["m00"]:
                    holes.append(float(moments["m01"] / moments["m00"] / 47.0))
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = float(2.0 * distance[ink > 0].mean() / 48.0) if np.any(ink) else 0.0
    handcrafted = np.asarray([
        top_bottom, left_right, centroid_y, aspect,
        min(len(holes), 3) / 3.0,
        float(np.mean(holes)) if holes else 0.0,
        stroke,
    ], dtype=np.float32)
    return np.concatenate([hog, handcrafted])


def load_classifier():
    """Load the optional joblib SVM once; report a clear fallback when sklearn is unavailable."""
    global _MODEL, _MODEL_CHECKED
    if _MODEL_CHECKED:
        return _MODEL
    _MODEL_CHECKED = True
    if not MODEL_PATH.is_file():
        return None
    try:
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
    except ImportError:
        print("scikit-learn/joblib is not installed; using Tesseract only.")
        _MODEL = None
    except Exception as error:
        print(f"Character model could not be loaded ({error}); using Tesseract only.")
        _MODEL = None
    return _MODEL


def classify_character(image, position):
    """Classify one glyph with HOG and restrict choices by plate position."""
    global _MODEL
    model = load_classifier()
    if model is None:
        return "", 0.0
    try:
        probabilities = model.predict_proba(hog_features(image).reshape(1, -1))[0]
    except ValueError as error:
        print(f"Character model feature layout is outdated ({error}); using Tesseract only.")
        _MODEL = None
        return "", 0.0
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    labels = [str(value) for value in getattr(model, "classes_", estimator.classes_)]
    allowed = [i for i, label in enumerate(labels)
               if (position < 3 and label.isalpha())
               or (position >= 3 and label.isdigit())]
    if not allowed:
        return "", 0.0
    best = max(allowed, key=lambda index: probabilities[index])
    return labels[best], float(probabilities[best])
