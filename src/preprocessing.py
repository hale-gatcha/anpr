"""Stage 2 - Image preprocessing.

Each function does ONE job so you can explain it on its own.
Only techniques that help plate localization are used.
"""

import cv2


def resize_to_width(image, width=800):
    """Resize to a fixed width, keeping the aspect ratio.

    Why: kernel sizes and area limits in later steps are tuned for one
    image size, so every photo must be brought to the same scale.
    """
    h, w = image.shape[:2]
    scale = width / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, int(h * scale)), interpolation=interp)


def to_grayscale(image):
    """Convert BGR to grayscale.

    Why: plate edges depend on brightness changes, not color, and one
    channel is faster to process than three.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    """Bilateral filter: smooths noise but keeps edges sharp.

    Why: a plain Gaussian blur also blurs the character edges we need.
    The bilateral filter only averages pixels that are close in both
    position and brightness, so edges survive.
    """
    return cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)


def enhance_contrast_if_needed(gray, std_threshold=40.0):
    """Apply CLAHE only when the image is low contrast.

    Returns (image, was_applied). The standard deviation of pixel
    intensity is used as a simple contrast measure.
    """
    if gray.std() >= std_threshold:
        return gray, False
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray), True


def detect_edges(gray, low=30, high=200):
    """Canny edge detection.

    Why: a plate is a rectangle full of character edges, so it shows up
    as a dense cluster of edges.
    """
    return cv2.Canny(gray, low, high)
