"""Classical image preprocessing operations for plate localization and OCR."""

import cv2
import numpy as np


def resize_to_width(image, width=800):
    """Resize while preserving aspect ratio for scale-stable morphology."""
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def to_grayscale(image):
    """Convert BGR to grayscale because localization uses intensity edges."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    """Apply bilateral filtering to suppress noise while preserving strokes."""
    return cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)


def enhance_contrast_if_needed(gray, std_threshold=40.0):
    """Use CLAHE local histogram equalization when global contrast is low."""
    if gray.std() >= std_threshold:
        return gray, False
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), True


def normalize_illumination(gray):
    """Divide by a Gaussian background estimate to flatten uneven lighting."""
    bg = cv2.GaussianBlur(gray, (0, 0), max(3, min(gray.shape) / 8))
    return cv2.divide(gray, np.maximum(bg, 1), scale=128)


def unsharp_mask(gray, amount=1.0):
    """Apply unsharp masking to restore local character edge contrast."""
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def blur_level(gray):
    """Estimate blur with variance of the Laplacian; lower means softer edges."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def deconvolve(gray, iterations=5):
    """Use Richardson-Lucy deconvolution with a small Gaussian PSF for blur recovery."""
    image = gray.astype(np.float32) / 255.0
    psf = cv2.getGaussianKernel(5, 1.0)
    psf = psf @ psf.T
    estimate = np.maximum(image, 1e-4)
    for _ in range(iterations):
        conv = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REPLICATE)
        ratio = image / np.maximum(conv, 1e-4)
        estimate *= cv2.filter2D(ratio, -1, psf[::-1, ::-1], borderType=cv2.BORDER_REPLICATE)
    return np.uint8(np.clip(estimate * 255, 0, 255))


def detect_edges(gray, low=None, high=None):
    """Apply median based auto-Canny so thresholds adapt to blur and exposure."""
    med = float(np.median(gray))
    low = max(0, int(0.66 * med)) if low is None else low
    high = min(255, int(1.33 * med)) if high is None else high
    return cv2.Canny(gray, low, max(low + 1, high))


def edge_maps(gray):
    """Combine auto-Canny, Sobel-X and blackhat/tophat morphology edge cues."""
    canny = detect_edges(gray)
    sx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    _, sx = cv2.threshold(sx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, gray.shape[1] // 40) | 1, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    morph = cv2.max(blackhat, tophat)
    _, morph = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"Canny": canny, "Sobel-X": sx, "Blackhat-tophat": morph}
