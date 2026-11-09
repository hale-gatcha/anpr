"""Helper functions: loading and validating images."""

import os

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(path):
    """Load an image from disk as a BGR NumPy array.

    Raises ValueError with a readable message if the file is not a
    supported type or cannot be decoded as an image.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Use JPG, JPEG or PNG.")
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    # np.fromfile + imdecode also works with non-ASCII Windows paths
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The file could not be read as an image (it may be corrupted).")
    return image


def decode_image_bytes(file_bytes):
    """Decode raw bytes (e.g. from a Streamlit upload) into a BGR image."""
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    return image


def save_image(path, image):
    """Save an image, creating the folder if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)
