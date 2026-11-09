"""Automatic Number Plate Recognition (ANPR) - Streamlit interface.

Run with:  python -m streamlit run app.py
"""

import cv2
import streamlit as st

from src.pipeline import run_pipeline
from src.utils import decode_image_bytes

st.set_page_config(page_title="ANPR", page_icon="🚗", layout="wide")


def to_rgb(image):
    """OpenCV uses BGR; Streamlit expects RGB. Gray images pass through."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ header
st.title("Automatic Number Plate Recognition (ANPR)")
st.caption(
    "Upload a photo of a vehicle. The system finds the license plate with "
    "digital image processing (OpenCV) and reads it with Tesseract OCR."
)

with st.expander("How it works: the image-processing steps"):
    st.markdown(
        """
**Finding the plate (our own image processing)**
1. **Resize** to a fixed width so every photo is handled at the same scale.
2. **Grayscale**: plates are found by brightness changes, not color.
3. **Bilateral filter**: removes noise but keeps edges sharp.
4. **CLAHE contrast boost**: applied only when the image is low contrast.
5. **Canny edge detection**: characters create many close edges.
6. **Morphological closing**: joins those edges into solid blocks.
7. **Contours + filtering**: keeps blocks with a plate-like shape and size.
8. **Character-row check**: prefers blocks that contain a row of
   similar-height characters (so windows and panels lose).
9. **Bounding box + crop** of the best block.

**Reading the plate**
10. **Clean the crop**: enlarge, keep the character row, Otsu threshold to
    black text on white, erase blobs that are not characters.
11. **Tesseract OCR** (external engine), run twice: once restricted to
    A-Z/0-9 for the text, once unrestricted to measure reliability.
12. **Pattern check**: look-alike characters are corrected only when the
    fix is clear. Doubtful characters are marked `?`.

**Limits:** blurry, small, angled or cluttered photos can fail. The
confidence values are rough indicators, not proven accuracy.
        """
    )

# ------------------------------------------------------------------ input
uploaded = st.file_uploader("Upload a vehicle image (JPG, JPEG or PNG)",
                            type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    image = decode_image_bytes(uploaded.getvalue())
except ValueError as err:
    st.error(f"Could not use this file: {err}")
    st.stop()

st.subheader("Original image")
st.image(to_rgb(image), use_container_width=True)
st.caption(f"{uploaded.name} - {image.shape[1]}x{image.shape[0]} px")

# ----------------------------------------------------------------- process
if not st.button("Recognize Plate", type="primary"):
    st.stop()

error_message = None
result = None
with st.status("Processing...", expanded=False) as status:
    try:
        status.update(label="Detecting the plate and reading characters...")
        result = run_pipeline(image)
        status.update(label="Processing complete", state="complete")
    except ValueError as err:                       # e.g. image too small
        error_message = str(err)
        status.update(label="Could not process the image", state="error")
    except Exception as err:                        # anything unexpected
        error_message = f"Something went wrong while processing: {err}"
        status.update(label="Processing failed", state="error")

# Errors are shown OUTSIDE the status box, which is collapsed by default.
if error_message:
    st.error(error_message)
    st.stop()

detection, ocr = result.detection, result.ocr

# --------------------------------------------------------------- detection
st.subheader("Detection result")
if not detection.found:
    st.error(detection.message)
    st.caption("Tip: use a sharper photo where the plate is clearly visible.")
else:
    left, right = st.columns(2)
    with left:
        st.markdown("**Processed image with detected plate**")
        st.image(to_rgb(detection.annotated), use_container_width=True)
    with right:
        st.markdown("**Cropped license plate**")
        st.image(to_rgb(detection.plate_crop), use_container_width=True)
    st.caption(detection.message +
               " This only shows where a plate-like region was found; "
               "it does not prove the region is a real plate.")

# --------------------------------------------------------------------- OCR
if ocr is not None:
    st.subheader("Recognized plate number")
    if not ocr.ok:
        st.warning(ocr.message)
    else:
        st.markdown(f"## `{ocr.text}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw OCR text", ocr.raw_text)
        c2.metric("Tesseract confidence", f"{ocr.confidence:.0f}%")
        c3.metric("Agreement of 2 passes", f"{ocr.agreement:.0%}")
        if ocr.message != "OCR completed.":
            st.warning(ocr.message)
        else:
            st.info("No warnings were raised, but this does not guarantee the "
                    "result is correct. Check it against the image.")
        st.caption(
            "Confidence and agreement are rough indicators, not a measured "
            "accuracy. Raw text is exactly what Tesseract returned; the "
            "large text has the position-based look-alike correction applied.")

# ----------------------------------------------------------- intermediate
with st.expander("Show intermediate processing steps"):
    steps = dict(detection.steps)
    if ocr is not None:
        steps.update(ocr.steps)
    names = list(steps)
    for i in range(0, len(names), 3):
        cols = st.columns(3)
        for col, name in zip(cols, names[i:i + 3]):
            col.image(to_rgb(steps[name]), caption=name, use_container_width=True)
