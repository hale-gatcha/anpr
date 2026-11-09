"""Streamlit interface for the local automatic number plate reader."""

import hashlib
import json
import math

import cv2
import streamlit as st

from anpr.ui import inject_css, render_chips, render_header, render_plate, render_steps
from src.pipeline import run_pipeline
from src.utils import decode_image_bytes


st.set_page_config(page_title="Plate Reader", page_icon="🚘", layout="wide")


@st.cache_data(show_spinner=False)
def cached_pipeline(file_bytes):
    """Decode and process uploaded bytes once, caching the pipeline result."""
    image_bgr = decode_image_bytes(file_bytes)
    return run_pipeline(image_bgr)


def to_rgb(image):
    """Convert OpenCV BGR or grayscale image data for Streamlit display."""
    if image is None or image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def friendly_error(title, message, kind="error"):
    """Render a readable styled message for input or processing errors."""
    icon = "⚠" if kind == "warning" else "✕"
    st.markdown(
        f'<div class="message-card {kind}"><span class="message-icon">{icon}</span>'
        f'<div><strong>{message_text(title)}</strong><br>{message_text(message)}</div></div>',
        unsafe_allow_html=True,
    )


def message_text(value):
    """Escape text before embedding it in a small HTML status card."""
    import html
    return html.escape(str(value), quote=True)


inject_css()
render_header()

with st.sidebar:
    st.markdown("### Display")
    plate_style = st.radio("Plate style", ["Green", "White"], horizontal=True)
    show_steps = st.checkbox("Show processing steps", value=False)
    st.markdown(
        '<div class="about-card"><strong>About this project</strong>'
        '<p>Digital Image Processing · SA1 Learning Evidence</p>'
        '<div class="pipeline-line"><span>Preprocess</span><b>›</b><span>Detect</span>'
        '<b>›</b><span>Clean</span><b>›</b><span>OCR</span></div></div>',
        unsafe_allow_html=True,
    )

with st.container(border=True):
    st.markdown('<div class="section-kicker">START HERE</div>', unsafe_allow_html=True)
    st.markdown("### Upload a vehicle photo")
    uploaded = st.file_uploader(
        "Drop your image here or browse files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    st.markdown('<div class="hint-line">Tip: a clear, close, front-facing photo reads best.</div>', unsafe_allow_html=True)

    image = None
    file_bytes = uploaded.getvalue() if uploaded is not None else None
    if uploaded is not None:
        try:
            image = decode_image_bytes(file_bytes)
        except ValueError as err:
            friendly_error("This image could not be opened", str(err))

    if image is not None:
        preview_col, button_col = st.columns([3, 1], vertical_alignment="center")
        with preview_col:
            st.image(to_rgb(image), caption=f"{uploaded.name} · {image.shape[1]} × {image.shape[0]} px", use_container_width=True)
        with button_col:
            read_clicked = st.button("Read Plate", type="primary", use_container_width=True)
    else:
        st.button("Read Plate", type="primary", disabled=True, use_container_width=True)
        read_clicked = False

file_key = hashlib.sha256(file_bytes).hexdigest() if file_bytes is not None else None
if read_clicked and file_key:
    try:
        with st.spinner("Reading plate..."):
            st.session_state["plate_result"] = cached_pipeline(file_bytes)
            st.session_state["plate_result_key"] = file_key
            st.session_state.pop("plate_error", None)
    except ValueError as err:
        st.session_state["plate_error"] = str(err)
        st.session_state["plate_result_key"] = file_key
    except Exception as err:
        detail = str(err)
        if "tesseract" in detail.casefold() or "tesseract" in type(err).__name__.casefold():
            detail = "Tesseract OCR is not available. Install it and make sure it is on PATH."
        st.session_state["plate_error"] = f"We couldn't finish reading this image. {detail}"
        st.session_state["plate_result_key"] = file_key

if file_key and st.session_state.get("plate_result_key") == file_key:
    if st.session_state.get("plate_error"):
        friendly_error("We couldn't process this photo", st.session_state["plate_error"])
    else:
        result = st.session_state.get("plate_result")
        if result is not None:
            detection, ocr = result.detection, result.ocr
            st.markdown('<div class="results-heading"><div class="section-kicker">YOUR RESULT</div><h2>Plate reading</h2></div>', unsafe_allow_html=True)
            if ocr is not None and ocr.ok:
                shown_text = ocr.text or ocr.raw_text
                confidence = ocr.confidence if math.isfinite(ocr.confidence) else 0.0
                render_plate(shown_text, [i for i, char in enumerate(shown_text) if char == "?"], plate_style, confidence)
                render_chips(ocr, result.winning_variant or ocr.winning_variant)
                if ocr.message and ocr.message != "OCR completed.":
                    friendly_error("Please verify this reading", ocr.message, "warning")
                elif "?" in shown_text:
                    friendly_error("Please verify this reading", "Some characters are uncertain. Compare them with the cropped plate.", "warning")
                st.caption("Confidence and agreement are rough indicators, not measured accuracy.")
                with st.expander("OCR details"):
                    st.write(f"Raw OCR text: {ocr.raw_text or 'No raw text returned.'}")
                    if result.alternatives:
                        st.write("Other candidate readings:")
                        st.dataframe(result.alternatives, hide_index=True, use_container_width=True)
                    if ocr.vote_table:
                        st.write("Per-character OCR votes:")
                        st.dataframe(ocr.vote_table, hide_index=True, use_container_width=True)
            else:
                render_plate("", [], plate_style, 0)
                if ocr is not None and ocr.message:
                    friendly_error("The plate was found, but the text is unclear", ocr.message, "warning")
                else:
                    friendly_error("No plate found yet", detection.message or "Try a sharper photo with the plate facing the camera.", "warning")

            if detection.found:
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown('<div class="image-card"><div class="image-title">Vehicle photo · detected plate</div>', unsafe_allow_html=True)
                    st.image(to_rgb(detection.annotated), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with right:
                    st.markdown('<div class="image-card"><div class="image-title">Plate crop</div>', unsafe_allow_html=True)
                    st.image(to_rgb(detection.plate_crop), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                if detection.message:
                    st.caption(detection.message)
            elif image is not None:
                st.markdown('<div class="image-card"><div class="image-title">Original photo</div>', unsafe_allow_html=True)
                st.image(to_rgb(image), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if detection.annotated is not None:
                ok, encoded = cv2.imencode(".jpg", detection.annotated)
                if ok:
                    download_col, _ = st.columns([1, 3])
                    with download_col:
                        st.download_button("Download annotated image", encoded.tobytes(), file_name="plate_reader_annotated.jpg", mime="image/jpeg", use_container_width=True)
            payload = {
                "plate": (ocr.text or ocr.raw_text) if ocr is not None and ocr.ok else "",
                "confidence": (ocr.confidence if ocr is not None and math.isfinite(ocr.confidence) else 0.0),
                "agreement": (ocr.agreement if ocr is not None and math.isfinite(ocr.agreement) else 0.0),
            }
            st.download_button("Download result JSON", json.dumps(payload, indent=2), file_name="plate_reader_result.json", mime="application/json")
            render_steps(detection, ocr, expanded=show_steps)

st.markdown('<footer class="app-footer">Digital Image Processing | SA1 Learning Evidence</footer>', unsafe_allow_html=True)
