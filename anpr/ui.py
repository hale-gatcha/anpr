"""Small, reusable Streamlit rendering helpers for the Plate Reader UI."""

import html
import re

import cv2
import streamlit as st


CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root { --green:#0B5D3B; --deep:#07422A; --fresh:#1E9E5A; --mint:#EAF6EF; --ink:#12261C; --amber:#E0A100; --red:#C0392B; }
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); }
.stApp { background:linear-gradient(180deg,#f6fbf7 0%,#EAF6EF 58%,#f7fbf8 100%); }
.block-container { max-width:1160px; padding-top:1.4rem; padding-bottom:1.5rem; }
.hero { background:linear-gradient(120deg,#07422A 0%,#0B5D3B 55%,#1E9E5A 100%); color:white; border-radius:20px; padding:1.45rem 1.8rem; margin-bottom:1.4rem; box-shadow:0 12px 28px #07422a25; display:flex; align-items:center; gap:1rem; }
.hero-icon { width:55px; height:39px; border:3px solid white; border-radius:8px; display:grid; place-items:center; font-size:23px; flex:none; box-shadow:inset 0 0 0 2px #ffffff35; }
.hero h1 { color:white; font-size:clamp(1.7rem,4vw,2.35rem); margin:0; font-weight:700; letter-spacing:-.035em; }
.hero p { color:#e5f5eb; margin:.18rem 0 0; font-size:.98rem; }
.section-kicker { color:#1E9E5A; font-size:.72rem; font-weight:700; letter-spacing:.13em; }
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stFileUploader"]) { background:#fff; border:2px dashed #61b783; border-radius:18px; padding:1.35rem 1.5rem 1.1rem; box-shadow:0 8px 24px #07422a0d; margin-bottom:1.5rem; }
[data-testid="stFileUploader"] { background:#f7fcf8; border:1px dashed #8bc8a4; border-radius:12px; padding:.8rem; }
[data-testid="stFileUploader"] section { border:0; }
.hint-line { color:#68786e; font-size:.84rem; margin:.45rem 0 .85rem; }
.stButton>button[kind="primary"] { background:var(--green); border:0; border-radius:11px; font-weight:700; min-height:2.8rem; box-shadow:0 5px 12px #0b5d3b27; }
.stButton>button[kind="primary"]:hover { background:var(--deep); border:0; }
.stButton>button { border-radius:10px; }
[data-testid="stSpinner"] svg { color:#1E9E5A; }
.results-heading { margin:1.4rem 0 .8rem; }
.results-heading h2 { margin:.1rem 0; color:var(--ink); }
.plate-wrap { max-width:690px; margin:.6rem auto 1rem; animation:plate-in .55s ease-out both; }
.plate { aspect-ratio:2.2/1; position:relative; border:9px solid #103e2a; border-radius:20px; padding:3.5% 5%; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 14px 28px #12261c2c,inset 0 0 0 3px #ffffffaa; overflow:hidden; }
.plate.white { background:linear-gradient(145deg,#fff 0%,#f1f5f2 100%); color:#123b28; }
.plate.green { background:linear-gradient(145deg,#0b6a42 0%,#07422a 100%); color:#fff; border-color:#073b26; }
.plate:before { content:""; position:absolute; inset:8px; border:2px solid currentColor; opacity:.55; border-radius:11px; pointer-events:none; }
.screw { position:absolute; width:10px; height:10px; top:13px; background:linear-gradient(135deg,#a5aaa7,#4b5550); border:1px solid #526159; border-radius:50%; box-shadow:inset 0 1px 2px #fff; z-index:1; }
.screw.left { left:18px; } .screw.right { right:18px; }
.plate-text { z-index:1; font-family:"Saira Stencil One","Black Ops One",Oswald,"Arial Black",Impact,sans-serif; font-size:clamp(2.1rem,8vw,5.4rem); font-weight:700; line-height:1; letter-spacing:.12em; white-space:nowrap; text-shadow:0 1px 0 #0001; }
.plate-group { display:inline-block; } .plate-gap { display:inline-block; width:.34em; }
.uncertain { color:#e0a100; text-decoration:underline dotted #e0a100 3px; text-underline-offset:5px; }
.plate.green .uncertain { color:#ffd36b; text-decoration-color:#ffd36b; }
.plate-caption { z-index:1; font-size:clamp(.52rem,1.5vw,.76rem); font-weight:700; letter-spacing:.3em; margin-top:1.3%; }
.chip-row { display:flex; align-items:stretch; justify-content:center; gap:.65rem; flex-wrap:wrap; margin:.75rem auto 1rem; }
.step-badge { display:inline-grid; place-items:center; width:25px; height:25px; margin-right:.55rem; border-radius:50%; background:#1E9E5A; color:#fff; font-size:.76rem; vertical-align:middle; }
.stat-chip { min-width:155px; padding:.62rem .8rem; background:#fff; border:1px solid #dfece3; border-radius:999px; box-shadow:0 3px 10px #07422a0a; font-size:.82rem; }
.stat-chip strong { color:#143b29; }
.mini-track { height:5px; background:#edf2ee; border-radius:5px; overflow:hidden; margin-top:.4rem; }
.mini-fill { height:100%; border-radius:5px; }
.fill-green { background:#1E9E5A; } .fill-amber { background:#E0A100; } .fill-red { background:#C0392B; }
.format-chip,.variant-chip { align-self:center; border-radius:999px; padding:.62rem .9rem; background:#eff8f2; color:#145c37; font-size:.82rem; font-weight:600; border:1px solid #cfe7d7; }
.format-chip.warn { background:#fff7df; color:#856000; border-color:#f0dfa8; }
.message-card { display:flex; align-items:flex-start; gap:.7rem; padding:.85rem 1rem; margin:.7rem auto; max-width:850px; border-radius:13px; line-height:1.5; }
.message-card.warning { background:#fff7df; color:#624b0b; border:1px solid #f0dfa8; }
.message-card.error { background:#fff1ef; color:#7f2822; border:1px solid #f1c5bf; }
.message-icon { font-size:1.15rem; font-weight:700; }
.image-card { background:white; border-radius:16px; padding:.8rem; border:1px solid #e0ece4; box-shadow:0 7px 22px #07422a0d; overflow:hidden; }
.image-title { font-weight:700; font-size:.9rem; margin:.15rem .15rem .65rem; }
.image-card img { border-radius:10px; }
.about-card { margin-top:1rem; padding:1rem; border:1px solid #d9eae0; background:#f7fcf8; border-radius:15px; line-height:1.45; }
.about-card p { color:#65756b; font-size:.81rem; margin:.35rem 0 .8rem; }
.pipeline-line { display:flex; flex-wrap:wrap; align-items:center; gap:.36rem; color:#145c37; font-size:.76rem; font-weight:700; }
.pipeline-line b { color:#85ae94; }
.app-footer { margin-top:2rem; background:linear-gradient(90deg,#07422A,#0B5D3B,#1E9E5A); color:#f5fff8; text-align:center; padding:.75rem 1rem; border-radius:12px; font-size:.8rem; letter-spacing:.02em; }
[data-testid="stSidebar"] { background:#f8fcf9; }
[data-testid="stExpander"] { background:#fff; border:1px solid #dfebe3; border-radius:14px; }
@keyframes plate-in { from { opacity:0; transform:translateY(13px); } to { opacity:1; transform:translateY(0); } }
@media (max-width:640px) { .block-container { padding:.8rem .8rem 1rem; } .hero { padding:1.1rem; gap:.75rem; } .hero-icon { width:46px; height:34px; } .upload-card { padding:1rem; } .plate { border-width:6px; border-radius:14px; } .screw { width:7px; height:7px; top:10px; } .screw.left { left:13px; } .screw.right { right:13px; } .stat-chip { min-width:135px; } }
</style>
"""
CSS = CSS.replace("</style>", "".join(f".width-{percent}{{width:{percent}%}}" for percent in range(0, 101, 5)) + "</style>")


DETECTION_EXPLANATIONS = {
    "Resized": "Scales the photo to a consistent working width.",
    "Grayscale": "Keeps brightness details used to locate plate edges.",
    "Bilateral filter": "Reduces noise while preserving sharp edges.",
    "CLAHE": "Improves local contrast when the photo needs it.",
    "edges": "Highlights strong boundaries and character-like details.",
    "Morphological closing": "Joins nearby edges into larger regions.",
    "All contours": "Outlines candidate shapes across the image.",
    "Detected plate": "Shows the selected plate region and its box.",
}
OCR_EXPLANATIONS = {
    "upscaled": "Enlarges the crop so character details are easier to read.",
    "gray": "Converts the plate crop to brightness values.",
    "horizontal_crop": "Keeps the row where the large plate characters sit.",
    "blur_variance": "Estimates how much fine detail is lost to blur.",
    "deconvolved": "Attempts to sharpen blur when the crop needs it.",
    "binary": "Separates dark characters from the bright background.",
    "final": "Prepares the main cleaned image for text recognition.",
    "variant_": "A second cleanup variant gives OCR another view to compare.",
    "characters_": "Shows the character-focused version used by OCR.",
    "segments_": "Marks the character regions found in this version.",
    "winning_variant": "The cleanup variant that produced the selected reading.",
}


def inject_css():
    """Inject the complete app stylesheet once during the current app render."""
    st.markdown(CSS, unsafe_allow_html=True)


def render_header():
    """Render the gradient banner and product title."""
    st.markdown(
        '<header class="hero"><div class="hero-icon" aria-hidden="true">▰</div>'
        '<div><h1>Plate Reader</h1><p>Upload a vehicle photo. We find the plate and read it.</p></div></header>',
        unsafe_allow_html=True,
    )


def render_plate(text, uncertain_positions, style, confidence):
    """Render OCR text inside a plate-shaped, escaped HTML display."""
    clean = re.sub(r"\s+", "", str(text or "")).upper()
    display = clean or "- - -"
    groups = re.fullmatch(r"([A-Z?]{3})([0-9?]{3,4})", display)
    if groups:
        chunks = [(0, groups.group(1)), (3, groups.group(2))]
        spans = []
        for group_index, (offset, value) in enumerate(chunks):
            chars = []
            for local_index, char in enumerate(value):
                pos = offset + local_index
                safe = html.escape(char, quote=True)
                if char == "?" or pos in uncertain_positions:
                    chars.append(f'<span class="uncertain" title="Please verify">{safe}</span>')
                else:
                    chars.append(safe)
            spans.append(f'<span class="plate-group">{"".join(chars)}</span>')
        text_html = f'{spans[0]}<span class="plate-gap">&nbsp;</span>{spans[1]}'
    else:
        chars = []
        for pos, char in enumerate(display):
            safe = html.escape(char, quote=True)
            if char == "?" or pos in uncertain_positions:
                chars.append(f'<span class="uncertain" title="Please verify">{safe}</span>')
            else:
                chars.append(safe)
        text_html = "".join(chars)
    high_confidence = isinstance(confidence, (int, float)) and confidence >= 80
    variant = "green" if str(style).lower() == "green" and high_confidence else "white"
    st.markdown(
        f'<div class="plate-wrap"><div class="plate {variant}"><span class="screw left"></span>'
        f'<span class="screw right"></span><div class="plate-text">{text_html}</div>'
        '<div class="plate-caption">PILIPINAS</div></div></div>',
        unsafe_allow_html=True,
    )


def render_chips(ocr, winning_variant=""):
    """Render confidence, agreement, format, and OCR-variant status chips."""
    confidence = max(0.0, min(100.0, float(ocr.confidence or 0)))
    agreement = max(0.0, min(1.0, float(ocr.agreement or 0))) * 100
    def level(value):
        """Choose the requested green, amber, or red metric color class."""
        return "green" if value >= 80 else "amber" if value >= 50 else "red"
    confidence_width = int(round(confidence / 5) * 5)
    agreement_width = int(round(agreement / 5) * 5)
    fmt_class = "" if ocr.format_ok else " warn"
    fmt_label = "✓ Format OK" if ocr.format_ok else "! Check format"
    variant = html.escape(str(winning_variant or "not reported"), quote=True)
    st.markdown(
        '<div class="chip-row">'
        f'<div class="stat-chip"><strong>Confidence {confidence:.0f}%</strong><div class="mini-track"><div class="mini-fill width-{confidence_width} fill-{level(confidence)}"></div></div></div>'
        f'<div class="stat-chip"><strong>Agreement {agreement:.0f}%</strong><div class="mini-track"><div class="mini-fill width-{agreement_width} fill-{level(agreement)}"></div></div></div>'
        f'<div class="format-chip{fmt_class}">{fmt_label}</div><div class="variant-chip">Variant: {variant}</div></div>',
        unsafe_allow_html=True,
    )


def render_steps(detection, ocr, expanded=False):
    """Render all detection and OCR intermediate images in ordered tabs."""
    with st.expander("See every processing step", expanded=expanded):
        detection_tab, ocr_tab = st.tabs(["Detection", "OCR cleanup"])
        with detection_tab:
            _render_step_grid(detection.steps or {}, DETECTION_EXPLANATIONS)
        with ocr_tab:
            _render_step_grid((ocr.steps if ocr is not None else {}) or {}, OCR_EXPLANATIONS)


def _render_step_grid(steps, explanations):
    """Display ordered step images with a number, title, and short explanation."""
    if not steps:
        st.caption("No images are available for this stage.")
        return
    items = list(steps.items())
    for start in range(0, len(items), 2):
        cols = st.columns(2, gap="large")
        for column_index, (name, image) in enumerate(items[start:start + 2]):
            step_number = start + column_index + 1
            title = re.sub(r"^\d+\s+", "", name).replace("_", " ")
            lower = name.casefold()
            explanation = next((value for key, value in explanations.items() if key.casefold() in lower), "Shows an intermediate view produced by this processing stage.")
            with cols[column_index]:
                st.markdown(f'<div class="image-card"><div class="image-title"><span class="step-badge">{step_number}</span>{html.escape(title)}</div>', unsafe_allow_html=True)
                st.image(_to_rgb(image), use_container_width=True)
                st.caption(explanation)
                st.markdown('</div>', unsafe_allow_html=True)


def _to_rgb(image):
    """Convert a grayscale or OpenCV BGR step image for Streamlit."""
    if image is None or getattr(image, "ndim", 0) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
