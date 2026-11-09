# Implementation Plan: Automatic Number Plate Recognition (ANPR)

## 1. Problem statement

Vehicle license plates must often be read manually from photos, for example for parking records, gate logs or traffic documentation. Manual reading is slow and error-prone. The problem is to build a program that takes a photo of a vehicle, finds the license plate, and extracts its characters automatically, using digital image processing.

## 2. Project objectives

1. Accept vehicle images (JPG, JPEG, PNG) and reject invalid files with clear messages.
2. Apply appropriate preprocessing and explain why each technique is used.
3. Detect the plate with a traditional image-processing algorithm and show a bounding box and cropped plate.
4. Recognize the characters with OCR, preceded by plate-specific cleanup, and report confidence and warnings.
5. Provide a local, user-friendly graphical interface.
6. Test the system on varied images and report detection and recognition results separately and honestly.
7. Document the system, its limitations and possible improvements.

## 3. Scope and limitations

**In scope**
- Still images of vehicles with one visible plate.
- Philippine plates (the test target), with look-alike correction written for the car format.
- A local application running on Windows through an IDE.
- Classical image processing as the main, explainable component.

**Out of scope**
- Video or real-time recognition.
- Multiple plates per image, and plates on two text lines.
- Cloud services, paid APIs, GPU use, and model training.
- Legal verification of plate numbers or any database lookup.

**Known limitations**
- Blur, low resolution, glare, strong shadow and steep angles reduce quality; a blurred plate in our own tests was not found correctly.
- Cluttered backgrounds can produce false plate regions.
- Tesseract is designed for document text and confuses similar characters on small plates (0/O/G, 1/I, 8/B, 5/S).
- The pattern correction assumes 3 letters + digits (cars). Motorcycle plates use other layouts, so correction can alter a correct read. Raw text is always shown.
- Confidence and agreement are heuristic indicators, not measured accuracy.

## 4. System requirements

**Hardware:** an ordinary laptop. No GPU or special hardware.

**Software**
- Windows 10 or 11, with Visual Studio Code or Positron.
- Python (developed on 3.14; 3.10 or newer is expected to work, but only 3.14 was used).
- Python packages: opencv-python, numpy, Pillow, pytesseract, streamlit (versions seen in development are in `docs/installation.md`).
- Tesseract OCR installed separately (Windows installer from UB Mannheim).
- A web browser, because Streamlit shows the interface locally.

**Input:** JPG, JPEG or PNG images, at least 60 px on the shortest side.

## 5. Software architecture and workflow

```
            +------------------------+
 Upload --> |  app.py  (Streamlit)   |  validation, preview, button, display
            +-----------+------------+
                        |
                +-------v--------+
                | src/pipeline.py |  run_pipeline(image)
                +----+------+-----+
                     |      |
        +------------v-+  +-v-------------+
        | detection.py |  |    ocr.py     |
        |  (find plate)|  | (read plate)  |
        +----+----+----+  +-------+-------+
             |    |               |
   preprocessing.py  characters.py  +--> Tesseract (external)
             |
          utils.py (load / save images)
```

**Workflow**
1. Load and validate the image.
2. Resize to 800 px wide.
3. Grayscale, bilateral filter, CLAHE only if the contrast is low.
4. Canny edge detection.
5. Morphological closing (three widths), contour detection.
6. Filter candidate regions by shape and size; score them (shape, size, position, character-row evidence).
7. Draw the box and crop the best region.
8. Clean the crop: enlarge, keep the character row, Otsu threshold, remove non-character blobs.
9. Run Tesseract twice (restricted text pass; unrestricted reliability pass).
10. Apply warnings and the pattern check; display results.

## 6. Selected algorithms and justification

| Step | Technique | Why this one |
|---|---|---|
| Resize | Fixed width 800 px | Makes sizes and kernels consistent across photos |
| Grayscale | Luminance conversion | Plates are located by brightness structure; fewer data |
| Noise reduction | Bilateral filter | Smooths noise while preserving edges, unlike a plain Gaussian blur |
| Contrast | CLAHE, conditional | Helps low-contrast images locally; skipped when unnecessary to avoid amplifying noise |
| Edges | Canny | Standard, well-understood edge detector; characters give dense edges |
| Joining characters | Morphological closing with a wide flat kernel | Merges close edges into a plate-shaped block; several widths because spacing varies |
| Localization | Contours + aspect ratio, area and fill filters | Plates are rectangles of a known proportion range |
| Ranking | Weighted score including a character-row check | Removes windows and panels that pass the shape filters |
| Binarization | Otsu's method, automatic polarity | Chooses the threshold automatically and handles light-on-dark or dark-on-light plates |
| Cleanup | Connected-component filtering | Removes logos, frame lines and specks that are not characters |
| OCR | Tesseract (`--psm 7`, A-Z/0-9) | Free, local, configurable, meets the brief |
| Reliability | Second unrestricted pass + agreement | A rough indicator; Tesseract's own confidence was not meaningful with a character restriction |

**Beyond basic course lessons** (as the brief asks): conditional contrast enhancement, multi-scale morphological closing, a character-row consistency check used for both ranking and cropping, connected-component cleanup, automatic polarity handling, and an honest uncertainty mechanism (agreement, warnings and `?` for ambiguous characters).

**Not used:** deep-learning detectors or OCR. They could improve results, but they would be an external component and the classical method is the main educational part (see Future enhancements).

## 7. Development phases

| Phase | Work | Output | Check |
|---|---|---|---|
| 1. Study and design | Choose stack and algorithm, folder structure, questions | Stage 1 plan | Approach confirmed |
| 2. Preprocessing and detection | `preprocessing.py`, `detection.py`, `utils.py`, `run_stage2.py` | Every step saved as an image | Box lands on the plate in sample photos |
| 3. OCR | `ocr.py`, `characters.py`, `run_stage3.py` | Text, confidence, warnings | Text printed for sample photos |
| 4. Interface | `pipeline.py`, `app.py` | Working Streamlit app | Upload, recognize, view results, errors handled |
| 5. Testing and documentation | `tests/evaluate.py`, test plan, README, references, demo guide | Real results from your photos | Numbers computed by the script |

Detection and OCR parameters were adjusted during phases 2 and 3 using a small number of sample photos. That is why the final test images must be new ones.

## 8. Testing methodology

- A dataset of at least 15 to 20 own photos across six categories: clear, angle, shadow, blurry, cluttered, absent.
- A `ground_truth.csv` with the true plate for each readable image.
- `tests/evaluate.py` runs the real pipeline, saves images, and computes metrics.
- Detection correctness is judged manually per image; recognition is computed automatically.
- Failures are recorded with their stage and cause. Full procedure: `docs/test_plan.md`.

## 9. Evaluation metrics

- Detection rate (manual `detection_ok = yes`).
- Exact plate match (raw and corrected).
- Character accuracy (1 - edit distance / longer length).
- Wrong reads that were not flagged.
- Correct rejection of images without a readable plate.
- Effect of the correction (raw vs corrected).

## 10. Expected outputs

- Working application (`app.py`).
- Source code with comments and a beginner explanation.
- Test dataset, ground truth, per-image results CSV, summary.
- This plan, README, test plan with results, references, demo guide, troubleshooting guide.

## 11. Risks and possible solutions

| Risk | Impact | Solution |
|---|---|---|
| Blurred or low-resolution photos | Detection or OCR fails | Take sharper photos for the main demo; keep blurred ones as documented failures; future: deblurring or a learned detector |
| Cluttered backgrounds | Wrong region chosen | Character-row check; closer photos; future: learned detector |
| Tesseract confuses similar characters | Wrong characters | Plate-specific cleanup; `?` for ambiguity; manual check message; future: plate-specific OCR |
| Pattern correction wrong for motorcycle plates | A correct read is altered | Show raw text beside corrected text; evaluate both; future: format-specific rules |
| Tesseract missing or in a different folder | OCR cannot run | Automatic search of default folders, `TESSERACT_CMD` override, clear message |
| Parameters tuned on a few photos | Results look better than reality | Test on new photos only; report limitations |
| Photos of other people's plates | Privacy issue | Own or permitted photos; do not publish identifiable plates |
| Different Tesseract versions | Results differ across computers | Record the version; demo on the same machine |
| Time pressure | Incomplete testing | Timeline below; test data collected early |

## 12. Estimated timeline

These durations are estimates. Adjust them to your actual deadline.

| Phase | Estimated time |
|---|---|
| 1. Study and design | 1 to 2 days |
| 2. Preprocessing and detection | 3 to 4 days |
| 3. OCR | 3 to 4 days |
| 4. Interface | 2 days |
| 5. Photo collection (can run in parallel from the start) | 2 to 3 days |
| 5. Testing and results | 2 to 3 days |
| Documentation and demo preparation | 2 to 3 days |

## 13. Future enhancements

- A pretrained plate detector (for example a YOLO-based model) for cluttered scenes. It would be an external dependency and would have to be described separately from our own algorithm.
- A stronger OCR engine (for example EasyOCR) compared against Tesseract on the same test set; it adds heavy dependencies.
- Perspective correction for angled plates.
- Deblurring or super-resolution for blurred and small plates.
- Support for several plate formats (motorcycle layouts, two-line plates) with format-specific validation based on current LTO rules.
- Character segmentation with a trained character classifier.
- Measuring detection with annotated boxes (overlap with the true box) instead of a manual check.
- Video input and multiple plates per image.
