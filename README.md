# Automatic Number Plate Recognition (ANPR)

A Digital Image Processing project (SA1 Learning Evidence). The user uploads a photo of a vehicle. The system finds the license plate with image processing, reads the characters with OCR, and shows the result with the intermediate images.

> **No accuracy figure is claimed in this README.** Results depend on the test photos. Real numbers are produced by `tests/evaluate.py` from your own test images (see `docs/test_plan.md`).

## What it does

1. Accepts JPG, JPEG or PNG images and rejects invalid or unreadable files.
2. Preprocesses the image (resize, grayscale, bilateral filtering, CLAHE, adaptive Canny/Sobel/morphology edge maps and scale-aware closing).
3. Locates the plate with Canny/Sobel contours, bright low-saturation quadrilaterals, and a Hough-line fallback. It checks for character evidence and rectifies slanted plates with a perspective homography.
4. Reads up to three scored candidates with multiple classical threshold variants and Tesseract page segmentation modes.
5. Shows everything in a Streamlit web interface that runs locally.

## Technology

| Tool | Used for |
|---|---|
| Python | Everything |
| OpenCV (`opencv-python`) | Image processing: resize, filtering, edges, morphology, contours, thresholding |
| NumPy | Images as arrays |
| Tesseract OCR + `pytesseract` | Per-character OCR votes (an external engine) |
| scikit-learn + joblib | Optional generated-data HOG/SVM glyph classifier |
| Streamlit | The graphical interface |
| Pillow | Listed in `requirements.txt`; not currently imported by the code |

No deep-learning model, cloud service or paid API is used. Detection and cleanup use classical image processing. Tesseract and the optional HOG/RBF-SVM classifier vote on individual characters; the classifier is trained from generated glyphs and local sample crops.

## Folder structure

```
anpr_project/
├── app.py                  Streamlit interface (run this)
├── requirements.txt
├── run_stage2.py           Test script: detection only, saves every step as an image
├── run_stage3.py           Test script: detection + OCR from the terminal
├── src/
│   ├── utils.py            Load and validate images, save images
│   ├── preprocessing.py    Resize, grayscale, filtering, CLAHE, Canny
│   ├── detection.py        Plate localization
│   ├── characters.py       Finds a row of similar-height characters (helper)
│   ├── ocr.py              Plate cleanup, Tesseract, confidence, pattern check
│   └── pipeline.py         Connects detection and OCR for the app
├── samples/                Your test photos
├── tests/
│   ├── ground_truth.csv    You fill in: image_id, filename, category, expected_plate
│   ├── test_images/        Your test photos for evaluation
│   └── evaluate.py         Computes the results from real runs
├── docs/                   Installation, code explanation, plans, references, demo, troubleshooting
└── results/                Generated output (never edited by hand)
```

## Quick start (Windows)

Full steps are in `docs/installation.md`. In short:

```
pip install -r requirements.txt
python -m streamlit run app.py
```

Tesseract OCR must be installed separately (see `docs/installation.md`).

## User interface

The Streamlit interface uses a Philippine plate-inspired green theme, with an upload preview, plate-style OCR result, confidence and format indicators, annotated and cropped images, and expandable detection/OCR processing steps. Use the sidebar to switch the plate display style or show the processing gallery. The result can be downloaded as an annotated JPG and a small JSON file.

Screenshots:

![Plate Reader upload and preview](docs/images/plate-reader-upload.png)

![Plate Reader result and processing steps](docs/images/plate-reader-result.png)

## How it works

```
Photo -> resize -> grayscale -> bilateral filter -> CLAHE (only if low contrast)
      -> auto-Canny + Sobel-X + blackhat/tophat -> scale-aware morphological closing
      -> filter by shape and size -> score (shape, size, position, character row)
      -> bright quadrilateral + Hough candidates -> character and color validation
      -> homography rectification to 440x200 -> residual shear search by projection contrast
      -> candidate score + OCR format/confidence ranking
      -> illumination normalization -> blur estimate / conditional deconvolution -> upscale
      -> Otsu, adaptive Gaussian, Sauvola, CLAHE/Otsu and sharpened/Otsu variants
      -> Tesseract (position whitelists) + optional HOG/SVM character vote
      -> pattern check -> plate text + warnings
```

## Train the optional character classifier

The training script renders A-Z and 0-9 from installed bold fonts, adds perspective, shear, rotation, blur, noise, contrast and slashed-zero variants, and extracts labeled glyphs from the local car samples when segmentation succeeds. No training images are downloaded.

```powershell
pip install scikit-learn joblib
python train_char_classifier.py
```

Build the labeled glyph dataset from local samples and train the source-held-out model:

```powershell
python tools/build_dataset.py
python train_char_classifier.py
```

The builder saves `data/chars/<label>/*.png` as 32x48 grayscale glyphs and supports optional corrections in `data/user_corrections.csv` (`filename,plate`). Training holds out complete plate/font sources, prints a per-class report and confusion matrix, then writes `models/char_clf.joblib`. If scikit-learn/joblib or the model is unavailable, inference uses Tesseract alone. `scikit-image` is optional; adaptive Gaussian thresholding is used when it is absent.

Why each technique was chosen is explained in `docs/implementation_plan.md` and `docs/code_explanation.md`.

## Limitations (honest list)

- **Blur, low resolution, glare, heavy shadow and steep angles** reduce detection and OCR quality. A blurred plate in our own tests was not found correctly.
- **Cluttered backgrounds**: text, grilles or stickers can look like a plate. The character-row check reduces, but does not remove, this problem.
- **Tesseract is built for documents, not plates.** Similar characters (0/O/G, 1/I, 8/B, 5/S) are confused on small or blurry plates.
- **Car-format correction is conditional** (3 letters + 3-4 digits; current Philippine car plates are 3 letters + 4 digits). Other alphanumeric layouts, including motorcycle and registration plates, are retained without position-based correction and are warned when they do not match the car pattern.
- **Single plate, single line**: the system returns one plate per image and one text line.
- The confidence and agreement values are rough indicators chosen heuristically, not measured accuracy.

## Testing

`tests/evaluate.py` runs the real pipeline on `samples/`, prints expected and predicted text per image, and reports exact accuracy. Filename labels are used when available. The supplied `car3.jpg` visibly reads `HVU253`; the requested `car3_CAX3200.jpg` file is not present in this checkout. Run `python tests/evaluate.py --synthetic-warps` to report OCR accuracy on 50 random perspective, shear and rotation warps of the local car3 plate crop.

## Documentation

| File | Contents |
|---|---|
| `docs/installation.md` | Windows, VS Code / Positron, Tesseract setup |
| `docs/code_explanation.md` | Every file explained for beginners |
| `docs/implementation_plan.md` | Problem, objectives, scope, architecture, algorithms, phases, risks, timeline |
| `docs/test_plan.md` | Test categories, procedure, results template |
| `docs/references.md` | Verified references with links |
| `docs/demo_guide.md` | How to present the working application |
| `docs/troubleshooting.md` | Common installation and runtime errors |

## Privacy and image use

License plates can identify people. Use photos you took yourself or have permission to use, avoid publishing identifiable plates in reports, and do not store other people's photos in public repositories.
