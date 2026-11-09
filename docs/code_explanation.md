# Code Explanation (for beginners)

This explains every file in simple terms, what tools it uses, and why it exists. The numbers quoted here match the code. If you change the code, update them.

## The big picture

The program reads a plate the way a person would: clean up the photo, find the plate, read the characters. Each step is a small function so it can be explained on its own.

```
app.py  ->  src/pipeline.py  ->  src/detection.py  (find the plate)
                              ->  src/ocr.py        (read the plate)
```

`detection.py` uses `preprocessing.py` and `characters.py`. `ocr.py` also uses `characters.py`. `utils.py` handles files.

## A few words used below

- **Image as numbers:** a photo is a grid of pixels. A color pixel is three numbers (blue, green, red in OpenCV). A gray pixel is one number from 0 (black) to 255 (white). NumPy holds these grids.
- **Edge:** a place where brightness changes sharply, such as the border of a letter.
- **Threshold:** a rule that turns gray into pure black or white at a chosen brightness.
- **Contour:** the outline of a connected white shape in a black-and-white image.
- **OCR:** software that turns a picture of text into text.

---

## requirements.txt

The shopping list of Python packages (`opencv-python`, `numpy`, `Pillow`, `pytesseract`, `streamlit`). `pip install -r requirements.txt` installs them all. Pillow is listed but no code uses it yet.

## src/utils.py (files)

Tools: `os`, NumPy, OpenCV.

- `load_image(path)`: checks that the file is JPG, JPEG or PNG and exists, then decodes it. Raises a clear error if it cannot be read. It reads the bytes with NumPy first so folder names with special characters (common in Windows) still work.
- `decode_image_bytes(bytes)`: the same decoding for files uploaded in the app.
- `save_image(path, image)`: saves a result image and creates the folder if needed.

## src/preprocessing.py (cleaning the photo)

Tool: OpenCV. One small function per technique.

| Function | What it does | Why |
|---|---|---|
| `resize_to_width` | Resizes to 800 px wide, keeping the shape | All later settings are tuned for one size, so every photo must be the same scale |
| `to_grayscale` | Color to gray | Plates are found by brightness changes; color is not needed and one channel is simpler |
| `denoise` | Bilateral filter (d=11, sigma 17/17) | Removes grain but keeps edges sharp. A normal blur would soften the character edges we need |
| `enhance_contrast_if_needed` | CLAHE (clip 2.0, 8x8 tiles), only when the gray values' standard deviation is below 40 | Helps dull, washed-out photos. It is skipped on photos that already have good contrast, because boosting those adds noise |
| `detect_edges` | Canny with thresholds 30 and 200 | Characters create many close edges, so the plate shows up as a dense cluster |

Techniques not used and why: Gaussian blur on the full image (the bilateral filter keeps edges better), a global threshold for detection (lighting varies too much across a whole photo).

## src/detection.py (finding the plate)

Tools: OpenCV, NumPy.

1. Runs the preprocessing steps above.
2. **Morphological closing** with a flat rectangle (width 9, 15 or 25 px, height 5 px). This joins nearby edges so a row of characters becomes one solid block. The right width depends on character spacing and photo distance, so three widths are tried.
3. **Contours:** finds the outline of every block.
4. **Filtering by shape and size** (`_filter_candidates`): keeps blocks with a width/height ratio between 1.5 and 5.5, covering between 0.2% and 60% of the image, and filling at least 30% of their bounding box.
5. **Scoring** (`_score`): each surviving block gets a score from shape fill (25%), how close its ratio is to 3:1 (20%), position near the middle (10%), size (20%), and whether it contains a row of characters (25%).
6. The best block gets a small padding, a green box is drawn on a copy of the image, and the plate is cropped.

Returns a `DetectionResult` that holds the box, the crop, the annotated image, a message and every intermediate image (used by the app's step viewer).

## src/characters.py (is there a row of characters?)

Tool: OpenCV.

`find_char_row` looks for 3 or more shapes of similar height lined up horizontally, which is what plate characters look like. It tries both black-on-white and white-on-black. It is used twice: in detection (windows and body panels do not contain such a row, so they lose) and in OCR (to crop tightly around the characters).

## src/ocr.py (reading the plate)

Tools: OpenCV, NumPy, **Tesseract through pytesseract**, `difflib`, `re`.

Cleanup (`prepare_plate_for_ocr`), all our own image processing:
1. Enlarge the crop so it is about 200 px tall (Tesseract reads small text poorly).
2. Gray + light Gaussian blur (3x3).
3. Find the character row and crop to it vertically (the dark handle or bumper above a plate would otherwise distort the threshold).
4. **Otsu threshold** to black text on white. It decides the threshold automatically and flips the picture if the text is the lighter color.
5. Erase blobs that cannot be characters (too small, too tall, too wide, or very thin), measured against the character height we found.
6. Add a white border.

Reading (`read_plate`):
- **Pass A:** Tesseract restricted to `A-Z` and `0-9`, single text line mode (`--psm 7`). This gives the text.
- **Pass B:** Tesseract unrestricted. Used to get a confidence number, and to measure **agreement** between the two reads (`difflib.SequenceMatcher`). With a character restriction, Tesseract's own confidence was not meaningful in our tests, so it is taken from pass B.
- **Warnings** are raised when fewer than 5 characters are read, when agreement is below 0.6 (a chosen rule of thumb, not a proven threshold), or when the text does not match the pattern.
- **Pattern correction** (`correct_with_pattern`): for 6 or 7 characters, assumes 3 letters followed by digits and swaps look-alikes by position (for example `O` to `0` in a digit position). A `G` in a digit position is marked `?` instead of guessed, because it could be a 0 or a 6. This assumes a car format and can be wrong for other formats. The raw text is always kept.

## src/pipeline.py (connecting the parts)

`run_pipeline(image)` checks the image is big enough (at least 60 px on the short side), runs detection, then OCR if a region was found, and returns both results. The app only calls this one function.

## app.py (the interface)

Tool: Streamlit. Shows the title, an explanation of the steps, the upload control, the original preview, the **Recognize Plate** button, the processed image with the box, the cropped plate, the recognized text (with raw text, confidence and agreement), warnings, and an expandable viewer of every intermediate image. Errors are displayed as messages instead of crashing.

## run_stage2.py and run_stage3.py (test scripts)

Terminal tools for developers. `run_stage2.py` runs detection only and saves every step image. `run_stage3.py` runs detection and OCR and prints the text. They are not the final application; the app is.

## tests/evaluate.py (measuring honestly)

Runs the real pipeline on every image in `tests/ground_truth.csv` and calculates:
- whether a region was returned (automatic),
- whether the detection was correct (**filled in by you** after looking at the image),
- exact plate match for raw and corrected text,
- **character accuracy** = 1 - (edit distance / length of the longer text). The edit distance is the smallest number of single-character changes (insert, delete, replace) needed to turn one text into the other. A `?` never counts as a match.
- whether wrong reads were flagged as unreliable.

---

## Questions you may be asked, with honest answers

**Which part did you build and which part is external?**
Detection and plate cleanup are our own OpenCV processing. Tesseract is an external OCR engine that reads the cleaned image.

**Why not deep learning?**
The course focuses on digital image processing, and a classical method can be explained step by step. A pretrained detector is listed as a future improvement.

**Why a bilateral filter?**
It smooths noise but keeps edges, which the next step needs.

**Why CLAHE only sometimes?**
It helps low-contrast images but adds noise to images that are already fine.

**Why does it fail on blurry plates?**
Blur weakens the edges, so the characters no longer join into one block, and Tesseract has less detail to read.

**How accurate is it?**
Only the evaluation results can answer that. Report the numbers from `tests/evaluate.py` with the number of images, and do not quote any other figure.
