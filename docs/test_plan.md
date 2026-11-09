# Test Plan and Results Template

**Rule:** every number in the final report must come from `tests/evaluate.py` run on your own images. Nothing in this file is a result. All results tables below are empty templates.

## 1. Purpose

Measure how well the system (a) finds the plate and (b) reads it, across realistic conditions, and record honestly where it fails.

Detection and recognition are reported separately. A system can find the plate correctly and still misread it, or return a region that is not a plate at all.

## 2. Test dataset

Take your own photos, or use photos you have permission to use. Plates identify people, so do not publish identifiable plates and do not copy other people's photos into public places.

Suggested minimum: **15 to 20 images**, more if possible. Aim for at least 3 per category.

| Category (`category` column) | What to photograph |
|---|---|
| `clear` | Plate facing the camera, good daylight, sharp |
| `angle` | Taken from the side or from above or below |
| `shadow` | Shadows across the plate, uneven light, backlit |
| `blurry` | Moving vehicle, out of focus, or far away / low resolution |
| `cluttered` | Busy background, stickers, text signs, grilles, many objects |
| `absent` | Plate missing, covered, or completely unreadable |

Tips: write the real plate number down at the time you take the photo, from the vehicle itself, so the expected value is certain. Use different vehicles and different times of day. Include both cars and motorcycles if you can, and mark them in `notes`.

## 3. Procedure

1. Put the photos in `tests/test_images/`.
2. Fill `tests/ground_truth.csv`, one row per image:

```
image_id,filename,category,expected_plate,notes
T01,photo1.jpg,clear,ABC1234,car
T02,photo2.jpg,blurry,XYZ567,motorcycle
T03,photo3.jpg,absent,,plate covered
```

   - `expected_plate`: the true plate. Spaces and hyphens are ignored. **Leave it empty** for the `absent` category.
   - Keep `image_id` unique.
3. Run:

```
python tests/evaluate.py
```

4. Open `results/eval/<image_id>/1_detected.png` for every image and decide whether the green box is on the real plate. Open `results/evaluation_results.csv` (Excel works) and fill the **`detection_ok`** column with `yes`, `no` or `partial`. Add notes in `observations` (reason for failure).
5. Recalculate the summary (your entries are kept if you re-run the evaluation too):

```
python tests/evaluate.py --summary-only
```

6. Copy the numbers from `results/evaluation_summary.txt` into section 5 below.

## 4. Metrics

| Metric | Definition |
|---|---|
| Region returned | The system gave a box. Automatic. Does not prove it is the plate. |
| Detection rate | Images where you marked `detection_ok = yes`, divided by images with a readable plate. |
| Exact match (raw) | The raw Tesseract text equals the expected plate. |
| Exact match (corrected) | The text after the pattern correction equals the expected plate. |
| Character accuracy | 1 - edit distance / length of the longer text, averaged over images. Counts wrong, missing and extra characters. |
| Wrong but not flagged | Wrong reads shown without any warning. The most harmful kind. |
| Correctly rejected | For `absent` images: nothing was found, or the result was flagged as unreliable. |
| Correction broke a read | The raw text was right but the corrected text is wrong. |

"Exact match" is strict: one wrong character counts as a failure. Character accuracy shows how close the failures were.

## 5. Results template (fill from your run)

### 5.1 Summary

| Item | Value (from evaluation_summary.txt) |
|---|---|
| Total images / with a readable plate / absent | |
| Region returned | |
| Detection correct (manual) | |
| Exact match, raw | |
| Exact match, corrected | |
| Mean character accuracy, raw | |
| Mean character accuracy, corrected | |
| Wrong reads not flagged | |
| Absent-plate images correctly rejected | |

### 5.2 By category

| Category | Images | Detection ok | Exact match (corrected) | Mean char accuracy |
|---|---|---|---|---|
| clear | | | | |
| angle | | | | |
| shadow | | | | |
| blurry | | | | |
| cluttered | | | | |
| absent | | n/a | n/a | n/a |

### 5.3 Per-image table (copy from `evaluation_results.csv`)

| Test ID | Category | Expected plate | Detected / cropped plate (image) | Raw OCR | Corrected OCR | Correct? | Observations / failure reason |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### 5.4 Failure analysis

For each failed image, state the **stage that failed** (detection or OCR), the **visible cause** (for example blur, glare, angle, clutter) and what could improve it. Group repeated causes. Failures belong in the report; a report with no failures suggests the test set was too easy.

### 5.5 Conclusions to write after the results

- Which conditions work best and worst, with the counts.
- Whether the confidence/agreement values were useful (did low values line up with wrong reads?).
- Whether the pattern correction helped or hurt (compare raw and corrected).
- The limits of the test: number of images, one phone camera, vehicles from one area, and so on.

## 6. Limits of this test plan

- A small sample (15 to 20 images) gives rough numbers. Report counts, not just percentages.
- Detection correctness is judged by eye, not by measured box overlap.
- The settings of the algorithm were adjusted while building it on a few sample photos. Test images should be **new photos not used while tuning**, otherwise the results will look better than they are.
