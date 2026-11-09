"""Stage 5 - Evaluation script.

Runs the real pipeline on every image listed in tests/ground_truth.csv and
computes the results FROM THOSE RUNS. Nothing here is typed in by hand.

Usage (from the anpr_project folder):
    python tests/evaluate.py                 run all images, then summarize
    python tests/evaluate.py --summary-only  re-summarize after you filled
                                             in the detection_ok column

Detection vs recognition are kept separate:
  * "a region was returned" is automatic, but it does NOT prove the region
    is the real plate.
  * detection_ok is filled in BY YOU (yes / no / partial) after looking at
    results/eval/<image_id>/1_detected.png. It is the real detection result.
  * Recognition (exact match, character accuracy) is computed against the
    expected_plate you wrote in ground_truth.csv.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from src.pipeline import run_pipeline  # noqa: E402
from src.utils import load_image, save_image  # noqa: E402

CATEGORIES = ["clear", "angle", "shadow", "blurry", "cluttered", "absent"]

RESULT_COLUMNS = [
    "image_id", "filename", "category", "expected_plate",
    "region_returned", "detection_ok",           # detection_ok is filled by hand
    "raw_ocr", "corrected_ocr", "ocr_flagged", "confidence", "agreement",
    "exact_raw", "exact_corrected", "char_acc_raw", "char_acc_corrected",
    "outcome", "message", "observations",        # observations is filled by hand
]


# --------------------------------------------------------------- metrics
def normalize(text):
    """Upper-case and drop spaces/hyphens. '?' is kept: it never matches."""
    return "".join(ch for ch in (text or "").upper() if ch not in " -")


def edit_distance(a, b):
    """Levenshtein distance: fewest single-character insertions, deletions
    or substitutions needed to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(expected, read):
    """1 - edit_distance / length of the longer string. Range 0..1.
    Extra or missing characters are penalised, not just wrong ones."""
    expected, read = normalize(expected), normalize(read)
    longest = max(len(expected), len(read))
    if longest == 0:
        return 1.0
    return 1.0 - edit_distance(expected, read) / longest


# ------------------------------------------------------------ running
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULT_COLUMNS})


def evaluate_one(item, images_dir, out_dir):
    """Run the pipeline on one ground-truth row; return a result dict."""
    row = OrderedDict(
        image_id=item["image_id"], filename=item["filename"],
        category=item.get("category", ""),
        expected_plate=normalize(item.get("expected_plate", "")),
        observations="",
    )
    path = os.path.join(images_dir, item["filename"])
    try:
        image = load_image(path)
        result = run_pipeline(image)
    except ValueError as err:
        row.update(region_returned="", outcome="error", message=str(err))
        return row

    det, ocr = result.detection, result.ocr
    img_dir = os.path.join(out_dir, "eval", item["image_id"])
    save_image(os.path.join(img_dir, "1_detected.png"), det.annotated)
    row["region_returned"] = "yes" if det.found else "no"

    if not det.found or ocr is None:
        raw = corrected = ""
        flagged = True
        row["message"] = det.message
    else:
        save_image(os.path.join(img_dir, "2_plate_crop.png"), det.plate_crop)
        if "ocr5_final" in ocr.steps:
            save_image(os.path.join(img_dir, "3_ocr_input.png"), ocr.steps["ocr5_final"])
        raw, corrected = ocr.raw_text, ocr.text
        flagged = (not ocr.ok) or ocr.message != "OCR completed."
        row["message"] = ocr.message
        row["confidence"] = f"{ocr.confidence:.0f}"
        row["agreement"] = f"{ocr.agreement:.2f}"

    row["raw_ocr"], row["corrected_ocr"] = raw, corrected
    row["ocr_flagged"] = "yes" if flagged else "no"

    expected = row["expected_plate"]
    if expected:                                   # a readable plate exists
        row["exact_raw"] = int(normalize(raw) == expected)
        row["exact_corrected"] = int(normalize(corrected) == expected)
        row["char_acc_raw"] = f"{char_accuracy(expected, raw):.3f}"
        row["char_acc_corrected"] = f"{char_accuracy(expected, corrected):.3f}"
        if row["exact_corrected"]:
            row["outcome"] = "correct"
        elif row["exact_raw"]:
            row["outcome"] = "raw correct, correction broke it"
        else:
            row["outcome"] = "wrong (flagged)" if flagged else "wrong (NOT flagged)"
    else:                                          # no readable plate expected
        row["outcome"] = "correctly rejected" if flagged else "false read (NOT flagged)"
    return row


def run_all(truth_path, images_dir, out_dir, results_path):
    truth = read_csv(truth_path)
    if not truth:
        print("ground_truth.csv has no rows. Add your images first (see docs/test_plan.md).")
        return False

    # Keep manual entries from an earlier run so re-running does not erase them
    old = {r["image_id"]: r for r in read_csv(results_path)} if os.path.isfile(results_path) else {}
    rows = []
    for item in truth:
        print(f"Running {item['image_id']} ({item['filename']}) ...")
        row = evaluate_one(item, images_dir, out_dir)
        prev = old.get(item["image_id"], {})
        row["detection_ok"] = prev.get("detection_ok", "")
        row["observations"] = prev.get("observations", "")
        rows.append(row)
    write_csv(results_path, rows)
    print(f"\nPer-image results written to {results_path}")
    return True


# ---------------------------------------------------------- summarizing
def _pct(num, den):
    return f"{100.0 * num / den:.1f}% ({num}/{den})" if den else "n/a (no images)"


def _mean(values):
    return f"{100.0 * sum(values) / len(values):.1f}%" if values else "n/a"


def summarize(results_path):
    rows = read_csv(results_path)
    lines = []
    add = lines.append
    valid = [r for r in rows if r.get("outcome") != "error"]
    errors = [r for r in rows if r.get("outcome") == "error"]
    with_plate = [r for r in valid if r["expected_plate"]]
    absent = [r for r in valid if not r["expected_plate"]]

    add("=== ANPR EVALUATION SUMMARY (computed from the actual runs) ===")
    add(f"Images listed: {len(rows)}  |  with a readable plate: {len(with_plate)}"
        f"  |  no/unreadable plate: {len(absent)}  |  failed to load: {len(errors)}")
    add("")

    # ---- detection
    add("--- DETECTION ---")
    returned = sum(r["region_returned"] == "yes" for r in with_plate)
    add(f"Region returned (automatic; does NOT prove it is the plate): "
        f"{_pct(returned, len(with_plate))}")
    judged = [r for r in with_plate if r["detection_ok"].strip().lower() in ("yes", "no", "partial")]
    if judged:
        yes = sum(r["detection_ok"].strip().lower() == "yes" for r in judged)
        part = sum(r["detection_ok"].strip().lower() == "partial" for r in judged)
        add(f"Detection correct (YOUR manual check): {_pct(yes, len(judged))}"
            f"  | partial: {part}  | no: {len(judged) - yes - part}")
    add(f"detection_ok still to fill in: {len(with_plate) - len(judged)} of {len(with_plate)}")
    add("")

    # ---- recognition
    add("--- RECOGNITION (images with a readable plate; compared to your expected_plate) ---")
    if with_plate:
        ex_raw = sum(int(r["exact_raw"]) for r in with_plate)
        ex_cor = sum(int(r["exact_corrected"]) for r in with_plate)
        ca_raw = [float(r["char_acc_raw"]) for r in with_plate]
        ca_cor = [float(r["char_acc_corrected"]) for r in with_plate]
        add(f"Exact plate match, raw OCR text:        {_pct(ex_raw, len(with_plate))}")
        add(f"Exact plate match, after correction:    {_pct(ex_cor, len(with_plate))}")
        add(f"Mean character accuracy, raw:           {_mean(ca_raw)}")
        add(f"Mean character accuracy, corrected:     {_mean(ca_cor)}")
        broke = sum(r["outcome"].startswith("raw correct") for r in with_plate)
        add(f"Times the correction broke a correct raw read: {broke}")
        wrong = [r for r in with_plate if r["outcome"].startswith("wrong")]
        unflagged = sum(r["outcome"] == "wrong (NOT flagged)" for r in wrong)
        add(f"Wrong reads: {len(wrong)}  | of these, NOT flagged as unreliable: {unflagged}")

        ok_det = [r for r in judged if r["detection_ok"].strip().lower() == "yes"]
        if ok_det:
            add("")
            add(f"Recognition when detection was correct ({len(ok_det)} images):")
            add(f"  Exact match, corrected: {_pct(sum(int(r['exact_corrected']) for r in ok_det), len(ok_det))}")
            add(f"  Mean character accuracy, corrected: {_mean([float(r['char_acc_corrected']) for r in ok_det])}")
    else:
        add("No images with an expected plate yet.")
    add("")

    # ---- absent plates
    add("--- IMAGES WITH NO / UNREADABLE PLATE ---")
    if absent:
        rej = sum(r["outcome"] == "correctly rejected" for r in absent)
        add(f"Correctly rejected or flagged as unreliable: {_pct(rej, len(absent))}")
        add(f"False reads presented without a warning: {len(absent) - rej}")
    else:
        add("None listed.")
    add("")

    # ---- per category
    add("--- BY CATEGORY (exact match after correction) ---")
    for cat in CATEGORIES:
        sub = [r for r in with_plate if r["category"] == cat]
        if sub:
            ok = sum(int(r["exact_corrected"]) for r in sub)
            add(f"{cat:<10} {_pct(ok, len(sub)):<16} mean char acc {_mean([float(r['char_acc_corrected']) for r in sub])}")
    if absent:
        add(f"{'absent':<10} rejected {sum(r['outcome'] == 'correctly rejected' for r in absent)}/{len(absent)}")
    if errors:
        add("")
        add("Failed to load: " + ", ".join(f"{r['image_id']} ({r['message']})" for r in errors))
    add("")
    add("Small samples give rough numbers. Report the counts, not just percentages.")

    text = "\n".join(lines)
    print("\n" + text)
    summary_path = os.path.join(os.path.dirname(results_path), "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the ANPR pipeline.")
    ap.add_argument("--truth", default=os.path.join(ROOT, "tests", "ground_truth.csv"))
    ap.add_argument("--images", default=os.path.join(ROOT, "samples"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--summary-only", action="store_true",
                    help="skip running; re-summarize the existing results CSV")
    args = ap.parse_args()

    results_path = os.path.join(args.out, "evaluation_results.csv")
    if not args.summary_only:
        if not os.path.isfile(args.truth):
            print(f"Cannot find {args.truth}")
            return
        truth = read_csv(args.truth)
        if not truth:
            # Filename convention: plate_<EXPECTED>.jpg; car1 is the supplied known case.
            truth = []
            known = {
                "car1": "NGJ391",
                "car2": "AAO8074",
                "lto_plate_number": "AHA8208",
                "car5": "PYX186",
            }
            for filename in sorted(os.listdir(args.images)):
                stem, ext = os.path.splitext(filename)
                if ext.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                expected = known.get(stem.lower(), "")
                if not expected and "_" in stem:
                    expected = stem.rsplit("_", 1)[1]
                truth.append({"image_id": stem, "filename": filename,
                              "category": "", "expected_plate": expected})
            print("Using sample images; expected text comes from the filename or known cases.")
            temp_truth = os.path.join(args.out, "sample_ground_truth.csv")
            os.makedirs(args.out, exist_ok=True)
            with open(temp_truth, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_id","filename","category","expected_plate"])
                writer.writeheader(); writer.writerows(truth)
            args.truth = temp_truth
        if not run_all(args.truth, args.images, args.out, results_path):
            return
    if not os.path.isfile(results_path):
        print("No results file yet. Run without --summary-only first.")
        return
    summarize(results_path)


if __name__ == "__main__":
    main()
