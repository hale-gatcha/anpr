"""Stage 3 check script: detect the plate, then read it with OCR.

Usage:  python run_stage3.py samples/car1.jpg
"""

import os
import sys

from src.detection import detect_plate
from src.ocr import read_plate
from src.utils import load_image, save_image


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_stage3.py <image path>")
        return
    path = sys.argv[1]
    try:
        image = load_image(path)
    except ValueError as err:
        print("Error:", err)
        return

    det = detect_plate(image)
    print("Detection:", det.message)
    if not det.found:
        return

    ocr = read_plate(det.plate_crop)
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join("results", name)
    save_image(os.path.join(out_dir, "plate_crop.png"), det.plate_crop)
    save_image(os.path.join(out_dir, "8_Detected_plate.png"), det.annotated)
    for step_name, img in ocr.steps.items():
        save_image(os.path.join(out_dir, f"{step_name}.png"), img)

    print("OCR status :", ocr.message)
    if ocr.ok:
        print("Raw OCR    :", ocr.raw_text)
        print("Corrected  :", ocr.text)
        print(f"Tesseract confidence (unrestricted pass): {ocr.confidence:.0f}%")
        print(f"Agreement between the two passes       : {ocr.agreement:.0%}")
        print("Format ok  :", ocr.format_ok)
    print("Images saved in:", out_dir)


if __name__ == "__main__":
    main()
