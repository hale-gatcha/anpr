"""Stage 2 check script.

Usage:  python run_stage2.py samples/car1.jpg
Saves every intermediate step into results/<image name>/ so you can
look at each stage of the algorithm.
"""

import os
import sys

from src.detection import detect_plate
from src.utils import load_image, save_image


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_stage2.py <image path>")
        return
    path = sys.argv[1]
    try:
        image = load_image(path)
    except ValueError as err:
        print("Error:", err)
        return

    result = detect_plate(image)
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join("results", name)

    for step_name, img in result.steps.items():
        safe = step_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_image(os.path.join(out_dir, f"{safe}.png"), img)
    if result.plate_crop is not None:
        save_image(os.path.join(out_dir, "plate_crop.png"), result.plate_crop)

    print(result.message)
    if result.found:
        print("Bounding box (x, y, w, h):", result.box)
    print("Step images saved in:", out_dir)


if __name__ == "__main__":
    main()
