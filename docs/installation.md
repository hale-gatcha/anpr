# Installation and Running (Windows, VS Code or Positron)

## 1. Check Python

Open a terminal in your editor (Terminal > New Terminal) and run:

```
python --version
```

You should see a version such as `Python 3.14.x`. If you see "not recognized", install Python from https://www.python.org/downloads/ and tick **Add python.exe to PATH** in the installer, then open a new terminal.

## 2. Open the project folder

In VS Code or Positron: **File > Open Folder** and choose the `anpr_project` folder. The terminal prompt should end with `...\anpr_project>`. If it does not:

```
cd "C:\path\to\anpr_project"
```

> If the folder is inside OneDrive, make sure the files have finished syncing before running anything.

## 3. Install the Python packages

```
pip install -r requirements.txt
```

If `pip` is "not recognized", use:

```
python -m pip install -r requirements.txt
```

Warnings such as "script ... is not on PATH" are harmless. They only mean commands like `streamlit` cannot be typed by themselves, which is why this project uses `python -m streamlit`.

An optional virtual environment keeps the project's packages separate from other projects:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

## 4. Install Tesseract OCR (separate program)

Tesseract is not a Python package. `pytesseract` only talks to it.

1. Download the Windows installer from the UB Mannheim page: https://github.com/UB-Mannheim/tesseract/wiki
2. Run it and keep the default folder `C:\Program Files\Tesseract-OCR`. English is included by default.
3. Check that the program exists (this works even if the `tesseract` command is not on PATH):

```
Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

`True` means it is installed. The project looks in that folder automatically, so **no PATH setup is needed**.

If you installed it somewhere else, tell the project where it is. For the current terminal only:

```
$env:TESSERACT_CMD = "C:\full\path\to\tesseract.exe"
```

Optionally, `tesseract --version` works once the folder is added to PATH or after you open a new terminal on some installs.

## 5. Run the application

```
python -m streamlit run app.py
```

Your browser opens at `http://localhost:8501`. If it does not, open that address yourself. Stop the app with **Ctrl+C** in the terminal.

Steps in the app:
1. Upload a JPG, JPEG or PNG vehicle photo.
2. Check the original preview.
3. Click **Recognize Plate**.
4. Read the detected plate image, cropped plate, recognized text, confidence and any warnings.
5. Open **Show intermediate processing steps** to see each image-processing stage.

## 6. Test scripts (optional, from the terminal)

```
python run_stage2.py samples/car1.jpg     detection only, saves every step in results/car1/
python run_stage3.py samples/car1.jpg     detection + OCR, prints the text and saves the images
```

## 7. Run the evaluation

See `docs/test_plan.md`. In short: fill `tests/ground_truth.csv`, put photos in `tests/test_images/`, then:

```
python tests/evaluate.py
```

## Versions seen on the development machine

These are the versions installed during development (from the pip log): `opencv-python 5.0.0.93`, `numpy 2.4.2`, `Pillow 12.1.1`, `pytesseract 0.3.13`, `streamlit 1.65.0`, on Python 3.14. The Tesseract version depends on the installer you downloaded (`tesseract --version` shows it). Results can differ slightly between Tesseract versions.
