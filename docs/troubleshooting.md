# Troubleshooting

## Installation and setup

| Message or symptom | Cause | Fix |
|---|---|---|
| `python : The term 'python' is not recognized` | Python is not installed or not on PATH | Install Python, tick "Add python.exe to PATH", open a new terminal. Try `py --version` as an alternative. |
| `pip : The term 'pip' is not recognized` | pip is not on PATH | Use `python -m pip install -r requirements.txt` |
| `WARNING: The script ... is not on PATH` | pip installed command-line tools in a folder not on PATH | Harmless. Use `python -m streamlit run app.py` |
| `streamlit : The term 'streamlit' is not recognized` | Same PATH issue | Use `python -m streamlit run app.py` |
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed in the Python you are running | `python -m pip install -r requirements.txt` in the same terminal you run the project from |
| `ModuleNotFoundError: No module named 'src'` | Running from the wrong folder | `cd` into `anpr_project` (the folder that contains `src` and `app.py`) |
| `can't open file '...run_stage3.py': No such file` | The file is not in the folder you are in | Move it into `anpr_project`, or `cd` to the right folder, then check with `dir` |
| Activate script blocked in PowerShell | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| Files seem missing or outdated | OneDrive has not finished syncing | Wait for the sync to finish, or move the project out of OneDrive |

## Tesseract

| Message or symptom | Cause | Fix |
|---|---|---|
| `Tesseract OCR was not found...` (shown by the app) | Tesseract is not installed, or is in an unusual folder | Install it (see `docs/installation.md`). Check `Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"`. If installed elsewhere, set `$env:TESSERACT_CMD` to the full path to `tesseract.exe`. |
| `tesseract : The term 'tesseract' is not recognized` | The installer did not add it to PATH | Normal. The project finds the default folder by itself. It only matters if you want to type `tesseract` in the terminal. |
| `TesseractNotFoundError` in a traceback | `pytesseract` cannot find the program | Same as above. |
| Results differ from another computer | Different Tesseract version or different image file | Compare `tesseract --version` and use the same original image. |

## Running the app

| Symptom | Cause | Fix |
|---|---|---|
| Browser does not open | Streamlit started without launching the browser | Open `http://localhost:8501` manually. |
| `Port 8501 is already in use` | An old app instance is still running | Close the other terminal, or run `python -m streamlit run app.py --server.port 8502` |
| Page keeps showing old code | Streamlit has not reloaded | Click "Rerun" in the page menu, or press Ctrl+C and start it again |
| "Could not use this file" | The upload is not a readable image (wrong type or corrupted) | Use a real JPG, JPEG or PNG. Renaming a file does not convert it. |
| "The image is too small" | The image is under 60 px on its shortest side | Use a larger image |
| App seems to hang after clicking the button | Very large photo, or first run | Wait a few seconds. Photos are resized to 800 px wide, so large files are still handled. |

## Results look wrong (not errors)

These are limitations of the method, not bugs. Record them in your test results.

| Symptom | Likely reason | What to try |
|---|---|---|
| "No plate-like region found" | Blur, tiny plate, strong glare, or plate very far away | A sharper, closer photo with the plate facing the camera |
| Green box on the wrong object | Clutter: windows, stickers, text that looks like a plate | Move closer or crop the photo so the plate is a larger part of the image |
| Box covers only part of the plate | Blur weakens the edges so the characters do not join | A sharper photo |
| Box is much bigger than the plate | Bumper bars, handles or frames joined to the plate | Check `ocr3_tight_crop.png` in the intermediate steps, since OCR crops tighter |
| "OCR returned no characters" | The crop was too blurry or too small to threshold | A sharper photo |
| `?` in the text | A character that could be a 0 or a 6 (G) was not guessed | Check it manually against the image |
| Warning that the format does not match | The text is not 3 letters + 3-4 digits (could be a motorcycle plate or a misread) | Compare with the image. Motorcycle plates use other formats. |
| Corrected text differs from raw text | The look-alike correction changed characters | Check both. The correction assumes a car format and can be wrong on other formats. |
| Text looks right but is wrong by one character | Similar characters confused (0/O/G, 1/I, 8/B, 5/S) | Check against the image. This is why the app says "check against the image". |

## Still stuck?

Collect these three things before asking for help:
1. The exact text of the error (copy it from the terminal).
2. The output of `python --version` and `pip list`.
3. For wrong results: the original photo and the images from `results/<photo name>/` (or the intermediate steps shown in the app).
