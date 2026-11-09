# Demonstration Guide

A suggested 5 to 8 minute demo. Practice it once with the real photos.

## Before the demo (checklist)

- [ ] Laptop charged, internet not needed (everything runs locally).
- [ ] Tesseract installed: `Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"` prints `True`.
- [ ] App starts: `python -m streamlit run app.py` opens in the browser.
- [ ] Photos for the demo are ready in one folder (see the list below).
- [ ] Screenshots of the app results saved as a backup in case something fails.
- [ ] `results/evaluation_summary.txt` and the per-image table open in another window.
- [ ] Browser zoom set so the text is readable from a distance.
- [ ] Project folder open in the editor (to show the code if asked).
- [ ] Same computer and same photo files as in your tests, because results can change with a different Tesseract version or file.

## Which photos to prepare

1. **A clear photo that works.** Shows the full pipeline succeeding.
2. **A photo with a cluttered background that works** (if you have one). Shows the character-row check at work.
3. **A photo that fails or is uncertain** (blurry or small plate). Shows the warnings and the honest limits.
4. **A non-image file**, such as a `.txt` renamed or a corrupted file, to show the error message.

## Script

**1. Introduce the problem (about 30 seconds)**
"Reading license plates by hand is slow and error-prone. Our system takes a vehicle photo, finds the plate with image processing, and reads it with OCR."

**2. Show the application (30 seconds)**
Point out the title, the upload control, and the button. State that it runs locally with Python, OpenCV, Tesseract and Streamlit.

**3. Run a clear photo (2 minutes)**
- Upload the photo; show the original preview.
- Click **Recognize Plate**.
- Show the green box on the processed image and the cropped plate.
- Show the recognized text, the raw text, confidence and agreement.
- Say: "The confidence and agreement values are rough indicators, not accuracy."

**4. Walk through the processing steps (2 minutes)**
Open **Show intermediate processing steps** and explain in order: resize, grayscale, bilateral filter (removes noise, keeps edges), CLAHE (only when contrast is low), Canny edges, morphological closing (joins character edges into a block), contours (candidate regions), then the plate crop and the cleaned black-and-white image given to Tesseract.

**5. Show a failure honestly (1 minute)**
Run the blurry photo. Show the warning or the unreadable result. Say what the cause is (blur weakens the edges) and that this is recorded in the test results.

**6. Show error handling (30 seconds)**
Upload the invalid file and show the message.

**7. Show the test results (1 minute)**
Show `evaluation_summary.txt`: detection rate, exact match, character accuracy, how many images, and the failure categories. Quote only these numbers.

**8. Close with limitations and improvements (30 seconds)**
Limitations: blur, clutter, angles, motorcycle formats. Improvements: a pretrained detector, a stronger OCR engine, perspective correction, deblurring.

## What to say about the technology

- "Detection and plate cleanup are our own OpenCV image processing. Tesseract is an external engine that only reads the cleaned crop."
- "We did not use any paid or cloud service, and no deep-learning model."

## Likely questions and honest answers

| Question | Answer |
|---|---|
| How accurate is it? | Give the numbers and the image counts from `evaluate.py`. Add that the sample is small. |
| Why did it get that character wrong? | Similar characters (0/O/G, 1/I) are confused on small or blurry plates. The system marks ambiguous characters with `?` and says to check manually. |
| Why not deep learning? | The course is about image processing, and the classical method can be explained step by step. It is listed as a future improvement. |
| What did you code and what is external? | Detection, cleanup, scoring, pipeline and interface are ours. Tesseract and the OpenCV functions are libraries. |
| Does it work on motorcycles? | Detection can work, but the text correction assumes a car format, so the raw text should be checked. Only say more if your tests included motorcycles. |
| Why is the confidence low even when the text is right? | Tesseract's confidence depends on the image quality and the mode. We use it as a rough signal with the agreement value. |
| What would you improve first? | Better handling of blurry plates, for example with a learned detector or deblurring. |

## Things not to claim

- Do not say "it is accurate", "it works on all plates" or give a percentage that is not in your results file.
- Do not call the green box "correct" without checking it against the photo.
- Do not describe `?` or the warnings as errors in the app; they are the safety feature.

## If something fails during the demo

1. App does not start: run `python -m streamlit run app.py` again from the `anpr_project` folder. See `docs/troubleshooting.md`.
2. "Tesseract OCR was not found": run the `Test-Path` check, or set `$env:TESSERACT_CMD`.
3. A photo gives a different result than in practice: say so, show the backup screenshot, and explain that results depend on image quality.
4. Last resort: present the saved screenshots and `results/eval/<image_id>/` images.
