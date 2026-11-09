# References

Every entry below was found through a web search or page lookup while preparing this project, and the link points to a page that exists. Before you submit, open each link and make sure the volume, page and author details match your citation style. Entries marked with a note are the ones where some details came from memory and should be confirmed on the linked page.

## A. Official documentation (techniques used in the code)

| Technique | Source |
|---|---|
| Canny edge detection | OpenCV. "Canny Edge Detection." https://docs.opencv.org/5.x/da/d22/tutorial_py_canny.html |
| Morphological closing | OpenCV. "More Morphology Transformations" (opening, closing, hats). https://docs.opencv.org/5.x/d3/dbe/tutorial_opening_closing_hats.html |
| Bilateral filter, smoothing | OpenCV. "Smoothing Images." https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html |
| CLAHE | OpenCV. "Histograms - 2: Histogram Equalization." https://docs.opencv.org/4.7.0/d5/daf/tutorial_py_histogram_equalization.html |
| Otsu thresholding | OpenCV. "Image Thresholding." https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html |
| Contours | OpenCV. "Contours: Getting Started." https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html |
| Improving Tesseract input (scaling, binarization, borders) | Tesseract OCR documentation. "Improving the quality of the output." https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality |
| Tesseract Windows installer | UB Mannheim. "Tesseract at UB Mannheim." https://github.com/UB-Mannheim/tesseract/wiki |
| Python wrapper for Tesseract | pytesseract on PyPI. https://pypi.org/project/pytesseract/ |
| File upload widget | Streamlit documentation. "st.file_uploader." https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader |

## B. Academic papers

**Plate recognition overviews**

1. Du, S., Ibrahim, M., Shehata, M., & Badawy, W. (2013). Automatic License Plate Recognition (ALPR): A State-of-the-Art Review. *IEEE Transactions on Circuits and Systems for Video Technology*, 23(2), 311-325. https://ieeexplore.ieee.org/document/6213519
2. Anagnostopoulos, C.-N. E., et al. (2008). License Plate Recognition From Still Images and Video Sequences: A Survey. *IEEE Transactions on Intelligent Transportation Systems*, 9(3), 377-391. https://dspace.lib.ntua.gr/xmlui/handle/123456789/18939 *(author list and pages from memory: confirm on the linked record)*
3. Lubna, Mufti, N., & Shah, S. A. A. (2021). Automatic Number Plate Recognition: A Detailed Survey of Relevant Algorithms. *Sensors*, 21(9), 3028. https://mdpi.com/1424-8220/21/9/3028

**Techniques**

4. Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), 679-698. https://cecas.clemson.edu/~ahoover/ece431/refs/Canny.pdf *(volume and pages from memory: confirm on the PDF)*
5. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66. https://doi.org/10.1109/TSMC.1979.4310076 *(volume and pages from memory: confirm on the page)*
6. Tomasi, C., & Manduchi, R. (1998). Bilateral Filtering for Gray and Color Images. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. https://mlanthology.org/iccv/1998/tomasi1998iccv-bilateral
7. Smith, R. (2007). An Overview of the Tesseract OCR Engine. *Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR)*. https://research.google/pubs/an-overview-of-the-tesseract-ocr-engine/

## C. Plate formats

8. Wikipedia. "Vehicle registration plates of the Philippines." https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_Philippines. It lists car plates as three letters followed by four digits since the 2014 series, and several different motorcycle layouts. Wikipedia is not an official source, so confirm the current rules with the Land Transportation Office (LTO).

## D. Textbook

9. Gonzalez, R. C., & Woods, R. E. *Digital Image Processing.* Pearson. Standard course textbook for the filtering, edge detection, thresholding and morphology topics. No link is given: use the edition and page numbers from your own course materials.

## E. Which source supports which part of the project

| Part of the project | References |
|---|---|
| What ANPR systems do, typical stages, known difficulties | 1, 2, 3 |
| Resizing, grayscale, filtering | OpenCV filtering page; 6 |
| Contrast enhancement (CLAHE) | OpenCV histogram page |
| Edge detection | 4; OpenCV Canny page |
| Morphological closing, contours | OpenCV morphology and contours pages |
| Thresholding | 5; OpenCV thresholding page |
| How OCR works, improving OCR input | 7; Tesseract quality page |
| Plate format assumptions | 8 (verify with LTO) |

## F. What "digital image processing" means here (short summary for your report)

Digital image processing is computing on pictures stored as grids of numbers, to improve them or extract information. In ANPR it is used in two places: preparing the photo (reducing noise, adjusting contrast, finding edges) so the plate stands out, and preparing the cropped plate (enlarging, thresholding, removing non-character shapes) so the OCR engine can read it. Tesseract's own recognition steps (finding lines and words, then classifying characters with a trained model) are described in reference 7.

## G. How to cite in your report

State clearly which parts are your own image-processing code (detection, cleanup) and which are external tools (Tesseract, OpenCV functions, Streamlit). Do not describe any result as measured unless it comes from `tests/evaluate.py` on your test images.
