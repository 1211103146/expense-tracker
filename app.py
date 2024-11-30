# REF: Initial OCR set up
# https://www.affinda.com/blog/how-to-convert-image-to-text-using-python 

# REF: Edge detection and cropping (needs to be on high contrast background)
# https://www.makeuseof.com/python-create-document-scanner/

# REF: Pytesseract best practices (paused at 4:11)
# https://www.youtube.com/watch?v=3BtLA75zKL0 

# TODO:
# - Test parameters for Tesseract
# - Ways to improve OCR results https://stackoverflow.com/questions/55140090/pytesseract-reading-receipt (apply deep learning)
# - Ways to improve OCR results https://nanonets.com/blog/ocr-with-tesseract/ (*apply thresholding)
# - BERT training https://www.shecodes.io/athena/38534-how-to-use-bert-for-text-classification-in-python 
#                 https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b 
# - Classification https://drojasug.medium.com/using-sci-kit-learn-to-categorize-personal-expenses-de07b6b385f5

import cv2
import imutils
# from PIL import Image, ImageEnhance
import numpy as np
from transform import perspective_transform

import pytesseract
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import pipeline

# Resize the image
# def reshape_image(image, width):
#     return imutils.resize(image, width=int(width))

# Preprocess the image
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image)
    blurred_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    return cv2.Canny(blurred_image, 75, 200)

def complete_quadrilteral():
    # Try Hough Line Transform to detect lines and complete the rectangle
    lines = cv2.HoughLinesP(edged_img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        # Draw detected lines to debug
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edged_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Try dilation
    if lines is None:
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(edged_img, kernel, iterations=1)


# Iterate over contours and find the quadrilateral
def find_quadrilateral(image):
    doc = None

    cnts, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            doc = approx
            return doc

# Load the image
image_path = 'C:\\Users\\Medieval\\College\\FYP\\FYP1\\Test\\Images\\archive_malaysian\\fullDataset\\images\\031.jpg'
# image_path = 'C:\\Users\\Medieval\\College\\FYP\\FYP1\\Test\\Images\\3input.png'
# image_path = 'C:\\Users\\Medieval\\College\\FYP\\FYP1\\Test\\Images\\3input.jpg'

image = cv2.imread(image_path)
image_copy = image.copy()

w = 600.0 # Value of image resize width
ratio = image.shape[1] / w

# Process image
# img_resize = reshape_image(image, w)
# edged_img = preprocess_image(img_resize)
edged_img = preprocess_image(image)
doc = find_quadrilateral(edged_img)

if doc is None:
    dilated = complete_quadrilteral()

if doc is not None:
    p = []

    for d in doc:
        tuple_point = tuple(d[0])
        cv2.circle(image, tuple_point, 3, (0, 0, 255), 4)
        p.append(tuple_point)

cv2.imshow('Circled corner points', image)
cv2.waitKey(0)

if doc is not None:
    # Warp image based on corner points
    warped_image = perspective_transform(image_copy, doc.reshape(4, 2))
    # warped_image = perspective_transform(image_copy, doc.reshape(4, 2) * ratio)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(warped_image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
    cv2.imshow("Warped Image", imutils.resize(warped_image, width=300))
    cv2.waitKey(0)

# PERFORM OCR
text = pytesseract.image_to_string(image, lang='eng') 
print(text)

open('output.txt', 'w').close()
f = open("output.txt", "a")
f.write(text)
f.close()

# Load pre-trained BERT model for NER
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
# nlp_ner = pipeline("ner", model=model_name, tokenizer=model_name, device=0)
pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0)

file = open('C:\\Users\\Medieval\\College\\FYP\\FYP1\\Test\\proj\\output.txt', 'r')
text = file.readlines()

# Extract entities
# entities = nlp_ner(text)
document = pipe(text)
print(document)







# # Load pre-trained BERT model for classification
# model_name = "bert-base-uncased"
# nlp_classification = pipeline("text-classification", model=model_name, tokenizer=model_name)

# # Classify transaction type
# result = nlp_classification(text)
# print(result)
