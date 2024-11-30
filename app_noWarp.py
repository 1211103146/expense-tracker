# FOR: Edge detection and cropping (needs to be on high contrast background)
# https://www.makeuseof.com/python-create-document-scanner/

import cv2
import imutils
# from PIL import Image, ImageEnhance
import numpy as np
from transform import perspective_transform

import pytesseract
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import pipeline

# Preprocess the image
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(gray_image)

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
# image_path = 'C:\\Users\\Medieval\\Downloads\\archive_Malaysian\\fullDataset\\images\\002.jpg'
# image_path = 'C:\\Users\\Medieval\\Desktop\\Test\\Images\\3input.png'
image_path = 'C:\\Users\\Medieval\\Desktop\\Test\\Images\\3input.jpg'

image = cv2.imread(image_path)
image_copy = image.copy()

w = 600.0 # Value of image resize width
ratio = image.shape[1] / w

# Process image
edged_img = preprocess_image(image)
# doc = find_quadrilateral(edged_img)

# if doc is None:
#     dilated = complete_quadrilteral()

# if doc is not None:
#     p = []

#     for d in doc:
#         tuple_point = tuple(d[0])
#         cv2.circle(image, tuple_point, 3, (0, 0, 255), 4)
#         p.append(tuple_point)

# cv2.imshow('Circled corner points', edged_img)
# cv2.waitKey(0)

# if doc is not None:
#     # Warp image based on corner points
#     warped_image = perspective_transform(image_copy, doc.reshape(4, 2) * ratio)
#     warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
#     # _, binary_image = cv2.threshold(warped_image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
cv2.imshow("Warped Image", imutils.resize(edged_img, width=600))
cv2.waitKey(0)

# PERFORM OCR
text = pytesseract.image_to_string(edged_img, lang='msa') 
print(text)







# # Load pre-trained BERT model for classification
# model_name = "bert-base-uncased"
# nlp_classification = pipeline("text-classification", model=model_name, tokenizer=model_name)

# # Classify transaction type
# result = nlp_classification(text)
# print(result)
