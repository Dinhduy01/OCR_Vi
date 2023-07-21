
import os
import cv2
import re
import pandas as pd
from modules import Preprocess, Detection, OCR, Retrieval, Correction
from tool.utils import natural_keys, visualize
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


img_id = "test"
class_mapping = {"SELLER":0, "ADDRESS":1, "TIMESTAMP":2, "TOTAL_COST":3, "NONE":4}
idx_mapping = {0:"SELLER", 1:"ADDRESS", 2:"TIMESTAMP", 3:"TOTAL_COST", 4:"NONE"}
det_weight = "C:/Users/Admin/Desktop/nhandienchu/ORC/OCR-20230719T060158Z-001/OCR/PANNet_best_map.pth"
ocr_weight = "C:/Users/Admin\Desktop/nhandienchu/ORC/OCR-20230719T060158Z-001/OCR/transformerocr.pth"

img = cv2.imread(f"C:/Users/Admin/Desktop/nhandienchu//test.jpg")

plt.imshow(img)
plt.show()

det_model = Detection(weight_path=det_weight)
ocr_model = OCR(weight_path=ocr_weight)
preproc = Preprocess(
    det_model=det_model,
    ocr_model=ocr_model,
    find_best_rotation=False)
retrieval = Retrieval(class_mapping, mode = 'all')
correction = Correction()


img1 = preproc(img)

plt.imshow(img1)
plt.show()

boxes, img2  = det_model(
    img1,
    crop_region=True,                               #Crop detected regions for OCR
    return_result=True,                             # Return plotted result
    output_path=f"C:/xampp/htdocs/Desktop/OCR_VI/results/{img_id}"   #Path to save cropped regions
)

plt.imshow(img2)
plt.show()

img_paths = os.listdir(f"C:/xampp/htdocs/Desktop/OCR_VI/results/{img_id}/crops")  # Cropped regions
img_paths.sort(key=natural_keys)
img_paths = [os.path.join(f"C:/xampp/htdocs/Desktop/OCR_VI/results/{img_id}/crops", i) for i in img_paths]
import csv
import re
def correct_spelling(dictionary_file, paragraph):
    dictionary = {}
    with open(dictionary_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            word = row[0]
            correction = row[1]
            dictionary[word] = correction
    words = re.findall(r'\b\w+\b|[^\w\s]', paragraph)
    fixed_words = []

    i = 0
    while i < len(words):
        if words[i] in dictionary:
            fixed_words.append(dictionary[words[i]])
            i += 1
        elif i < len(words) - 1 and f"{words[i]} {words[i+1]}" in dictionary:
            fixed_words.append(dictionary[f"{words[i]} {words[i+1]}"])
            i += 2
        else:
            fixed_words.append(words[i])
            i += 1

    fixed_sentence = ' '.join(fixed_words)
    return fixed_sentence

texts, probs = ocr_model.predict_folder(img_paths, return_probs=True)  # OCR
#texts = correction(texts)
dictionary_file = 'dic.csv'
for corrected_text in texts:
    print( correct_spelling(dictionary_file,corrected_text))
