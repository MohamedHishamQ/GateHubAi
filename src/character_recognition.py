import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
import cv2
import os
from tensorflow.keras.models import load_model
from pathlib import Path


PADDING = 3
PLATE_WIDTH = 1200
MIN_CHAR_RATIO = 0.2
MAX_CHAR_RATIO = 1.3
dim=30
CHARACTER_DIM = (dim, dim)
CLASS_ARA = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
    'alf': 'أ', 'beh': 'ب', 'dal': 'د', 'fa2': 'ف', 'gem': 'ج', 'hah': 'ح', 'heh': 'ه', 'kaf': 'ق', 'kha': 'خ',
    'lam': 'ل', 'mem': 'م', 'non': 'ن', 'ra2': 'ر', 'sad': 'ص', 'sen': 'س', 'ta2': 'ط', 'wow': 'و', 'ya2': 'ى',
    '3en': 'ع'
}

# Load CNN model and label encoder
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR=BASE_DIR / 'models\\CNN_CharacterRecogention.h5'
model = load_model(MODEL_DIR)
ENCODER_DIR=BASE_DIR / 'models\encoder\label_encoder.npy'
label_encoder = np.load(ENCODER_DIR, allow_pickle=True)


def mapClassToChar(charClass):
    return CLASS_ARA[charClass]


def recognizeChar(img):
    try:
        # Resize and normalize image
        image = cv2.resize(img, CHARACTER_DIM) / 255.0

        # Reshape for CNN
        image = image.reshape(1, dim, dim, 1)

        # Get prediction
        predictions = model.predict(image)
        predicted_class = label_encoder[np.argmax(predictions)]
        confidence_score = np.max(predictions)

        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error in character recognition: {str(e)}")
        return '?', 0.0


def crop_image(image):
    Base_Image = image
    # loop from top left corner until a white pixel is found and save its row
    for i in range(0, Base_Image.shape[0]):
        for j in range(0, Base_Image.shape[1]):
            if Base_Image[i, j] == 255:
                break
        if Base_Image[i, j] == 255:
            break
    top_row = i

    # loop from bottom left corner until a white pixel is found and save its row
    for i in range(Base_Image.shape[0] - 1, 0, -1):
        for j in range(0, Base_Image.shape[1]):
            if Base_Image[i, j] == 255:
                break
        if Base_Image[i, j] == 255:
            break
    bottom_row = i

    segment_1 = Base_Image[top_row:bottom_row, :]
    return segment_1


def segmentPlate2(image, cropFrom):
    # Note
    # img is the image after preprocessing
    # cropFrom the image to take characters from
    # Finding contours
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda tup: cv2.boundingRect(tup)[0])
    chars = []
    X_s = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (x > PADDING and y > PADDING):
            continue
        x -= PADDING
        y -= PADDING
        w += 2 * PADDING
        h += 2 * PADDING
        if (w / h > MIN_CHAR_RATIO and w / h < MAX_CHAR_RATIO):
            cropped = cropFrom[y:y + h, x:x + w]
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                chars.append(cropped)
                X_s.append(x + (w / 2))
    chars2 = []
    X_s2 = []
    for i in range(len(chars)):
        if chars[i].shape[1] > 105:
            chars2.append(chars[i])
            X_s2.append(X_s[i])
    return chars2


def preProcessPlate(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 60))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    return dilation


def resizePlate(img):
    # resize plate to constant width
    factor = PLATE_WIDTH / img.shape[0]
    width = int(img.shape[0] * factor)
    height = int(img.shape[1] * factor)

    dim = (height, width)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def readPlate2(image):
    Gray_image = resizePlate(image)
    preprocessed = preProcessPlate(Gray_image)
    segments = segmentPlate2(preprocessed, Gray_image)
    kernel = np.ones((4, 7), np.uint8)

    # Process segments
    for i in range(len(segments)):
        segments[i] = cv2.threshold(segments[i], 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        segments[i] = cv2.morphologyEx(segments[i], cv2.MORPH_CLOSE, kernel)
        segments[i] = crop_image(segments[i])

    text = ""
    for segment in segments:
        char, score = recognizeChar(segment)
        if score > 0.6:
            text += mapClassToChar(char) + ' '
    return text[::-1]  # reverse String