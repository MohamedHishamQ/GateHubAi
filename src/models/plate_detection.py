import cv2
from ultralytics import YOLO
from src.character_recognition import  readPlate2
import logging
import torch

class PlateDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.model.fuse()

    def detect_and_read(self, img):
        try:
            results = self.model(img)
            if not results or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = map(int, boxes[0])

            plate_crop = self._process_plate_crop(img, x1, y1, x2, y2)
            if plate_crop is None:
                return None

            plate_text = readPlate2(plate_crop)
            if plate_text and len(plate_text.strip()) > 0:
                return {
                    'text': plate_text,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(results[0].boxes.conf[0])
                }

            return None

        except Exception as e:
            logging.error(f"Plate detection error: {e}")
            return None

    def _process_plate_crop(self, img, x1, y1, x2, y2):
        plate_height = y2 - y1
        crop_start_y = y1 + int(plate_height * 0.25)
        plate_crop = img[crop_start_y:y2, x1:x2]

        if plate_crop.size == 0 or min(plate_crop.shape[:2]) < 20:
            return None

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary