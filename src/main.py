import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from character_recognition import readPlate2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR=BASE_DIR / 'models\YOLO_PlateDetector.pt'
def process_image(image_path):
    # Load models
    plate_detector = YOLO(MODEL_DIR)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    # Detect plates
    results = plate_detector(img)

    # Process detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            print("No plates detected")
            return

        # Process first detected plate
        x1, y1, x2, y2 = map(int, boxes[0])
        # Calculate the height of the plate and the starting y-coordinate for cropping
        plate_height = y2 - y1
        crop_start_y = y1 + int(plate_height * 0.25)  # Skip the top 25%

        # Crop the plate excluding the top 25%
        plate_crop = img[crop_start_y:y2, x1:x2]

        # Convert to grayscale
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Character recognition
        plate_text = readPlate2(plate_gray)
        print(f"Detected Plate: {plate_text}")

        # Visualization
        plt.figure(figsize=(12, 4))

        # Original image with detection
        plt.subplot(1, 3, 1)
        detected_img = img.copy()
        cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Plate Location')
        plt.axis('off')

        # Plate crop
        plt.subplot(1, 3, 2)
        plt.imshow(plate_gray, cmap='gray')
        plt.title('Cropped Plate')
        plt.axis('off')

        # Recognition results
        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.5, f"Recognized Text:\n{plate_text}",
                 fontsize=14, ha='left', va='center')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    process_image(BASE_DIR /"data\\image.png")