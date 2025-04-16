import cv2
from src.models.vehicle_detection import VehicleDetector
from src.models.plate_detection import PlateDetector
from src.utils.api_client import APIClient


class ImageProcessor:
    def __init__(self, vehicle_detector, plate_detector, api_client):
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.api_client = api_client

    def process_image(self, image_path, gate_id=1):
        img = cv2.imread(str(image_path))
        if img is None:
            print("❌ Failed to load image")
            return None

        # Direct plate detection
        plate_data = self.plate_detector.detect_and_read(img)
        if plate_data:
            self._handle_plate_detection(plate_data, gate_id)
            return [plate_data]

        # Vehicle detection approach
        vehicles = self.vehicle_detector.detect(img)
        results = []

        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_crop = img[y1:y2, x1:x2]
            plate_data = self.plate_detector.detect_and_read(vehicle_crop)

            if plate_data:
                self._handle_plate_detection(plate_data, gate_id, vehicle)
                results.append(plate_data)

        return results

    def _handle_plate_detection(self, plate_data, gate_id, vehicle=None):
        print(f"📝 Plate Number: {plate_data['text']}")
        print(f"📊 Confidence: {plate_data['confidence']:.2f}")

        if vehicle:
            print(f"🚗 Vehicle Type: {vehicle['type']}")

        self.api_client.send_fine_data(plate_data['text'], gate_id)