import torch
from ultralytics import YOLO
import logging


class VehicleDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.model.fuse()

    @torch.no_grad()
    def detect(self, img):
        try:
            results = self.model(
                img,
                conf=0.6,
                iou=0.5,
                classes=[2, 3, 5, 7],
                verbose=False
            )

            vehicles = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    class_name = self.model.names[int(cls)]
                    vehicles.append({
                        'bbox': [x1, y1, x2, y2],
                        'type': class_name,
                        'confidence': float(conf)
                    })

            return vehicles

        except Exception as e:
            logging.error(f"Vehicle detection error: {e}")
            return []