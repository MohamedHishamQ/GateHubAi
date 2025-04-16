from pathlib import Path
from models.vehicle_detection import VehicleDetector
from models.plate_detection import PlateDetector
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from utils.api_client import APIClient
import logging

logging.basicConfig(level=logging.ERROR)


def main():
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / 'models'

    # Initialize components
    vehicle_detector = VehicleDetector(MODELS_DIR / 'yolo11x.pt')
    plate_detector = PlateDetector(MODELS_DIR / 'YOLO_PlateDetector.pt')
    api_client = APIClient()  # Now this works with default base_url

    # Choose processing mode
    mode = input("Choose mode (1 for image, 2 for video): ")

    if mode == "1":
        # Image processing
        image_processor = ImageProcessor(vehicle_detector, plate_detector, api_client)
        image_path = BASE_DIR / "data/image.png"
        print(f"\n📸 Processing image: {image_path}")
        results = image_processor.process_image(image_path)

        if results:
            print("\n📋 Results Summary:")
            for idx, result in enumerate(results, 1):
                print(f"Vehicle {idx}:")
                print(f"  - Plate Number: {result['text']}")
                if 'vehicle_type' in result:
                    print(f"  - Vehicle Type: {result['vehicle_type']}")
                print(f"  - Confidence: {result['confidence']:.2f}")

    elif mode == "2":
        # Video processing
        video_processor = VideoProcessor(vehicle_detector, plate_detector, api_client)

        # Ask for video source
        source = input("Enter video source (0 for webcam, or path to video file): ")
        video_source = 0 if source == "0" else source

        # Start processing
        video_processor.process_video(video_source, gate_id=1, display=True)

    else:
        print("❌ Invalid mode selected")


if __name__ == "__main__":
    main()