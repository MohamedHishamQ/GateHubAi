import cv2
import time
from collections import deque
from datetime import datetime
import numpy as np


class VideoProcessor:
    def __init__(self, vehicle_detector, plate_detector, api_client):
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.api_client = api_client

        # Video processing parameters
        self.frame_buffer_size = 30
        self.process_every_n_frames = 5
        self.min_detection_confidence = 0.6

        # Track processed plates to avoid duplicates
        self.processed_plates = set()
        self.last_detection_time = {}
        self.detection_cooldown = 10  # seconds

    def calculate_frame_quality(self, frame):
        """Calculate frame quality score based on multiple metrics"""
        try:
            if frame is None:
                return 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate various quality metrics
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)

            # Combine metrics into a quality score
            quality_score = (blur_score * 0.5 + brightness * 0.25 + contrast * 0.25) / 100
            return quality_score

        except Exception as e:
            print(f"Error calculating frame quality: {e}")
            return 0

    def should_process_plate(self, plate_number):
        """Determine if plate should be processed based on cooldown"""
        current_time = time.time()
        if plate_number in self.last_detection_time:
            time_since_last = current_time - self.last_detection_time[plate_number]
            if time_since_last < self.detection_cooldown:
                return False

        self.last_detection_time[plate_number] = current_time
        return True

    def process_video(self, video_source=0, gate_id=1, display=True):
        """Process video stream with improved detection and display"""
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print("❌ Failed to open video source")
                return

            print("\n🎥 Starting video processing...")
            frame_buffer = deque(maxlen=self.frame_buffer_size)
            frame_count = 0
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_fps = frame_count / (time.time() - start_time)

                # Add frame to buffer with quality score
                quality = self.calculate_frame_quality(frame)
                frame_buffer.append((frame.copy(), quality))

                # Process every nth frame
                if frame_count % self.process_every_n_frames == 0:
                    # Get best quality frame from buffer
                    best_frame, _ = max(frame_buffer, key=lambda x: x[1])

                    # Detect vehicles
                    vehicles = self.vehicle_detector.detect(best_frame)

                    for vehicle in vehicles:
                        x1, y1, x2, y2 = vehicle['bbox']
                        vehicle_crop = best_frame[y1:y2, x1:x2]

                        # Detect plate
                        plate_data = self.plate_detector.detect_and_read(vehicle_crop)

                        if plate_data and plate_data['confidence'] > self.min_detection_confidence:
                            plate_number = plate_data['text']

                            # Check if we should process this plate
                            if self.should_process_plate(plate_number):
                                print(f"\n🚗 New vehicle detected!")
                                print(f"📝 Plate Number: {plate_number}")
                                print(f"🚙 Vehicle Type: {vehicle['type']}")
                                print(f"📊 Confidence: {plate_data['confidence']:.2f}")

                                # Send to API
                                api_data = {
                                    "plateNumber": plate_number,
                                    "fineValue": 0,
                                    "fineType": "No Fine",
                                    "gateId": gate_id
                                }

                                self.api_client.send_data(api_data)

                                # Draw detection on frame
                                if display:
                                    self.draw_detection(frame, vehicle, plate_number)

                # Display processing info
                if display:
                    self.draw_info(frame, frame_count, current_fps)
                    cv2.imshow('Vehicle Detection', frame)

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Cleanup
            cap.release()
            if display:
                cv2.destroyAllWindows()

            # Print summary
            print(f"\n📊 Processing Summary:")
            print(f"Total frames processed: {frame_count}")
            print(f"Unique plates detected: {len(self.processed_plates)}")
            print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")

        except Exception as e:
            print(f"❌ Error processing video: {e}")
            if cap is not None:
                cap.release()
            if display:
                cv2.destroyAllWindows()

    def draw_detection(self, frame, vehicle, plate_number):
        """Draw detection boxes and info on frame"""
        try:
            x1, y1, x2, y2 = vehicle['bbox']

            # Draw vehicle box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw vehicle info
            info_text = f"{vehicle['type']} | {plate_number}"
            cv2.putText(frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error drawing detection: {e}")

    def draw_info(self, frame, frame_count, fps):
        """Draw processing information on frame"""
        try:
            height = frame.shape[0]
            info_text = f"Frame: {frame_count} | FPS: {fps:.2f} | Plates: {len(self.processed_plates)}"
            cv2.putText(frame, info_text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error drawing info: {e}")

    def cleanup(self):
        """Cleanup resources"""
        self.processed_plates.clear()
        self.last_detection_time.clear()