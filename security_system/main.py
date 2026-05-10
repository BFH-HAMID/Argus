"""
AI-Powered Home Security System
Main Application Entry Point

Author: Security System
Version: 1.0.0
"""

import cv2
import numpy as np
import threading
import time
import logging
import sys
import os
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
import config

# Import modules
from modules.object_detection import ObjectDetector
from modules.face_recognition_module import FaceRecognitionModule
from modules.sound_detection import SoundDetector
from modules.telegram_alert import TelegramAlert
from modules.night_mode import NightModeEnhancer
from modules.storage_manager import StorageManager
from modules.video_recorder import VideoRecorder, TimedRecordingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('security_system.log')
    ]
)
logger = logging.getLogger(__name__)


class SecuritySystem:
    """Main security system class that orchestrates all components."""

    def __init__(self):
        """Initialize the security system."""
        logger.info("Initializing Security System...")
        
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Alert cooldown tracking
        self.last_person_alert = 0
        self.last_vehicle_alert = 0
        self.last_sound_alert = 0
        self.alert_cooldown = config.COOLDOWN_PERIOD
        
        # Initialize components
        self._init_storage()
        self._init_camera()
        self._init_ai_modules()
        self._init_alerts()
        
        logger.info("Security System initialized successfully")

    def _init_storage(self) -> None:
        """Initialize storage management."""
        # Ensure directories exist
        os.makedirs(config.STORAGE_PATH, exist_ok=True)
        os.makedirs(config.KNOWN_FACES_PATH, exist_ok=True)
        
        self.storage_manager = StorageManager(
            storage_path=config.STORAGE_PATH,
            max_storage_gb=config.MAX_STORAGE_GB,
            auto_delete_days=config.AUTO_DELETE_DAYS
        )
        
        self.video_recorder = VideoRecorder(
            storage_path=config.STORAGE_PATH,
            frame_width=config.FRAME_WIDTH,
            frame_height=config.FRAME_HEIGHT,
            fps=config.FPS
        )
        
        self.timed_recorder = TimedRecordingManager(
            recorder=self.video_recorder,
            duration=config.RECORDING_DURATION,
            cooldown=config.COOLDOWN_PERIOD
        )

    def _init_camera(self) -> None:
        """Initialize camera capture."""
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Camera not available")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.FPS)
        
        logger.info(f"Camera initialized: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT} @ {config.FPS}fps")

    def _init_ai_modules(self) -> None:
        """Initialize AI detection modules."""
        # Object Detection (YOLO)
        logger.info("Loading YOLO model...")
        self.object_detector = ObjectDetector(
            model_path="yolov8n.pt",
            confidence=config.YOLO_CONFIDENCE
        )
        
        # Face Recognition
        logger.info("Loading face recognition...")
        self.face_recognizer = FaceRecognitionModule(
            known_faces_path=config.KNOWN_FACES_PATH,
            tolerance=config.FACE_RECOGNITION_TOLERANCE
        )
        
        # Night Mode Enhancer
        self.night_enhancer = NightModeEnhancer(
            night_start=config.NIGHT_MODE_START,
            night_end=config.NIGHT_MODE_END,
            clip_limit=config.CLAHE_CLIP_LIMIT,
            tile_grid_size=config.CLAHE_TILE_GRID_SIZE
        )
        
        # Sound Detection
        self.sound_detector = SoundDetector(
            threshold=config.SOUND_THRESHOLD,
            sample_rate=config.SOUND_RATE,
            chunk_size=config.SOUND_CHUNK,
            callback=self._on_sound_trigger
        )

    def _init_alerts(self) -> None:
        """Initialize alert system."""
        self.telegram = TelegramAlert(
            token=config.TELEGRAM_TOKEN,
            chat_id=config.CHAT_ID
        )

    def _on_sound_trigger(self, sound_level: int) -> None:
        """Callback for sound threshold trigger."""
        current_time = time.time()
        
        if current_time - self.last_sound_alert < self.alert_cooldown:
            return
        
        self.last_sound_alert = current_time
        logger.warning(f"Sound trigger activated! Level: {sound_level}")
        
        # Send telegram alert
        self.telegram.send_sound_alert(sound_level)
        
        # Trigger emergency recording
        with self.frame_lock:
            if self.current_frame is not None:
                self.telegram.send_snapshot(
                    self.current_frame.copy(),
                    "SOUND ALERT",
                    f"Loud sound detected: level {sound_level}"
                )
        
        # Start recording
        self.timed_recorder.trigger_recording(
            get_frame_func=self._get_current_frame,
            callback=lambda path: self.telegram.send_video_clip(path, "Sound Triggered Recording")
        )

    def _get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame thread-safely."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def _process_detections(self, frame: np.ndarray, detections: list, faces: list) -> np.ndarray:
        """Process detections and handle alerts."""
        current_time = time.time()
        display_frame = frame.copy()
        
        # Draw object detections
        display_frame = self.object_detector.draw_detections(display_frame, detections)
        
        # Check for persons
        if self.object_detector.has_person(detections):
            if current_time - self.last_person_alert > self.alert_cooldown:
                self.last_person_alert = current_time
                
                # Check if any faces are strangers
                stranger_detected = any(f['is_stranger'] for f in faces)
                
                if stranger_detected:
                    logger.warning("Stranger detected!")
                    self.telegram.send_person_alert(frame, "Unknown", True)
                    
                    # Trigger recording for strangers
                    self.timed_recorder.trigger_recording(
                        get_frame_func=self._get_current_frame,
                        callback=lambda path: self.telegram.send_video_clip(path, "Stranger Detection")
                    )
                elif faces:
                    # Known person
                    for face in faces:
                        if not face['is_stranger']:
                            self.telegram.send_person_alert(frame, face['name'], False)
        
        # Check for vehicles
        if self.object_detector.has_vehicle(detections):
            if current_time - self.last_vehicle_alert > self.alert_cooldown:
                self.last_vehicle_alert = current_time
                
                vehicles = self.object_detector.get_vehicles(detections)
                for vehicle in vehicles:
                    self.telegram.send_vehicle_alert(frame, vehicle['class'].upper())
        
        # Draw face recognition results
        for face in faces:
            top, right, bottom, left = face['location']
            
            # Color: red for stranger, green for known
            color = (0, 0, 255) if face['is_stranger'] else (0, 255, 0)
            
            # Draw face box
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Draw name label
            label = face['name']
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                display_frame,
                (left, bottom),
                (left + label_size[0], bottom + label_size[1] + 10),
                color,
                -1
            )
            cv2.putText(
                display_frame,
                label,
                (left, bottom + label_size[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return display_frame

    def _draw_status_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw status information on frame."""
        overlay = frame.copy()
        
        # Status bar background
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Current time
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Recording status
        if self.video_recorder.is_recording:
            cv2.circle(frame, (frame.shape[1] - 30, 20), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Night mode status
        if self.night_enhancer.is_night_time():
            cv2.putText(frame, "NIGHT MODE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Sound level
        sound_level = self.sound_detector.get_current_level()
        sound_percentage = min(100, (sound_level / config.SOUND_THRESHOLD) * 100)
        bar_width = int(200 * (sound_percentage / 100))
        
        cv2.rectangle(frame, (10, 60), (210, 75), (100, 100, 100), -1)
        
        bar_color = (0, 255, 0) if sound_percentage < 70 else (0, 255, 255) if sound_percentage < 90 else (0, 0, 255)
        cv2.rectangle(frame, (10, 60), (10 + bar_width, 75), bar_color, -1)
        cv2.putText(frame, f"Sound: {sound_level}", (220, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Storage info
        storage_info = self.storage_manager.get_storage_info()
        storage_str = f"Storage: {storage_info['recordings_usage_gb']:.1f}GB / {storage_info['max_storage_gb']}GB"
        cv2.putText(
            frame, 
            storage_str, 
            (frame.shape[1] - 250, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (255, 255, 255), 
            1
        )
        
        return frame

    def run(self) -> None:
        """Main loop of the security system."""
        logger.info("Starting Security System...")
        
        self.running = True
        
        # Start background services
        self.storage_manager.start()
        self.sound_detector.start()
        self.telegram.start()
        
        # Send startup notification
        self.telegram.send_startup_message()
        
        # Frame timing
        frame_time = 1.0 / config.FPS
        last_detection_time = 0
        detection_interval = 0.1  # Run AI detection every 100ms
        
        logger.info("Security System running. Press 'q' to quit, 'r' to record, 's' to snapshot.")
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Capture frame
                ret, frame = self.camera.read()
                
                if not ret:
                    logger.error("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Apply night mode enhancement if needed
                enhanced_frame = self.night_enhancer.enhance(frame)
                
                # Store current frame thread-safely
                with self.frame_lock:
                    self.current_frame = enhanced_frame.copy()
                
                # Add frame to recording if active
                self.video_recorder.add_frame(enhanced_frame)
                
                # Run AI detection periodically
                current_time = time.time()
                detections = []
                faces = []
                
                if current_time - last_detection_time >= detection_interval:
                    last_detection_time = current_time
                    
                    # Object detection
                    detections = self.object_detector.detect(enhanced_frame)
                    
                    # Face recognition (only if persons detected)
                    if self.object_detector.has_person(detections):
                        faces = self.face_recognizer.identify_faces(enhanced_frame)
                
                # Process detections and get display frame
                display_frame = self._process_detections(enhanced_frame, detections, faces)
                
                # Draw status overlay
                display_frame = self._draw_status_overlay(display_frame)
                
                # Show frame
                cv2.imshow("AI Security System", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit command received")
                    break
                elif key == ord('r'):
                    # Toggle recording
                    if self.video_recorder.is_recording:
                        filepath = self.video_recorder.stop_recording()
                        logger.info(f"Recording saved: {filepath}")
                    else:
                        filepath = self.video_recorder.start_recording()
                        logger.info(f"Recording started: {filepath}")
                elif key == ord('s'):
                    # Take snapshot
                    filepath = self.video_recorder.save_snapshot(enhanced_frame)
                    logger.info(f"Snapshot saved: {filepath}")
                elif key == ord('n'):
                    # Toggle night mode
                    self.night_enhancer.set_force_night_mode(
                        not self.night_enhancer.force_night_mode
                    )
                elif key == ord('f'):
                    # Reload known faces
                    self.face_recognizer.reload_known_faces()
                    logger.info("Reloaded known faces")
                
                # Maintain frame rate
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the security system."""
        logger.info("Stopping Security System...")
        
        self.running = False
        
        # Stop recording if active
        if self.video_recorder.is_recording:
            self.video_recorder.stop_recording()
        
        # Stop background services
        self.sound_detector.stop()
        self.storage_manager.stop()
        
        # Send shutdown notification
        self.telegram.send_shutdown_message()
        time.sleep(1)  # Allow message to send
        self.telegram.stop()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        logger.info("Security System stopped")


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         AI-Powered Home Security System v1.0              ║
    ║                                                           ║
    ║  Features:                                                ║
    ║  • YOLOv8 Person & Vehicle Detection                      ║
    ║  • Face Recognition (Known vs Stranger)                   ║
    ║  • Telegram Alerts with Snapshots                         ║
    ║  • Sound-Triggered Recording                              ║
    ║  • Night Mode Enhancement                                 ║
    ║  • Auto-Delete Old Recordings                             ║
    ║                                                           ║
    ║  Controls:                                                ║
    ║  [Q] Quit  [R] Record  [S] Snapshot                       ║
    ║  [N] Toggle Night Mode  [F] Reload Faces                  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        system = SecuritySystem()
        system.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
