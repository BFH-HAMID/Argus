"""
Object Detection Module using YOLOv8
Detects persons and vehicles, ignoring background motion
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    # COCO class IDs for objects we care about
    PERSON_CLASS_ID = 0
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_path: Path to YOLO model weights (will download if not exists)
            confidence: Minimum confidence threshold for detections
        """
        self.confidence = confidence
        self.model = None
        self._load_model(model_path)
        
        # Class names mapping
        self.class_names = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

    def _load_model(self, model_path: str) -> None:
        """Load the YOLO model."""
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons and vehicles in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detections with format:
            [{'class': str, 'confidence': float, 'bbox': (x1, y1, x2, y2)}]
        """
        if self.model is None:
            return []
        
        detections = []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.confidence)
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only process persons and vehicles
                    if class_id == self.PERSON_CLASS_ID or class_id in self.VEHICLE_CLASS_IDS:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'class': self.class_names.get(class_id, 'unknown'),
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        })
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: BGR image from OpenCV
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Color based on class
            if class_name == 'person':
                color = (0, 0, 255)  # Red for persons
            else:
                color = (255, 165, 0)  # Orange for vehicles
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame_copy, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame_copy, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        return frame_copy

    def has_person(self, detections: List[Dict]) -> bool:
        """Check if any person is detected."""
        return any(d['class'] == 'person' for d in detections)

    def has_vehicle(self, detections: List[Dict]) -> bool:
        """Check if any vehicle is detected."""
        return any(d['class'] in ['car', 'motorcycle', 'bus', 'truck'] for d in detections)

    def get_persons(self, detections: List[Dict]) -> List[Dict]:
        """Get only person detections."""
        return [d for d in detections if d['class'] == 'person']

    def get_vehicles(self, detections: List[Dict]) -> List[Dict]:
        """Get only vehicle detections."""
        return [d for d in detections if d['class'] in ['car', 'motorcycle', 'bus', 'truck']]
