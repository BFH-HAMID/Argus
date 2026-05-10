"""
Face Recognition Module
Handles loading known faces and identifying persons in frames
"""

import os
import face_recognition
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionModule:
    def __init__(self, known_faces_path: str, tolerance: float = 0.6):
        """
        Initialize the face recognition module.
        
        Args:
            known_faces_path: Path to directory containing known face images
            tolerance: How much distance between faces to consider it a match (lower = stricter)
        """
        self.known_faces_path = known_faces_path
        self.tolerance = tolerance
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        """Load all known faces from the specified directory."""
        if not os.path.exists(self.known_faces_path):
            os.makedirs(self.known_faces_path)
            logger.warning(f"Created known_faces directory at {self.known_faces_path}")
            return

        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(self.known_faces_path):
            if filename.lower().endswith(supported_formats):
                filepath = os.path.join(self.known_faces_path, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        # Use filename without extension as the person's name
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        logger.info(f"Loaded face: {name}")
                    else:
                        logger.warning(f"No face found in {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

        logger.info(f"Loaded {len(self.known_face_encodings)} known faces")

    def identify_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Identify faces in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of dictionaries containing face info:
            [{'name': str, 'location': (top, right, bottom, left), 'is_stranger': bool}]
        """
        # Convert BGR to RGB for face_recognition
        rgb_frame = frame[:, :, ::-1]
        
        # Resize frame for faster processing (1/4 size)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        # Find faces and encodings
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            name = "Stranger"
            is_stranger = True
            
            if self.known_face_encodings:
                # Compare face with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=self.tolerance
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        is_stranger = False
            
            results.append({
                'name': name,
                'location': (top, right, bottom, left),
                'is_stranger': is_stranger
            })
        
        return results

    def reload_known_faces(self) -> None:
        """Reload known faces from directory (useful for runtime updates)."""
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self._load_known_faces()

    def add_known_face(self, image_path: str, name: str) -> bool:
        """
        Add a new known face.
        
        Args:
            image_path: Path to the face image
            name: Name to associate with the face
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(name)
                logger.info(f"Added new known face: {name}")
                return True
            else:
                logger.warning(f"No face found in {image_path}")
                return False
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False


# Import cv2 here to avoid circular imports
import cv2
