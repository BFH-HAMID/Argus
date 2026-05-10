"""
Video Recorder Module
Handles video recording with threading support
"""

import cv2
import numpy as np
import threading
import time
from queue import Queue
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(
        self,
        storage_path: str,
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: int = 30
    ):
        """
        Initialize the video recorder.
        
        Args:
            storage_path: Path to save recordings
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Frames per second
        """
        self.storage_path = storage_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        self.is_recording = False
        self.current_writer = None
        self.current_file = None
        
        self.frame_queue = Queue(maxsize=100)
        self.recording_thread = None
        self.running = False

    def start_recording(self, filename: Optional[str] = None) -> str:
        """
        Start a new recording.
        
        Args:
            filename: Optional custom filename (without extension)
            
        Returns:
            Path to the recording file
        """
        if self.is_recording:
            logger.warning("Already recording, stopping current recording")
            self.stop_recording()
        
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}"
        
        self.current_file = f"{self.storage_path}/{filename}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.current_writer = cv2.VideoWriter(
            self.current_file,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.current_writer.isOpened():
            logger.error(f"Failed to create video writer for {self.current_file}")
            return ""
        
        self.is_recording = True
        self.running = True
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._write_frames, daemon=True)
        self.recording_thread.start()
        
        logger.info(f"Started recording: {self.current_file}")
        return self.current_file

    def stop_recording(self) -> str:
        """
        Stop the current recording.
        
        Returns:
            Path to the completed recording
        """
        if not self.is_recording:
            return ""
        
        self.is_recording = False
        self.running = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=5)
            self.recording_thread = None
        
        # Release writer
        if self.current_writer:
            self.current_writer.release()
            self.current_writer = None
        
        completed_file = self.current_file
        self.current_file = None
        
        logger.info(f"Stopped recording: {completed_file}")
        return completed_file

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the recording queue.
        
        Args:
            frame: BGR image from OpenCV
        """
        if not self.is_recording:
            return
        
        try:
            # Resize frame if necessary
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Add to queue (non-blocking)
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame.copy())
        except Exception as e:
            logger.error(f"Error adding frame: {e}")

    def _write_frames(self) -> None:
        """Background thread to write frames to file."""
        while self.running or not self.frame_queue.empty():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    if self.current_writer and self.current_writer.isOpened():
                        self.current_writer.write(frame)
                else:
                    time.sleep(0.01)
            except Exception as e:
                if self.running:
                    logger.error(f"Frame write error: {e}")

    def record_for_duration(self, duration: int, get_frame_func) -> str:
        """
        Record for a specific duration.
        
        Args:
            duration: Recording duration in seconds
            get_frame_func: Function that returns the current frame
            
        Returns:
            Path to the completed recording
        """
        filepath = self.start_recording()
        
        if not filepath:
            return ""
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if not self.is_recording:
                break
            
            frame = get_frame_func()
            if frame is not None:
                self.add_frame(frame)
            
            time.sleep(1.0 / self.fps)
        
        return self.stop_recording()

    def save_snapshot(self, frame: np.ndarray, filename: Optional[str] = None) -> str:
        """
        Save a single frame as an image.
        
        Args:
            frame: BGR image from OpenCV
            filename: Optional custom filename (without extension)
            
        Returns:
            Path to the saved snapshot
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}"
        
        filepath = f"{self.storage_path}/{filename}.jpg"
        
        try:
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            logger.info(f"Saved snapshot: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return ""


class TimedRecordingManager:
    """Manages automatic timed recordings."""
    
    def __init__(self, recorder: VideoRecorder, duration: int = 30, cooldown: int = 10):
        """
        Initialize the timed recording manager.
        
        Args:
            recorder: VideoRecorder instance
            duration: Recording duration in seconds
            cooldown: Minimum time between recordings
        """
        self.recorder = recorder
        self.duration = duration
        self.cooldown = cooldown
        
        self.last_recording_time = 0
        self.recording_thread = None

    def trigger_recording(self, get_frame_func, callback=None) -> bool:
        """
        Trigger a timed recording if cooldown has passed.
        
        Args:
            get_frame_func: Function that returns the current frame
            callback: Optional function to call when recording completes
            
        Returns:
            True if recording was started
        """
        current_time = time.time()
        
        if current_time - self.last_recording_time < self.cooldown:
            logger.debug("Recording cooldown active")
            return False
        
        if self.recorder.is_recording:
            logger.debug("Already recording")
            return False
        
        self.last_recording_time = current_time
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_with_callback,
            args=(get_frame_func, callback),
            daemon=True
        )
        self.recording_thread.start()
        
        return True

    def _record_with_callback(self, get_frame_func, callback):
        """Record and call callback when complete."""
        filepath = self.recorder.record_for_duration(self.duration, get_frame_func)
        
        if callback and filepath:
            try:
                callback(filepath)
            except Exception as e:
                logger.error(f"Recording callback error: {e}")
