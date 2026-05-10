"""
Sound Detection Module
Monitors ambient noise and triggers alerts on loud sounds
"""

import pyaudio
import numpy as np
import threading
import time
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class SoundDetector:
    def __init__(
        self, 
        threshold: int = 2000, 
        sample_rate: int = 44100, 
        chunk_size: int = 1024,
        callback: Optional[Callable] = None
    ):
        """
        Initialize the sound detector.
        
        Args:
            threshold: Amplitude threshold to trigger alert
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
            callback: Function to call when threshold is exceeded
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.callback = callback
        
        self.audio = None
        self.stream = None
        self.running = False
        self.thread = None
        
        self.current_level = 0
        self.is_triggered = False
        self.last_trigger_time = 0
        self.cooldown = 5  # Seconds between triggers

    def start(self) -> bool:
        """Start the sound monitoring thread."""
        if self.running:
            return True
        
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            
            logger.info("Sound detector started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting sound detector: {e}")
            self._cleanup()
            return False

    def stop(self) -> None:
        """Stop the sound monitoring."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        
        self._cleanup()
        logger.info("Sound detector stopped")

    def _cleanup(self) -> None:
        """Clean up audio resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
            self.audio = None

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in a separate thread."""
        while self.running:
            try:
                if self.stream is None:
                    break
                
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate amplitude
                self.current_level = int(np.abs(audio_data).mean())
                
                # Check threshold
                current_time = time.time()
                if (self.current_level > self.threshold and 
                    current_time - self.last_trigger_time > self.cooldown):
                    
                    self.is_triggered = True
                    self.last_trigger_time = current_time
                    
                    logger.warning(f"Sound threshold exceeded! Level: {self.current_level}")
                    
                    if self.callback:
                        try:
                            self.callback(self.current_level)
                        except Exception as e:
                            logger.error(f"Sound callback error: {e}")
                else:
                    self.is_triggered = False
                
                time.sleep(0.01)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                if self.running:
                    logger.error(f"Sound monitoring error: {e}")
                break

    def get_current_level(self) -> int:
        """Get the current sound level."""
        return self.current_level

    def set_threshold(self, threshold: int) -> None:
        """Update the detection threshold."""
        self.threshold = max(0, threshold)
        logger.info(f"Sound threshold set to {self.threshold}")

    def set_callback(self, callback: Callable) -> None:
        """Set or update the trigger callback function."""
        self.callback = callback

    def get_level_percentage(self) -> float:
        """Get sound level as percentage of threshold."""
        if self.threshold == 0:
            return 0.0
        return min(100.0, (self.current_level / self.threshold) * 100)
