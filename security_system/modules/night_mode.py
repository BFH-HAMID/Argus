"""
Night Mode Enhancement Module
Enhances low-light visibility using CLAHE and brightness adjustments
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class NightModeEnhancer:
    def __init__(
        self,
        night_start: int = 18,
        night_end: int = 6,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        """
        Initialize the night mode enhancer.
        
        Args:
            night_start: Hour when night mode starts (24h format)
            night_end: Hour when night mode ends (24h format)
            clip_limit: CLAHE clip limit for contrast enhancement
            tile_grid_size: CLAHE tile grid size
        """
        self.night_start = night_start
        self.night_end = night_end
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        
        self.force_night_mode = False
        self.auto_detect_low_light = True
        self.brightness_threshold = 50  # Average brightness below this triggers enhancement

    def is_night_time(self) -> bool:
        """Check if current time is within night hours."""
        if self.force_night_mode:
            return True
        
        current_hour = datetime.now().hour
        
        if self.night_start > self.night_end:
            # Night spans midnight (e.g., 18:00 to 06:00)
            return current_hour >= self.night_start or current_hour < self.night_end
        else:
            # Night within same day
            return self.night_start <= current_hour < self.night_end

    def is_low_light(self, frame: np.ndarray) -> bool:
        """
        Detect if the frame has low light conditions.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            True if low light is detected
        """
        # Convert to grayscale and calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < self.brightness_threshold

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply night mode enhancement to the frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Enhanced frame
        """
        # Determine if enhancement is needed
        should_enhance = self.is_night_time()
        
        if self.auto_detect_low_light and not should_enhance:
            should_enhance = self.is_low_light(frame)
        
        if not should_enhance:
            return frame
        
        return self._apply_enhancement(frame)

    def _apply_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE and brightness enhancement."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Optional: Apply slight brightness boost
        enhanced = self._adjust_brightness(enhanced, 1.1, 10)
        
        return enhanced

    def _adjust_brightness(
        self, 
        frame: np.ndarray, 
        alpha: float = 1.0, 
        beta: int = 0
    ) -> np.ndarray:
        """
        Adjust brightness and contrast.
        
        Args:
            frame: Input image
            alpha: Contrast control (1.0 = no change)
            beta: Brightness control (0 = no change)
            
        Returns:
            Adjusted frame
        """
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply denoising for night footage.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Denoised frame
        """
        return cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)

    def enhance_full(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply full enhancement pipeline (slower but better quality).
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Fully enhanced frame
        """
        enhanced = self._apply_enhancement(frame)
        denoised = self.denoise(enhanced)
        return denoised

    def set_night_hours(self, start: int, end: int) -> None:
        """Update night mode hours."""
        self.night_start = start % 24
        self.night_end = end % 24
        logger.info(f"Night mode hours set: {self.night_start}:00 - {self.night_end}:00")

    def set_force_night_mode(self, force: bool) -> None:
        """Force night mode on/off regardless of time."""
        self.force_night_mode = force
        logger.info(f"Force night mode: {'ON' if force else 'OFF'}")

    def update_clahe_params(
        self, 
        clip_limit: float = None, 
        tile_grid_size: Tuple[int, int] = None
    ) -> None:
        """Update CLAHE parameters."""
        if clip_limit is not None:
            self.clip_limit = clip_limit
        if tile_grid_size is not None:
            self.tile_grid_size = tile_grid_size
        
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        logger.info(f"CLAHE params updated: clip={self.clip_limit}, grid={self.tile_grid_size}")
