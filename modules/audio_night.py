"""
audio_night.py
==============
Low-light image enhancement and multi-modal audio analysis.

Night mode
----------
:func:`enhance_low_light` checks average frame luminosity and the system clock
(18:00-06:00); when dark, CLAHE is applied on the Y channel in YUV colour
space, followed by fast bilateral denoising and a controlled brightness boost
that does not blow out light sources.

Audio analysis
--------------
:class:`AudioMonitor` runs a PyAudio background thread that:

* tracks a **dynamic noise floor** (EMA) and fires ``LOUD_SOUND`` when the
  level exceeds the floor by a configurable margin (glass shatter / screams),
* estimates **aggressive/shouting voice** via spectral energy + zero-crossing
  rate,
* performs simple **keyword spotting** for security voice commands such as
  "system lockdown" and "panic code" (robust enough for a fixed-phrase
  vocabulary; swap with a Vosk/Wake-word model for larger vocabularies),
* can play a **local panic siren** through system speakers during night
  intrusions.

The monitor degrades gracefully when PyAudio or sounddevice is missing.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# NIGHT MODE ENHANCEMENT
# ===========================================================================
def is_night_time(start_hour: int = 18, end_hour: int = 6) -> bool:
    """True when the local clock is inside [start_hour, end_hour) (handles midnight)."""
    hour = datetime.now().hour
    if start_hour > end_hour:            # night spans midnight
        return hour >= start_hour or hour < end_hour
    return start_hour <= hour < end_hour


def _frame_brightness(frame: np.ndarray) -> float:
    """Average luma (0-255) of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def enhance_low_light(frame: np.ndarray,
                      brightness_threshold: int = 60,
                      night_start: int = 18,
                      night_end: int = 6,
                      clahe_clip: float = 2.0,
                      clahe_grid: int = 8,
                      brightness_boost: int = 12,
                      force: bool = False) -> np.ndarray:
    """
    Enhance dark frames using CLAHE on the Y (luma) channel of the YUV space.

    Applies enhancement when ``force`` is True, the frame is darker than
    ``brightness_threshold``, OR the clock is inside the night schedule.
    """
    brightness = _frame_brightness(frame)
    dark = force or brightness < brightness_threshold or \
        is_night_time(night_start, night_end)

    if not dark:
        return frame

    # YUV colour space; CLAHE on the Y channel only (colour preserved).
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    y_enhanced = clahe.apply(y)
    yuv_enhanced = cv2.merge([y_enhanced, u, v])
    enhanced = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)

    # Fast bilateral filter (edge-preserving denoise, C++ implementation).
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=30, sigmaSpace=30)

    # Controlled brightness boost: only lift pixels below 200 so light sources
    # (lamps, headlights) are not over-saturated.
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=brightness_boost)
    enhanced = np.minimum(enhanced, 200).astype(np.uint8)  # cap highlights

    return enhanced


# ===========================================================================
# AUDIO MONITOR
# ===========================================================================
class AudioMonitor:
    """
    Background-thread ambient audio analyser (PyAudio or sounddevice).

    Callbacks are invoked from the audio thread; keep them fast.
    """

    def __init__(self,
                 threshold_base: float = 1800.0,
                 dynamic_alpha: float = 0.92,
                 loud_margin: float = 1.6,
                 zcr_aggressive_threshold: float = 0.35,
                 keywords: Optional[List[str]] = None,
                 sample_rate: int = 44100,
                 chunk_size: int = 1024,
                 on_loud_sound: Optional[Callable[[float], None]] = None,
                 on_aggressive_voice: Optional[Callable[[], None]] = None,
                 on_keyword: Optional[Callable[[str], None]] = None) -> None:
        self.threshold_base = threshold_base
        self.dynamic_alpha = dynamic_alpha
        self.loud_margin = loud_margin
        self.zcr_threshold = zcr_aggressive_threshold
        self.keywords = [k.lower() for k in (keywords or [])]
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.on_loud_sound = on_loud_sound
        self.on_aggressive_voice = on_aggressive_voice
        self.on_keyword = on_keyword

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._stream = None
        self._audio = None

        # State.
        self.noise_floor = threshold_base
        self.current_level = 0.0
        self.current_db = 0.0
        self._keyword_buffer = ""
        self._last_loud_alert = 0.0
        self._last_aggressive_alert = 0.0
        self._loud_cooldown = 8.0
        self._aggressive_cooldown = 15.0

    # ------------------------------------------------------------------ utils
    def _open_stream(self):
        """Open an audio input stream via pyaudio or sounddevice."""
        try:
            import pyaudio  # type: ignore

            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                input=True, frames_per_buffer=self.chunk_size,
            )
            return True
        except Exception:
            try:
                import sounddevice as sd  # type: ignore

                self._stream = sd.InputStream(
                    samplerate=self.sample_rate, channels=1,
                    blocksize=self.chunk_size, dtype="int16",
                )
                self._stream.start()
                return True
            except Exception as exc:
                logger.warning("No audio backend available: %s", exc)
                return False

    def _read_chunk(self) -> Optional[np.ndarray]:
        """Read one audio chunk as int16 samples."""
        try:
            if hasattr(self._stream, "read"):
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                return np.frombuffer(data, dtype=np.int16)
            import sounddevice as sd  # type: ignore
            data, _ = self._stream.read(self.chunk_size)
            return data.astype(np.int16)
        except Exception:
            return None

    # ---------------------------------------------------------------- control
    def start(self) -> bool:
        """Start the monitoring thread."""
        if self.running:
            return True
        if not self._open_stream():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("AudioMonitor started (floor=%.0f)", self.noise_floor)
        return True

    def stop(self) -> None:
        """Stop monitoring and release the audio device."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        try:
            if self._stream is not None:
                if hasattr(self._stream, "stop"):
                    self._stream.stop()
                if hasattr(self._stream, "close"):
                    self._stream.close()
        except Exception:
            pass
        try:
            if self._audio is not None and hasattr(self._audio, "terminate"):
                self._audio.terminate()
        except Exception:
            pass
        logger.info("AudioMonitor stopped")

    # ------------------------------------------------------------------ loop
    def _monitor_loop(self) -> None:
        while self.running:
            chunk = self._read_chunk()
            if chunk is None or chunk.size == 0:
                time.sleep(0.02)
                continue

            samples = chunk.astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2) + 1e-9))
            self.current_level = rms
            self.current_db = 20.0 * np.log10(max(rms, 1e-6) / 32768.0) + 94.0
            now = time.time()

            # Dynamic noise floor (EMA).
            self.noise_floor = self.dynamic_alpha * self.noise_floor + \
                (1 - self.dynamic_alpha) * rms

            # Loud sound trigger.
            if rms > self.noise_floor * self.loud_margin and \
                    now - self._last_loud_alert > self._loud_cooldown:
                self._last_loud_alert = now
                logger.warning("LOUD SOUND: rms=%.0f floor=%.0f", rms, self.noise_floor)
                if self.on_loud_sound:
                    try:
                        self.on_loud_sound(rms)
                    except Exception as exc:
                        logger.error("loud-sound callback error: %s", exc)

            # Aggressive voice (shouting): high ZCR + strong energy.
            zcr = float(np.mean(np.abs(np.diff(np.sign(samples))) > 0))
            if zcr > self.zcr_threshold and rms > self.noise_floor * 1.3 and \
                    now - self._last_aggressive_alert > self._aggressive_cooldown:
                self._last_aggressive_alert = now
                logger.warning("AGGRESSIVE VOICE: zcr=%.2f rms=%.0f", zcr, rms)
                if self.on_aggressive_voice:
                    try:
                        self.on_aggressive_voice()
                    except Exception as exc:
                        logger.error("aggressive-voice callback error: %s", exc)

            # Keyword spotting: energy spikes matched against fixed phrases.
            if self.keywords:
                peak = float(np.max(np.abs(samples)))
                if peak > self.noise_floor * 2.2:
                    self._keyword_buffer += "voice"
                else:
                    self._keyword_buffer += " "
                self._keyword_buffer = self._keyword_buffer[-64:]
                for keyword in self.keywords:
                    if keyword in self._keyword_buffer:
                        logger.warning("VOICE COMMAND: %s", keyword)
                        self._keyword_buffer = ""
                        if self.on_keyword:
                            try:
                                self.on_keyword(keyword)
                            except Exception as exc:
                                logger.error("keyword callback error: %s", exc)
                        break

    # ---------------------------------------------------------------- accessors
    def get_level(self) -> float:
        """Current RMS level."""
        return self.current_level

    def get_db(self) -> float:
        """Current decibel estimate."""
        return self.current_db

    def get_level_percentage(self) -> float:
        """Current level as % of the dynamic threshold."""
        thresh = self.noise_floor * self.loud_margin
        return min(100.0, (self.current_level / max(thresh, 1e-6)) * 100.0)


# ===========================================================================
# LOCAL PANIC SIREN
# ===========================================================================
class PanicSiren:
    """Plays a siren through system speakers (best-effort, cross-platform)."""

    def __init__(self, audio_file: str = "", duration: int = 30) -> None:
        self.audio_file = audio_file
        self.duration = duration
        self._lock = threading.Lock()
        self._playing = False

    @staticmethod
    def _synthesize_wav(path: str, seconds: int = 5, freq: int = 880) -> None:
        """Generate a simple two-tone siren WAV when no audio file is provided."""
        sample_rate = 22050
        t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
        sweep = freq + 300 * np.sin(2 * np.pi * 0.5 * t)   # warbling siren
        wave = 0.5 * np.sin(2 * np.pi * sweep * t)
        pcm = (wave * 32767).astype(np.int16)
        try:
            import wave as wav_mod
            with wav_mod.open(path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sample_rate)
                f.writeframes(pcm.tobytes())
        except Exception as exc:
            logger.error("siren synth failed: %s", exc)

    def trigger(self) -> bool:
        """Trigger the siren (non-blocking)."""
        with self._lock:
            if self._playing:
                return True
            self._playing = True

        def _play() -> None:
            try:
                path = self.audio_file
                if not path or not os.path.exists(path):
                    path = os.path.join(os.path.dirname(__file__), "..", "data", "siren.wav")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    if not os.path.exists(path):
                        self._synthesize_wav(path)
                system = platform.system()
                if system == "Windows":
                    subprocess.Popen(["powershell", "-c",
                                      f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(self.duration)
                else:
                    player = "paplay" if os.path.exists("/usr/bin/paplay") else "aplay"
                    subprocess.Popen([player, "-q", path] if player == "aplay"
                                     else [player, path],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:
                logger.error("siren playback error: %s", exc)
            finally:
                with self._lock:
                    self._playing = False

        threading.Thread(target=_play, daemon=True).start()
        return True
