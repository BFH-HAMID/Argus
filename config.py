"""
config.py
=========
Central configuration for the Enterprise Multi-Camera CCTV AI Security System.

This module defines everything the rest of the system needs:

* ``CAMERAS``          - a list of camera descriptors (RTSP streams, zones, features)
* YOLO settings        - object detection model, confidence / IOU thresholds
* Face recognition     - known-faces folder, tolerance, detector model
* Alert settings       - Telegram bot credentials, cooldowns, escalation rules
* Storage settings     - recording root, size limits, retention policy
* Processing settings  - frame skip factor, target FPS, stream quality
* System parameters    - night-mode schedule, audio thresholds, siren behaviour

All *sensitive* values (tokens, passwords) are read from environment variables
with safe fallback defaults, so the file can be committed to git without leaking
secrets.  Create a ``.env`` file next to this module (see ``.env.example``) to
override the defaults at runtime.

Author: Argus Security Suite
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Environment loading (dotenv)
# ---------------------------------------------------------------------------
try:  # python-dotenv is a soft dependency; the system still runs without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when dotenv is missing
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

# Base directory of the security system (folder containing this file).
BASE_DIR: Path = Path(__file__).resolve().parent

# Load `.env` file if present (keys in the real environment take precedence).
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    """Read a float environment variable with a fallback default."""
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean environment variable with a fallback default."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ===========================================================================
# 1. CAMERA DEFINITIONS
# ===========================================================================
# The system supports 2 to 20+ cameras.  Add or remove dictionaries from the
# list below; the orchestration engine (main.py) spawns one worker process per
# camera automatically.  Five realistic examples are provided covering home,
# yard, road and farm use-cases.
#
# Available ``active_features`` (enable per camera):
#   object_detection, face_recognition, anti_spoofing, anpr,
#   fire_smoke, pose_fall, night_mode, zone_intrusion, loitering,
#   tailgating, tampering, audio
#
# ``active_zones``        - polygon vertices (x, y) that mark restricted zones.
# ``privacy_mask_zones``  - polygon vertices that will be blurred before any
#                           AI processing (e.g. neighbour's house on a road cam).

CAMERAS: List[Dict[str, Any]] = [
    {
        "id": "CAM-01",
        "name": "Home Front Gate",
        "type": "outdoor",                    # indoor / outdoor / road / farm
        "url": _env("CAM_01_URL", "rtsp://user:pass@192.168.1.101:554/stream1"),
        "resolution": (1920, 1080),
        "fps": 25,
        # Restricted zone: the driveway / gate area (polygon in frame coords).
        "active_zones": [
            [(420, 300), (980, 300), (980, 720), (420, 720)],
        ],
        # Blur the public street strip at the top of the frame.
        "privacy_mask_zones": [
            [(0, 0), (1920, 0), (1920, 220), (0, 220)],
        ],
        "active_features": [
            "object_detection", "face_recognition", "anti_spoofing",
            "zone_intrusion", "loitering", "tailgating", "tampering", "audio",
        ],
    },
    {
        "id": "CAM-02",
        "name": "Backyard Yard",
        "type": "outdoor",
        "url": _env("CAM_02_URL", "rtsp://user:pass@192.168.1.102:554/stream1"),
        "resolution": (2560, 1440),
        "fps": 20,
        "active_zones": [
            [(600, 400), (1400, 400), (1400, 900), (600, 900)],
        ],
        "privacy_mask_zones": [
            # Neighbour's property along the left edge.
            [(0, 300), (260, 300), (260, 900), (0, 900)],
        ],
        "active_features": [
            "object_detection", "face_recognition", "anti_spoofing",
            "fire_smoke", "night_mode", "zone_intrusion", "loitering",
            "tampering", "audio",
        ],
    },
    {
        "id": "CAM-03",
        "name": "Main Road",
        "type": "road",
        "url": _env("CAM_03_URL", "rtsp://user:pass@192.168.1.103:554/stream1"),
        "resolution": (1920, 1080),
        "fps": 30,
        "active_zones": [
            [(200, 700), (1720, 700), (1720, 1080), (200, 1080)],
        ],
        "privacy_mask_zones": [
            [(0, 0), (1920, 0), (1920, 260), (0, 260)],   # sky / neighbouring flats
        ],
        "active_features": [
            "object_detection", "anpr", "fire_smoke", "tampering",
            "zone_intrusion", "loitering",
        ],
    },
    {
        "id": "CAM-04",
        "name": "Farm Perimeter",
        "type": "farm",
        "url": _env("CAM_04_URL", "rtsp://user:pass@192.168.1.104:554/stream1"),
        "resolution": (1920, 1080),
        "fps": 20,
        "active_zones": [
            [(300, 500), (1500, 500), (1500, 1080), (300, 1080)],
        ],
        "privacy_mask_zones": [],
        "active_features": [
            "object_detection", "fire_smoke", "night_mode",
            "zone_intrusion", "loitering", "tampering",
        ],
    },
    {
        "id": "CAM-05",
        "name": "Living Room",
        "type": "indoor",
        "url": _env("CAM_05_URL", "rtsp://user:pass@192.168.1.105:554/stream1"),
        "resolution": (1280, 720),
        "fps": 25,
        "active_zones": [
            [(200, 200), (1080, 200), (1080, 720), (200, 720)],
        ],
        "privacy_mask_zones": [],
        "active_features": [
            "object_detection", "face_recognition", "anti_spoofing",
            "pose_fall", "night_mode", "audio",
        ],
    },
]


def get_camera(camera_id: str) -> Optional[Dict[str, Any]]:
    """Return the camera descriptor for ``camera_id`` or ``None`` if unknown."""
    for cam in CAMERAS:
        if cam["id"] == camera_id:
            return cam
    return None


def active_features(cam: Dict[str, Any]) -> List[str]:
    """Return the enabled feature names for a camera descriptor."""
    return list(cam.get("active_features", []))


# ===========================================================================
# 2. YOLO / VISION SETTINGS
# ===========================================================================
YOLO_MODEL_PATH: str = _env("YOLO_MODEL_PATH", "yolov8n.pt")   # or yolov8m.pt
YOLO_CONFIDENCE: float = _env_float("YOLO_CONFIDENCE", 0.5)    # det. confidence
YOLO_IOU: float = _env_float("YOLO_IOU", 0.45)                 # NMS IOU threshold
YOLO_DEVICE: str = _env("YOLO_DEVICE", "auto")                 # auto | cpu | 0

# Optional dedicated fire/smoke model (e.g. a fine-tuned yolov8n).  Leave empty
# to fall back to the built-in colour/texture heuristic detector.
FIRE_SMOKE_MODEL_PATH: str = _env("FIRE_SMOKE_MODEL_PATH", "")

# YOLO-Pose model used for fall / wall-climb detection.
POSE_MODEL_PATH: str = _env("POSE_MODEL_PATH", "yolov8n-pose.pt")
POSE_CONFIDENCE: float = _env_float("POSE_CONFIDENCE", 0.5)

# COCO class ids the system cares about.
CLASS_PERSON: int = 0
CLASS_VEHICLES: List[int] = [2, 3, 5, 7]     # car, motorcycle, bus, truck
CLASS_ANIMALS: List[int] = [16, 17, 18, 19]  # dog, horse, sheep, cow

# ===========================================================================
# 3. FACE RECOGNITION SETTINGS
# ===========================================================================
KNOWN_FACES_PATH: str = _env(
    "KNOWN_FACES_PATH", str(BASE_DIR / "known_faces")
)
FACE_RECOGNITION_TOLERANCE: float = _env_float(
    "FACE_RECOGNITION_TOLERANCE", 0.45
)
FACE_RECOGNITION_MODEL: str = _env("FACE_RECOGNITION_MODEL", "hog")  # hog | cnn
FACE_DETECT_DOWNSAMPLE: int = _env_int("FACE_DETECT_DOWNSAMPLE", 2)

# ===========================================================================
# 4. ALERT SETTINGS (Telegram)
# ===========================================================================
TELEGRAM_TOKEN: str = _env("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str = _env("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

ALERT_COOLDOWN_SECONDS: float = _env_float("ALERT_COOLDOWN_SECONDS", 10.0)
# Longer cooldown applied to escalated (multi-factor) high-priority alerts.
HIGH_PRIORITY_ESCALATION_COOLDOWN: float = _env_float(
    "HIGH_PRIORITY_ESCALATION_COOLDOWN", 60.0
)
# A visual alert is escalated only if an audio trigger happened within this
# window (multi-factor validation to suppress false alarms).
MULTI_FACTOR_WINDOW_SECONDS: float = _env_float(
    "MULTI_FACTOR_WINDOW_SECONDS", 8.0
)
TELEGRAM_POLL_TIMEOUT: int = _env_int("TELEGRAM_POLL_TIMEOUT", 25)

# ===========================================================================
# 5. STORAGE SETTINGS
# ===========================================================================
STORAGE_ROOT: str = _env("STORAGE_ROOT", str(BASE_DIR / "recordings"))
MAX_STORAGE_GB: float = _env_float("MAX_STORAGE_GB", 500.0)
CLEANUP_DAYS: int = _env_int("CLEANUP_DAYS", 7)            # max video age (days)
FREE_SPACE_THRESHOLD_PERCENT: int = _env_int(
    "FREE_SPACE_THRESHOLD_PERCENT", 80                      # purge down to 80%
)
EVENT_DB_PATH: str = _env("EVENT_DB_PATH", str(BASE_DIR / "data" / "events.db"))
STORAGE_CHECK_INTERVAL: int = _env_int("STORAGE_CHECK_INTERVAL", 300)

# Rolling pre-trigger buffer & event clip duration (seconds).
PRE_TRIGGER_BUFFER_SECONDS: int = _env_int("PRE_TRIGGER_BUFFER_SECONDS", 5)
EVENT_CLIP_SECONDS: int = _env_int("EVENT_CLIP_SECONDS", 30)

# Adaptive FPS recording: 1 FPS when idle, ramp to RECORDING_FPS on motion.
IDLE_RECORD_FPS: int = _env_int("IDLE_RECORD_FPS", 1)
MOTION_RECORD_FPS: int = _env_int("MOTION_RECORD_FPS", 30)

# ===========================================================================
# 6. PROCESSING SETTINGS
# ===========================================================================
# Process 1 out of every N frames (1 = process everything, 3 = one third).
FRAME_SKIP_FACTOR: int = _env_int("FRAME_SKIP_FACTOR", 3)
TARGET_PROCESSING_FPS: int = _env_int("TARGET_PROCESSING_FPS", 10)
RECORDING_FPS: int = _env_int("RECORDING_FPS", 30)          # capture/record FPS
STREAM_JPEG_QUALITY: int = _env_int("STREAM_JPEG_QUALITY", 70)

# ===========================================================================
# 7. SYSTEM PARAMETERS
# ===========================================================================
# Night-mode schedule (24h).
NIGHT_MODE_START_HOUR: int = _env_int("NIGHT_MODE_START_HOUR", 18)  # 6 PM
NIGHT_MODE_END_HOUR: int = _env_int("NIGHT_MODE_END_HOUR", 6)       # 6 AM
NIGHT_BRIGHTNESS_THRESHOLD: int = _env_int("NIGHT_BRIGHTNESS_THRESHOLD", 60)

# Audio analysis (see modules/audio_night.py).
SOUND_SAMPLE_RATE: int = _env_int("SOUND_SAMPLE_RATE", 44100)
SOUND_CHUNK_SIZE: int = _env_int("SOUND_CHUNK_SIZE", 1024)
SOUND_THRESHOLD_BASE: float = _env_float("SOUND_THRESHOLD_BASE", 1800.0)
SOUND_DYNAMIC_ALPHA: float = _env_float("SOUND_DYNAMIC_ALPHA", 0.92)
LOUD_SOUND_MARGIN: float = _env_float("LOUD_SOUND_MARGIN", 1.6)  # x baseline
ZCR_AGGRESSIVE_THRESHOLD: float = _env_float("ZCR_AGGRESSIVE_THRESHOLD", 0.35)
VOICE_KEYWORDS: List[str] = ["system lockdown", "panic code"]
SIREN_AUDIO_FILE: str = _env("SIREN_AUDIO_FILE", "")         # optional .wav
SIREN_DURATION_SECONDS: int = _env_int("SIREN_DURATION_SECONDS", 30)

# Behaviour analytics thresholds.
LOITERING_SECONDS: int = _env_int("LOITERING_SECONDS", 120)
TAILGATE_WINDOW_SECONDS: float = _env_float("TAILGATE_WINDOW_SECONDS", 3.0)
TAMPER_DARKNESS_THRESHOLD: int = _env_int("TAMPER_DARKNESS_THRESHOLD", 25)
TAMPER_BLUR_THRESHOLD: float = _env_float("TAMPER_BLUR_THRESHOLD", 60.0)
TAMPER_FROZEN_SECONDS: int = _env_int("TAMPER_FROZEN_SECONDS", 10)

# Dashboard / web GUI.
DASHBOARD_HOST: str = _env("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT: int = _env_int("DASHBOARD_PORT", 5000)
DASHBOARD_PASSWORD: str = _env("DASHBOARD_PASSWORD", "admin123")
DASHBOARD_AUTH_MODE: str = _env("DASHBOARD_AUTH_MODE", "session")  # session|basic

# System health / remote watchdog.
HEARTBEAT_URL: str = _env("HEARTBEAT_URL", "")   # e.g. https://status.example.com/ping
HEARTBEAT_INTERVAL: int = _env_int("HEARTBEAT_INTERVAL", 60)
GEOFENCE_ROUTER_IP: str = _env("GEOFENCE_ROUTER_IP", "192.168.1.1")
GEOFENCE_PING_INTERVAL: int = _env_int("GEOFENCE_PING_INTERVAL", 30)
GEOFENCE_MISS_THRESHOLD: int = _env_int("GEOFENCE_MISS_THRESHOLD", 3)

# Process / system-wide toggles (defaults).
ARMED_BY_DEFAULT: bool = _env_bool("ARMED_BY_DEFAULT", True)
WORKER_RESTART_BACKOFF: float = _env_float("WORKER_RESTART_BACKOFF", 15.0)

# Number of seconds to wait between Telegram retries etc.
TELEGRAM_RETRY_WAIT: float = _env_float("TELEGRAM_RETRY_WAIT", 5.0)
