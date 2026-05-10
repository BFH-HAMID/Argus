"""
Configuration file for AI-Powered Home Security System
Update these values according to your setup
"""

# Telegram Bot Configuration
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

# Storage Configuration
STORAGE_PATH = "./recordings"
KNOWN_FACES_PATH = "./known_faces"
MAX_STORAGE_GB = 50  # Maximum storage in GB before cleanup
AUTO_DELETE_DAYS = 7  # Delete files older than this

# Detection Thresholds
YOLO_CONFIDENCE = 0.5  # Confidence threshold for YOLO detection
FACE_RECOGNITION_TOLERANCE = 0.6  # Lower = more strict matching

# Sound Detection
SOUND_THRESHOLD = 2000  # Amplitude threshold for sound trigger
SOUND_RATE = 44100  # Audio sample rate
SOUND_CHUNK = 1024  # Audio chunk size

# Night Mode Settings
NIGHT_MODE_START = 18  # 6 PM
NIGHT_MODE_END = 6  # 6 AM
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Camera Settings
CAMERA_INDEX = 0  # Default camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Recording Settings
RECORDING_DURATION = 30  # Seconds to record after trigger
COOLDOWN_PERIOD = 10  # Seconds between alerts
