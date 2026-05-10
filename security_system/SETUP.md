# AI-Powered Home Security System - Setup Guide

## System Requirements

- **OS**: Windows 10/11 (also works on Linux/macOS)
- **Python**: 3.8 or higher
- **Camera**: USB webcam or built-in camera
- **Microphone**: For sound detection
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA (optional, for faster AI processing)

## Installation Steps

### 1. Install Python Dependencies

```bash
# Navigate to the security_system directory
cd security_system

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Install Additional System Dependencies

#### For Face Recognition (dlib)

**Windows:**
```bash
# Install Visual Studio Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select "Desktop development with C++"

# Then install cmake
pip install cmake

# Install dlib and face_recognition
pip install dlib face_recognition
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
pip install dlib face_recognition
```

**macOS:**
```bash
brew install cmake
pip install dlib face_recognition
```

#### For PyAudio (Sound Detection)

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

### 3. Setup Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the instructions
3. Copy the **API Token** provided
4. Get your **Chat ID**:
   - Search for `@userinfobot` on Telegram
   - Start the bot and it will show your Chat ID
5. Update `config.py`:
   ```python
   TELEGRAM_TOKEN = "your_bot_token_here"
   CHAT_ID = "your_chat_id_here"
   ```

### 4. Setup Known Faces Directory

1. Create the `known_faces` folder (auto-created on first run)
2. Add photos of known people:
   ```
   known_faces/
   ├── John_Doe.jpg
   ├── Jane_Smith.jpg
   ├── Family_Member.png
   └── ...
   ```
3. **Naming Convention**: Use the person's name as the filename
   - The filename (without extension) becomes the display name
   - Use underscores for spaces: `John_Doe.jpg` → "John_Doe"
4. **Photo Requirements**:
   - Clear, front-facing photo
   - Good lighting
   - One face per image
   - Supported formats: JPG, JPEG, PNG, BMP

### 5. Configure Settings

Edit `config.py` to customize:

```python
# Telegram Bot Configuration
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

# Storage Configuration
STORAGE_PATH = "./recordings"      # Where recordings are saved
KNOWN_FACES_PATH = "./known_faces" # Where known face photos are stored
MAX_STORAGE_GB = 50                # Auto-cleanup when exceeded
AUTO_DELETE_DAYS = 7               # Delete recordings older than this

# Detection Thresholds
YOLO_CONFIDENCE = 0.5              # Lower = more detections, more false positives
FACE_RECOGNITION_TOLERANCE = 0.6   # Lower = stricter matching

# Sound Detection
SOUND_THRESHOLD = 2000             # Adjust based on your environment

# Night Mode Settings
NIGHT_MODE_START = 18              # 6 PM
NIGHT_MODE_END = 6                 # 6 AM

# Camera Settings
CAMERA_INDEX = 0                   # Change if you have multiple cameras
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
```

## Running the System

```bash
# Make sure virtual environment is activated
cd security_system
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `R` | Start/Stop manual recording |
| `S` | Take a snapshot |
| `N` | Toggle night mode |
| `F` | Reload known faces |

## Directory Structure

```
security_system/
├── main.py                    # Main application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── SETUP.md                   # This file
├── security_system.log        # Application logs
├── modules/
│   ├── __init__.py
│   ├── object_detection.py    # YOLOv8 detection
│   ├── face_recognition_module.py
│   ├── sound_detection.py
│   ├── telegram_alert.py
│   ├── night_mode.py
│   ├── storage_manager.py
│   └── video_recorder.py
├── known_faces/               # Add known face photos here
│   ├── Person1.jpg
│   └── Person2.jpg
└── recordings/                # Recordings and snapshots saved here
    ├── recording_20240115_120000.mp4
    └── snapshot_20240115_120030.jpg
```

## Troubleshooting

### Camera Not Found
- Check if camera is properly connected
- Try different `CAMERA_INDEX` values (0, 1, 2...)
- Ensure no other application is using the camera

### Face Recognition Not Working
- Ensure known_faces folder has clear photos
- Check that faces are visible and well-lit in photos
- Try adjusting `FACE_RECOGNITION_TOLERANCE` (lower = stricter)

### Sound Detection Issues
- Check microphone permissions
- Adjust `SOUND_THRESHOLD` based on your environment
- Test with `python -c "import pyaudio; print('OK')"`

### Telegram Alerts Not Sending
- Verify `TELEGRAM_TOKEN` and `CHAT_ID` are correct
- Ensure internet connection is active
- Start a conversation with your bot first (send `/start`)

### High CPU Usage
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT`
- Lower `FPS` value
- Increase detection interval in code

### GPU Acceleration (NVIDIA)
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Security Notes

- Keep your `config.py` secure (contains API tokens)
- Don't share your Telegram bot token
- Regularly review and delete old recordings
- Consider network security if accessing remotely

## License

This software is provided for personal/educational use.
