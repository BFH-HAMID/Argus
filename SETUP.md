# Argus Security Suite — Setup Guide

Enterprise Multi-Camera CCTV AI Security System (2–20+ RTSP cameras).

## System requirements

- **OS**: Windows 10/11, Linux (Ubuntu/Debian), macOS
- **Python**: 3.9 or higher
- **Camera**: IP cameras with RTSP streams (H.264 recommended)
- **Microphone**: for sound/voice detection
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: NVIDIA with CUDA (optional, speeds up AI inference)

## 1. Install Python dependencies

```bash
cd Argus

# Create & activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

## 2. Platform extras

### Face recognition (dlib)
**Windows**
```bash
# Install "Desktop development with C++" from Visual Studio Build Tools first
pip install cmake dlib face_recognition
```

**Linux**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
pip install dlib face_recognition
```

**macOS**
```bash
brew install cmake
pip install dlib face_recognition
```

### PyAudio (sound detection)
**Windows**
```bash
pip install pipwin
pipwin install pyaudio
```

**Linux**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS**
```bash
brew install portaudio
pip install pyaudio
```

### Tesseract (ANPR licence plates)
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS
brew install tesseract
```

### GPU acceleration (optional, NVIDIA)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Set `YOLO_DEVICE=auto` in `.env` (auto-detected) or `0` for the first GPU.

## 3. Telegram bot setup

1. Open Telegram, search for **@BotFather**, send `/newbot`, follow instructions.
2. Copy the **API token**.
3. Search for **@userinfobot** and send `/start` to get your **Chat ID**.
4. Edit `.env`:
   ```
   TELEGRAM_TOKEN=123456:ABC-DEF...
   TELEGRAM_CHAT_ID=123456789
   ```
5. Send `/start` to your bot so it can message you.

### Interactive bot commands
| Command | Action |
|---------|--------|
| `/status` | System health overview (workers alive, events, armed state) |
| `/reload_faces` | Re-index `known_faces/` in all workers |
| `/arm` | Arm the system |
| `/disarm` | Disarm the system |
| *photo upload* | Save the photo into `known_faces/` (caption = person name) & re-index |

## 4. Configure cameras & `.env`

```bash
cp .env.example .env
# then edit .env: camera RTSP urls, telegram token, thresholds...
```

Edit `config.py` to add/remove cameras in the `CAMERAS` list:

```python
{
    "id": "CAM-06",
    "name": "Garage Door",
    "type": "outdoor",
    "url": "rtsp://user:pass@192.168.1.106:554/stream1",
    "resolution": (1920, 1080),
    "fps": 25,
    "active_zones": [[(400, 300), (1500, 300), (1500, 1080), (400, 1080)]],
    "privacy_mask_zones": [],
    "active_features": ["object_detection", "face_recognition",
                        "zone_intrusion", "tampering", "audio"],
},
```

Available `active_features`: `object_detection`, `face_recognition`,
`anti_spoofing`, `anpr`, `fire_smoke`, `pose_fall`, `night_mode`,
`zone_intrusion`, `loitering`, `tailgating`, `tampering`, `audio`.

## 5. Known faces

```bash
mkdir -p known_faces
```
Drop a clear front-facing photo per person into the folder; the filename
(without extension) becomes the person's name: `John_Doe.jpg` → “John_Doe”.

## 6. Run

```bash
python main.py
```

Open the web dashboard: **http://localhost:5000**
(default login password `admin123` — change `DASHBOARD_PASSWORD` in `.env`).

Useful flags: `python main.py --no-dashboard`, `python main.py --no-health`,
`python main.py --smoke` (hardware-free self-test).

## 7. Directory structure

```
Argus/
├── main.py                  # orchestrator: spawns 1 process per camera
├── config.py                # all configuration
├── .env                     # secrets (git-ignored)
├── known_faces/             # known-person photos (name.jpg)
├── recordings/<CAM_ID>/     # event clips & snapshots
├── data/events.db           # SQLite event index
├── security_system.log
└── modules/
    ├── vision_pipeline.py   # YOLO / faces / ANPR / fire / pose / spoof
    ├── behavior_tracking.py # zones, loitering, tailgating, tamper, re-id
    ├── audio_night.py       # night-mode + sound/voice/siren
    ├── alert_manager.py     # Telegram alerts + bot
    ├── storage_engine.py    # recorder + cleanup + event search
    ├── system_health.py     # heartbeat / geofence / power / recap
    └── dashboard_server.py  # Flask web GUI
```

## 8. Troubleshooting

- **No camera frames**: verify the RTSP URL with VLC; check `CAMERA_INDEX` style
  URLs are correct; the worker auto-reconnects with backoff, watch
  `security_system.log`.
- **Face recognition fails**: use clear, well-lit, front-facing photos; lower
  `FACE_RECOGNITION_TOLERANCE` for stricter matching.
- **No sound triggers**: check mic permissions; adjust `SOUND_THRESHOLD_BASE`.
- **Telegram not sending**: verify token/chat-id; message your bot `/start` once;
  check internet.
- **ANPR empty**: install tesseract (section 2).
- **High CPU**: increase `FRAME_SKIP_FACTOR`, lower `TARGET_PROCESSING_FPS`,
  use smaller `YOLO_MODEL_PATH` (`yolov8n.pt`).
- **Spawn errors on Windows**: run `python main.py` from the project folder
  (never from an interactive REPL).

## Security notes

- Keep `.env` private — never commit tokens.
- Change the dashboard password before exposing the box to a network.
- Optionally point `HEARTBEAT_URL` at an external watchdog to detect power/network
  loss remotely.
- Provided for personal/educational use.
