# Argus — Enterprise Multi-Camera CCTV AI Security System

**Argus** is a production-oriented, multi-process CCTV AI security platform that
handles **2 to 20+ RTSP cameras** across home, yard, road and farm sites. Every
camera runs in its own isolated `multiprocessing` process with per-camera AI
features, Telegram alerts, event recording, and a real-time web dashboard.

> Full rewrite of the original single-camera `CCCamera-FULL-SECURITY` project.

---

## ✨ Features

### Vision pipeline (`modules/vision_pipeline.py`)
- **YOLOv8 detection** — persons (red), vehicles (orange), farm animals (yellow)
- **Face recognition** — known vs `STRANGER` (green/red boxes, 128-d encodings)
- **Anti-spoofing** — mask/cap concealment + photo/screen (no-blink) detection
- **ANPR** — licence-plate OCR on vehicle crops (pytesseract)
- **Fire & smoke** — dedicated YOLO sub-model or colour/texture heuristic
- **Fall & pose** — fallen-body and wall-climb detection (YOLO-Pose)
- **Privacy masks** — polygon regions blurred *before* any AI processing
- **CUDA auto-detection** and per-camera feature toggling

### Behaviour analytics (`modules/behavior_tracking.py`)
- Virtual tripline **zone intrusion** (polygon containment)
- **Loitering** (IoU tracker dwell time > configurable seconds)
- **Tailgating** (stranger within 3 s behind a recognised person)
- **Camera tampering** (lens covered / blur / frozen frames)
- **Cross-camera Re-ID** stub correlating embeddings into persistent person IDs

### Night mode & audio (`modules/audio_night.py`)
- CLAHE-on-Y (YUV) low-light enhancement with denoise & highlight capping
- Dynamic-threshold **loud sound** (glass/scream), **aggressive voice** (ZCR),
  keyword commands ("system lockdown", "panic code") and a **local panic siren**

### Alerts (`modules/alert_manager.py`)
- Queue-backed async sender — video loop never blocks on network
- **Multi-factor validation** (visual + audio/zone) suppresses false alarms
- Snapshots, 30 s MP4 clips, dynamic per-camera cooldowns (10 s / 60 s escalated)
- Interactive bot: `/status`, `/reload_faces`, `/arm`, `/disarm`, and
  **photo upload → auto-save into `known_faces/` + re-index**

### Recording & storage (`modules/storage_engine.py`)
- Adaptive-FPS recorder (1 FPS idle → 30 FPS on motion) with a **5 s pre-trigger
  buffer** flushed into event clips
- Disk maintenance: purge > 7 days, cap at 500 GB (delete oldest to 80 %)
- **SQLite event search** (`events.db`) — query by date range / camera / event type

### System health (`modules/system_health.py`)
- 60 s **heartbeat** to an external watchdog URL (offline detection)
- Wi-Fi **geofencing** → auto ARMED/DISARMED when you leave/return home
- UPS **power-saver** mode (drops ANPR/live-encoding, keeps intruder detection)
- Dynamic JPEG quality when upload latency is high
- **Daily recap** at 00:00 posted to Telegram (APScheduler)

### Dashboard (`modules/dashboard_server.py`)
- MJPEG live feeds per camera (`/video_feed/<cam_id>`)
- Auto-grid UI (2×2 → 5×4) with FPS / REC / NIGHT / load overlays
- Control panel: record, snapshot, night-mode toggle, panic siren
- Live event-log sidebar via **SSE**
- Password protection (session cookies or Basic Auth)

---

## 🚀 Quick start

```bash
# 1. Install dependencies (Python 3.9+)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure (copy & edit)
cp .env.example .env                # fill in Telegram token, RTSP URLs

# 3. Add known faces (optional)
mkdir -p known_faces                # name photos: John_Doe.jpg  (name = filename)

# 4. Run
python main.py
```

Open the dashboard at **http://localhost:5000** (default password `admin123`,
change it via `DASHBOARD_PASSWORD` in `.env`).

> GPU users: install a CUDA-enabled torch first, e.g.
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Command line

| Flag | Effect |
|------|--------|
| `--no-dashboard` | skip the web GUI |
| `--no-health`    | disable heartbeat / geofence / power-saver / recap |
| `--smoke`        | run a hardware-free self-test and exit |

---

## 🧩 Architecture

```
main.py ── orchestrator (parent process)
 ├── mp.Process per camera ── camera_worker()
 │     ├── RTSP capture (CAP_PROP_BUFFERSIZE=1, reconnect w/ backoff)
 │     ├── vision_pipeline   (YOLO / faces / ANPR / fire / pose / spoof)
 │     ├── behavior_tracking (zones / loitering / tailgating / tamper / re-id)
 │     ├── audio_night       (loud sound / voice / siren)
 │     ├── storage_engine    (pre-trigger recorder + events.db)
 │     └── out_queue ───────▶ dashboard MJPEG
 ├── TelegramDispatcher (queue + interactive bot, multi-factor validation)
 ├── DashboardServer (Flask: MJPEG grid, SSE log, controls)
 ├── SystemHealth (heartbeat, geofence, power saver, daily recap)
 ├── StorageManager (cleanup / retention)
 └── supervisor threads (restart crashed workers, stats, status, logs)
```

Key configuration lives in **`config.py`** — add/remove cameras in the `CAMERAS`
list; all secrets come from environment variables (`.env`) with safe defaults.

---

## 📁 Project layout

```
├── main.py                  # orchestration engine (spawn workers, supervise)
├── config.py                # cameras, YOLO/face/alert/storage/processing config
├── .env.example             # copy to .env and fill in secrets
├── requirements.txt
├── SETUP.md                 # detailed install guide (Windows/Linux/macOS)
└── modules/
    ├── vision_pipeline.py   # YOLOv8 / faces / anti-spoof / ANPR / fire / pose
    ├── behavior_tracking.py # zones, loitering, tailgating, tamper, re-id
    ├── audio_night.py       # night-mode CLAHE + audio monitor + siren
    ├── alert_manager.py     # async Telegram alerts + interactive bot
    ├── storage_engine.py    # adaptive recorder + storage manager + event DB
    ├── system_health.py     # heartbeat, geofence, power, bitrate, recap
    └── dashboard_server.py  # Flask MJPEG grid + SSE log + controls
```

---

## ⚠️ Notes & limitations

- **Dependencies are heavy**: `face_recognition`/`dlib` need C++ build tools, and
  `ultralytics` pulls in PyTorch. See `SETUP.md` for per-OS instructions.
- **ANPR** requires the system `tesseract` binary, **audio** needs a working
  PyAudio/sounddevice backend.
- Face anti-spoofing is a *heuristic* layer (blink + texture analysis) — an IR
  depth camera or a dedicated liveness model is recommended for full protection.
- This software is provided for personal/educational use.
