<div align="center">

<!-- ═══════════════ ANIMATED HEADER ═══════════════ -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=ARGUS&fontSize=90&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Enterprise%20Multi-Camera%20CCTV%20AI%20Security%20System&descAlignY=62&descSize=20" alt="ARGUS header"/>

<!-- ═══════════════ TYPING ANIMATION ═══════════════ -->
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=22D3EE&center=true&vCenter=true&width=800&lines=👁+2+to+20%2B+RTSP+cameras%2C+one+process+each;🧠+YOLOv8+%2B+Face+Recognition+%2B+ANPR+%2B+Anti-Spoofing;🔥+Fire+%26+Smoke+%2B+Fall+Detection+%2B+Night+Vision;📲+Telegram+alerts+with+video+clips+%26+snapshots;📊+Real-time+web+dashboard+with+live+event+log" alt="Typing animation"/>

<br/>

<!-- ═══════════════ BADGES ═══════════════ -->
<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+"/>
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8"/>
<img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
<img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
<img src="https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram"/>
<br/>
<img src="https://img.shields.io/badge/CUDA-Auto_Detect-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA"/>
<img src="https://img.shields.io/badge/Cameras-2_to_20%2B-red?style=flat-square&logo=cctv&logoColor=white" alt="Cameras"/>
<img src="https://img.shields.io/badge/Platform-Windows_|_Linux_|_macOS-blue?style=flat-square" alt="Platform"/>
<img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs welcome"/>
<img src="https://img.shields.io/badge/Maintained-yes-green?style=flat-square" alt="Maintained"/>

</div>

---

<div align="center">

![Argus Banner](assets/banner.png)

*🎯 One process per camera · Zero blocked frames · Production-oriented from day one*

> A full rewrite of the original single-camera `CCCamera-FULL-SECURITY` project —
> rebuilt for **scale, isolation and reliability**.

</div>

---

## 📖 Table of Contents

- [✨ Why Argus?](#-why-argus)
- [🧠 AI Feature Matrix](#-ai-feature-matrix)
- [🖥️ Dashboard Preview](#️-dashboard-preview)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [🎥 Camera Coverage](#-camera-coverage)
- [🤖 Telegram Bot](#-telegram-bot)
- [⚙️ Configuration](#️-configuration)
- [📁 Project Layout](#-project-layout)
- [📊 Performance & Scaling](#-performance--scaling)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [⚠️ Notes & Limitations](#️-notes--limitations)

---

## ✨ Why Argus?

<table>
<tr>
<td width="33%" align="center">
<h3>🔒 Isolated</h3>
Every camera runs in its <b>own OS process</b>. One camera crashes? The supervisor <b>auto-restarts</b> it — the other 19 never blink.
</td>
<td width="33%" align="center">
<h3>⚡ Non-blocking</h3>
A <b>queue-backed alert engine</b> with multi-factor validation means the video loop <b>never waits on the network</b>. Ever.
</td>
<td width="33%" align="center">
<h3>🌙 24/7 Aware</h3>
CLAHE night vision, audio sensing, UPS power-saver, Wi-Fi geofencing and a <b>midnight daily recap</b> — it watches while you sleep.
</td>
</tr>
</table>

<details>
<summary>🎬 <b>What happens when an intruder is detected? (click to expand)</b></summary>
<br/>

```mermaid
sequenceDiagram
    autonumber
    participant 📷 as 📷 Camera Worker
    participant 🧠 as 🧠 Vision Pipeline
    participant 🎧 as 🎧 Audio Monitor
    participant ✅ as ✅ Multi-Factor Check
    participant 📲 as 📲 Telegram
    participant 💾 as 💾 Recorder

    📷->>🧠: frame (privacy-masked)
    🧠->>🧠: YOLO + face + zone check
    🧠-->>✅: 🚨 STRANGER in restricted zone
    🎧-->>✅: 🔊 loud sound detected
    ✅->>✅: visual + audio within 8 s window?
    ✅->>📲: 📸 snapshot + 🎬 30 s clip (5 s pre-buffer!)
    ✅->>💾: index event in SQLite
    📲-->>📷: user taps /status → all-clear ✅
```

</details>

---

## 🧠 AI Feature Matrix

### 👁️ Vision Pipeline — `modules/vision_pipeline.py`

| Feature | What it does | Visual |
|---|---|---|
| 🧍 **YOLOv8 detection** | Persons, vehicles, farm animals | 🔴 persons · 🟠 vehicles · 🟡 animals |
| 🙂 **Face recognition** | Known vs `STRANGER` (128-d encodings) | 🟢 known · 🔴 stranger |
| 🎭 **Anti-spoofing** | Mask/cap concealment + photo/screen (no-blink) detection | ⚠️ spoof alert |
| 🔢 **ANPR** | Licence-plate OCR on vehicle crops (`pytesseract`) | 🚗 `ABC-1234` |
| 🔥 **Fire & smoke** | Dedicated YOLO sub-model or colour/texture heuristic fallback | 🔥 fire box |
| 🧎 **Fall & pose** | Fallen-body and wall-climb detection (YOLO-Pose) | 🆘 fall alert |
| ⬛ **Privacy masks** | Polygon regions blurred *before* any AI processing | 🌫️ neighbour-safe |
| 🎛️ **CUDA auto-detect** | GPU when available, CPU fallback + per-camera toggles | ⚡ fast path |

### 🕵️ Behaviour Analytics — `modules/behavior_tracking.py`

![zone](https://img.shields.io/badge/🚧-Zone_Intrusion-e11d48?style=flat-square)
![loiter](https://img.shields.io/badge/⏱️-Loitering-f59e0b?style=flat-square)
![tailgate](https://img.shields.io/badge/👥👤-Tailgating-8b5cf6?style=flat-square)
![tamper](https://img.shields.io/badge/📷-Tamper_Detect-0ea5e9?style=flat-square)
![reid](https://img.shields.io/badge/🔗-Cross_Cam_Re--ID-10b981?style=flat-square)

- **🚧 Virtual tripline zone intrusion** — polygon containment per camera
- **⏱️ Loitering** — IoU tracker dwell time > configurable seconds
- **👥👤 Tailgating** — stranger within 3 s behind a recognised person
- **📷 Camera tampering** — lens covered / blur / frozen-frame detection
- **🔗 Cross-camera Re-ID** — stub correlating embeddings into persistent person IDs

### 🌙 Night Mode & Audio — `modules/audio_night.py`

![night](https://img.shields.io/badge/🌙-CLAHE_Night_Vision-1e1b4b?style=flat-square&logoColor=white)
![audio](https://img.shields.io/badge/🔊-Sound_Detection-dc2626?style=flat-square)
![voice](https://img.shields.io/badge/🎙️-Voice_Analysis-7c3aed?style=flat-square)
![siren](https://img.shields.io/badge/🚨-Panic_Siren-f97316?style=flat-square)

- **🌙 CLAHE-on-Y (YUV)** low-light enhancement with denoise & highlight capping
- **🔊 Dynamic-threshold loud-sound** detection (glass break / scream)
- **🎙️ Aggressive voice** (ZCR analysis) + keyword commands (`"system lockdown"`, `"panic code"`)
- **🚨 Local panic siren** triggerable from dashboard or bot

### 📲 Alerts — `modules/alert_manager.py`

- **⚡ Queue-backed async sender** — video loop never blocks on network
- **✅ Multi-factor validation** (visual + audio/zone) suppresses false alarms
- **📸 Snapshots + 🎬 30 s MP4 clips**, dynamic per-camera cooldowns (10 s / 60 s escalated)
- **🤖 Interactive bot** — `/status`, `/reload_faces`, `/arm`, `/disarm`, and **photo upload → auto-save into `known_faces/` + re-index** 🪄

### 💾 Recording & Storage — `modules/storage_engine.py`

- **🎞️ Adaptive-FPS recorder** — `1 FPS` idle → `30 FPS` on motion, with a **5 s pre-trigger buffer** flushed into event clips
- **🧹 Disk maintenance** — purge > 7 days, cap at 500 GB (delete oldest down to 80 %)
- **🔍 SQLite event search** (`events.db`) — query by date range / camera / event type

### ❤️ System Health — `modules/system_health.py`

- **💓 60 s heartbeat** to an external watchdog URL (offline detection)
- **📶 Wi-Fi geofencing** → auto ARMED/DISARMED when you leave/return home
- **🔋 UPS power-saver** mode (drops ANPR/live-encoding, keeps intruder detection)
- **🎚️ Dynamic JPEG quality** when upload latency is high
- **📰 Daily recap** at 00:00 posted to Telegram (APScheduler)

---

## 🖥️ Dashboard Preview

<div align="center">

![Dashboard Preview](assets/dashboard-preview.png)

*🧩 Auto-grid UI (2×2 → 5×4) · FPS / REC / NIGHT overlays · SSE live event log · password-protected*

</div>

| Endpoint | Description |
|---|---|
| `GET /` | 🔲 Auto-grid dashboard (2×2 → 5×4) with FPS / REC / NIGHT / load overlays |
| `GET /video_feed/<cam_id>` | 📹 MJPEG live feed per camera |
| `GET /events` | 📡 Live event-log sidebar via **SSE** |
| `POST /control/<action>` | 🎛️ Record · snapshot · night-mode toggle · panic siren |

---

## 🏗️ Architecture

### 🔌 System Overview

```mermaid
flowchart TB
    subgraph ORCH["🖥️ main.py — Orchestrator (parent process)"]
        direction TB
        SUP["👮 Supervisor threads<br/>restart crashed workers · stats · logs"]
    end

    subgraph CAMS["📷 One mp.Process per camera"]
        direction LR
        W1["🎥 camera_worker<br/>CAM-01"]
        W2["🎥 camera_worker<br/>CAM-02"]
        WD["⋯"]
        WN["🎥 camera_worker<br/>CAM-N"]
    end

    subgraph PIPE["🧠 Per-worker pipeline"]
        direction TB
        CAP["📡 RTSP capture<br/>buffer=1 · reconnect w/ backoff"]
        VIS["👁️ vision_pipeline<br/>YOLO · faces · ANPR · fire · pose · spoof"]
        BEH["🕵️ behavior_tracking<br/>zones · loitering · tailgating · tamper"]
        AUD["🎧 audio_night<br/>loud sound · voice · siren"]
        STO["💾 storage_engine<br/>pre-trigger recorder + events.db"]
        CAP --> VIS --> BEH --> STO
        AUD -.-> STO
    end

    TG["📲 TelegramDispatcher<br/>queue + interactive bot"]
    WEB["🌐 DashboardServer<br/>Flask MJPEG grid · SSE log"]
    HLT["❤️ SystemHealth<br/>heartbeat · geofence · power · recap"]
    MGR["🧹 StorageManager<br/>cleanup · retention"]

    ORCH --> CAMS
    W1 -.-> PIPE
    PIPE -- "out_queue 🎞️" --> WEB
    PIPE -- "alert queue 🚨" --> TG
    HLT -.-> ORCH
    MGR -.-> STO

    style ORCH fill:#1e1b4b,stroke:#22d3ee,color:#fff
    style CAMS fill:#3f1d1d,stroke:#f87171,color:#fff
    style PIPE fill:#14291f,stroke:#34d399,color:#fff
    style TG fill:#0c2a4a,stroke:#38bdf8,color:#fff
    style WEB fill:#2a1a3f,stroke:#c084fc,color:#fff
```

### 🗂️ Classic Tree View

<details>
<summary><b>Click to expand the process tree</b></summary>

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

</details>

---

## 🚀 Quick Start

### 1️⃣ Install

```bash
# Clone the repo
git clone https://github.com/BFH-HAMID/Argus.git
cd Argus

# Install dependencies (Python 3.9+)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> 🖥️ **GPU users:** install a CUDA-enabled torch first, e.g.
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### 2️⃣ Configure

```bash
# Copy & edit — fill in Telegram token, RTSP URLs
cp .env.example .env
```

### 3️⃣ Add known faces *(optional)*

```bash
mkdir -p known_faces    # name photos: John_Doe.jpg  (name = filename)
```

### 4️⃣ Run 🎉

```bash
python main.py
```

<div align="center">

### 🎊 That's it! Open the dashboard at **http://localhost:5000**

*Default password `admin123` — change it via `DASHBOARD_PASSWORD` in `.env` 🔐*

</div>

### 🚩 Command-line Flags

| Flag | Effect |
|------|--------|
| `--no-dashboard` | 🚫 Skip the web GUI |
| `--no-health` | 🚫 Disable heartbeat / geofence / power-saver / recap |
| `--smoke` | 🧪 Run a hardware-free self-test and exit |

> 📚 **Need detailed per-OS instructions?** See **[SETUP.md](SETUP.md)** — Windows / Linux / macOS walkthroughs, Telegram bot creation, troubleshooting & more.

---

## 🎥 Camera Coverage

Ships with **5 realistic presets** — home, yard, road and farm. Add up to **20+** by extending the `CAMERAS` list in [`config.py`](config.py):

| ID | 📍 Location | 🏷️ Type | 🎯 Active Features |
|----|-------------|---------|-------------------|
| `CAM-01` | 🏠 Home Front Gate | outdoor | detection · faces · spoof · zones · loitering · tailgating · tamper · audio |
| `CAM-02` | 🌳 Backyard Yard | outdoor | detection · faces · spoof · 🔥 fire · 🌙 night · zones · loitering · tamper · audio |
| `CAM-03` | 🛣️ Main Road | road | detection · 🔢 ANPR · 🔥 fire · tamper · zones · loitering |
| `CAM-04` | 🚜 Farm Perimeter | farm | detection · 🔥 fire · 🌙 night · zones · loitering · tamper |
| `CAM-05` | 🛋️ Living Room | indoor | detection · faces · spoof · 🧎 pose/fall · 🌙 night · audio |

<details>
<summary><b>➕ How to add a camera (click to expand)</b></summary>
<br/>

Drop a dict into the `CAMERAS` list — a worker process spawns automatically:

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

Available features: `object_detection` · `face_recognition` · `anti_spoofing` ·
`anpr` · `fire_smoke` · `pose_fall` · `night_mode` · `zone_intrusion` ·
`loitering` · `tailgating` · `tampering` · `audio`

</details>

---

## 🤖 Telegram Bot

<div align="center">

![telegram](https://img.shields.io/badge/📲-Real_Time_Alerts-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)

</div>

| Command | Action |
|---------|--------|
| `/status` | 💓 System health overview (workers alive, events, armed state) |
| `/reload_faces` | 🔄 Re-index `known_faces/` in all workers |
| `/arm` | 🔒 Arm the system |
| `/disarm` | 🔓 Disarm the system |
| 📷 *photo upload* | 🪄 Save into `known_faces/` (caption = person name) & re-index |

```mermaid
flowchart LR
    E["🚨 Event detected"] --> Q["📥 Alert queue"]
    Q --> V{"✅ Multi-factor<br/>validation"}
    V -- "single factor" --> C["⏳ 10 s cooldown"]
    V -- "visual + audio/zone<br/>within 8 s" --> H["🔺 Escalated<br/>60 s cooldown"]
    C --> S["📲 Telegram:<br/>snapshot + 30 s clip"]
    H --> S
    style V fill:#3f1d1d,stroke:#f87171,color:#fff
    style H fill:#3f1d1d,stroke:#f87171,color:#fff
    style S fill:#0c2a4a,stroke:#38bdf8,color:#fff
```

---

## ⚙️ Configuration

All secrets come from **environment variables** (`.env`) with safe defaults — `config.py` is git-safe to commit. 🔐

| Area | Key settings |
|------|--------------|
| 🎥 Cameras | `CAM_01_URL` … `CAM_0N_URL` — one RTSP URL per camera |
| 🧠 YOLO | `YOLO_MODEL_PATH` · `YOLO_CONFIDENCE` · `YOLO_IOU` · `YOLO_DEVICE` (`auto`/`cpu`/`0`) |
| 🙂 Faces | `KNOWN_FACES_PATH` · `FACE_RECOGNITION_TOLERANCE` · `FACE_RECOGNITION_MODEL` (`hog`/`cnn`) |
| 📲 Alerts | `TELEGRAM_TOKEN` · `TELEGRAM_CHAT_ID` · `ALERT_COOLDOWN_SECONDS` · `MULTI_FACTOR_WINDOW_SECONDS` |
| 💾 Storage | `STORAGE_ROOT` · `MAX_STORAGE_GB` · `CLEANUP_DAYS` · `PRE_TRIGGER_BUFFER_SECONDS` · `EVENT_CLIP_SECONDS` |
| 🎞️ Processing | `FRAME_SKIP_FACTOR` · `TARGET_PROCESSING_FPS` · `RECORDING_FPS` · `STREAM_JPEG_QUALITY` |
| 🌙 System | `NIGHT_MODE_START_HOUR` · `SOUND_THRESHOLD_BASE` · `LOITERING_SECONDS` · `DASHBOARD_PASSWORD` · `HEARTBEAT_URL` |

> 📝 Full reference with defaults lives in [`config.py`](config.py) — every knob is documented inline.

---

## 📁 Project Layout

```
Argus/
├── 🖼️  assets/                  # README banner & dashboard artwork
├── 🧠  main.py                  # orchestration engine (spawn workers, supervise)
├── ⚙️  config.py                 # cameras, YOLO/face/alert/storage/processing config
├── 🔐  .env.example             # copy to .env and fill in secrets
├── 📦  requirements.txt
├── 📚  SETUP.md                 # detailed install guide (Windows/Linux/macOS)
└── 📂  modules/
    ├── 👁️  vision_pipeline.py   # YOLOv8 / faces / anti-spoof / ANPR / fire / pose
    ├── 🕵️  behavior_tracking.py # zones, loitering, tailgating, tamper, re-id
    ├── 🌙  audio_night.py       # night-mode CLAHE + audio monitor + siren
    ├── 📲  alert_manager.py     # async Telegram alerts + interactive bot
    ├── 💾  storage_engine.py    # adaptive recorder + storage manager + event DB
    ├── ❤️  system_health.py     # heartbeat, geofence, power, bitrate, recap
    └── 🌐  dashboard_server.py  # Flask MJPEG grid + SSE log + controls
```

---

## 📊 Performance & Scaling

| 🖥️ Setup | 📷 Cameras | 🧠 Inference | 🎞️ Recording | 💡 Notes |
|----------|-----------|--------------|--------------|---------|
| 🟢 CPU only (8 GB RAM) | 2–4 @ 720p | `yolov8n.pt` + `FRAME_SKIP_FACTOR=3` | 1→30 FPS adaptive | Raise frame-skip if CPU > 80 % |
| 🟡 CPU (16 GB RAM) | 4–8 @ 1080p | `yolov8n.pt`, tuned FPS | 1→30 FPS adaptive | Recommended minimum for multi-cam |
| 🔴 NVIDIA CUDA GPU | **20+** @ 1080p+ | `yolov8m.pt`, `YOLO_DEVICE=0` | Full 30 FPS | One process/camera scales linearly |

---

## 🗺️ Roadmap

- [x] ✅ Multi-process per-camera isolation with supervisor restarts
- [x] ✅ Telegram alerts with snapshots, clips & interactive bot
- [x] ✅ MJPEG dashboard grid with SSE event log
- [x] ✅ Night mode, audio sensing, geofencing, UPS power-saver
- [ ] 🔲 Web UI zone editor (draw polygons in the browser)
- [ ] 🔲 Full cross-camera Re-ID with persistent person gallery
- [ ] 🔲 ONVIF auto-discovery for plug-and-play camera setup
- [ ] 🔲 Docker image + `docker-compose` one-liner deploy
- [ ] 🔲 Mobile-friendly PWA dashboard

---

## 🤝 Contributing

Contributions are welcome! 🎉

```bash
# Fork, then:
git checkout -b feature/my-cool-detector
# ... make your changes ...
git commit -m "feat: add my cool detector"
git push origin feature/my-cool-detector
# ... open a Pull Request 🚀
```

Please keep per-camera features toggleable via `active_features` and avoid blocking the video loop. 🙏

---

## ⚠️ Notes & Limitations

- **🏋️ Dependencies are heavy**: `face_recognition`/`dlib` need C++ build tools, and `ultralytics` pulls in PyTorch. See [`SETUP.md`](SETUP.md) for per-OS instructions.
- **🔢 ANPR** requires the system `tesseract` binary; **🎧 audio** needs a working PyAudio/sounddevice backend.
- **🎭 Face anti-spoofing** is a *heuristic* layer (blink + texture analysis) — an IR depth camera or a dedicated liveness model is recommended for full protection.
- **🧑‍🎓 This software is provided for personal/educational use.**

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=140&section=footer" alt="footer"/>

<b>👁️ Argus is watching — so you don't have to. 🛡️</b>
<br/><br/>
<img src="https://img.shields.io/badge/Built_with-🐍_Python_•_🧠_YOLOv8_•_📷_OpenCV-22d3ee?style=flat-square"/>
<br/><br/>
⭐ <i>If this project helps secure your space, give it a star!</i> ⭐

</div>
