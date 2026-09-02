"""
main.py
=======
Central orchestration engine for the Enterprise Multi-Camera CCTV AI Security
System (2 to 20+ cameras).

Architecture
------------
Each camera defined in ``config.CAMERAS`` runs in its own isolated
``multiprocessing.Process`` ("camera worker").  Workers:

* open the RTSP stream with OpenCV (``CAP_PROP_BUFFERSIZE=1``),
* auto-reconnect with exponential backoff on disconnects,
* process every Nth frame (``FRAME_SKIP_FACTOR``) through the vision pipeline
  (:mod:`modules.vision_pipeline`) and behaviour analytics
  (:mod:`modules.behavior_tracking`),
* forward alert events to the parent (queue-backed Telegram dispatcher),
* write event clips through :mod:`modules.storage_engine`,
* publish annotated frames to shared ``multiprocessing.Queue`` objects consumed
  by the web dashboard (:mod:`modules.dashboard_server`).

The parent process
------------------
* sets the start method to ``spawn`` for multi-platform compatibility,
* monitors worker health and restarts crashed workers with backoff,
* runs the Telegram dispatcher + interactive bot, system-health jobs and the
  dashboard server,
* handles graceful shutdown on SIGINT / SIGTERM / KeyboardInterrupt.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

# Ensure `config` and `modules.*` resolve when launched from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "security_system.log")),
    ],
)
logger = logging.getLogger("argus.main")

# Shared bookkeeping (parent process).
FRAME_QUEUES: Dict[str, Any] = {}

SYSTEM_STATE: Dict[str, Any] = {
    "armed": True,
    "alerts_sent": 0,
    "frames_processed": 0,
    "workers_alive": 0,
    "known_events": 0,
    "stranger_events": 0,
    "vehicle_events": 0,
}


# ===========================================================================
# CAMERA WORKER
# ===========================================================================
def camera_worker(camera: Dict[str, Any],
                  out_queue: Any,
                  cmd_queue: Any,
                  alert_queue: Any,
                  stats_queue: Any,
                  log_queue: Any,
                  armed_event: Any,
                  status_queue: Any) -> None:
    """
    Process entry point for one camera.

    ``armed_event`` is a shared ``multiprocessing.Event`` flipped by the parent
    when the system arms / disarms (geofence or /arm /disarm commands).
    """
    import queue as _q

    from modules.audio_night import AudioMonitor, PanicSiren, enhance_low_light
    from modules.behavior_tracking import BehaviorAnalyzer
    from modules.storage_engine import EventSearch, VideoRecorder
    from modules.vision_pipeline import VisionPipeline

    cam_id = camera["id"]
    cam_name = camera.get("name", cam_id)
    resolution = camera.get("resolution", (1280, 720))
    fps = camera.get("fps", 25)
    features = list(camera.get("active_features", []))

    def _log(level: str, msg: str) -> None:
        try:
            log_queue.put_nowait({"cam": cam_id, "level": level, "msg": msg,
                                  "time": datetime.now().isoformat()})
        except Exception:
            pass

    def _publish(frame: np.ndarray) -> None:
        """Non-blocking frame publish to the dashboard (drop on overflow)."""
        try:
            out_queue.put_nowait(("frame", cam_id, frame))
        except Exception:
            pass

    def _stats(**kw: Any) -> None:
        try:
            stats_queue.put_nowait({"cam": cam_id, **kw})
        except Exception:
            pass

    logger.info("[%s] worker started (features=%s)", cam_id, features)

    # --- Vision pipeline -----------------------------------------------------
    pipeline = VisionPipeline(
        camera_id=cam_id, camera_config=camera,
        known_faces_path=config.KNOWN_FACES_PATH,
        tolerance=config.FACE_RECOGNITION_TOLERANCE,
        model=config.FACE_RECOGNITION_MODEL,
        device=config.YOLO_DEVICE,
    )

    # --- Behaviour analytics -------------------------------------------------
    behavior = BehaviorAnalyzer(cam_id, camera)

    # --- Storage engine ------------------------------------------------------
    recorder = VideoRecorder(
        storage_path=os.path.join(config.STORAGE_ROOT, cam_id),
        frame_width=int(resolution[0]), frame_height=int(resolution[1]),
        pre_buffer_seconds=config.PRE_TRIGGER_BUFFER_SECONDS,
        event_seconds=config.EVENT_CLIP_SECONDS,
        idle_fps=config.IDLE_RECORD_FPS, motion_fps=config.MOTION_RECORD_FPS,
    )
    events_db = EventSearch(config.EVENT_DB_PATH)

    # --- Audio monitor ---------------------------------------------------------
    audio: Optional[AudioMonitor] = None
    siren = PanicSiren(audio_file=config.SIREN_AUDIO_FILE,
                       duration=config.SIREN_DURATION_SECONDS)

    last_alert_ts: Dict[str, float] = {}

    def _push_event(event: Dict[str, Any]) -> None:
        """Rate-limit, then forward an event to DB + parent + log."""
        now = time.time()
        ev_type = event.get("type", "EVENT")
        # Per-camera/per-type cooldown to avoid spam (fire heuristics, etc.).
        if now - last_alert_ts.get(ev_type, 0.0) < config.ALERT_COOLDOWN_SECONDS:
            return
        last_alert_ts[ev_type] = now

        try:
            alert_queue.put_nowait({"kind": "event", "cam_id": cam_id,
                                    "cam_name": cam_name, "event": event})
        except Exception:
            pass
        events_db.log_event(cam_id, ev_type, message=event.get("message", ""))
        _log("ALERT", f"{ev_type}: {event.get('message', '')}")

    def _on_loud_sound(level: float) -> None:
        _push_event({"type": "LOUD_SOUND", "priority": "high",
                     "message": f"Loud sound level {level:.0f}"})
        recorder.trigger_event("LOUD_SOUND", cam_id)

    def _on_aggressive_voice() -> None:
        _push_event({"type": "AGGRESSIVE_VOICE", "priority": "high",
                     "message": "Aggressive / shouting voice detected"})

    def _on_keyword(keyword: str) -> None:
        _push_event({"type": "VOICE_COMMAND", "priority": "high",
                     "message": f"Voice command recognised: {keyword}"})
        if "lockdown" in keyword or "panic" in keyword:
            try:
                cmd_queue.put_nowait({"type": "system", "action": "lockdown",
                                      "by": cam_id})
            except Exception:
                pass

    if "audio" in features:
        audio = AudioMonitor(
            threshold_base=config.SOUND_THRESHOLD_BASE,
            dynamic_alpha=config.SOUND_DYNAMIC_ALPHA,
            loud_margin=config.LOUD_SOUND_MARGIN,
            zcr_aggressive_threshold=config.ZCR_AGGRESSIVE_THRESHOLD,
            keywords=config.VOICE_KEYWORDS,
            sample_rate=config.SOUND_SAMPLE_RATE,
            chunk_size=config.SOUND_CHUNK_SIZE,
            on_loud_sound=_on_loud_sound,
            on_aggressive_voice=_on_aggressive_voice,
            on_keyword=_on_keyword,
        )
        audio.start()

    # --- capture helpers -------------------------------------------------------
    def _open_capture() -> Optional[cv2.VideoCapture]:
        """Open RTSP with a 1-frame buffer to eliminate frame latency."""
        c = cv2.VideoCapture(camera["url"])
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
            c.set(cv2.CAP_PROP_FPS, int(fps))
        except Exception:
            pass
        return c

    cap = _open_capture()
    retry_delay = 1.0
    max_retry_delay = 60.0
    frame_idx = 0
    night_flag = "night_mode" in features
    night_force = False
    last_status_ts = 0.0
    last_annotated: Optional[np.ndarray] = None

    def _handle_commands() -> None:
        """Drain dashboard / bot / voice commands (works even while offline)."""
        nonlocal night_force
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except _q.Empty:
                return
            except Exception:
                return
            if cmd.get("type") == "camera" and cmd.get("cam") == cam_id:
                action = cmd.get("action")
                if action == "record":
                    clip = recorder.trigger_event("MANUAL", cam_id)
                    if clip:
                        _log("INFO", f"manual recording started: {clip}")
                elif action == "snapshot":
                    if last_annotated is not None:
                        path = recorder.save_snapshot(last_annotated, cam_id)
                        events_db.log_event(cam_id, "SNAPSHOT", file_path=path)
                        _log("INFO", f"snapshot saved: {path}")
                    else:
                        _log("WARNING", "snapshot requested but no frames yet")
                elif action == "siren":
                    siren.trigger()
                    _log("INFO", "panic siren triggered (dashboard)")
                elif action == "night":
                    night_force = not night_force
                    _log("INFO", f"night mode force {'ON' if night_force else 'OFF'}")
            elif cmd.get("type") == "system":
                if cmd.get("action") == "lockdown":
                    siren.trigger()
                    events_db.log_event(cam_id, "LOCKDOWN",
                                        message=f"by {cmd.get('by', '?')}")
                elif cmd.get("action") == "reload_faces":
                    count = pipeline.reload_known_faces()
                    _log("INFO", f"reloaded faces: {count}")

    try:
        while True:
            loop_start = time.time()
            _handle_commands()

            # --- reconnect with exponential backoff ----------------------------
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                logger.warning("[%s] stream lost - retrying in %.1fs", cam_id, retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
                cap = _open_capture()
                if cap is not None and cap.isOpened():
                    retry_delay = 1.0
                    _log("INFO", "stream reconnected")
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                cap = None
                continue
            retry_delay = 1.0
            frame_idx += 1

            # --- night mode pre-enhancement -------------------------------------
            if night_flag or night_force:
                frame = enhance_low_light(
                    frame,
                    brightness_threshold=config.NIGHT_BRIGHTNESS_THRESHOLD,
                    night_start=config.NIGHT_MODE_START_HOUR,
                    night_end=config.NIGHT_MODE_END_HOUR,
                    force=night_force,
                )
            last_annotated = frame

            # --- armed state (geofence / /arm /disarm) ---------------------------
            if not armed_event.is_set():
                _publish(frame)
                continue

            # --- frame skip factor (CPU/GPU optimisation) -------------------------
            if frame_idx % config.FRAME_SKIP_FACTOR != 0:
                _publish(frame)
                _heartbeat(cam_id, status_queue, frame_idx, loop_start,
                           last_status_ts, recorder, night_flag, audio)
                last_status_ts = time.time()
                continue

            # --- vision pipeline ---------------------------------------------------
            annotated, alerts, debug = pipeline.process_camera_frame(frame)
            last_annotated = annotated

            # --- behaviour analytics -------------------------------------------------
            behavior_alerts = behavior.analyze(annotated, debug)
            all_alerts = alerts + behavior_alerts

            # --- storage: adaptive-fps recording --------------------------------------
            motion = bool(debug.get("detections"))
            recorder.add_frame(annotated, motion=motion)

            # --- alert dispatch ----------------------------------------------------------
            for ev in all_alerts:
                ev = dict(ev)
                ev_type = ev.get("type", "EVENT")

                # Attach the annotated frame as the alert snapshot.
                ev.setdefault("frame", annotated)

                # Start an event clip for high-priority triggers (pre-trigger
                # buffer is flushed automatically by the recorder).
                if ev.get("priority") == "high" and ev_type in (
                        "STRANGER", "FIRE", "SMOKE", "FALL_DETECTED"):
                    clip = recorder.trigger_event(ev_type, cam_id)
                    if clip:
                        ev["clip_path"] = clip

                _push_event(ev)

            # --- periodic stats to parent ----------------------------------------------
            _stats(frames=config.FRAME_SKIP_FACTOR)
            if any(e.get("type") == "STRANGER" for e in all_alerts):
                _stats(strangers=1)
            if any(e.get("type") == "ANPR" for e in all_alerts):
                _stats(vehicles=1)
            if any(e.get("type") == "KNOWN_PERSON" for e in all_alerts):
                _stats(known=1)

            # --- dashboard frame ------------------------------------------------------------
            _publish(annotated)
            _heartbeat(cam_id, status_queue, frame_idx, loop_start,
                       last_status_ts, recorder, night_flag, audio)
            last_status_ts = time.time()

            # --- FPS pacing --------------------------------------------------------------------
            target_interval = 1.0 / max(1, config.TARGET_PROCESSING_FPS)
            elapsed = time.time() - loop_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    except KeyboardInterrupt:
        logger.info("[%s] interrupted", cam_id)
    except Exception:
        logger.exception("[%s] worker crashed", cam_id)
    finally:
        if cap is not None:
            cap.release()
        recorder.stop()
        if audio is not None:
            audio.stop()
        _log("INFO", "worker stopped")
        logger.info("[%s] worker stopped", cam_id)


def _heartbeat(cam_id: str, status_queue: Any, frame_idx: int,
               loop_start: float, last_status_ts: float, recorder: Any,
               night_flag: bool, audio: Any) -> None:
    """Report worker status to the dashboard overlay every ~2s."""
    now = time.time()
    if now - last_status_ts < 2.0:
        return
    item = {
        "cam": cam_id,
        "fps": float(1.0 / max(1e-6, now - loop_start)),
        "rec": recorder.is_recording,
        "night": night_flag,
        "frame_idx": frame_idx,
        "time": datetime.now().isoformat(),
    }
    if audio is not None:
        item["sound_db"] = round(audio.get_db(), 1)
        item["sound_pct"] = round(audio.get_level_percentage(), 1)
    try:
        status_queue.put_nowait(item)
    except Exception:
        pass


# ===========================================================================
# PARENT-SIDE SERVICES
# ===========================================================================
def _set_armed(workers: Dict[str, Any], armed: bool) -> None:
    """Flip the armed event of every worker (geofence / bot commands)."""
    SYSTEM_STATE["armed"] = armed
    for w in workers.values():
        ev = w["armed_event"]
        if armed:
            ev.set()
        else:
            ev.clear()
    logger.info("System %s", "ARMED" if armed else "DISARMED")


class TelegramDispatcher:
    """Parent-process Telegram sender + interactive bot command handler."""

    def __init__(self, workers: Dict[str, Any]) -> None:
        from modules.alert_manager import AlertManager

        self._workers = workers
        self.manager = AlertManager(
            token=config.TELEGRAM_TOKEN, chat_id=config.TELEGRAM_CHAT_ID,
            cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
            escalation_cooldown=config.HIGH_PRIORITY_ESCALATION_COOLDOWN,
            multi_factor_window=config.MULTI_FACTOR_WINDOW_SECONDS,
            known_faces_path=config.KNOWN_FACES_PATH,
            poll_timeout=config.TELEGRAM_POLL_TIMEOUT,
        )
        self.manager.on_status = self._status_text
        self.manager.on_reload_faces = self._reload_faces
        self.manager.on_arm = lambda armed: _set_armed(self._workers, armed)

    # ------------------------------------------------------------------ status
    def _status_text(self) -> str:
        lines = [
            f"🖥️ Workers alive: {SYSTEM_STATE['workers_alive']}/{len(self._workers)}",
            f"📊 Frames processed: {SYSTEM_STATE['frames_processed']}",
            f"⚠️ Stranger events: {SYSTEM_STATE['stranger_events']}",
            f"✅ Known events: {SYSTEM_STATE['known_events']}",
            f"🚗 Vehicle/ANPR events: {SYSTEM_STATE['vehicle_events']}",
            f"🔒 Armed: {SYSTEM_STATE['armed']}",
        ]
        for cam_id, w in self._workers.items():
            proc = w["proc"]
            state = "alive" if proc.is_alive() else "DEAD"
            lines.append(f"• {cam_id}: {state}")
        return "\n".join(lines)

    def _reload_faces(self) -> int:
        """Ask every worker to reload its face database."""
        count = 0
        for w in self._workers.values():
            try:
                w["cmd_queue"].put_nowait({"type": "system", "action": "reload_faces"})
                count += 1
            except Exception:
                pass
        return count

    # ------------------------------------------------------------------ routes
    def route(self, cam_id: str, cam_name: str, event: Dict[str, Any]) -> None:
        """Validate, cooldown-check and dispatch one event to Telegram."""
        if not event:
            return
        ev_type = event.get("type", "")
        # Multi-factor corroboration timestamps (visual + audio/zone).
        if ev_type in ("LOUD_SOUND", "AGGRESSIVE_VOICE"):
            self.manager.note_audio_trigger()
        if ev_type == "ZONE_INTRUSION":
            self.manager.note_zone_trigger()

        self.manager.send_event(event, cam_id, cam_name)
        SYSTEM_STATE["alerts_sent"] += 1
        if ev_type == "STRANGER":
            SYSTEM_STATE["stranger_events"] += 1
        elif ev_type == "ANPR":
            SYSTEM_STATE["vehicle_events"] += 1
        elif ev_type == "KNOWN_PERSON":
            SYSTEM_STATE["known_events"] += 1

    def start(self) -> None:
        self.manager.start()

    def stop(self) -> None:
        self.manager.stop()


def collector_loop(dispatcher: TelegramDispatcher, workers: Dict[str, Any],
                   dashboard: Any, run_event: Any) -> None:
    """Forward alert events from worker queues to the Telegram dispatcher."""
    while run_event.is_set():
        for w in workers.values():
            try:
                item = w["alert_queue"].get_nowait()
            except Exception:
                continue
            if item is None:
                continue
            if item.get("kind") == "event":
                dispatcher.route(item["cam_id"], item["cam_name"], item["event"])
                if dashboard is not None:
                    try:
                        dashboard.log_event(item["cam_id"], item["event"]["type"],
                                            item["event"].get("message", ""))
                    except Exception:
                        pass
            elif item.get("kind") == "text":
                dispatcher.manager.send_text(item.get("text", ""))
        time.sleep(0.05)


def stats_loop(workers: Dict[str, Any], run_event: Any) -> None:
    """Aggregate per-worker stats (frames/events) into SYSTEM_STATE."""
    while run_event.is_set():
        for w in workers.values():
            try:
                item = w["stats_queue"].get_nowait()
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            SYSTEM_STATE["frames_processed"] += item.get("frames", 0)
            SYSTEM_STATE["known_events"] += item.get("known", 0)
            SYSTEM_STATE["stranger_events"] += item.get("strangers", 0)
            SYSTEM_STATE["vehicle_events"] += item.get("vehicles", 0)
        time.sleep(0.3)


def status_loop(workers: Dict[str, Any], dashboard: Any, run_event: Any) -> None:
    """Collect worker heartbeats into the dashboard status registry."""
    while run_event.is_set():
        for cam_id, w in workers.items():
            try:
                item = w["status_queue"].get_nowait()
            except Exception:
                continue
            if item:
                item["alive"] = w["proc"].is_alive()
                dashboard.status[cam_id] = item
        time.sleep(0.2)


def log_loop(workers: Dict[str, Any], dashboard: Any, run_event: Any) -> None:
    """Mirror worker log lines into the dashboard event log."""
    while run_event.is_set():
        for w in workers.values():
            try:
                item = w["log_queue"].get_nowait()
            except Exception:
                continue
            if item and item.get("level") in ("ALERT", "WARNING", "INFO"):
                dashboard.log_event(item["cam"], item["level"], item.get("msg", ""))
        time.sleep(0.2)


# ===========================================================================
# WORKER PROCESS MANAGEMENT
# ===========================================================================
def start_worker(camera: Dict[str, Any]) -> Dict[str, Any]:
    """Spawn one camera worker process with its queue set."""
    cam_id = camera["id"]
    out_queue = mp.Queue(maxsize=10)      # dashboard frames (drop on overflow)
    cmd_queue = mp.Queue(maxsize=128)     # parent -> worker commands
    alert_queue = mp.Queue(maxsize=256)   # worker -> parent alert events
    stats_queue = mp.Queue(maxsize=256)   # worker -> parent counters
    log_queue = mp.Queue(maxsize=512)     # worker -> parent logs
    status_queue = mp.Queue(maxsize=128)  # worker -> parent status heartbeats
    armed_event = mp.Event()
    if SYSTEM_STATE["armed"]:
        armed_event.set()

    proc = mp.Process(
        target=camera_worker,
        args=(camera, out_queue, cmd_queue, alert_queue, stats_queue,
              log_queue, armed_event, status_queue),
        name=f"cam-{cam_id}",
        daemon=True,
    )
    proc.start()

    FRAME_QUEUES[cam_id] = out_queue

    return {
        "proc": proc, "out_queue": out_queue, "cmd_queue": cmd_queue,
        "alert_queue": alert_queue, "stats_queue": stats_queue,
        "log_queue": log_queue, "armed_event": armed_event,
        "status_queue": status_queue, "camera": camera,
    }


def monitor_workers(workers: Dict[str, Any], run_event: Any) -> None:
    """
    Health monitor: restart any crashed worker.  Repeated crashes within a
    short window pause with ``WORKER_RESTART_BACKOFF``.
    """
    failures: Dict[str, int] = {}
    while run_event.is_set():
        time.sleep(2)
        for cam_id, w in list(workers.items()):
            proc = w["proc"]
            if proc.is_alive():
                failures[cam_id] = 0
                continue
            failures[cam_id] = failures.get(cam_id, 0) + 1
            if failures[cam_id] > 3:
                logger.error("[%s] crashed %d times - waiting %ds before restart",
                             cam_id, failures[cam_id], config.WORKER_RESTART_BACKOFF)
                time.sleep(config.WORKER_RESTART_BACKOFF)
                failures[cam_id] = 0
            logger.warning("[%s] worker died - restarting (attempt %d)",
                           cam_id, failures[cam_id])
            try:
                proc.terminate()
                proc.join(timeout=3)
            except Exception:
                pass
            new_worker = start_worker(w["camera"])
            workers[cam_id] = new_worker
            FRAME_QUEUES[cam_id] = new_worker["out_queue"]


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    """Entry point: spawn workers, start services, supervise until shutdown."""
    parser = argparse.ArgumentParser(description="Argus multi-camera security engine")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not start the web dashboard")
    parser.add_argument("--no-health", action="store_true",
                        help="disable heartbeat/geofence/power-saver jobs")
    parser.add_argument("--smoke", action="store_true",
                        help="run a quick import/self-test and exit")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        return

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            ARGUS - Enterprise Multi-Camera CCTV AI           ║
    ║            Security System  v2.0                             ║
    ║                                                              ║
    ║  • YOLOv8 detection (person/vehicle/animal)                  ║
    ║  • Face recognition + anti-spoofing + ANPR                   ║
    ║  • Fire/smoke, fall/pose, tampering, loitering               ║
    ║  • Zone intrusion, tailgating, cross-camera re-ID            ║
    ║  • Night mode + audio analytics + panic siren                ║
    ║  • Telegram alerts with snapshots & clips (bot commands)     ║
    ║  • Adaptive-FPS event recording + SQLite event search        ║
    ║  • Heartbeat, geofencing, power-saver, daily recap           ║
    ║  • Web dashboard: MJPEG grid, controls, live event log       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # --- multiprocessing setup ------------------------------------------------
    mp.set_start_method("spawn", force=True)
    run_event = mp.Event()
    run_event.set()

    # --- spawn workers -----------------------------------------------------------
    SYSTEM_STATE["armed"] = config.ARMED_BY_DEFAULT
    logger.info("Starting %d camera worker(s)...", len(config.CAMERAS))
    workers: Dict[str, Any] = {}
    for camera in config.CAMERAS:
        workers[camera["id"]] = start_worker(camera)

    # --- dashboard -----------------------------------------------------------------
    dashboard = None
    if not args.no_dashboard:
        try:
            from modules.dashboard_server import DashboardServer

            dashboard = DashboardServer(
                cameras=config.CAMERAS, frame_queues=FRAME_QUEUES,
                host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT,
                password=config.DASHBOARD_PASSWORD,
                auth_mode=config.DASHBOARD_AUTH_MODE,
            )

            def _forward_cmd(cam_id: str, action: str) -> None:
                """Push a dashboard control command to the camera's worker."""
                w = workers.get(cam_id)
                if w is None:
                    return
                try:
                    w["cmd_queue"].put_nowait({"type": "camera", "cam": cam_id,
                                               "action": action})
                except Exception:
                    pass

            dashboard.set_command_sender(_forward_cmd)
            dashboard.start()
        except Exception as exc:
            logger.error("dashboard failed to start: %s", exc)
            dashboard = None

    # --- Telegram dispatcher ---------------------------------------------------------
    dispatcher = TelegramDispatcher(workers)
    dispatcher.start()

    # --- system health / storage maintenance -------------------------------------------
    health = None
    storage = None
    if not args.no_health:
        try:
            from modules.storage_engine import EventSearch, StorageManager
            from modules.system_health import SystemHealth

            storage = StorageManager(
                storage_path=config.STORAGE_ROOT,
                max_storage_gb=config.MAX_STORAGE_GB,
                cleanup_days=config.CLEANUP_DAYS,
                free_space_percent=config.FREE_SPACE_THRESHOLD_PERCENT,
                check_interval=config.STORAGE_CHECK_INTERVAL,
            )
            storage.start()

            def _recap_stats() -> Dict[str, Any]:
                db = EventSearch(config.EVENT_DB_PATH)
                counts = db.count_by_type()
                return {
                    "known": counts.get("KNOWN_PERSON", 0),
                    "strangers": counts.get("STRANGER", 0),
                    "vehicles": counts.get("ANPR", 0),
                    "storage_gb": storage.get_info()["usage_gb"] if storage else 0.0,
                    "events_by_type": counts,
                }

            health = SystemHealth(
                heartbeat_url=config.HEARTBEAT_URL,
                heartbeat_interval=config.HEARTBEAT_INTERVAL,
                router_ip=config.GEOFENCE_ROUTER_IP,
                geofence_interval=config.GEOFENCE_PING_INTERVAL,
                geofence_miss_threshold=config.GEOFENCE_MISS_THRESHOLD,
                telegram_sender=dispatcher.manager.send_text,
                status_provider=dispatcher._status_text,
                armed_setter=lambda armed: _set_armed(workers, armed),
            )
            health.start()
            if health._scheduler is not None:
                health._scheduler.add_job(
                    lambda: health.daily_recap(_recap_stats),
                    "cron", hour=0, minute=0,
                )
        except Exception as exc:
            logger.error("system health failed to start: %s", exc)

    # --- supervisor threads --------------------------------------------------------------
    threading.Thread(target=collector_loop,
                     args=(dispatcher, workers, dashboard, run_event),
                     daemon=True, name="alert-collector").start()
    threading.Thread(target=monitor_workers,
                     args=(workers, run_event), daemon=True,
                     name="worker-monitor").start()
    threading.Thread(target=stats_loop,
                     args=(workers, run_event), daemon=True,
                     name="stats-loop").start()
    if dashboard is not None:
        threading.Thread(target=status_loop,
                         args=(workers, dashboard, run_event), daemon=True,
                         name="status-loop").start()
        threading.Thread(target=log_loop,
                         args=(workers, dashboard, run_event), daemon=True,
                         name="log-loop").start()

    # --- graceful shutdown -----------------------------------------------------------------
    stop_requested = {"flag": False}

    def _shutdown(signum=None, frame=None) -> None:  # noqa: ANN001
        if stop_requested["flag"]:
            return
        stop_requested["flag"] = True
        logger.info("Shutdown requested (signal=%s) - terminating workers...", signum)
        run_event.clear()
        for cam_id, w in workers.items():
            try:
                w["proc"].terminate()
                w["proc"].join(timeout=5)
            except Exception:
                pass
            if w["proc"].is_alive():  # force-kill stragglers
                try:
                    w["proc"].kill()
                    w["proc"].join(timeout=3)
                except Exception:
                    pass
            try:
                w["out_queue"].close()
                w["alert_queue"].close()
            except Exception:
                pass
        if dispatcher.manager._configured:
            try:
                dispatcher.manager.send_text("🔴 <b>Argus Security System offline</b>")
                time.sleep(1)
            except Exception:
                pass
        dispatcher.stop()
        if health is not None:
            health.stop()
        if storage is not None:
            storage.stop()
        logger.info("Shutdown complete - exiting cleanly")
        # Clean interpreter exit lets multiprocessing reap daemon children.
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Argus running - %d cameras, dashboard=%s",
                len(config.CAMERAS), dashboard is not None)
    try:
        while not stop_requested["flag"]:
            alive = sum(1 for w in workers.values() if w["proc"].is_alive())
            SYSTEM_STATE["workers_alive"] = alive
            time.sleep(5)
            if alive == 0 and all(w["proc"].exitcode is not None
                                  for w in workers.values()):
                logger.error("All camera workers exited - stopping.")
                _shutdown()
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT)


def _smoke_test() -> None:
    """Import & logic self-test (no camera/hardware required)."""
    import queue as _q
    import tempfile

    print("=== Argus smoke test ===")
    from modules.alert_manager import AlertManager
    from modules.audio_night import PanicSiren, enhance_low_light, is_night_time
    from modules.behavior_tracking import BehaviorAnalyzer, bbox_iou, point_in_polygon
    from modules.dashboard_server import DashboardServer
    from modules.storage_engine import EventSearch, StorageManager, VideoRecorder
    from modules.system_health import SystemHealth
    from modules.vision_pipeline import VisionPipeline

    # Geometry.
    poly = [(100, 100), (400, 100), (400, 400), (100, 400)]
    assert point_in_polygon((200, 200), poly) is True
    assert point_in_polygon((900, 900), poly) is False
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) > 0.14
    print("geometry OK")

    # Night mode on a synthetic dark frame.
    dark = np.full((240, 320, 3), 12, dtype=np.uint8)
    bright = enhance_low_light(dark)
    assert bright.mean() > dark.mean()
    print("night-mode enhancement OK (is_night=%s)" % is_night_time())

    # Storage: recorder clip + snapshot + event DB search.
    tmp = tempfile.mkdtemp(prefix="argus_smoke_")
    rec = VideoRecorder(tmp, 320, 240, pre_buffer_seconds=1, event_seconds=1,
                        idle_fps=1, motion_fps=5)
    for _ in range(15):
        rec.add_frame(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                      motion=True)
    clip = rec.trigger_event("STRANGER", "CAM-X")
    rec.stop()
    print("recorder OK clip=%s" % clip)

    db = EventSearch(os.path.join(tmp, "events.db"))
    db.log_event("CAM-X", "STRANGER", file_path=clip, message="test")
    assert len(db.search(event_type="STRANGER")) >= 1
    print("event search OK")

    sm = StorageManager(tmp, max_storage_gb=0.0001, cleanup_days=7)
    sm.cleanup()
    print("storage manager OK")

    # Behavior analyzer + tamper (dark frame).
    cam_cfg = {"id": "CAM-X", "name": "Test", "active_zones": [poly],
               "active_features": ["zone_intrusion", "loitering", "tampering"]}
    ba = BehaviorAnalyzer("CAM-X", cam_cfg)
    ba.analyze(dark, {"detections": [], "faces": []})
    print("behavior analyzer OK")

    # Dashboard server import/registry + HTML rendering (catches f-string bugs).
    q = _q.Queue()
    ds = DashboardServer([cam_cfg], {"CAM-X": q}, port=5999)
    assert ds.app is not None
    html = ds._render_index()
    assert "Argus Security — Camera Grid" in html
    assert "item.type.toLowerCase()" in html  # JS intact after f-string escaping
    print("dashboard server OK (index renders)")

    # Health + siren + alert manager objects constructible.
    h = SystemHealth(heartbeat_url="")
    p = PanicSiren()
    a = AlertManager("X", "Y")
    assert h is not None and p is not None and a is not None
    print("system-health / siren / alert OK")

    # Vision pipeline loads without optional models.
    vp = VisionPipeline("CAM-X", cam_cfg,
                        known_faces_path=os.path.join(tmp, "known_faces"),
                        tolerance=0.45, model="hog", device="cpu")
    ann, alerts, debug = vp.process_camera_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    assert ann.shape == (240, 320, 3)
    print("vision pipeline OK (alerts=%d)" % len(alerts))

    print("=== smoke test PASSED ===")


if __name__ == "__main__":
    main()
