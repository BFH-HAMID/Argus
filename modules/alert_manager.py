"""
alert_manager.py
================
Asynchronous Telegram notification manager.

Design
------
* **Threaded queue** - all network I/O happens in a background daemon thread,
  so video processing is never blocked by a slow Telegram request.
* **Multi-factor validation** - an event is *escalated* (high priority) only
  when both a visual trigger (e.g. STRANGER) and an audio/zone trigger occur
  inside ``config.MULTI_FACTOR_WINDOW_SECONDS``; suppresses false alarms.
* **Smart dispatcher** - formatted HTML messages with snapshot photos and
  optional 30 s MP4 clips; per-camera/per-type dynamic cooldowns.
* **Interactive bot** - a polling loop (long-poll ``getUpdates``) answers
  ``/status``, ``/reload_faces``, ``/arm``, ``/disarm`` and accepts an
  uploaded photo to add a face to ``known_faces/`` with a re-index.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class AlertManager:
    """Threaded Telegram alert sender + interactive command bot."""

    def __init__(self, token: str, chat_id: str,
                 cooldown_seconds: float = 10.0,
                 escalation_cooldown: float = 60.0,
                 multi_factor_window: float = 8.0,
                 known_faces_path: str = "known_faces",
                 poll_timeout: int = 25) -> None:
        self.token = token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self.escalation_cooldown = escalation_cooldown
        self.multi_factor_window = multi_factor_window
        self.known_faces_path = known_faces_path
        self.poll_timeout = poll_timeout

        self.base_url = f"https://api.telegram.org/bot{token}"
        self._configured = bool(token) and "YOUR_" not in token and \
            bool(chat_id) and "YOUR_" not in chat_id

        # Outbound queue (main thread enqueues, sender thread posts).
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=512)
        self._sender_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False

        # Cooldown bookkeeping: key=(cam_id, event_type) -> last_sent_ts
        self._last_sent: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

        # Multi-factor corroboration timestamps (audio / zone triggers).
        self._last_audio_trigger: float = 0.0
        self._last_zone_trigger: float = 0.0

        # Callbacks used by interactive commands (set by main orchestrator).
        self.on_status: Optional[Callable[[], str]] = None
        self.on_reload_faces: Optional[Callable[[], int]] = None
        self.on_arm: Optional[Callable[[bool], None]] = None

        # Offsets for long polling.
        self._last_update_id = 0

        if not self._configured:
            logger.warning("Telegram credentials not configured - alerts disabled")

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        """Start the sender and command-polling threads."""
        if self._running:
            return
        self._running = True
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()
        if self._configured:
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
        logger.info("AlertManager started (configured=%s)", self._configured)

    def stop(self) -> None:
        """Stop both threads."""
        self._running = False
        if self._sender_thread:
            self._sender_thread.join(timeout=3)
            self._sender_thread = None
        if self._poll_thread:
            self._poll_thread.join(timeout=3)
            self._poll_thread = None
        logger.info("AlertManager stopped")

    # ------------------------------------------------------------- cooldowns
    def _can_send(self, cam_id: str, event_type: str, priority: str) -> bool:
        """Dynamic cooldown: 10 s normal, 60 s for escalated high-priority."""
        key = (cam_id, event_type)
        now = time.time()
        with self._lock:
            last = self._last_sent.get(key, 0.0)
            window = (self.escalation_cooldown if priority == "high"
                      else self.cooldown_seconds)
            if now - last < window:
                return False
            self._last_sent[key] = now
            return True

    # ---------------------------------------------------- multi-factor logic
    def _validate(self, event: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Multi-factor validation: escalate visual events only when an audio or
        zone trigger happened inside the window.  Returns (valid, reason).
        """
        event_type = event.get("type", "")
        priority = event.get("priority", "medium")

        if priority != "high":
            return True, "normal"

        if event_type in ("STRANGER", "MASKED_FACE", "SPOOF_SCREEN", "TAILGATING"):
            # Needs corroboration: audio or zone intrusion in the window.
            corroborated = False
            if self._last_audio_trigger and \
                    time.time() - self._last_audio_trigger <= self.multi_factor_window:
                corroborated = True
            if self._last_zone_trigger and \
                    time.time() - self._last_zone_trigger <= self.multi_factor_window:
                corroborated = True
            if corroborated:
                return True, "escalated (visual+audio/zone)"
            return True, "unverified (visual only)"
        return True, "normal"

    # Called by the audio/zone components to record trigger timestamps.
    def note_audio_trigger(self) -> None:
        self._last_audio_trigger = time.time()

    def note_zone_trigger(self) -> None:
        self._last_zone_trigger = time.time()

    # ----------------------------------------------------------------- enqueue
    def send_event(self, event: Dict[str, Any], cam_id: str, cam_name: str) -> None:
        """Validate, cooldown-check and enqueue an alert event for a camera."""
        if not self._configured:
            return

        valid, reason = self._validate(event)
        if not valid:
            return
        priority = event.get("priority", "medium")

        if not self._can_send(cam_id, event.get("type", "EVENT"), priority):
            return

        event_type = event.get("type", "EVENT")
        message = event.get("message", f"{event_type} at {cam_name}")

        # Snapshot from crop or annotated frame.
        crop = event.get("crop_img")
        frame = event.get("frame")
        image = crop if (crop is not None and getattr(crop, "size", 0)) else frame

        caption = (f"🚨 <b>{event_type}</b> at <b>{cam_id}: {cam_name}</b>\n"
                   f"{message}\n"
                   f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"<i>factor: {reason}</i>")

        if image is not None:
            self._queue.put({"kind": "photo", "image": image, "caption": caption})
        else:
            self._queue.put({"kind": "text", "text": caption})

        # Attach the recorded clip if one is referenced (e.g. by storage engine).
        clip = event.get("clip_path")
        if clip and os.path.exists(clip):
            self._queue.put({"kind": "video", "video_path": clip,
                             "caption": f"📹 <b>{event_type}</b> {cam_id}"})

    def send_text(self, text: str) -> None:
        """Enqueue a raw text message."""
        if self._configured:
            self._queue.put({"kind": "text", "text": text})

    def send_image(self, image: np.ndarray, caption: str = "") -> None:
        """Enqueue a snapshot photo."""
        if self._configured and image is not None:
            self._queue.put({"kind": "photo", "image": image, "caption": caption})

    # -------------------------------------------------------------- sender
    def _sender_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if item["kind"] == "text":
                    self._send_text(item["text"])
                elif item["kind"] == "photo":
                    self._send_photo(item["image"], item["caption"])
                elif item["kind"] == "video":
                    self._send_video(item["video_path"], item["caption"])
            except Exception as exc:
                logger.error("sender error: %s", exc)

    def _send_text(self, text: str) -> bool:
        try:
            r = requests.post(f"{self.base_url}/sendMessage",
                              data={"chat_id": self.chat_id, "text": text,
                                    "parse_mode": "HTML"}, timeout=30)
            return r.status_code == 200
        except Exception as exc:
            logger.error("telegram text failed: %s", exc)
            return False

    def _send_photo(self, image: np.ndarray, caption: str) -> bool:
        try:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return False
            bio = io.BytesIO(buf.tobytes())
            bio.name = "snapshot.jpg"
            r = requests.post(f"{self.base_url}/sendPhoto",
                              data={"chat_id": self.chat_id, "caption": caption,
                                    "parse_mode": "HTML"},
                              files={"photo": bio}, timeout=60)
            return r.status_code == 200
        except Exception as exc:
            logger.error("telegram photo failed: %s", exc)
            return False

    def _send_video(self, video_path: str, caption: str) -> bool:
        try:
            with open(video_path, "rb") as f:
                r = requests.post(f"{self.base_url}/sendVideo",
                                  data={"chat_id": self.chat_id, "caption": caption,
                                        "parse_mode": "HTML"},
                                  files={"video": f}, timeout=120)
            return r.status_code == 200
        except Exception as exc:
            logger.error("telegram video failed: %s", exc)
            return False

    # ------------------------------------------------------- interactive bot
    def _poll_loop(self) -> None:
        """Long-poll Telegram for commands & uploaded photos."""
        while self._running:
            try:
                params = {"timeout": self.poll_timeout,
                          "offset": self._last_update_id + 1}
                r = requests.get(f"{self.base_url}/getUpdates", params=params,
                                 timeout=self.poll_timeout + 10)
                if r.status_code != 200:
                    time.sleep(5)
                    continue
                for update in r.json().get("result", []):
                    self._last_update_id = update["update_id"]
                    self._handle_update(update)
            except Exception as exc:
                logger.error("telegram poll error: %s", exc)
                time.sleep(5)

    def _handle_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return
        text = (message.get("text") or "").strip()
        chat_id = message["chat"]["id"]

        # Only respond to the configured chat.
        if str(chat_id) != str(self.chat_id):
            return

        # Uploaded photo -> save into known_faces/ and re-index.
        if message.get("photo") or message.get("document"):
            self._save_uploaded_face(message)
            return

        if text.startswith("/"):
            self._handle_command(text, chat_id)

    def _handle_command(self, text: str, chat_id: int) -> None:
        cmd = text.split()[0].lower()
        if cmd == "/status":
            status = self.on_status() if self.on_status else "System running"
            self._send_text(f"🖥️ <b>System Status</b>\n{status}")
        elif cmd == "/reload_faces":
            count = self.on_reload_faces() if self.on_reload_faces else 0
            self._send_text(f"🔄 Reloaded face database: <b>{count}</b> face(s) loaded.")
        elif cmd == "/arm":
            if self.on_arm:
                self.on_arm(True)
            self._send_text("🔒 <b>System ARMED</b> - full monitoring active.")
        elif cmd == "/disarm":
            if self.on_arm:
                self.on_arm(False)
            self._send_text("🔓 <b>System DISARMED</b> - monitoring suspended.")
        else:
            self._send_text("Available commands:\n/status\n/reload_faces\n/arm\n/disarm")

    def _save_uploaded_face(self, message: Dict[str, Any]) -> None:
        """Download a photo the user sent and add it to known_faces/."""
        try:
            file_id = None
            caption = (message.get("caption") or "").strip()
            if message.get("photo"):
                file_id = message["photo"][-1]["file_id"]  # largest size
            elif message.get("document"):
                file_id = message["document"]["file_id"]

            if not file_id:
                return

            # Resolve file path.
            r = requests.get(f"{self.base_url}/getFile",
                             params={"file_id": file_id}, timeout=30)
            file_path = r.json()["result"]["file_path"]
            data = requests.get(f"https://api.telegram.org/file/bot{self.token}/{file_path}",
                                timeout=60)
            if data.status_code != 200:
                return

            os.makedirs(self.known_faces_path, exist_ok=True)
            # Name from caption, else timestamp.
            safe_name = "".join(c for c in caption if c.isalnum() or c in " _-").strip() \
                or datetime.now().strftime("face_%Y%m%d_%H%M%S")
            dest = os.path.join(self.known_faces_path, f"{safe_name}.jpg")
            with open(dest, "wb") as f:
                f.write(data.content)

            count = self.on_reload_faces() if self.on_reload_faces else 0
            self._send_text(f"✅ Face saved as <b>{safe_name}</b> and re-indexed. "
                            f"<b>{count}</b> face(s) in database.")
            logger.info("Saved uploaded face to %s", dest)
        except Exception as exc:
            logger.error("failed to save uploaded face: %s", exc)
            self._send_text("❌ Could not save that photo as a face.")
