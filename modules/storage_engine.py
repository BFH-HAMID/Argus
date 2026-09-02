"""
storage_engine.py
=================
Background recording and disk-maintenance engine.

* :class:`VideoRecorder` - adaptive-FPS MP4 recorder with a rolling 5-second
  pre-trigger buffer.  On an event trigger it writes the buffered frames plus
  ``EVENT_CLIP_SECONDS`` of live footage in a worker thread.  When idle it
  records at 1 FPS; it ramps to 30 FPS on motion/event.
* :class:`StorageManager` - monitors ``recordings/`` disk usage, purges files
  older than ``CLEANUP_DAYS`` and deletes the oldest clips when usage exceeds
  ``MAX_STORAGE_GB`` until capacity drops to ``FREE_SPACE_THRESHOLD_PERCENT``.
* :class:`EventSearch` - lightweight SQLite index (``events.db``) storing
  (timestamp, camera_id, event_type, file_path, metadata) with search helpers
  for date-range / camera / event-category queries.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# ADAPTIVE FPS VIDEO RECORDER
# ===========================================================================
class VideoRecorder:
    """
    Writes MP4 clips in a worker thread.

    * Rolling pre-trigger ring buffer (``pre_buffer_seconds`` of frames).
    * ``trigger_event`` flushes the buffer + live frames for ``event_seconds``.
    * Adaptive FPS: ``idle_fps`` normally, ``motion_fps`` while an event runs
      or motion is high.
    """

    def __init__(self, storage_path: str, frame_width: int, frame_height: int,
                 pre_buffer_seconds: int = 5, event_seconds: int = 30,
                 idle_fps: int = 1, motion_fps: int = 30) -> None:
        self.storage_path = storage_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pre_buffer_seconds = pre_buffer_seconds
        self.event_seconds = event_seconds
        self.idle_fps = idle_fps
        self.motion_fps = motion_fps

        os.makedirs(storage_path, exist_ok=True)

        # Pre-trigger ring buffer (timestamped frames).
        self._buffer: Deque[Tuple[float, np.ndarray]] = deque(
            maxlen=max(1, pre_buffer_seconds * motion_fps))

        self._queue: "Deque[np.ndarray]" = deque()
        self._queue_lock = threading.Lock()
        self._writer_thread: Optional[threading.Thread] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._clip_path: Optional[str] = None
        self._last_frame_ts: float = 0.0
        self._event_until: float = 0.0
        self._clip_start: Optional[float] = None
        self.is_recording = False

    # ------------------------------------------------------------------ api
    def add_frame(self, frame: np.ndarray, motion: bool = False) -> None:
        """Feed the recorder one frame (called from the camera worker)."""
        now = time.time()
        self._buffer.append((now, frame.copy()))

        # Adaptive FPS decision.
        target_fps = self.motion_fps if (motion or now < self._event_until) \
            else self.idle_fps
        interval = 1.0 / max(1, target_fps)

        if self.is_recording and now - self._last_frame_ts >= interval:
            with self._queue_lock:
                self._queue.append(frame.copy())
            self._last_frame_ts = now

            # Event clip finished?  (stop flag; the writer thread finalizes.)
            if self._clip_start is not None and now - self._clip_start >= self.event_seconds:
                self.is_recording = False
                self._clip_start = None

    def trigger_event(self, event_type: str, camera_id: str) -> Optional[str]:
        """
        Start an event clip: flush the pre-trigger buffer, then keep recording
        for ``event_seconds``.  Returns the clip path once writing starts.
        """
        if self.is_recording:
            return self._clip_path

        # Make sure a previous writer thread has fully finished.
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_type = "".join(c for c in event_type if c.isalnum() or c in "-_")
        self._clip_path = os.path.join(
            self.storage_path, f"{camera_id}_{safe_type}_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._clip_path, fourcc, float(self.motion_fps),
            (self.frame_width, self.frame_height))
        if not self._writer.isOpened():
            logger.error("could not open VideoWriter for %s", self._clip_path)
            self._clip_path = None
            self._writer = None
            return None

        # Seed with pre-trigger frames.
        with self._queue_lock:
            for _, buffered in list(self._buffer):
                self._queue.append(buffered.copy())

        self.is_recording = True
        self._clip_start = time.time()
        self._event_until = self._clip_start + self.event_seconds
        self._last_frame_ts = time.time()

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        logger.info("[%s] event clip started: %s", camera_id, self._clip_path)
        return self._clip_path

    def _writer_loop(self) -> None:
        """
        The ONLY thread that touches the cv2.VideoWriter.  Drains queued frames;
        when recording is finished and the queue is empty it releases the writer
        and exits — this avoids concurrent write/release segfaults.
        """
        while True:
            with self._queue_lock:
                if self._queue:
                    frame = self._queue.popleft()
                else:
                    frame = None
                    still_recording = self.is_recording
            if frame is not None:
                try:
                    if self._writer is not None:
                        self._writer.write(frame)
                except Exception as exc:
                    logger.error("frame write error: %s", exc)
                    still_recording = False
                    with self._queue_lock:
                        self._queue.clear()
            elif not still_recording:
                break
            else:
                time.sleep(0.005)

        # Finalize: drain anything left, then release the writer.
        with self._queue_lock:
            leftover = list(self._queue)
            self._queue.clear()
        for f in leftover:
            try:
                if self._writer is not None:
                    self._writer.write(f)
            except Exception:
                break
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        logger.info("clip finalized: %s", self._clip_path)

    def save_snapshot(self, frame: np.ndarray, camera_id: str,
                      prefix: str = "snapshot") -> str:
        """Save a JPEG snapshot; returns its path."""
        os.makedirs(self.storage_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.storage_path, f"{camera_id}_{prefix}_{timestamp}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path

    def stop(self) -> None:
        """Stop recording and close the writer safely."""
        self.is_recording = False
        self._clip_start = None
        self._event_until = 0.0
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10)
            self._writer_thread = None
        if self._writer is not None:  # belt & braces if the loop never ran
            self._writer.release()
            self._writer = None
        logger.info("recorder stopped")


# ===========================================================================
# STORAGE MANAGER (disk maintenance)
# ===========================================================================
class StorageManager:
    """Monitors recordings/ usage, purges old files, enforces a size cap."""

    def __init__(self, storage_path: str, max_storage_gb: float = 500.0,
                 cleanup_days: int = 7, free_space_percent: int = 80,
                 check_interval: int = 300) -> None:
        self.storage_path = storage_path
        self.max_storage_gb = max_storage_gb
        self.cleanup_days = cleanup_days
        self.free_space_percent = free_space_percent
        self.check_interval = check_interval

        os.makedirs(storage_path, exist_ok=True)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ api
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("StorageManager started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            try:
                self.cleanup()
            except Exception as exc:
                logger.error("storage cleanup error: %s", exc)
            for _ in range(self.check_interval):
                if not self._running:
                    return
                time.sleep(1)

    def usage_bytes(self) -> int:
        total = 0
        for root, _, files in os.walk(self.storage_path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total

    def get_info(self) -> Dict[str, Any]:
        usage = self.usage_bytes()
        cap = self.max_storage_gb * 1024 ** 3
        return {
            "usage_gb": usage / 1024 ** 3,
            "max_storage_gb": self.max_storage_gb,
            "usage_percent": (usage / cap * 100.0) if cap else 0.0,
            "cleanup_days": self.cleanup_days,
        }

    def cleanup(self) -> Dict[str, Any]:
        """Purge old files, then enforce the size cap. Returns stats."""
        deleted = 0
        freed = 0

        # 1. Files older than CLEANUP_DAYS.
        cutoff = time.time() - self.cleanup_days * 86400
        for filepath in self._all_media_files():
            try:
                if os.path.getmtime(filepath) < cutoff:
                    freed += os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted += 1
            except OSError:
                pass

        # 2. Enforce MAX_STORAGE_GB -> delete oldest until 80% capacity.
        cap = self.max_storage_gb * 1024 ** 3
        target = cap * (self.free_space_percent / 100.0)
        usage = self.usage_bytes()
        files = sorted(self._all_media_files(), key=os.path.getmtime)  # oldest first
        for filepath in files:
            if usage <= target:
                break
            try:
                size = os.path.getsize(filepath)
                os.remove(filepath)
                usage -= size
                freed += size
                deleted += 1
            except OSError:
                pass

        if deleted:
            logger.info("cleanup: removed %d file(s), freed %.1f MB",
                        deleted, freed / 1024 ** 2)
        return {"deleted": deleted, "freed_bytes": freed}

    def _all_media_files(self) -> List[str]:
        files = []
        for root, _, names in os.walk(self.storage_path):
            for name in names:
                if name.lower().endswith((".mp4", ".jpg", ".jpeg", ".png")):
                    files.append(os.path.join(root, name))
        return files


# ===========================================================================
# EVENT SEARCH / BROWSER ENGINE (SQLite)
# ===========================================================================
class EventSearch:
    """SQLite-backed event metadata index with query helpers."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                file_path TEXT,
                message TEXT,
                metadata TEXT
            )""")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_cam ON events(camera_id)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        self._conn.commit()

    # ------------------------------------------------------------------ write
    def log_event(self, camera_id: str, event_type: str, file_path: Optional[str] = None,
                  message: str = "", metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert one event row; returns row id."""
        import json
        cur = self._conn.execute(
            "INSERT INTO events (timestamp, camera_id, event_type, file_path, message, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), camera_id,
             event_type, file_path, message, json.dumps(metadata or {})),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------ query
    def search(self, camera_id: Optional[str] = None,
               event_type: Optional[str] = None,
               start: Optional[str] = None, end: Optional[str] = None,
               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query clips by date range, camera id and/or event category.

        ``start``/``end`` are ISO strings like ``2026-09-01T00:00:00``.
        """
        import json
        sql = ("SELECT timestamp, camera_id, event_type, file_path, message, metadata "
               "FROM events WHERE 1=1")
        params: List[Any] = []
        if camera_id:
            sql += " AND camera_id = ?"
            params.append(camera_id)
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        if start:
            sql += " AND timestamp >= ?"
            params.append(start)
        if end:
            sql += " AND timestamp <= ?"
            params.append(end)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "timestamp": r[0], "camera_id": r[1], "event_type": r[2],
                "file_path": r[3], "message": r[4],
                "metadata": json.loads(r[5] or "{}"),
            }
            for r in rows
        ]

    def count_by_type(self, start: Optional[str] = None) -> Dict[str, int]:
        """Event counts grouped by type (optionally since ``start``)."""
        sql = "SELECT event_type, COUNT(*) FROM events WHERE 1=1"
        params: List[Any] = []
        if start:
            sql += " AND timestamp >= ?"
            params.append(start)
        sql += " GROUP BY event_type"
        rows = self._conn.execute(sql, params).fetchall()
        return {row[0]: row[1] for row in rows}
