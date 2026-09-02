"""
behavior_tracking.py
====================
Spatio-temporal behaviour analytics across multi-camera streams.

Implements:

* **Virtual tripline & zone intrusion** - polygon containment tests on tracked
  object centres (restricted zones from ``camera_config['active_zones']``).
* **Loitering detection** - an IoU tracker keeps per-object IDs; when a person
  stays inside a zone longer than ``config.LOITERING_SECONDS`` a
  ``LOITERING_DETECTED`` high-priority event fires.
* **Tailgating detection** - a stranger entering an entry zone within
  ``config.TAILGATE_WINDOW_SECONDS`` *behind* a recognised person triggers a
  ``TAILGATING`` event.
* **Camera tampering detection** - monitors brightness (lens covered), blur
  (Laplacian variance, lens spray/unfocused) and zero-variance (frozen) frames.
* **Cross-camera re-identification (stub)** - ``ReIdManager`` correlates face
  embeddings across camera ids so one person keeps a persistent global ID.

The tracker is a lightweight IoU tracker (ByteTrack-style matching with
greedy assignment).  It can be swapped for a full ByteTrack / DeepSORT backend
without changing the public API.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def point_in_polygon(point: Tuple[float, float],
                     polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test for convex/concave polygons."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Centre point of an (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-9)


# ---------------------------------------------------------------------------
# Lightweight IoU tracker (ByteTrack-style greedy matching)
# ---------------------------------------------------------------------------
class IoUTracker:
    """Assigns stable integer IDs to moving boxes via greedy IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match detections to existing tracks and return tracked objects::

            [{"track_id": 3, "bbox": (x1,y1,x2,y2), "class": "person", ...}]
        """
        if detections:
            # Greedy assignment: for each track pick best unseen detection.
            matched_dets: set = set()
            for track_id in list(self.tracks.keys()):
                track = self.tracks[track_id]
                best_iou, best_idx = 0.0, -1
                for i, det in enumerate(detections):
                    if i in matched_dets:
                        continue
                    iou = bbox_iou(track["bbox"], det["bbox"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                if best_idx >= 0 and best_iou >= self.iou_threshold:
                    det = detections[best_idx]
                    track["bbox"] = det["bbox"]
                    track["last_seen"] = time.time()
                    track["age"] = 0
                    track.update(det)
                    matched_dets.add(best_idx)

            # New tracks for unmatched detections.
            for i, det in enumerate(detections):
                if i not in matched_dets:
                    self.tracks[self._next_id] = {
                        "track_id": self._next_id, "bbox": det["bbox"],
                        "first_seen": time.time(), "last_seen": time.time(),
                        "age": 0, **det,
                    }
                    self._next_id += 1

        # Age out lost tracks.
        now = time.time()
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1
            if now - self.tracks[track_id]["last_seen"] > self.max_age * 0.1 \
                    or self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]

        return [dict(t) for t in self.tracks.values()]


# ---------------------------------------------------------------------------
# Camera tampering / feed-quality monitor
# ---------------------------------------------------------------------------
class TamperMonitor:
    """
    Detects lens covering (darkness), blur/spray (low Laplacian variance) and
    frozen frames (near-zero inter-frame variance) over a rolling window.
    """

    def __init__(self, darkness_threshold: int = 25, blur_threshold: float = 60.0,
                 frozen_seconds: int = 10) -> None:
        self.darkness_threshold = darkness_threshold
        self.blur_threshold = blur_threshold
        self.frozen_seconds = frozen_seconds
        self._prev_gray: Optional[np.ndarray] = None
        self._frozen_since: Optional[float] = None
        self._dark_since: Optional[float] = None
        self._blur_since: Optional[float] = None

    def analyze(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Return tampering events; one per condition, rate-limited to once/30s."""
        events: List[Dict[str, Any]] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()

        brightness = float(np.mean(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 1. Lens covered -> sudden darkness.
        if brightness < self.darkness_threshold:
            self._dark_since = self._dark_since or now
        else:
            self._dark_since = None
        if self._dark_since and now - self._dark_since >= 3:
            events.append({"type": "CAMERA_TAMPERING",
                           "message": f"Lens covered / darkness (brightness={brightness:.0f})"})
            self._dark_since = now + 30  # rate-limit

        # 2. Severe blur -> unfocused / lens sprayed.
        if lap_var < self.blur_threshold:
            self._blur_since = self._blur_since or now
        else:
            self._blur_since = None
        if self._blur_since and now - self._blur_since >= 3:
            events.append({"type": "CAMERA_TAMPERING",
                           "message": f"Severe blur / lens obscured (lapvar={lap_var:.0f})"})
            self._blur_since = now + 30

        # 3. Frozen / zero-variance frame.
        if self._prev_gray is not None and gray.shape == self._prev_gray.shape:
            diff = float(cv2.absdiff(gray, self._prev_gray).mean())
            if diff < 0.5:
                self._frozen_since = self._frozen_since or now
            else:
                self._frozen_since = None
            if self._frozen_since and now - self._frozen_since >= self.frozen_seconds:
                events.append({"type": "CAMERA_TAMPERING",
                               "message": "Frozen / zero-variance stream detected"})
                self._frozen_since = now + 60
        self._prev_gray = gray

        return events


# ---------------------------------------------------------------------------
# Cross-camera re-identification manager (embedding store stub)
# ---------------------------------------------------------------------------
class ReIdManager:
    """
    Correlates face embeddings (128-d) across camera ids to keep one persistent
    global person ID.  In-memory store; swap for a vector DB for scale.
    """

    def __init__(self, match_threshold: float = 0.45) -> None:
        self.match_threshold = match_threshold
        self._embeddings: List[np.ndarray] = []
        self._person_ids: List[int] = []
        self._next_person = 1

    def register(self, camera_id: str, track_id: int,
                 embedding: np.ndarray) -> Tuple[int, bool]:
        """
        Register/associate an embedding.  Returns ``(global_person_id, is_new)``.
        """
        if embedding is None:
            return 0, False

        # Cosine distance matching.
        if self._embeddings:
            emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            gallery = np.vstack(self._embeddings).astype(np.float32)
            gallery /= (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-9)
            emb /= (np.linalg.norm(emb) + 1e-9)
            dists = 1.0 - (gallery @ emb.T).ravel()
            best = int(np.argmin(dists))
            if dists[best] <= self.match_threshold:
                return self._person_ids[best], False

        pid = self._next_person
        self._next_person += 1
        self._embeddings.append(np.asarray(embedding, dtype=np.float32))
        self._person_ids.append(pid)
        logger.debug("[reid] new global person %d from %s track %d",
                     pid, camera_id, track_id)
        return pid, True


# ---------------------------------------------------------------------------
# Behaviour analyzer (one per camera worker)
# ---------------------------------------------------------------------------
class BehaviorAnalyzer:
    """Combines zone/loitering/tailgating/tampering/re-id logic per camera."""

    def __init__(self, camera_id: str, camera_config: Dict[str, Any]) -> None:
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.features = set(camera_config.get("active_features", []))

        import config

        self.loitering_seconds = config.LOITERING_SECONDS
        self.tailgate_window = config.TAILGATE_WINDOW_SECONDS
        self.tracker = IoUTracker()
        self.tamper = TamperMonitor(
            darkness_threshold=config.TAMPER_DARKNESS_THRESHOLD,
            blur_threshold=config.TAMPER_BLUR_THRESHOLD,
            frozen_seconds=config.TAMPER_FROZEN_SECONDS,
        )
        self.reid = ReIdManager(match_threshold=config.FACE_RECOGNITION_TOLERANCE)

        # Zone-entry bookkeeping.
        self._zone_entry: Dict[int, float] = {}       # track_id -> entered at
        self._zone_alerted: Dict[int, bool] = {}      # loiter alert fired?
        self._known_entries: Deque[Tuple[float, float]] = deque(maxlen=8)
        self._last_loiter_alert: float = 0.0

        self._active_zones = [
            [(float(x), float(y)) for (x, y) in poly]
            for poly in camera_config.get("active_zones", [])
        ]

    # ------------------------------------------------------------------
    def analyze(self, frame: np.ndarray, debug_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run behaviour analytics on a processed frame; return alert events."""
        events: List[Dict[str, Any]] = []
        now = time.time()

        # -- Tracking -------------------------------------------------------
        detections = [dict(d) for d in debug_info.get("detections", [])]
        tracked = self.tracker.update(detections)

        # -- Tampering ------------------------------------------------------
        if "tampering" in self.features:
            events.extend(self.tamper.analyze(frame))

        # -- Zone intrusion & loitering --------------------------------------
        if "zone_intrusion" in self.features and self._active_zones:
            for obj in tracked:
                cx, cy = bbox_center(obj["bbox"])
                in_zone = any(
                    point_in_polygon((cx, cy), poly) for poly in self._active_zones
                )
                if in_zone:
                    if obj["track_id"] not in self._zone_entry:
                        self._zone_entry[obj["track_id"]] = now
                        self._zone_alerted[obj["track_id"]] = False
                        if obj.get("class") == "person":
                            events.append({
                                "type": "ZONE_INTRUSION",
                                "priority": "high",
                                "message": (f"Person entered restricted zone at "
                                            f"{self.camera_config.get('name')}"),
                                "bbox": obj["bbox"],
                            })
                    else:
                        dwell = now - self._zone_entry[obj["track_id"]]
                        if ("loitering" in self.features and obj.get("class") == "person"
                                and dwell > self.loitering_seconds
                                and not self._zone_alerted[obj["track_id"]]
                                and now - self._last_loiter_alert > 60):
                            self._zone_alerted[obj["track_id"]] = True
                            self._last_loiter_alert = now
                            events.append({
                                "type": "LOITERING_DETECTED",
                                "priority": "high",
                                "message": (f"Person loitering {dwell:.0f}s in zone "
                                            f"at {self.camera_config.get('name')}"),
                                "bbox": obj["bbox"],
                            })
                else:
                    self._zone_entry.pop(obj["track_id"], None)
                    self._zone_alerted.pop(obj["track_id"], None)

        # -- Tailgating -------------------------------------------------------
        if "tailgating" in self.features:
            faces = debug_info.get("faces", [])
            strangers = [f for f in faces if f.get("is_stranger")]
            known = [f for f in faces if not f.get("is_stranger")]

            for k in known:
                top, right, bottom, left = k["location"]
                self._known_entries.append((now, (left + right) / 2))

            # A stranger whose entry is within the window of a known entry and
            # horizontally close is flagged as tailgating.
            for s in strangers:
                top, right, bottom, left = s["location"]
                sx = (left + right) / 2
                for entry_time, kx in self._known_entries:
                    if now - entry_time <= self.tailgate_window and abs(sx - kx) < 0.3 * frame.shape[1]:
                        events.append({
                            "type": "TAILGATING",
                            "priority": "high",
                            "message": "Unknown person tailgating behind recognised person",
                            "bbox": (left, top, right, bottom),
                        })
                        break

        # -- Cross-camera re-id (embedding correlation) ------------------------
        for face in debug_info.get("faces", []):
            if face.get("encoding") is not None:
                pid, _ = self.reid.register(self.camera_id, 0, face["encoding"])
                debug_info.setdefault("tracks_meta", {})[pid] = {
                    "camera": self.camera_id, "name": face["name"],
                }

        return events
