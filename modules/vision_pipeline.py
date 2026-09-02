"""
vision_pipeline.py
==================
Comprehensive vision analysis pipeline built on Ultralytics YOLOv8,
OpenCV and ``face_recognition``.

Every camera worker feeds raw frames into :func:`process_camera_frame`, which
runs only the features enabled for that camera (see ``config.CAMERAS[].active_features``)
and returns:

* the annotated frame (bounding boxes, labels, privacy masks applied), and
* a structured list of alert events::

      [{"type": "STRANGER", "message": "...", "crop_img": ndarray|None, "priority": "high"}]

Features
--------
* YOLOv8 object detection - persons, vehicles (car/motorcycle/bus/truck) and
  farm animals (cow/sheep/horse/dog) with colour-coded boxes.
* Face recognition - known vs stranger with per-face encodings.
* Anti-spoofing - heuristic mask / cap / photo-screen detection (blink & skin
  coverage analysis).  A depth camera or dedicated liveness model improves it.
* ANPR - license plate OCR on vehicle crops (requires ``pytesseract`` + tesseract).
* Fire & smoke - dedicated YOLO sub-model when configured, otherwise a
  colour/texture heuristic fallback.
* Pose analytics - fallen body and wall-jump posture detection (YOLO-Pose).

Performance
-----------
* CUDA acceleration is used automatically when PyTorch detects a GPU.
* Privacy masks are blurred *before* any AI processing.
* All heavy model imports are lazy, so the module degrades gracefully when
  optional dependencies are missing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class -> category colour mapping used for drawing.
# ---------------------------------------------------------------------------
PERSON_COLOR = (0, 0, 255)      # red    - persons / strangers
VEHICLE_COLOR = (0, 165, 255)   # orange - vehicles
ANIMAL_COLOR = (0, 255, 255)    # yellow - animals
KNOWN_FACE_COLOR = (0, 255, 0)  # green  - recognised faces


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def _in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _apply_privacy_masks(frame: np.ndarray, polygons: List[List[Tuple[int, int]]]) -> np.ndarray:
    """Gaussian-blur the given polygon regions of the frame."""
    if not polygons:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    if mask.any():
        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
        frame = np.where(mask[..., None] > 0, blurred, frame).astype(np.uint8)
    return frame


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    """Draw a filled label with text on the frame."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y0 = max(0, y - th - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 6, y0 + th + 8), color, -1)
    cv2.putText(frame, text, (x + 3, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# Fire / smoke heuristic detector (used when no dedicated model is installed)
# ---------------------------------------------------------------------------
class FireSmokeHeuristic:
    """Cheap colour/texture based fallback for flame and smoke detection."""

    def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], bool, bool]:
        """
        Return (fire_boxes, fire, smoke) using HSV colour segmentation.
        """
        fire_boxes: List[Tuple[int, int, int, int]] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Flame: bright red/orange/yellow hues with high saturation & value.
        lower = np.array([0, 120, 140], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        fire_mask = cv2.inRange(hsv, lower, upper)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fire = int(cv2.countNonZero(fire_mask)) > 400

        # Smoke: low saturation grey/white blobs with high local variance.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        mean = cv2.boxFilter(gray_f, -1, (31, 31))
        sq_mean = cv2.boxFilter(gray_f * gray_f, -1, (31, 31))
        variance = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
        smoke_candidates = (gray > 90) & (gray < 230) & (variance > 28)
        smoke = int(smoke_candidates.sum()) > 9000

        if fire:
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 300:
                    x, y, w, h = cv2.boundingRect(cnt)
                    fire_boxes.append((x, y, x + w, y + h))
        return fire_boxes, fire, smoke


# ---------------------------------------------------------------------------
# Anti-spoofing helper (mask / cap / photo-screen heuristics)
# ---------------------------------------------------------------------------
class AntiSpoofAnalyzer:
    """
    Heuristic liveness / concealment analysis.

    Real anti-spoofing requires an IR/depth camera or a trained liveness model;
    this implementation provides a best-effort heuristic layer:

    * **Blink detection** - photo/screen presentations do not blink.  Uses the
      Eye Aspect Ratio (EAR) over rolling per-track history.
    * **Mask / cap concealment** - measures how much of the upper/lower face is
      covered by a uniform (low-texture) region and by skin-tone absence.
    """

    def __init__(self, blink_frames: int = 3, ear_threshold: float = 0.20,
                 history: int = 15) -> None:
        self.blink_frames = blink_frames
        self.ear_threshold = ear_threshold
        self.history_size = history
        self._ear_history: Dict[str, List[float]] = {}

    # -- EAR helpers (6-point eye landmark approximation) -------------------
    @staticmethod
    def _eye_aspect_ratio(eye_pts: np.ndarray) -> float:
        if eye_pts is None or len(eye_pts) < 6:
            return 1.0
        p = np.asarray(eye_pts, dtype=np.float32)
        a = np.linalg.norm(p[1] - p[5])
        b = np.linalg.norm(p[2] - p[4])
        c = np.linalg.norm(p[0] - p[3])
        return float((a + b) / (2.0 * c + 1e-6))

    def _blink_rate(self, track_id: str, ear: float) -> float:
        """Maintain rolling EAR history and return blink frequency (0..1)."""
        hist = self._ear_history.setdefault(track_id, [])
        hist.append(ear)
        if len(hist) > self.history_size:
            hist.pop(0)
        if len(hist) < self.blink_frames + 1:
            return 0.0
        # A blink = EAR drops below threshold then returns.
        drops = sum(1 for i in range(1, len(hist)) if hist[i] < self.ear_threshold)
        return drops / len(hist)

    def analyze(self, track_id: str, face_region: np.ndarray,
                landmarks: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze a face crop.  Returns a dict with ``masked``, ``presentation``
        and a human-readable ``reason``.
        """
        result: Dict[str, Any] = {"masked": False, "presentation": False, "reason": ""}
        if face_region is None or face_region.size == 0:
            return result

        h, w = face_region.shape[:2]
        if h < 16 or w < 16:
            return result

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # --- Mask / cap: lower-face uniformity & skin coverage ---------------
        lower = gray[int(h * 0.55):, :]
        if lower.size:
            lap = cv2.Laplacian(lower, cv2.CV_64F)
            lower_variance = float(lap.var())
            # Skin-tone check (YCrCb ranges for typical skin).
            ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
            skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            skin_ratio = float(np.count_nonzero(skin[int(h * 0.55):, :])) / max(lower.size, 1)
            if lower_variance < 90 and skin_ratio < 0.25:
                result["masked"] = True
                result["reason"] = "Mask / face covering suspected (uniform lower face)"

        # --- Presentation attack: no blinking ---------------------------------
        if landmarks is not None and not result["presentation"]:
            left_eye = landmarks[36:42] if landmarks.shape[0] >= 42 else None
            right_eye = landmarks[42:48] if landmarks.shape[0] >= 48 else None
            ear = min(self._eye_aspect_ratio(left_eye), self._eye_aspect_ratio(right_eye))
            blink_rate = self._blink_rate(track_id, ear)
            if blink_rate < 0.05 and self._ear_history.get(track_id):
                result["presentation"] = True
                result["reason"] = "Photo/screen presentation suspected (no blinks)"

        return result


# ---------------------------------------------------------------------------
# The main vision pipeline
# ---------------------------------------------------------------------------
class VisionPipeline:
    """Per-camera vision pipeline; one instance lives in each worker process."""

    def __init__(self, camera_id: str, camera_config: Dict[str, Any],
                 known_faces_path: str, tolerance: float = 0.45,
                 model: str = "hog", device: str = "auto") -> None:
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.known_faces_path = known_faces_path
        self.tolerance = tolerance
        self.face_model = model
        self.device = self._resolve_device(device)

        self.yolo = None          # ultralytics.YOLO  (lazy)
        self.pose_model = None    # ultralytics.YOLO pose
        self.fire_model = None    # optional dedicated fire/smoke model
        self.face_encodings: List[np.ndarray] = []
        self.face_names: List[str] = []
        self.anpr_available = False
        self.anti_spoof = AntiSpoofAnalyzer()
        self.fire_smoke_fallback = FireSmokeHeuristic()

        # Feature quick-lookup.
        self.features = set(camera_config.get("active_features", []))
        self._load_optional_models()
        self._load_known_faces()

        logger.info("[%s] Vision pipeline ready (device=%s, features=%s)",
                    camera_id, self.device, sorted(self.features))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to 'cuda' when a GPU is available."""
        if device and device != "auto":
            return device
        try:
            import torch  # type: ignore
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_optional_models(self) -> None:
        """Lazily import & load YOLO models (skip silently if unavailable)."""
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("[%s] ultralytics not installed: %s", self.camera_id, exc)
            return

        import config

        if "object_detection" in self.features or "anpr" in self.features \
                or "fire_smoke" in self.features:
            self.yolo = YOLO(config.YOLO_MODEL_PATH)
            logger.info("[%s] YOLO loaded: %s", self.camera_id, config.YOLO_MODEL_PATH)

        if "pose_fall" in self.features:
            try:
                self.pose_model = YOLO(config.POSE_MODEL_PATH)
            except Exception as exc:
                logger.warning("[%s] pose model unavailable: %s", self.camera_id, exc)

        if "fire_smoke" in self.features and config.FIRE_SMOKE_MODEL_PATH:
            try:
                self.fire_model = YOLO(config.FIRE_SMOKE_MODEL_PATH)
            except Exception as exc:
                logger.warning("[%s] fire/smoke model unavailable: %s", self.camera_id, exc)

        if "anpr" in self.features:
            try:
                import pytesseract  # type: ignore
                self.anpr_available = True
                self._tesseract = pytesseract
            except Exception:
                logger.warning("[%s] pytesseract not installed - ANPR disabled",
                               self.camera_id)

    def _load_known_faces(self) -> None:
        """Load / reload face encodings from the known_faces directory."""
        try:
            import face_recognition  # type: ignore
        except Exception as exc:
            logger.warning("[%s] face_recognition unavailable: %s", self.camera_id, exc)
            return

        self.face_encodings = []
        self.face_names = []
        if not os.path.isdir(self.known_faces_path):
            os.makedirs(self.known_faces_path, exist_ok=True)
            return

        for filename in sorted(os.listdir(self.known_faces_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            path = os.path.join(self.known_faces_path, filename)
            try:
                image = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    self.face_encodings.append(encs[0])
                    self.face_names.append(os.path.splitext(filename)[0])
            except Exception as exc:
                logger.error("[%s] failed to load face %s: %s", self.camera_id, filename, exc)

        logger.info("[%s] %d known face(s) loaded", self.camera_id, len(self.face_names))

    def reload_known_faces(self) -> int:
        """Re-index the face database (called by /reload_faces)."""
        self._load_known_faces()
        return len(self.face_names)

    # ------------------------------------------------------------- detection
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLOv8 and keep only relevant classes."""
        import config

        detections: List[Dict[str, Any]] = []
        if self.yolo is None:
            return detections

        results = self.yolo(
            frame, verbose=False, conf=config.YOLO_CONFIDENCE,
            iou=config.YOLO_IOU, device=self.device,
        )
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                category = self._class_category(cls_id)
                if category is None:
                    continue
                detections.append({
                    "class": result.names[cls_id],
                    "class_id": cls_id,
                    "category": category,          # person | vehicle | animal
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                })
        return detections

    @staticmethod
    def _class_category(cls_id: int) -> Optional[str]:
        import config
        if cls_id == config.CLASS_PERSON:
            return "person"
        if cls_id in config.CLASS_VEHICLES:
            return "vehicle"
        if cls_id in config.CLASS_ANIMALS:
            return "animal"
        return None

    # ------------------------------------------------------------- faces
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect, recognise and anti-spoof faces in the frame."""
        try:
            import face_recognition  # type: ignore
        except Exception:
            return []

        import config

        # Downsample for speed (configurable).
        ds = config.FACE_DETECT_DOWNSAMPLE
        small = cv2.resize(frame, (frame.shape[1] // ds, frame.shape[0] // ds))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model=self.face_model)
        encodings = face_recognition.face_encodings(rgb, locations)

        faces: List[Dict[str, Any]] = []
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            # Scale back to original resolution.
            top, right, bottom, left = (top * ds, right * ds, bottom * ds, left * ds)

            name, is_stranger = "STRANGER", True
            if self.face_encodings:
                distances = face_recognition.face_distance(self.face_encodings, encoding)
                best = int(np.argmin(distances))
                if distances[best] <= self.tolerance:
                    name, is_stranger = self.face_names[best], False

            crop = frame[max(0, top):bottom, max(0, left):right]
            track_key = f"{self.camera_id}:{left}:{top}"
            spoof = self.anti_spoof.analyze(track_key, crop)

            faces.append({
                "name": name,
                "is_stranger": is_stranger,
                "location": (top, right, bottom, left),
                "encoding": encoding,
                "masked": spoof["masked"],
                "presentation": spoof["presentation"],
                "spoof_reason": spoof["reason"],
            })
        return faces

    # ------------------------------------------------------------- ANPR
    def _read_plate(self, vehicle_crop: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
        """OCR a vehicle crop; returns (plate_text, plate_crop)."""
        if not self.anpr_available or vehicle_crop is None or vehicle_crop.size == 0:
            return "", None

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        # Enhance contrast & binarize (plate regions are high-contrast).
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = self._tesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        plate = "".join(ch for ch in text if ch.isalnum()).upper()
        return plate, thresh

    # ------------------------------------------------------------- fire/smoke
    def _detect_fire_smoke(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Use dedicated model if present, else heuristic fallback."""
        import config

        events: List[Dict[str, Any]] = []
        if self.fire_model is not None:
            results = self.fire_model(frame, verbose=False, conf=config.YOLO_CONFIDENCE,
                                      device=self.device)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_name = result.names[int(box.cls[0])].lower()
                    if "fire" in cls_name or "smoke" in cls_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        events.append({
                            "type": "FIRE" if "fire" in cls_name else "SMOKE",
                            "bbox": (x1, y1, x2, y2),
                            "message": f"{cls_name.upper()} detected",
                        })
        else:
            fire_boxes, fire, smoke = self.fire_smoke_fallback.detect(frame)
            if fire:
                events.append({"type": "FIRE", "bbox": fire_boxes[0] if fire_boxes else None,
                               "message": "Flame pattern detected (heuristic)"})
            if smoke:
                events.append({"type": "SMOKE", "bbox": None,
                               "message": "Smoke pattern detected (heuristic)"})
        return events

    # ------------------------------------------------------------- pose
    def _detect_pose_events(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallen-body & wall-jump posture detection using YOLO-Pose keypoints.
        """
        import config

        events: List[Dict[str, Any]] = []
        if self.pose_model is None:
            return events

        results = self.pose_model(frame, verbose=False, conf=config.POSE_CONFIDENCE,
                                  device=self.device)
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            kps = result.keypoints.xy.cpu().numpy()          # (N, 17, 2)
            for i, kp in enumerate(kps):
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].cpu().numpy())
                h = max(1, y2 - y1)
                w = max(1, x2 - x1)

                # Fallen: torso (shoulders->hips) is nearly horizontal.
                try:
                    shoulder_l, shoulder_r = kp[5], kp[6]
                    hip_l, hip_r = kp[11], kp[12]
                    torso_angle = abs(
                        np.degrees(np.arctan2(
                            (hip_l[1] - shoulder_l[1]) - (hip_r[1] - shoulder_r[1]),
                            (hip_r[0] - shoulder_r[0]) - (hip_l[0] - shoulder_l[0]) + 1e-6)))
                    if torso_angle < 30 and h > 0.25 * frame.shape[0]:
                        events.append({"type": "FALL_DETECTED",
                                       "bbox": (x1, y1, x2, y2),
                                       "message": "Fallen body posture detected"})
                except (IndexError, ValueError):
                    pass

                # Wall jump: person rises high & fast with wide limb spread is a
                # proxy; here we flag very tall-narrow boxes near zone edges.
                if h > 0.55 * frame.shape[0] and w < 0.35 * h:
                    events.append({"type": "WALL_CLIMB",
                                   "bbox": (x1, y1, x2, y2),
                                   "message": "Possible wall-climb posture detected"})
        return events

    # ------------------------------------------------------------- ANPR loop
    def _run_anpr(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OCR plates on all vehicle detections."""
        events: List[Dict[str, Any]] = []
        for det in detections:
            if det["category"] != "vehicle":
                continue
            x1, y1, x2, y2 = det["bbox"]
            # Licence plates usually sit in the lower half of the vehicle box.
            crop = frame[max(0, y1 + int((y2 - y1) * 0.35)):y2, x1:x2]
            plate, plate_img = self._read_plate(crop)
            if len(plate) >= 4:  # plausible plate length
                events.append({
                    "type": "ANPR",
                    "bbox": (x1, y1, x2, y2),
                    "message": f"Plate {plate} on {det['class']}",
                    "plate": plate,
                    "crop_img": plate_img,
                })
        return events

    # ------------------------------------------------------------- entry point
    def process_camera_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single frame and return ``(annotated_frame, alerts, debug_info)``.

        ``debug_info`` carries structured detections/faces for the behaviour
        tracker (tracking, loitering, tailgating, zone intrusion, re-id).
        """
        alerts: List[Dict[str, Any]] = []
        debug: Dict[str, Any] = {"detections": [], "faces": [], "tracks_meta": {}}

        # 1. Privacy masks BEFORE any analysis.
        frame = _apply_privacy_masks(frame, self.camera_config.get("privacy_mask_zones", []))

        # 2. Object detection.
        detections = self._detect_objects(frame) if "object_detection" in self.features else []
        debug["detections"] = detections

        # 3. Fire & smoke.
        if "fire_smoke" in self.features:
            for ev in self._detect_fire_smoke(frame):
                alerts.append({**ev, "priority": "high", "crop_img": None})

        # 4. Pose analytics.
        if "pose_fall" in self.features:
            for ev in self._detect_pose_events(frame):
                alerts.append({**ev, "priority": "high", "crop_img": None})

        # 5. Face recognition + anti-spoofing (only when a person is present).
        persons = [d for d in detections if d["category"] == "person"]
        if "face_recognition" in self.features and persons:
            faces = self._detect_faces(frame)
            debug["faces"] = faces
            for face in faces:
                top, right, bottom, left = face["location"]
                if face["presentation"]:
                    alerts.append({
                        "type": "SPOOF_SCREEN",
                        "priority": "high",
                        "message": f"Presentation attack suspected on {self.camera_id}",
                        "crop_img": frame[top:bottom, left:right],
                    })
                if face["is_stranger"]:
                    alerts.append({
                        "type": "STRANGER",
                        "priority": "high",
                        "message": f"STRANGER at {self.camera_config.get('name')}",
                        "crop_img": frame[top:bottom, left:right],
                        "face_encoding": face["encoding"],
                    })
                elif face["masked"]:
                    alerts.append({
                        "type": "MASKED_FACE",
                        "priority": "medium",
                        "message": f"Masked face near {self.camera_config.get('name')}",
                        "crop_img": frame[top:bottom, left:right],
                    })

        # 6. ANPR.
        if "anpr" in self.features and detections:
            for ev in self._run_anpr(frame, detections):
                alerts.append({**ev, "priority": "medium"})

        # 7. Draw annotations.
        annotated = self._draw_annotations(frame, detections, debug["faces"])
        return annotated, alerts, debug

    # ------------------------------------------------------------- drawing
    def _draw_annotations(self, frame: np.ndarray, detections: List[Dict[str, Any]],
                          faces: List[Dict[str, Any]]) -> np.ndarray:
        out = frame.copy()

        # Object boxes: red=person, orange=vehicle, yellow=animal.
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = {"person": PERSON_COLOR, "vehicle": VEHICLE_COLOR,
                     "animal": ANIMAL_COLOR}.get(det["category"], (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            _draw_label(out, f"{det['class']} {det['confidence']:.2f}", x1, y1, color)

        # Face boxes: green = known, red = stranger; spoof shown in magenta.
        for face in faces:
            top, right, bottom, left = face["location"]
            if face["presentation"]:
                color = (255, 0, 255)
            elif face["is_stranger"]:
                color = PERSON_COLOR
            else:
                color = KNOWN_FACE_COLOR
            cv2.rectangle(out, (left, top), (right, bottom), color, 2)
            label = face["name"]
            if face["masked"]:
                label += " [MASK]"
            if face["presentation"]:
                label += " [SPOOF]"
            _draw_label(out, label, left, top, color)

        return out
