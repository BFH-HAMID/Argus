"""
dashboard_server.py
===================
Flask web server providing the real-time Multi-Camera Grid View GUI.

* ``/video_feed/<cam_id>`` - MJPEG streams pulling frames from the shared
  multiprocessing queues.
* ``/`` - responsive grid dashboard (auto 2x2 / 3x3 / 4x4 / 5x4 grid based on
  camera count) with per-tile overlays (name, FPS, REC, night mode, load),
  a control panel (record / snapshot / night-mode / panic siren) and an
  SSE event-log sidebar.
* Basic-auth or session-cookie password protection.

The server runs in the *main* process; frames arrive from camera workers
through ``multiprocessing.Queue`` objects registered in a registry dict.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _auth_required(func):
    """Decorator enforcing password auth on the wrapped Flask route."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import flask

        app = flask.current_app
        password = app.config.get("DASHBOARD_PASSWORD", "admin123")
        mode = app.config.get("AUTH_MODE", "session")

        if mode == "basic":
            import base64
            auth = flask.request.headers.get("Authorization", "")
            if auth.startswith("Basic "):
                try:
                    decoded = base64.b64decode(auth[6:]).decode()
                    if decoded.split(":", 1)[1] == password:
                        return func(*args, **kwargs)
                except Exception:
                    pass
            return ("Unauthorized", 401,
                    {"WWW-Authenticate": 'Basic realm="Argus"'})
        # session-cookie mode
        if flask.session.get("authed") is True:
            return func(*args, **kwargs)
        if flask.request.method == "POST" and \
                flask.request.form.get("password") == password:
            flask.session["authed"] = True
            return func(*args, **kwargs)
        return ("<html><body style='font-family:sans-serif;background:#0b0f14;"
                "color:#e6edf3'><h2>🛡 Argus Dashboard</h2>"
                "<form method='post' action='/login'><input type='password' "
                "name='password' placeholder='password'>"
                "<button type='submit'>Login</button></form></body></html>", 401)
    return wrapper


class DashboardServer:
    """Flask app wrapper: MJPEG feeds, grid UI, controls, SSE log."""

    def __init__(self, cameras: list, frame_queues: Dict[str, "queue.Queue"],
                 host: str = "0.0.0.0", port: int = 5000,
                 password: str = "admin123", auth_mode: str = "session") -> None:
        self.cameras = cameras
        self.frame_queues = frame_queues
        self.host = host
        self.port = port
        self.password = password
        self.auth_mode = auth_mode

        # Runtime status registry (updated by workers/other modules).
        self.status: Dict[str, Dict[str, Any]] = {}
        self.controls: Dict[str, Dict[str, bool]] = {}   # e.g. force night mode
        self.manual_recording: Dict[str, bool] = {}
        self.event_log: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=500)
        self._log_lock = threading.Lock()

        # Forwarder: fn(cam_id, action) that pushes commands into the camera
        # worker's command queue (wired by main orchestrator).
        self._cmd_sender = None
        self._build_app()

    def set_command_sender(self, fn) -> None:
        """Set the callback used to forward control commands to camera workers."""
        self._cmd_sender = fn

    # ---------------------------------------------------------------- app
    def _build_app(self) -> None:
        import flask

        app = flask.Flask(__name__)
        app.secret_key = "argus-dashboard-secret"  # override in production
        app.config["DASHBOARD_PASSWORD"] = self.password
        app.config["AUTH_MODE"] = self.auth_mode
        self.app = app

        @app.route("/login", methods=["GET", "POST"])
        def login():
            if flask.request.method == "POST" and \
                    flask.request.form.get("password") == self.password:
                flask.session["authed"] = True
                return flask.redirect("/")
            return ("<html><body style='font-family:sans-serif;background:#0b0f14;"
                    "color:#e6edf3'><h2>🛡 Argus Dashboard</h2>"
                    "<form method='post'><input type='password' name='password'>"
                    "<button type='submit'>Login</button></form></body></html>", 401)

        @app.route("/")
        @_auth_required
        def index():
            return flask.Response(self._render_index(), mimetype="text/html")

        @app.route("/video_feed/<cam_id>")
        @_auth_required
        def video_feed(cam_id):
            return flask.Response(self._mjpeg_generator(cam_id),
                                  mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/api/status")
        @_auth_required
        def api_status():
            return flask.jsonify({
                "cameras": self.status,
                "controls": self.controls,
                "system_time": datetime.now().isoformat(),
            })

        @app.route("/api/eventlog")
        @_auth_required
        def event_log_stream():
            return flask.Response(self._sse_generator(),
                                  mimetype="text/event-stream",
                                  headers={"Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})

        # --- Control panel endpoints (POST) ---
        @app.route("/api/control/<cam_id>/record", methods=["POST"])
        @_auth_required
        def control_record(cam_id):
            self.manual_recording[cam_id] = not self.manual_recording.get(cam_id, False)
            if self._cmd_sender:
                self._cmd_sender(cam_id, "record")
            return flask.jsonify({"ok": True, "recording": self.manual_recording[cam_id]})

        @app.route("/api/control/<cam_id>/snapshot", methods=["POST"])
        @_auth_required
        def control_snapshot(cam_id):
            self.controls.setdefault(cam_id, {})["snapshot_requested"] = True
            if self._cmd_sender:
                self._cmd_sender(cam_id, "snapshot")
            return flask.jsonify({"ok": True})

        @app.route("/api/control/<cam_id>/night", methods=["POST"])
        @_auth_required
        def control_night(cam_id):
            ctrl = self.controls.setdefault(cam_id, {})
            ctrl["force_night"] = not ctrl.get("force_night", False)
            if self._cmd_sender:
                self._cmd_sender(cam_id, "night")
            return flask.jsonify({"ok": True, "force_night": ctrl["force_night"]})

        @app.route("/api/control/siren", methods=["POST"])
        @_auth_required
        def control_siren():
            self.controls.setdefault("global", {})["panic_siren"] = True
            if self._cmd_sender:
                for cam_id in [c["id"] for c in self.cameras]:
                    self._cmd_sender(cam_id, "siren")
            return flask.jsonify({"ok": True, "message": "Panic siren triggered"})

    # ---------------------------------------------------------------- utils
    def log_event(self, cam_id: str, event_type: str, message: str = "") -> None:
        """Publish an event to the SSE log (thread-safe, from any process/module)."""
        item = {"time": datetime.now().strftime("%H:%M:%S"),
                "cam_id": cam_id, "type": event_type, "message": message}
        with self._log_lock:
            try:
                self.event_log.put_nowait(item)
            except queue.Full:
                try:
                    self.event_log.get_nowait()
                    self.event_log.put_nowait(item)
                except queue.Empty:
                    pass

    def _render_index(self) -> str:
        """Render the grid dashboard HTML with embedded CSS/JS."""
        cam_ids = [c["id"] for c in self.cameras]
        n = max(1, len(cam_ids))
        if n <= 4:
            grid_cols, grid_rows = 2, 2
        elif n <= 9:
            grid_cols, grid_rows = 3, 3
        elif n <= 16:
            grid_cols, grid_rows = 4, 4
        else:
            grid_cols, grid_rows = 5, 4

        cam_json = str([c["id"] for c in self.cameras]).replace("'", '"')
        cam_names = {c["id"]: c.get("name", c["id"]) for c in self.cameras}
        names_json = str(cam_names).replace("'", '"')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Argus - Multi-Camera Security</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0b0f14; color: #e6edf3; font-family: 'Segoe UI', system-ui, sans-serif; }}
  header {{ padding: 12px 18px; background: #11161d; border-bottom: 1px solid #1f2937;
           display: flex; justify-content: space-between; align-items: center; }}
  h1 {{ font-size: 18px; color: #7dd3fc; }} .dim {{ color: #8b98a5; font-size: 12px; }}
  main {{ display: grid; grid-template-columns: 1fr 300px; gap: 0; min-height: calc(100vh - 50px); }}
  #grid {{ display: grid; grid-template-columns: repeat({grid_cols}, 1fr);
          grid-auto-rows: minmax(180px, auto); gap: 2px; padding: 2px; }}
  .tile {{ position: relative; background: #000; overflow: hidden; }}
  .tile img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .overlay {{ position: absolute; inset: 0; pointer-events: none; }}
  .ov-top {{ position: absolute; top: 6px; left: 8px; right: 8px; display: flex;
            justify-content: space-between; font-size: 12px; text-shadow: 0 1px 3px #000; }}
  .rec {{ color: #f87171; font-weight: 700; animation: blink 1s infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0.2; }} }}
  .badge {{ background: rgba(0,0,0,.55); padding: 2px 7px; border-radius: 4px; }}
  .ov-bot {{ position: absolute; bottom: 4px; left: 8px; right: 8px; font-size: 11px;
            color: #cbd5e1; display: flex; justify-content: space-between; }}
  #sidebar {{ background: #11161d; border-left: 1px solid #1f2937; display: flex;
             flex-direction: column; }}
  #log {{ flex: 1; overflow-y: auto; padding: 8px; font-size: 12px; }}
  .log-item {{ padding: 5px 8px; border-bottom: 1px solid #1a2230; }}
  .log-time {{ color: #64748b; margin-right: 6px; }}
  .log-type {{ font-weight: 700; }} .t-stranger {{ color:#f87171; }} .t-vehicle {{ color:#fb923c; }}
  .t-fire,.t-loud {{ color:#facc15; }} .t-known {{ color:#4ade80; }}
  .controls {{ padding: 8px; border-top: 1px solid #1f2937; }}
  .btn {{ background:#1d4ed8; color:#fff; border:0; border-radius:6px; padding:6px 10px;
         font-size:12px; margin:2px; cursor:pointer; }} .btn.red {{ background:#b91c1c; }}
  .btn:hover {{ filter: brightness(1.15); }}
</style>
</head>
<body>
<header>
  <h1>🛡 Argus Security — Camera Grid</h1>
  <span class="dim" id="sysclock"></span>
</header>
<main>
  <div id="grid"></div>
  <aside id="sidebar">
    <div id="log"><div class="dim" style="padding:8px">Connecting to event feed…</div></div>
    <div class="controls">
      <button class="btn red" onclick="triggerSiren()">🚨 Panic Siren</button>
      <button class="btn" onclick="refreshStatus()">⟳ Refresh</button>
      <div class="dim" style="margin-top:6px">Click a tile's ⏺ button for manual control.</div>
    </div>
  </aside>
</main>
<script>
  const CAMERAS = {cam_json};
  const NAMES = {names_json};
  const grid = document.getElementById('grid');

  CAMERAS.forEach(id => {{
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.innerHTML = `
      <img src="/video_feed/${{id}}" alt="${{id}}">
      <div class="overlay">
        <div class="ov-top">
          <span class="badge">📷 ${{NAMES[id]}} <span id="fps-${{id}}"></span></span>
          <span id="flags-${{id}}"></span>
        </div>
        <div class="ov-bot">
          <span id="rec-${{id}}"></span>
          <span><button class="btn" style="padding:2px 8px" onclick="toggleRecord('${{id}}')">⏺</button>
                 <button class="btn" style="padding:2px 8px" onclick="snapshot('${{id}}')">📸</button>
                 <button class="btn" style="padding:2px 8px" onclick="toggleNight('${{id}}')">🌙</button></span>
        </div>
      </div>`;
    grid.appendChild(tile);
  }});

  async function refreshStatus() {{
    try {{
      const r = await fetch('/api/status');
      const data = await r.json();
      for (const id of CAMERAS) {{
        const s = data.cameras[id] || {{}};
        const el = document.getElementById('fps-' + id);
        if (el) el.textContent = s.fps ? s.fps.toFixed(1) + ' fps' : '';
        const flags = document.getElementById('flags-' + id);
        if (flags) {{
          const parts = [];
          if (s.night) parts.push('🌙 NIGHT');
          if (s.load !== undefined) parts.push('⚡ ' + Math.round(s.load * 100) + '%');
          flags.textContent = parts.join(' ');
        }}
        const rec = document.getElementById('rec-' + id);
        if (rec) rec.innerHTML = s.rec ? '<span class="rec">● REC</span>' : '';
      }}
    }} catch (e) {{}}
  }}

  // Server-Sent Events for the log sidebar.
  const logEl = document.getElementById('log');
  const es = new EventSource('/api/eventlog');
    es.onmessage = e => {{
    const item = JSON.parse(e.data);
    const div = document.createElement('div');
    div.className = 'log-item';
    div.innerHTML = `<span class="log-time">${{item.time}}</span>` +
      `<span class="log-type t-${{item.type.toLowerCase()}}">[${{item.type}}]</span> ` +
      `<b>${{item.cam_id}}</b> ${{item.message}}`;
    logEl.prepend(div);
    while (logEl.children.length > 150) logEl.removeChild(logEl.lastChild);
  }};

  async function post(path) {{ await fetch(path, {{method:'POST'}}); refreshStatus(); }}
  function toggleRecord(id)  {{ post('/api/control/' + id + '/record'); }}
  function snapshot(id)      {{ post('/api/control/' + id + '/snapshot'); }}
  function toggleNight(id)   {{ post('/api/control/' + id + '/night'); }}
  function triggerSiren()    {{ post('/api/control/siren'); }}

  setInterval(refreshStatus, 3000);
  setInterval(() => {{
    document.getElementById('sysclock').textContent = new Date().toLocaleString();
  }}, 1000);
</script>
</body>
</html>"""

    # ----------------------------------------------------------- MJPEG feed
    def _mjpeg_generator(self, cam_id: str):
        """Yield JPEG frames from the camera's shared queue."""
        q = self.frame_queues.get(cam_id)
        quality = 70
        while True:
            frame = None
            if q is not None:
                try:
                    payload = q.get(timeout=1.0)
                    if isinstance(payload, tuple):
                        # Workers publish ("frame", cam_id, ndarray).
                        frame = payload[2] if len(payload) > 2 else None
                    else:
                        frame = payload
                except queue.Empty:
                    pass
            if frame is None:
                # Blank "no signal" tile.
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"{cam_id} - no signal", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                frame = blank
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")

    # ------------------------------------------------------------- SSE feed
    def _sse_generator(self):
        """Push new events to connected dashboards."""
        import json
        while True:
            try:
                item = self.event_log.get(timeout=10)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        """Start Flask in a background thread (does not block)."""
        threading.Thread(target=lambda: self.app.run(
            host=self.host, port=self.port, threaded=True,
            debug=False, use_reloader=False), daemon=True).start()
        logger.info("Dashboard listening on http://%s:%d", self.host, self.port)
