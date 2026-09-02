"""
system_health.py
================
Reliability, automation and diagnostics for the security suite.

* **Heartbeat & offline monitor** - pings an external watchdog URL every 60 s
  so a remote server can alert the user if this machine goes offline.
* **Geofencing & arm/disarm automation** - pings the home router; when home
  devices are reachable the system auto-DISARMS, when away it auto-ARMS.
* **Battery / UPS power saver** - on battery (via ``psutil``) it disables the
  live stream encoding and non-critical AI modules (e.g. ANPR) while keeping
  primary intruder detection active.
* **Dynamic bitrate / speed sync** - when upload latency is high, JPEG quality
  is lowered before snapshots are sent to Telegram.
* **Daily security recap** - APScheduler job at 00:00 posts a summary
  (known / stranger / vehicle counts, storage used) to Telegram.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import requests

logger = logging.getLogger(__name__)

try:  # optional dependency
    from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
    APSCHEDULER_AVAILABLE = True
except Exception:  # pragma: no cover
    APSCHEDULER_AVAILABLE = False


class SystemHealth:
    """Heartbeat, geofencing, power saving, bitrate tuning, daily recap."""

    def __init__(self,
                 heartbeat_url: str = "",
                 heartbeat_interval: int = 60,
                 router_ip: str = "192.168.1.1",
                 geofence_interval: int = 30,
                 geofence_miss_threshold: int = 3,
                 telegram_sender: Optional[Callable[[str], None]] = None,
                 status_provider: Optional[Callable[[], str]] = None,
                 armed_setter: Optional[Callable[[bool], None]] = None,
                 jpeg_quality_setter: Optional[Callable[[int], None]] = None) -> None:
        self.heartbeat_url = heartbeat_url
        self.heartbeat_interval = heartbeat_interval
        self.router_ip = router_ip
        self.geofence_interval = geofence_interval
        self.geofence_miss_threshold = geofence_miss_threshold

        self.telegram_sender = telegram_sender
        self.status_provider = status_provider
        self.armed_setter = armed_setter
        self.jpeg_quality_setter = jpeg_quality_setter

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._scheduler = None

        # Geofence state.
        self._miss_count = 0
        self.home_detected = False
        self.armed = True

        # Network state.
        self.last_latency_ms: float = 0.0
        self.current_jpeg_quality = 85

        # Battery state.
        self.on_battery = False

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        """Start heartbeat + geofence threads and the daily recap job."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        if APSCHEDULER_AVAILABLE:
            try:
                self._scheduler = BackgroundScheduler()
                self._scheduler.add_job(self.daily_recap, "cron", hour=0, minute=0)
                self._scheduler.start()
                logger.info("daily recap scheduled for 00:00")
            except Exception as exc:
                logger.warning("APScheduler unavailable: %s", exc)
        logger.info("SystemHealth started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._scheduler:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass

    # ------------------------------------------------------------ main loop
    def _main_loop(self) -> None:
        last_heartbeat = 0.0
        last_geofence = 0.0
        last_latency = 0.0
        while self._running:
            now = time.time()
            if self.heartbeat_url and now - last_heartbeat >= self.heartbeat_interval:
                last_heartbeat = now
                self._send_heartbeat()
            if now - last_geofence >= self.geofence_interval:
                last_geofence = now
                self._geofence_check()
            if now - last_latency >= 60:
                last_latency = now
                self._check_network_and_tune()
            self._check_power()
            time.sleep(1)

    # ------------------------------------------------------------- heartbeat
    def _send_heartbeat(self) -> None:
        try:
            requests.get(self.heartbeat_url, timeout=10,
                         params={"system": "argus", "time": datetime.now().isoformat()})
        except Exception as exc:
            logger.error("heartbeat failed: %s", exc)

    # -------------------------------------------------------------- geofence
    def _ping_router(self) -> bool:
        """Ping the home router; returns True when the home network is present."""
        try:
            if subprocess.run(["ping", "-c", "1", "-W", "1", self.router_ip],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=3).returncode == 0:
                return True
        except Exception:
            pass
        return False

    def _geofence_check(self) -> None:
        reachable = self._ping_router()
        if reachable:
            self._miss_count = 0
            self.home_detected = True
        else:
            self._miss_count += 1
            self.home_detected = self._miss_count < self.geofence_miss_threshold

        new_armed = not self.home_detected
        if new_armed != self.armed:
            self.armed = new_armed
            state = "ARMED" if new_armed else "DISARMED"
            logger.info("Geofence: home=%s -> system %s", self.home_detected, state)
            if self.armed_setter:
                self.armed_setter(new_armed)
            if self.telegram_sender:
                self.telegram_sender(
                    f"📡 <b>Geofence update</b>: system is now <b>{state}</b>"
                    f" (home network {'detected' if self.home_detected else 'not detected'}).")

    # ------------------------------------------------------------- power
    def _check_power(self) -> None:
        try:
            import psutil  # type: ignore

            battery = psutil.sensors_battery()
            if battery is not None:
                self.on_battery = battery.power_plugged is False and battery.percent < 100
                if self.on_battery:
                    logger.info("On battery (%.0f%%) - power saver active", battery.percent)
        except Exception:
            pass

    def power_saver_features(self, features: list) -> list:
        """Drop non-critical features when on battery (keep intruder detection)."""
        if not self.on_battery:
            return features
        non_critical = {"anpr", "tailgating", "reid", "fire_smoke", "pose_fall"}
        kept = [f for f in features if f not in non_critical]
        logger.info("power saver: dropped %s", sorted(set(features) - set(kept)))
        return kept

    # ---------------------------------------------------- dynamic bitrate
    def _check_network_and_tune(self) -> None:
        """Probe upload latency; lower JPEG quality when the link is slow."""
        try:
            start = time.time()
            requests.get("https://api.telegram.org", timeout=5)
            latency = (time.time() - start) * 1000
            self.last_latency_ms = latency

            quality = self.current_jpeg_quality
            if latency > 800:
                quality = min(quality, 45)
            elif latency > 300:
                quality = min(quality, 60)
            else:
                quality = 85
            if quality != self.current_jpeg_quality:
                self.current_jpeg_quality = quality
                logger.info("link latency %.0f ms -> JPEG quality %d", latency, quality)
                if self.jpeg_quality_setter:
                    self.jpeg_quality_setter(quality)
        except Exception as exc:
            logger.error("network probe error: %s", exc)

    # ------------------------------------------------------------ recap
    def daily_recap(self, stats_provider: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        """
        Build & post the daily security recap.  ``stats_provider`` should return
        a dict like ``{"known": 3, "strangers": 5, "vehicles": 12,
        "storage_gb": 12.4, "events_by_type": {...}}``.
        """
        if not self.telegram_sender:
            return
        stats = stats_provider() if stats_provider else {}
        date = datetime.now().strftime("%Y-%m-%d")
        text = (
            f"📊 <b>Daily Security Recap - {date}</b>\n"
            f"👤 Known persons: <b>{stats.get('known', 0)}</b>\n"
            f"⚠️ Strangers: <b>{stats.get('strangers', 0)}</b>\n"
            f"🚗 Vehicles: <b>{stats.get('vehicles', 0)}</b>\n"
            f"📦 Storage used: <b>{stats.get('storage_gb', 0):.1f} GB</b>\n"
            f"🔔 Events: {stats.get('events_by_type', {})}"
        )
        try:
            self.telegram_sender(text)
            logger.info("daily recap posted")
        except Exception as exc:
            logger.error("daily recap failed: %s", exc)
