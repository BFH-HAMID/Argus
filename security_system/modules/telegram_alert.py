"""
Telegram Alert Module
Sends alerts and snapshots to Telegram
"""

import requests
import cv2
import numpy as np
import io
import threading
from queue import Queue
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramAlert:
    def __init__(self, token: str, chat_id: str):
        """
        Initialize the Telegram alert system.
        
        Args:
            token: Telegram Bot API token
            chat_id: Target chat ID for alerts
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        
        self.message_queue = Queue()
        self.running = False
        self.thread = None
        
        self._validate_credentials()

    def _validate_credentials(self) -> bool:
        """Validate Telegram bot credentials."""
        if not self.token or self.token == "YOUR_TELEGRAM_BOT_TOKEN":
            logger.warning("Telegram token not configured")
            return False
        
        if not self.chat_id or self.chat_id == "YOUR_CHAT_ID":
            logger.warning("Telegram chat ID not configured")
            return False
        
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            if response.status_code == 200:
                bot_info = response.json()
                logger.info(f"Telegram bot connected: @{bot_info['result']['username']}")
                return True
            else:
                logger.error(f"Telegram validation failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram connection error: {e}")
            return False

    def start(self) -> None:
        """Start the message sending thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._send_loop, daemon=True)
        self.thread.start()
        logger.info("Telegram alert system started")

    def stop(self) -> None:
        """Stop the message sending thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        logger.info("Telegram alert system stopped")

    def _send_loop(self) -> None:
        """Process queued messages."""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message_data = self.message_queue.get(timeout=1)
                    self._process_message(message_data)
                else:
                    import time
                    time.sleep(0.1)
            except Exception as e:
                if self.running:
                    logger.error(f"Message send loop error: {e}")

    def _process_message(self, message_data: dict) -> None:
        """Process a queued message."""
        msg_type = message_data.get('type', 'text')
        
        if msg_type == 'text':
            self._send_text(message_data['text'])
        elif msg_type == 'photo':
            self._send_photo(message_data['image'], message_data.get('caption', ''))
        elif msg_type == 'video':
            self._send_video(message_data['video_path'], message_data.get('caption', ''))

    def _send_text(self, text: str) -> bool:
        """Send a text message."""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                data={
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': 'HTML'
                },
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending text: {e}")
            return False

    def _send_photo(self, image: np.ndarray, caption: str = "") -> bool:
        """Send a photo."""
        try:
            # Encode image to JPEG
            _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_bytes = io.BytesIO(img_encoded.tobytes())
            img_bytes.name = 'snapshot.jpg'
            
            response = requests.post(
                f"{self.base_url}/sendPhoto",
                data={
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                },
                files={'photo': img_bytes},
                timeout=60
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending photo: {e}")
            return False

    def _send_video(self, video_path: str, caption: str = "") -> bool:
        """Send a video file."""
        try:
            with open(video_path, 'rb') as video_file:
                response = requests.post(
                    f"{self.base_url}/sendVideo",
                    data={
                        'chat_id': self.chat_id,
                        'caption': caption,
                        'parse_mode': 'HTML'
                    },
                    files={'video': video_file},
                    timeout=120
                )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending video: {e}")
            return False

    def send_alert(self, message: str) -> None:
        """Queue a text alert."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"🚨 <b>SECURITY ALERT</b>\n\n{message}\n\n🕐 {timestamp}"
        self.message_queue.put({'type': 'text', 'text': formatted_message})

    def send_snapshot(self, image: np.ndarray, alert_type: str, details: str = "") -> None:
        """Queue a snapshot with caption."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caption = f"🚨 <b>{alert_type}</b>\n{details}\n🕐 {timestamp}"
        self.message_queue.put({'type': 'photo', 'image': image, 'caption': caption})

    def send_video_clip(self, video_path: str, alert_type: str) -> None:
        """Queue a video clip."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caption = f"📹 <b>{alert_type}</b>\n🕐 {timestamp}"
        self.message_queue.put({'type': 'video', 'video_path': video_path, 'caption': caption})

    def send_person_alert(self, image: np.ndarray, person_name: str, is_stranger: bool) -> None:
        """Send alert for person detection."""
        if is_stranger:
            alert_type = "STRANGER DETECTED"
            details = "⚠️ Unknown person detected on camera!"
        else:
            alert_type = "KNOWN PERSON"
            details = f"✅ {person_name} detected on camera"
        
        self.send_snapshot(image, alert_type, details)

    def send_vehicle_alert(self, image: np.ndarray, vehicle_type: str) -> None:
        """Send alert for vehicle detection."""
        self.send_snapshot(image, "VEHICLE DETECTED", f"🚗 {vehicle_type} detected")

    def send_sound_alert(self, sound_level: int) -> None:
        """Send alert for loud sound."""
        self.send_alert(f"🔊 <b>LOUD SOUND DETECTED</b>\nSound level: {sound_level}")

    def send_startup_message(self) -> None:
        """Send system startup notification."""
        self.message_queue.put({
            'type': 'text',
            'text': "🟢 <b>Security System Online</b>\n\nMonitoring started."
        })

    def send_shutdown_message(self) -> None:
        """Send system shutdown notification."""
        self._send_text("🔴 <b>Security System Offline</b>\n\nMonitoring stopped.")
