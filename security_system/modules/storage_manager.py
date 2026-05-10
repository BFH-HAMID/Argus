"""
Storage Manager Module
Handles recording storage, auto-deletion, and disk space management
"""

import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class StorageManager:
    def __init__(
        self,
        storage_path: str,
        max_storage_gb: float = 50,
        auto_delete_days: int = 7
    ):
        """
        Initialize the storage manager.
        
        Args:
            storage_path: Path to recordings directory
            max_storage_gb: Maximum storage in GB before cleanup
            auto_delete_days: Delete files older than this many days
        """
        self.storage_path = storage_path
        self.max_storage_gb = max_storage_gb
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024
        self.auto_delete_days = auto_delete_days
        
        self.running = False
        self.thread = None
        self.check_interval = 3600  # Check every hour
        
        self._ensure_storage_directory()

    def _ensure_storage_directory(self) -> None:
        """Create storage directory if it doesn't exist."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            logger.info(f"Created storage directory: {self.storage_path}")

    def start(self) -> None:
        """Start the background storage monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Storage manager started")

    def stop(self) -> None:
        """Stop the storage monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        logger.info("Storage manager stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                self.cleanup()
                
                # Sleep in intervals to allow clean shutdown
                for _ in range(self.check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Storage monitor error: {e}")
                time.sleep(60)

    def cleanup(self) -> Tuple[int, int]:
        """
        Perform cleanup: delete old files and free space if needed.
        
        Returns:
            Tuple of (files_deleted, bytes_freed)
        """
        files_deleted = 0
        bytes_freed = 0
        
        # Delete old files
        old_deleted, old_bytes = self._delete_old_files()
        files_deleted += old_deleted
        bytes_freed += old_bytes
        
        # Check disk space and delete if needed
        space_deleted, space_bytes = self._free_disk_space()
        files_deleted += space_deleted
        bytes_freed += space_bytes
        
        if files_deleted > 0:
            logger.info(f"Cleanup complete: {files_deleted} files deleted, {bytes_freed / 1024 / 1024:.2f} MB freed")
        
        return files_deleted, bytes_freed

    def _delete_old_files(self) -> Tuple[int, int]:
        """Delete files older than auto_delete_days."""
        files_deleted = 0
        bytes_freed = 0
        
        cutoff_time = datetime.now() - timedelta(days=self.auto_delete_days)
        
        for filename in os.listdir(self.storage_path):
            filepath = os.path.join(self.storage_path, filename)
            
            if not os.path.isfile(filepath):
                continue
            
            try:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_mtime < cutoff_time:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    files_deleted += 1
                    bytes_freed += file_size
                    logger.debug(f"Deleted old file: {filename}")
                    
            except Exception as e:
                logger.error(f"Error deleting {filename}: {e}")
        
        return files_deleted, bytes_freed

    def _free_disk_space(self) -> Tuple[int, int]:
        """Delete oldest files if storage exceeds maximum."""
        files_deleted = 0
        bytes_freed = 0
        
        current_usage = self.get_storage_usage()
        
        if current_usage <= self.max_storage_bytes:
            return files_deleted, bytes_freed
        
        # Get all files sorted by modification time (oldest first)
        files = self._get_files_by_age()
        
        for filepath, file_size, _ in files:
            if current_usage <= self.max_storage_bytes * 0.8:  # Free until 80% capacity
                break
            
            try:
                os.remove(filepath)
                files_deleted += 1
                bytes_freed += file_size
                current_usage -= file_size
                logger.debug(f"Deleted to free space: {os.path.basename(filepath)}")
            except Exception as e:
                logger.error(f"Error freeing space: {e}")
        
        return files_deleted, bytes_freed

    def _get_files_by_age(self) -> List[Tuple[str, int, float]]:
        """
        Get all files sorted by modification time.
        
        Returns:
            List of tuples: (filepath, size, mtime)
        """
        files = []
        
        for filename in os.listdir(self.storage_path):
            filepath = os.path.join(self.storage_path, filename)
            
            if os.path.isfile(filepath):
                try:
                    file_size = os.path.getsize(filepath)
                    file_mtime = os.path.getmtime(filepath)
                    files.append((filepath, file_size, file_mtime))
                except:
                    pass
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[2])
        
        return files

    def get_storage_usage(self) -> int:
        """Get total storage usage in bytes."""
        total_size = 0
        
        for filename in os.listdir(self.storage_path):
            filepath = os.path.join(self.storage_path, filename)
            if os.path.isfile(filepath):
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        return total_size

    def get_storage_info(self) -> dict:
        """Get storage information."""
        usage = self.get_storage_usage()
        
        # Get disk space
        try:
            total, used, free = shutil.disk_usage(self.storage_path)
        except:
            total, used, free = 0, 0, 0
        
        file_count = len([f for f in os.listdir(self.storage_path) 
                         if os.path.isfile(os.path.join(self.storage_path, f))])
        
        return {
            'recordings_usage_bytes': usage,
            'recordings_usage_gb': usage / 1024 / 1024 / 1024,
            'max_storage_gb': self.max_storage_gb,
            'usage_percentage': (usage / self.max_storage_bytes) * 100 if self.max_storage_bytes > 0 else 0,
            'disk_total_gb': total / 1024 / 1024 / 1024,
            'disk_free_gb': free / 1024 / 1024 / 1024,
            'file_count': file_count,
            'auto_delete_days': self.auto_delete_days
        }

    def get_recording_path(self, prefix: str = "recording") -> str:
        """
        Generate a unique recording file path.
        
        Args:
            prefix: Prefix for the filename
            
        Returns:
            Full path for the new recording
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.mp4"
        return os.path.join(self.storage_path, filename)

    def get_snapshot_path(self, prefix: str = "snapshot") -> str:
        """
        Generate a unique snapshot file path.
        
        Args:
            prefix: Prefix for the filename
            
        Returns:
            Full path for the new snapshot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        return os.path.join(self.storage_path, filename)
