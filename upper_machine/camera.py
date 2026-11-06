"""Threaded camera reader for non-blocking frame capture."""

import threading
import time

import cv2
import numpy as np


class ThreadedCamera:
    """Non-blocking threaded camera reader that continuously grabs the latest frame."""

    def __init__(self, src: int | str, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.ret: bool = False
        self.running: bool = False
        self.thread: threading.Thread | None = None

    def start(self) -> bool:
        if not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
            time.sleep(0.002)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap.isOpened():
            self.cap.release()
