"""Threaded serial byte transport built on pyserial."""

from __future__ import annotations

import threading
from collections.abc import Callable

from .errors import SerialTransportError


class SerialTransport:
    """Owns a pyserial port and delivers received bytes to an asynchronous callback."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.1,
                 on_bytes: Callable[[bytes], None] | None = None) -> None:
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.on_bytes = on_bytes
        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._write_lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    def open(self) -> None:
        if self.is_open:
            return
        try:
            import serial
        except ImportError as exc:
            raise SerialTransportError(
                "pyserial is required; please run `pip install pyserial`") from exc
        try:
            self._serial = serial.Serial(
                port=self.port_name, baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=self.timeout,
                xonxoff=False, rtscts=False, dsrdtr=False,
            )
        except Exception as exc:
            raise SerialTransportError(f"Cannot open serial port {self.port_name}: {exc}") from exc
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="serial-rx-thread", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.timeout * 3))
        self._thread = None
        if self._serial:
            self._serial.close()
        self._serial = None

    def write(self, data: bytes) -> None:
        if not self.is_open:
            raise SerialTransportError("Serial port is not open")
        try:
            with self._write_lock:
                self._serial.write(bytes(data))
        except Exception as exc:
            raise SerialTransportError(f"Serial write failed: {exc}") from exc

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._serial.read(self._serial.in_waiting or 1)
            except Exception:
                if not self._stop.is_set():
                    return
                return
            if data and self.on_bytes:
                self.on_bytes(bytes(data))
