"""RDLC-over-serial link combining serial transport with streaming RDLC codec."""

from __future__ import annotations

import queue

from .codec import RdlcCodec
from .config import SerialConfig
from .frame import RdlcFrame
from .transport import SerialTransport


class RdlcSerialLink:
    """Combines threaded serial transport with streaming RDLC frame codec."""

    def __init__(self, config: SerialConfig) -> None:
        self.config = config
        self.codec = RdlcCodec(config.max_payload_size, config.escape_enabled)
        self._frames: queue.Queue[RdlcFrame] = queue.Queue()
        self.transport = SerialTransport(
            config.port, config.baudrate, config.timeout, self._on_bytes
        )

    def open(self) -> None:
        self.transport.open()

    def close(self) -> None:
        self.transport.close()

    def send(self, payload: bytes, *, dst: int | None = None, src: int | None = None) -> bytes:
        frame = self.codec.encode(
            self.config.local_address if src is None else src,
            self.config.remote_address if dst is None else dst,
            payload,
        )
        self.transport.write(frame)
        return frame

    def receive(self, timeout: float | None = None) -> RdlcFrame:
        return self._frames.get(timeout=timeout)

    def _on_bytes(self, data: bytes) -> None:
        for frame in self.codec.feed(data):
            self._frames.put(frame)
