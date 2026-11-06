"""Streaming RDLC frame encoder and decoder."""

from __future__ import annotations

from enum import Enum, auto

from .constants import (
    DEFAULT_MAX_PAYLOAD,
    FRAME_ESCAPE,
    FRAME_HEAD,
    FRAME_TAIL,
)
from .crc import crc16_modbus
from .errors import RdlcDecodeError, RdlcEncodeError
from .frame import RdlcFrame


class _State(Enum):
    WAIT_HEAD = auto()
    SRC = auto()
    DST = auto()
    LEN_LOW = auto()
    LEN_HIGH = auto()
    PAYLOAD = auto()
    CRC_LOW = auto()
    CRC_HIGH = auto()
    TAIL = auto()


class RdlcCodec:
    """Encodes frames and provides incremental streaming decoding for serial bytes."""

    def __init__(self, max_payload_size: int = DEFAULT_MAX_PAYLOAD,
                 escape_enabled: bool = False) -> None:
        if not 0 <= max_payload_size <= 0xFFFF - 6:
            raise ValueError("max_payload_size out of range")
        self.max_payload_size = max_payload_size
        self.escape_enabled = escape_enabled
        self.reset()

    def reset(self) -> None:
        """Reset parser state machine, dropping partial frame data."""
        self._state = _State.WAIT_HEAD
        self._escaped = False
        self._body = bytearray()
        self._payload_size = 0
        self._crc_low = 0

    def encode(self, src: int, dst: int, payload: bytes) -> bytes:
        """Encode src, dst, and payload into a complete wire frame."""
        payload = bytes(payload)
        if not 0 <= src <= 0xFF or not 0 <= dst <= 0xFF:
            raise RdlcEncodeError("Addresses must be in 0..255")
        if len(payload) > self.max_payload_size:
            raise RdlcEncodeError(
                f"Payload is {len(payload)} bytes; max is {self.max_payload_size}")
        body = bytes((src, dst, len(payload) & 0xFF, len(payload) >> 8)) + payload
        crc = crc16_modbus(payload)
        body += bytes((crc & 0xFF, crc >> 8))
        if not self.escape_enabled:
            return bytes((FRAME_HEAD,)) + body + bytes((FRAME_TAIL,))
        return bytes((FRAME_ESCAPE, FRAME_HEAD)) + self._escape(body) + bytes(
            (FRAME_ESCAPE, FRAME_TAIL))

    @staticmethod
    def _escape(data: bytes) -> bytes:
        encoded = bytearray()
        for byte in data:
            if byte in (FRAME_ESCAPE, FRAME_HEAD, FRAME_TAIL):
                encoded.append(FRAME_ESCAPE)
            encoded.append(byte)
        return bytes(encoded)

    def feed(self, data: bytes, *, strict: bool = False) -> list[RdlcFrame]:
        """Feed received byte chunks and yield all successfully decoded frames."""
        frames: list[RdlcFrame] = []
        for byte in bytes(data):
            try:
                frame = self._feed_byte(byte)
            except RdlcDecodeError:
                self.reset()
                if strict:
                    raise
                continue
            if frame is not None:
                frames.append(frame)
        return frames

    def _feed_byte(self, byte: int) -> RdlcFrame | None:
        if self.escape_enabled:
            if self._escaped:
                self._escaped = False
                if byte not in (FRAME_ESCAPE, FRAME_HEAD, FRAME_TAIL):
                    raise RdlcDecodeError("Invalid RDLC escape sequence")
                if self._state is _State.WAIT_HEAD:
                    if byte != FRAME_HEAD:
                        raise RdlcDecodeError("Invalid escaped frame head")
                    self._start_frame()
                    return None
                if self._state is _State.TAIL and byte == FRAME_TAIL:
                    return self._finish_frame()
                return self._consume_data(byte)
            if byte == FRAME_ESCAPE:
                self._escaped = True
                return None
            if self._state is _State.WAIT_HEAD:
                return None
            if byte == FRAME_HEAD:
                self._start_frame()
                return None
            if byte == FRAME_TAIL:
                raise RdlcDecodeError("Unexpected frame tail")
            return self._consume_data(byte)

        # Standard non-escaped mode
        if self._state is _State.WAIT_HEAD:
            if byte == FRAME_HEAD:
                self._start_frame()
            return None
        if byte == FRAME_HEAD:
            self._start_frame()
            return None
        if self._state is _State.TAIL:
            if byte != FRAME_TAIL:
                raise RdlcDecodeError("Missing RDLC frame tail")
            return self._finish_frame()
        if byte == FRAME_TAIL:
            raise RdlcDecodeError("Unexpected RDLC frame tail")
        return self._consume_data(byte)

    def _start_frame(self) -> None:
        self._state = _State.SRC
        self._body.clear()
        self._payload_size = 0
        self._crc_low = 0

    def _consume_data(self, byte: int) -> RdlcFrame | None:
        if self._state is _State.SRC:
            self._body.append(byte)
            self._state = _State.DST
        elif self._state is _State.DST:
            self._body.append(byte)
            self._state = _State.LEN_LOW
        elif self._state is _State.LEN_LOW:
            self._body.append(byte)
            self._payload_size = byte
            self._state = _State.LEN_HIGH
        elif self._state is _State.LEN_HIGH:
            self._body.append(byte)
            self._payload_size |= byte << 8
            if self._payload_size > self.max_payload_size:
                raise RdlcDecodeError("RDLC payload exceeded max allowed size")
            self._state = _State.PAYLOAD if self._payload_size else _State.CRC_LOW
        elif self._state is _State.PAYLOAD:
            self._body.append(byte)
            if len(self._body) == 4 + self._payload_size:
                self._state = _State.CRC_LOW
        elif self._state is _State.CRC_LOW:
            self._crc_low = byte
            self._state = _State.CRC_HIGH
        elif self._state is _State.CRC_HIGH:
            received = self._crc_low | (byte << 8)
            if received != crc16_modbus(bytes(self._body[4:])):
                raise RdlcDecodeError("RDLC CRC verification failed")
            self._state = _State.TAIL
        else:
            raise RdlcDecodeError("Invalid RDLC parser state")
        return None

    def _finish_frame(self) -> RdlcFrame:
        if len(self._body) != 4 + self._payload_size:
            raise RdlcDecodeError("Incomplete RDLC frame")
        frame = RdlcFrame(self._body[0], self._body[1], self._body[4:])
        self.reset()
        return frame
