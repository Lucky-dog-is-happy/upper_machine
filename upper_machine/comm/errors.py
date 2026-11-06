"""RDLC and serial transport exceptions."""


class RdlcError(Exception):
    """Base exception for all RDLC protocol errors."""


class RdlcEncodeError(RdlcError, ValueError):
    """Encoding error: invalid address, payload overflow, etc."""


class RdlcDecodeError(RdlcError, ValueError):
    """Decoding error: CRC mismatch, invalid frame markers, corrupted frame."""


class SerialTransportError(RuntimeError):
    """Raised when a serial port cannot be opened, read, or written."""
