"""Self-contained RDLC serial communication module."""

from .codec import RdlcCodec
from .config import SerialConfig
from .constants import DEFAULT_BAUDRATE, DEFAULT_MAX_PAYLOAD, FRAME_HEAD, FRAME_TAIL
from .crc import crc16_modbus
from .errors import RdlcDecodeError, RdlcEncodeError, RdlcError, SerialTransportError
from .frame import RdlcFrame
from .rdlc_link import RdlcSerialLink
from .transport import SerialTransport

__all__ = [
    "DEFAULT_BAUDRATE",
    "DEFAULT_MAX_PAYLOAD",
    "FRAME_HEAD",
    "FRAME_TAIL",
    "crc16_modbus",
    "RdlcCodec",
    "RdlcDecodeError",
    "RdlcEncodeError",
    "RdlcError",
    "SerialTransportError",
    "RdlcFrame",
    "SerialConfig",
    "SerialTransport",
    "RdlcSerialLink",
]
