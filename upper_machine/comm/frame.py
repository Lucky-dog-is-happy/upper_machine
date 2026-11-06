"""RDLC frame representation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RdlcFrame:
    """An RDLC frame with source, destination, and binary payload."""

    src: int
    dst: int
    payload: bytes

    def __post_init__(self) -> None:
        if not 0 <= self.src <= 0xFF or not 0 <= self.dst <= 0xFF:
            raise ValueError("RDLC addresses must be in 0..255")
        object.__setattr__(self, "payload", bytes(self.payload))
