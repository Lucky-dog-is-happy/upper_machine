"""CRC-16/Modbus implementation consistent with MCU RDLC."""


def crc16_modbus(data: bytes) -> int:
    """Calculate CRC-16/Modbus checksum (init 0xFFFF, poly 0xA001)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc = crc >> 1
    return crc & 0xFFFF
