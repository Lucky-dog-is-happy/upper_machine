"""RDLC protocol constants."""

FRAME_ESCAPE = 0xFF        # 转义字节（转义模式下使用）
FRAME_HEAD = 0xC0          # 帧头标记
FRAME_TAIL = 0x0C          # 帧尾标记
DEFAULT_BAUDRATE = 115200  # 默认串口波特率
DEFAULT_MAX_PAYLOAD = 128  # 帧载荷最大字节数（发送与接收防御上限）
