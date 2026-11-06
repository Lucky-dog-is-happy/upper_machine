"""Serial communication via RDLC protocol."""

import os
import struct
import sys
from queue import Empty

from dataclasses import dataclass
from .config import BlockInfo, RingInfo, TaskCode

try:
    from .comm import SerialConfig, RdlcSerialLink
    HAS_SERIAL = True
except ImportError as exc:
    HAS_SERIAL = False
    print(f"Warning: Serial communication module unavailable: {exc}", file=sys.stderr)


# ── Protocol constants ───────────────────────────────────────────────────────

MSG_VISION = 0x01         # 物块检测结果（批量，上位机->下位机）
MSG_RING = 0x02           # 色环检测结果（批量，上位机->下位机）
MSG_TASK_CODE = 0x03      # 任务码（上位机->下位机）
MSG_MODE_CMD = 0x04       # 检测模式指令（下位机->上位机）
MSG_GRIPPER_COUNT = 0x05  # 抓取与放置统计指令（下位机->上位机）

MODE_BLOCK = 0            # 识别物料
MODE_RING = 1             # 识别色环

# MSG_GRIPPER_COUNT action codes
GRIPPER_PICK_INC = 0x01   # 抓取成功 +1
GRIPPER_PICK_DEC = 0x02   # 抓取成功 -1
GRIPPER_PLACE_INC = 0x03  # 放置成功 +1
GRIPPER_PLACE_DEC = 0x04  # 放置成功 -1
GRIPPER_RESET = 0x05      # 计数值清零
GRIPPER_SET = 0x10        # 直接同步计数 [0x05, 0x10, pick, place]


@dataclass
class SerialRxEvents:
    new_mode: int | None = None
    pick_delta: int = 0
    place_delta: int = 0
    set_pick: int | None = None
    set_place: int | None = None


class SerialSender:
    """RDLC-based serial sender for vision data."""

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.link = None

    def open(self) -> bool:
        if not HAS_SERIAL:
            return False
        try:
            config = SerialConfig(
                port=self.port,
                baudrate=self.baudrate,
                local_address=0x01,
                remote_address=0x02,
            )
            self.link = RdlcSerialLink(config)
            self.link.open()
            print(f"Serial opened: {self.port} @ {self.baudrate} (RDLC)")
            return True
        except Exception as e:
            print(f"Serial open failed: {e}", file=sys.stderr)
            return False

    def close(self):
        if self.link:
            self.link.close()

    def send_task_code(self, tc: TaskCode):
        if not self.link:
            return
        raw_bytes = tc.raw.encode("utf-8")
        payload = struct.pack("<B", MSG_TASK_CODE) + raw_bytes
        try:
            self.link.send(payload, dst=0x02, src=0x01)
            print(f"[TX] TaskCode: {tc.raw}")
        except Exception as e:
            print(f"Send task_code error: {e}", file=sys.stderr)

    def send_blocks(self, blocks: list[BlockInfo]):
        if not self.link:
            return
        count = min(len(blocks), 3)
        block_data = b""
        for b in blocks[:count]:
            block_data += struct.pack("<Bfff", b.color_id,
                                      float(b.cx), float(b.cy), float(b.radius))
        payload = struct.pack("<BB", MSG_VISION, count) + block_data
        try:
            self.link.send(payload, dst=0x02, src=0x01)
            desc = [f"#{b.color_id}@({b.cx},{b.cy})" for b in blocks[:count]]
            if desc:
                print(f"[TX] Blocks ({count}): {', '.join(desc)}")
            else:
                print(f"[TX] Blocks (0): none")
        except Exception as e:
            print(f"Send vision error: {e}", file=sys.stderr)

    def send_rings(self, rings: list[RingInfo]):
        if not self.link:
            return
        count = min(len(rings), 3)
        ring_data = b""
        desc_list = []
        for r in rings[:count]:
            target_id = getattr(r, "digit_id", r.color_id)
            ring_data += struct.pack("<Bfff", target_id,
                                     float(r.cx), float(r.cy), float(r.radius))
            desc_list.append(f"#{target_id}@({r.cx},{r.cy})")
        payload = struct.pack("<BB", MSG_RING, count) + ring_data
        try:
            self.link.send(payload, dst=0x02, src=0x01)
            if desc_list:
                print(f"[TX] Rings ({count}): {', '.join(desc_list)}")
            else:
                print(f"[TX] Rings (0): none")
        except Exception as e:
            print(f"Send ring error: {e}", file=sys.stderr)

    def poll_events(self) -> SerialRxEvents:
        """Non-blocking read of all incoming lower-board commands (mode switch and gripper stats).

        Drains the receive queue and accumulates all events.
        """
        events = SerialRxEvents()
        if not self.link:
            return events

        while True:
            try:
                frame = self.link.receive(timeout=0)
                payload = frame.payload
                if len(payload) < 2:
                    continue

                msg_id = payload[0]
                if msg_id == MSG_MODE_CMD:
                    mode = payload[1]
                    if mode in (MODE_BLOCK, MODE_RING):
                        events.new_mode = mode

                elif msg_id == MSG_GRIPPER_COUNT:
                    action = payload[1]
                    if action == GRIPPER_PICK_INC:
                        events.pick_delta += 1
                    elif action == GRIPPER_PICK_DEC:
                        events.pick_delta -= 1
                    elif action == GRIPPER_PLACE_INC:
                        events.place_delta += 1
                    elif action == GRIPPER_PLACE_DEC:
                        events.place_delta -= 1
                    elif action == GRIPPER_RESET:
                        events.set_pick = 0
                        events.set_place = 0
                    elif action == GRIPPER_SET and len(payload) >= 4:
                        events.set_pick = int(payload[2])
                        events.set_place = int(payload[3])

            except Empty:
                break
            except Exception as e:
                print(f"[RX Error] Frame decode error: {e}", file=sys.stderr)
                break

        return events

    def poll_command(self) -> int | None:
        """Backward compatibility wrapper: returns latest mode command if any."""
        events = self.poll_events()
        return events.new_mode
