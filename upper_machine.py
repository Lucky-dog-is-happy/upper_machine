#!/usr/bin/env python3
"""Upper machine: QR code + color block detection with serial output.

Detects QR codes (task codes) and colored circular blocks,
sends results to lower board via USB-TTL serial.

Task code format: 156+123+516+231
  - Group 1: batch1 colors (1=red,2=yellow,3=blue,4=green,5=black,6=sky-blue)
  - Group 2: batch1 placement positions (ring numbers)
  - Group 3: batch2 colors
  - Group 4: batch2 placement positions
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

import cv2
import numpy as np

import struct

try:
    from upper_machine.comm import SerialConfig, RdlcSerialLink
    HAS_SERIAL = True
except ImportError as exc:
    HAS_SERIAL = False
    print(f"Warning: Serial communication module unavailable: {exc}", file=sys.stderr)

try:
    from upper_machine.digit import (
        DigitClassifier, extract_digit_scores_from_ring, match_rings_to_digits,
    )
    HAS_DIGIT_CLASSIFIER = True
except ImportError:
    HAS_DIGIT_CLASSIFIER = False


# ── Color config ──────────────────────────────────────────────────────────────

COLOR_MAP = {
    1: "red",
    2: "yellow",
    3: "blue",
    4: "green",
    5: "black",
    6: "sky-blue",
}

COLOR_HSV = {
    # 红色：双段保持优秀配置
    "red": {
        "lower": np.array([0, 70, 70]),
        "upper": np.array([10, 255, 255]),
        "bgr": (0, 0, 255),
    },
    "red2": {
        "lower": np.array([156, 70, 70]),
        "upper": np.array([180, 255, 255]),
        "bgr": (0, 0, 255),
    },
    # 黄色：适当微调上限至 40，消除与绿色的过度断层
    "yellow": {
        "lower": np.array([15, 75, 70]),
        "upper": np.array([40, 255, 255]),
        "bgr": (0, 255, 255),
    },
    # 绿色：下限延伸至 41 覆盖草绿，上限放宽至 85，Vmax 放宽至 255 解决高光穿孔
    "green": {
        "lower": np.array([41, 70, 50]),
        "upper": np.array([85, 255, 255]),
        "bgr": (0, 255, 0),
    },
    # 浅蓝：提升 S 下限至 45 杜绝白地误检，S 上限设为 160 彻底与深蓝隔离
    "sky-blue": {
        "lower": np.array([90, 45, 60]),
        "upper": np.array([125, 160, 255]),
        "bgr": (255, 255, 0),
    },
    # 深蓝：S 从 165 起步，与浅蓝产生 5 像素的安全隔离带
    "blue": {
        "lower": np.array([95, 165, 50]),
        "upper": np.array([125, 255, 255]),
        "bgr": (255, 0, 0),
    },
    # 黑色：限制 S 必须在 110 以下，防止把阴影下的暗色物料误判为黑
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([179, 110, 90]),
        "bgr": (50, 50, 50),
    },
}

MIN_BLOCK_RADIUS = 20

COLOR_NAME_TO_ID = {
    "red": 1, "yellow": 2, "blue": 3, "green": 4, "black": 5, "sky-blue": 6,
}

# Fraction of the ring radius kept when sampling ring colors: only the annulus
# between RING_BAND_INNER*r and r is voted on, ignoring inner/surrounding pixels.
RING_BAND_INNER = 0.7

# Color-ring thresholds. Ring colors match the material palette (same 6 colors)
# but are kept separate so ring-only tuning (e.g. green ring) does not affect blocks.
RING_HSV = {
    "red": {
        "lower": np.array([0, 70, 70]),
        "upper": np.array([10, 255, 255]),
        "bgr": (0, 0, 255),
    },
    "red2": {
        "lower": np.array([156, 70, 70]),
        "upper": np.array([180, 255, 255]),
        "bgr": (0, 0, 255),
    },
    "yellow": {
        "lower": np.array([15, 75, 70]),
        "upper": np.array([35, 255, 255]),
        "bgr": (0, 255, 255),
    },
    "blue": {
        "lower": np.array([98, 179, 0]),
        "upper": np.array([121, 255, 255]),
        "bgr": (255, 0, 0),
    },
    "green": {
        "lower": np.array([40, 30, 50]),
        "upper": np.array([90, 255, 255]),
        "bgr": (0, 255, 0),
    },
    "sky-blue": {
        "lower": np.array([98, 0, 0]),
        "upper": np.array([121, 190, 255]),
        "bgr": (255, 255, 0),
    },
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([179, 255, 96]),
        "bgr": (50, 50, 50),
    },
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BlockInfo:
    color: str
    color_id: int
    cx: int
    cy: int
    radius: int


@dataclass
class RingInfo:
    color: str
    color_id: int          # target_id: 1, 2, 3 (digit) or 5 (black ring unknown)
    cx: int
    cy: int
    radius: int
    digit_id: int = 5      # 1, 2, 3 (digit) or 5 (black ring without digit)
    confidence: float = 1.0


@dataclass
class TaskCode:
    raw: str = ""
    batch1_colors: list[int] = field(default_factory=list)
    batch1_positions: list[int] = field(default_factory=list)
    batch2_colors: list[int] = field(default_factory=list)
    batch2_positions: list[int] = field(default_factory=list)
    valid: bool = False


# ── Threaded Camera ───────────────────────────────────────────────────────────

class ThreadedCamera:
    """Non-blocking threaded camera reader that continuously grabs the latest frame."""

    def __init__(self, src: int, width: int = 640, height: int = 480):
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


# ── QR code ───────────────────────────────────────────────────────────────────

def init_qr_detector():
    cv_version = tuple(int(x) for x in cv2.__version__.split(".")[:2])
    if cv_version >= (5, 0):
        return cv2.wechat_qrcode_WeChatQRCode()
    models_dir = os.path.expanduser("~/.opencv/wechat_qrcode")
    model_names = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    model_paths = [os.path.join(models_dir, n) for n in model_names]
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing QR model files: {missing}", file=sys.stderr)
        print("Run: python qr_scanner.py --download-models", file=sys.stderr)
        sys.exit(1)
    return cv2.wechat_qrcode_WeChatQRCode(*model_paths)


def parse_task_code(text: str) -> TaskCode:
    tc = TaskCode(raw=text)
    text = text.strip()
    parts = text.split("+")
    if len(parts) != 4:
        return tc
    try:
        nums = [int(p) for p in parts]
        if not all(100 <= n <= 999 for n in nums):
            return tc
        tc.batch1_colors = [int(d) for d in str(nums[0])]
        tc.batch1_positions = [int(d) for d in str(nums[1])]
        tc.batch2_colors = [int(d) for d in str(nums[2])]
        tc.batch2_positions = [int(d) for d in str(nums[3])]
        if not all(1 <= c <= 6 for c in tc.batch1_colors + tc.batch2_colors):
            return tc
        if not all(1 <= p <= 6 for p in tc.batch1_positions + tc.batch2_positions):
            return tc
        tc.valid = True
    except (ValueError, IndexError):
        pass
    return tc


# ── Color detection ───────────────────────────────────────────────────────────

def get_mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             kernel_size: tuple[int, int] = (5, 5)) -> np.ndarray:
    mask = cv2.inRange(hsv, lower, upper)
    if kernel_size[0] > 1 and kernel_size[1] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def dedup_blocks(blocks: list[BlockInfo], tolerance: int = 20) -> list[BlockInfo]:
    """Remove overlapping block detections across different color masks."""
    remaining = list(blocks)
    result = []
    while remaining:
        target = remaining[0]
        group = [b for b in remaining
                 if (b.cx - target.cx) ** 2 + (b.cy - target.cy) ** 2 <= tolerance ** 2]
        result.append(max(group, key=lambda b: b.radius))
        remaining = [b for b in remaining
                     if (b.cx - target.cx) ** 2 + (b.cy - target.cy) ** 2 > tolerance ** 2]
    return result


def detect_blocks(hsv: np.ndarray,
                  min_radius: int = 20,
                  max_radius: int = 150,
                  min_fill_ratio: float = 0.60,
                  min_circularity: float = 0.60,
                  min_aspect_ratio: float = 0.65) -> list[BlockInfo]:
    blocks = []
    for name, cfg in COLOR_HSV.items():
        if name == "red2":
            continue
        mask = get_mask(hsv, cfg["lower"], cfg["upper"])
        if name == "red":
            mask2 = get_mask(hsv, COLOR_HSV["red2"]["lower"], COLOR_HSV["red2"]["upper"])
            mask = cv2.bitwise_or(mask, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * min_radius * min_radius:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < min_radius or r > max_radius:
                continue

            # 1. 宽高比校验 (滤除长条、桌边反光、背景光带)
            x, y, w, h = cv2.boundingRect(cnt)
            if max(w, h) == 0:
                continue
            aspect_ratio = min(w, h) / max(w, h)
            if aspect_ratio < min_aspect_ratio:
                continue

            # 2. 面积饱满度校验 (轮廓面积 / 最小外接圆面积)
            circle_area = np.pi * (r ** 2)
            fill_ratio = area / circle_area if circle_area > 0 else 0
            if fill_ratio < min_fill_ratio:
                continue

            # 3. 几何圆度校验 (4 * pi * area / perimeter^2)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < min_circularity:
                continue

            color_id = COLOR_NAME_TO_ID.get(name, 0)
            blocks.append(BlockInfo(
                color=name, color_id=color_id,
                cx=int(cx), cy=int(cy), radius=int(r),
            ))

    if len(blocks) > 1:
        blocks = dedup_blocks(blocks, tolerance=20)

    return blocks


def dedup_concentric_circles(circles: list[tuple[int, int, int]],
                             tolerance: int = 5) -> list[tuple[int, int, int]]:
    """Keep only the largest circle from each group of concentric detections."""
    remaining = list(circles)
    result = []
    while remaining:
        x0, y0, _ = remaining[0]
        group = [c for c in remaining
                 if abs(c[0] - x0) <= tolerance and abs(c[1] - y0) <= tolerance]
        result.append(max(group, key=lambda c: c[2]))
        remaining = [c for c in remaining
                     if abs(c[0] - x0) > tolerance or abs(c[1] - y0) > tolerance]
    return result


def detect_rings(frame: np.ndarray, hsv: np.ndarray,
                 min_radius: int, max_radius: int, min_color_area: int,
                 param2: float = 0.8,
                 digit_classifier: "DigitClassifier | None" = None,
                 max_rings: int = 3) -> list[RingInfo]:
    """Detect ground black circular rings and classify inner digits 1, 2, 3."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    circles = cv2.HoughCircles(
        gauss, cv2.HOUGH_GRADIENT_ALT, 1.2, minDist=max(50, int(min_radius * 0.8)),
        param1=100, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []

    circles = np.round(circles.reshape((-1, 3))).astype(int)
    circles = dedup_concentric_circles([(int(c[0]), int(c[1]), int(c[2])) for c in circles])

    # Direct black mask for ring verification
    black_cfg = RING_HSV["black"]
    black_mask = get_mask(hsv, black_cfg["lower"], black_cfg["upper"], kernel_size=(3, 3))

    h, w = gray.shape
    candidates: list[tuple[int, int, int, int]] = []
    for cx, cy, r in circles:
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            continue

        band = np.zeros(gray.shape, np.uint8)
        cv2.circle(band, (cx, cy), r, 255, -1)
        cv2.circle(band, (cx, cy), max(1, int(r * RING_BAND_INNER)), 0, -1)
        band_roi = band[y1:y2, x1:x2]

        black_area = int(cv2.countNonZero(cv2.bitwise_and(black_mask[y1:y2, x1:x2], band_roi)))

        adaptive_threshold = max(30, int(2 * np.pi * r * 0.35))
        effective_min_area = min(min_color_area, adaptive_threshold)
        if black_area < effective_min_area:
            continue

        candidates.append((int(cx), int(cy), int(r), black_area))

    if not candidates:
        return []

    candidates.sort(key=lambda c: c[2], reverse=True)
    selected = candidates[:max_rings]

    if digit_classifier is not None and HAS_DIGIT_CLASSIFIER:
        scores_list = [
            extract_digit_scores_from_ring(frame, cx, cy, r, digit_classifier)
            for cx, cy, r, _ in selected
        ]
        assigned_ids = match_rings_to_digits(scores_list, min_confidence=0.45)
    else:
        assigned_ids = [5] * len(selected)

    rings = []
    for idx, (cx, cy, r, _) in enumerate(selected):
        target_id = assigned_ids[idx]
        conf = 1.0
        if digit_classifier is not None and HAS_DIGIT_CLASSIFIER and target_id in (1, 2, 3):
            conf = scores_list[idx].get(target_id, 1.0)
        rings.append(RingInfo(
            color="black",
            color_id=target_id,
            cx=cx,
            cy=cy,
            radius=r,
            digit_id=target_id,
            confidence=conf,
        ))
    return rings


# ── Serial communication (RDLC protocol) ──────────────────────────────────────

# Message IDs
MSG_VISION = 0x01      # 物块检测结果（批量）
MSG_RING = 0x02        # 色环检测结果（批量）
MSG_TASK_CODE = 0x03   # 任务码
MSG_MODE_CMD = 0x04    # 下位机->上位机 检测模式指令: [0x04][mode]

# Detection modes (controlled by the lower board via MSG_MODE_CMD)
MODE_BLOCK = 0         # 识别物料
MODE_RING = 1          # 识别色环

# Protocol layout (inside RDLC payload):
#   [msg_id(1B)] [data...]
#
# MSG_VISION data:
#   [count(1B)] [color_id(1B) cx(f32) cy(f32) radius(f32)] * count
#
# MSG_RING data:
#   [count(1B)] [color_id(1B) cx(f32) cy(f32) radius(f32)] * count
#
# MSG_TASK_CODE data:
#   [raw string bytes]
#
# MSG_MODE_CMD data:
#   [mode(1B)]  (0 = block, 1 = ring)


class SerialSender:
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.link = None

    def open(self):
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
        # payload: msg_id(1B) + data (task code bytes)
        payload = struct.pack("<B", MSG_TASK_CODE) + raw_bytes
        try:
            self.link.send(payload, dst=0x02, src=0x01)
            print(f"[TX] TaskCode: {tc.raw}")
        except Exception as e:
            print(f"Send task_code error: {e}", file=sys.stderr)

    def send_blocks(self, blocks: list[BlockInfo]):
        # Each block: color_id(1B) + cx(4B) + cy(4B) + radius(4B) = 13B
        # Max 3 blocks: 1(count) + 3*13 = 40B
        count = min(len(blocks), 3)
        block_data = b""
        for b in blocks[:count]:
            block_data += struct.pack("<Bfff", b.color_id,
                                      float(b.cx), float(b.cy), float(b.radius))
        # payload: msg_id(1B) + count(1B) + block_data
        payload = struct.pack("<BB", MSG_VISION, count) + block_data
        try:
            self.link.send(payload, dst=0x02, src=0x01)
        except Exception as e:
            print(f"Send vision error: {e}", file=sys.stderr)

    def send_rings(self, rings: list[RingInfo]):
        # Same layout as MSG_VISION, only the msg_id differs (MSG_RING)
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
        except Exception as e:
            print(f"Send ring error: {e}", file=sys.stderr)

    def poll_command(self) -> int | None:
        """Non-blocking read of a lower-board mode command.

        Drains the receive queue to always fetch the latest command without lag.
        Returns MODE_BLOCK / MODE_RING if an MSG_MODE_CMD frame arrived, else None.
        """
        if not self.link:
            return None
        latest_mode = None
        while True:
            try:
                frame = self.link.receive(timeout=0)
                payload = frame.payload
                if len(payload) >= 2 and payload[0] == MSG_MODE_CMD:
                    mode = payload[1]
                    if mode in (MODE_BLOCK, MODE_RING):
                        latest_mode = mode
            except Empty:
                break
        return latest_mode


# ── Display helpers ───────────────────────────────────────────────────────────

def draw_blocks(display: np.ndarray, blocks: list[BlockInfo]):
    for b in blocks:
        cfg = COLOR_HSV.get(b.color, COLOR_HSV["red"])
        bgr = cfg["bgr"]
        cv2.circle(display, (b.cx, b.cy), b.radius, bgr, 2)
        cv2.circle(display, (b.cx, b.cy), 3, bgr, -1)
        label = f"{b.color} ({b.cx},{b.cy}) r={b.radius}"
        cv2.putText(display, label, (b.cx - b.radius, b.cy - b.radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)


def draw_rings(display: np.ndarray, rings: list[RingInfo]):
    for r in rings:
        digit = getattr(r, "digit_id", 5)
        conf = getattr(r, "confidence", 1.0)
        if digit in (1, 2, 3):
            outline_color = (0, 255, 0)
            label = f"Target #{digit} ({r.cx},{r.cy}) r={r.radius} [{conf:.2f}]"
            center_text = str(digit)
        else:
            outline_color = (0, 255, 255)
            label = f"Ring (unknown) ({r.cx},{r.cy}) r={r.radius}"
            center_text = "?"

        cv2.circle(display, (r.cx, r.cy), r.radius, outline_color, 2)
        cv2.circle(display, (r.cx, r.cy), max(1, int(r.radius * 0.65)), (100, 100, 100), 1)
        cv2.circle(display, (r.cx, r.cy), 4, (0, 0, 255), -1)
        cv2.putText(display, label, (r.cx - r.radius, max(15, r.cy - r.radius - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 1)
        cv2.putText(display, center_text, (r.cx - 6, r.cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_task_code(display: np.ndarray, tc: TaskCode):
    if tc.valid:
        text = f"Task: {tc.raw}"
        color = (0, 255, 0)
    elif tc.raw:
        text = f"QR: {tc.raw} (invalid)"
        color = (0, 0, 255)
    else:
        text = "Waiting for QR code..."
        color = (0, 255, 255)
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if tc.valid:
        names1 = [COLOR_MAP.get(c, "?") for c in tc.batch1_colors]
        names2 = [COLOR_MAP.get(c, "?") for c in tc.batch2_colors]
        info1 = f"Batch1: {'/'.join(names1)} -> ring {'/'.join(map(str, tc.batch1_positions))}"
        info2 = f"Batch2: {'/'.join(names2)} -> ring {'/'.join(map(str, tc.batch2_positions))}"
        cv2.putText(display, info1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, info2, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upper machine: QR + color block detector")
    parser.add_argument("--qr-camera", type=int, default=0, help="Camera index for QR code")
    parser.add_argument("--block-camera", type=int, default=2, help="Camera index for color blocks")
    parser.add_argument("--no-qr", action="store_true",
                        help="Disable QR code scanning (run with only one camera for blocks/rings)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--serial-port", type=str, default=None,
                        help="Serial port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    parser.add_argument("--no-serial", action="store_true", help="Disable serial output")
    parser.add_argument("--send-interval", type=float, default=0.5,
                        help="Seconds between serial sends")
    parser.add_argument("--mode", choices=["block", "ring"], default="block",
                        help="Initial detection mode (lower board overrides via MSG_MODE_CMD)")
    parser.add_argument("--block-min-radius", type=int, default=20, help="Min block radius (px)")
    parser.add_argument("--block-max-radius", type=int, default=150, help="Max block radius (px)")
    parser.add_argument("--block-min-circularity", type=float, default=0.60,
                        help="Min circularity 4*pi*area/perimeter^2 (0.0~1.0)")
    parser.add_argument("--block-min-fill", type=float, default=0.60,
                        help="Min fill ratio area/(pi*r^2) (0.0~1.0)")
    parser.add_argument("--block-min-aspect", type=float, default=0.65,
                        help="Min bounding rect aspect ratio (0.0~1.0)")
    parser.add_argument("--ring-min-radius", type=int, default=60, help="Min ring radius (px, default 60)")
    parser.add_argument("--ring-max-radius", type=int, default=200, help="Max ring radius (px)")
    parser.add_argument("--ring-min-area", type=int, default=100,
                        help="Min ring color area (px) to accept a ring (default 100, adaptive for small rings)")
    parser.add_argument("--ring-param2", type=float, default=0.8,
                        help="HoughCircles param2 (circle perfectness, default 0.8) for ring detection")
    args = parser.parse_args()

    if args.list_ports:
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            ports = []
        print("Available serial ports:")
        for p in ports:
            print(f"  {p}")
        if not ports:
            print("  (none)")
        return

    digit_classifier = DigitClassifier() if HAS_DIGIT_CLASSIFIER else None
    qr_detector = None
    cam_qr = None
    qr_active = False

    if not args.no_qr and args.qr_camera >= 0:
        if args.qr_camera == args.block_camera:
            print(f"Warning: --qr-camera and --block-camera are both {args.qr_camera}.", file=sys.stderr)
            print("OpenCV cannot open the same camera twice simultaneously.", file=sys.stderr)
            print("Auto-disabling QR camera for this run. (Tip: add '--no-qr' for single-camera mode)", file=sys.stderr)
            args.no_qr = True
        else:
            qr_detector = init_qr_detector()
            cam_qr = ThreadedCamera(args.qr_camera, args.width, args.height)
            qr_active = cam_qr.start()
            if not qr_active:
                print(f"Cannot open QR camera {args.qr_camera}, QR scan disabled.", file=sys.stderr)

    if args.no_qr:
        print("QR scan disabled (single camera mode).")

    cam_block = ThreadedCamera(args.block_camera, args.width, args.height)
    if not cam_block.start():
        print(f"Cannot open block camera {args.block_camera}", file=sys.stderr)
        if cam_qr and qr_active:
            cam_qr.stop()
        sys.exit(1)

    qr_info_str = f"{args.qr_camera} (active={qr_active})" if qr_active else "disabled"
    print(f"QR camera: {qr_info_str} | Block camera: {args.block_camera}")

    sender = None
    if not args.no_serial and args.serial_port:
        sender = SerialSender(args.serial_port, args.baudrate)
        sender.open()

    window_qr = "QR Camera (ESC quit)"
    window_block = "Block Camera (ESC quit)"
    if qr_active:
        cv2.namedWindow(window_qr)
    cv2.namedWindow(window_block)

    last_task_code = TaskCode()
    qr_candidate = ""
    qr_match_count = 0
    last_send_time = 0.0
    consecutive_block_fails = 0
    mode = MODE_RING if args.mode == "ring" else MODE_BLOCK

    print(f"Running. mode={'RING' if mode == MODE_RING else 'BLOCK'}. "
          "ESC quit, m toggle mode.\n")

    while True:
        # ── Lower-board mode command (block vs ring) ──
        if sender:
            cmd = sender.poll_command()
            if cmd is not None and cmd != mode:
                mode = cmd
                print(f"[RX] Mode -> {'RING' if mode == MODE_RING else 'BLOCK'}")

        # ── QR camera (active until 3 identical valid detections) ──
        if qr_active:
            ret_qr, frame_qr = cam_qr.read()
            if ret_qr and frame_qr is not None:
                display_qr = frame_qr.copy()
                results, points = qr_detector.detectAndDecode(frame_qr)
                if results:
                    for text, pt in zip(results, points):
                        tc = parse_task_code(text)
                        if tc.valid:
                            if tc.raw == qr_candidate:
                                qr_match_count += 1
                            else:
                                qr_candidate = tc.raw
                                qr_match_count = 1

                            print(f"[QR] Candidate: {tc.raw} ({qr_match_count}/3)")
                            if qr_match_count >= 3:
                                last_task_code = tc
                                print(f"[QR] Confirmed 3 times: {tc.raw}! Closing QR scan.")
                                if sender:
                                    sender.send_task_code(tc)
                                qr_active = False
                                cam_qr.stop()
                                cv2.destroyWindow(window_qr)
                                break

                        pts = pt.astype(int).reshape((-1, 1, 2))
                        cv2.polylines(display_qr, [pts], True, (0, 255, 0), 2)
                        cx, cy = pts[:, 0, :].mean(axis=0).astype(int)
                        cv2.putText(display_qr, f"QR: {text}", (cx, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if qr_active:
                    draw_task_code(display_qr, last_task_code)
                    cv2.imshow(window_qr, display_qr)

        # ── Detection camera (block or color ring, non-blocking latest frame) ──
        ret_block, frame_block = cam_block.read()
        if not ret_block or frame_block is None:
            consecutive_block_fails += 1
            if consecutive_block_fails > 100:
                print("Block camera failed continuously. Exiting...", file=sys.stderr)
                break
            time.sleep(0.005)
            continue
        consecutive_block_fails = 0

        display_block = frame_block.copy()
        hsv = cv2.cvtColor(frame_block, cv2.COLOR_BGR2HSV)
        blocks: list[BlockInfo] = []
        rings: list[RingInfo] = []
        if mode == MODE_RING:
            rings = detect_rings(frame_block, hsv, args.ring_min_radius,
                                 args.ring_max_radius, args.ring_min_area,
                                 args.ring_param2,
                                 digit_classifier=digit_classifier if HAS_DIGIT_CLASSIFIER else None,
                                 max_rings=3)
            draw_rings(display_block, rings)
            target_ids = [str(r.digit_id) if r.digit_id in (1, 2, 3) else "?" for r in rings]
            ring_desc = f" [{', '.join(target_ids)}]" if target_ids else ""
            status = f"Mode: RING (black){ring_desc} | Rings: {len(rings)}"
        else:
            blocks = detect_blocks(
                hsv,
                min_radius=args.block_min_radius,
                max_radius=args.block_max_radius,
                min_fill_ratio=args.block_min_fill,
                min_circularity=args.block_min_circularity,
                min_aspect_ratio=args.block_min_aspect,
            )
            draw_blocks(display_block, blocks)
            status = f"Mode: BLOCK | Blocks: {len(blocks)}"

        if sender and sender.link:
            status += " | Serial: OK (RDLC)"
        elif not args.no_serial:
            status += " | Serial: off"

        if last_task_code.valid:
            status += f" | Task: {last_task_code.raw}"
        elif qr_active:
            status += f" | QR: scanning ({qr_match_count}/3)"
        else:
            status += " | QR: off"

        cv2.putText(display_block, status, (10, display_block.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(window_block, display_block)

        # ── Serial send or console log (task code + current mode result) ──
        now = time.time()
        if now - last_send_time >= args.send_interval:
            if sender:
                if last_task_code.valid:
                    sender.send_task_code(last_task_code)
                if mode == MODE_RING:
                    if rings:
                        sender.send_rings(rings)
                elif blocks:
                    sender.send_blocks(blocks)
            else:
                if mode == MODE_RING and rings:
                    desc_list = [
                        f"Target #{r.digit_id}(conf={r.confidence:.2f})@({r.cx},{r.cy},r={r.radius})"
                        if r.digit_id in (1, 2, 3) else f"Ring unknown@({r.cx},{r.cy},r={r.radius})"
                        for r in rings
                    ]
                    print(f"[Vision RING] {len(rings)} ring(s): {', '.join(desc_list)}")
                elif mode == MODE_BLOCK and blocks:
                    desc_list = [f"{b.color}@({b.cx},{b.cy},r={b.radius})" for b in blocks]
                    print(f"[Vision BLOCK] {len(blocks)} block(s): {', '.join(desc_list)}")
            last_send_time = now

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("m"):
            mode = MODE_BLOCK if mode == MODE_RING else MODE_RING
            print(f"[key] Mode -> {'RING' if mode == MODE_RING else 'BLOCK'}")
        elif key == ord("p"):
            if mode == MODE_RING:
                if rings:
                    print(f"[Current Detections: {len(rings)} Ring(s)]")
                    for r in rings:
                        target_str = f"Target #{r.digit_id} (conf={r.confidence:.2f})" if r.digit_id in (1, 2, 3) else "unknown digit"
                        print(f"  ring [{r.color:5s}] {target_str:25s} center=({r.cx:4d}, {r.cy:4d})  r={r.radius:3d}")
                else:
                    print("  [Vision RING] No rings detected in current frame.")
            else:
                if blocks:
                    print(f"[Current Detections: {len(blocks)} Block(s)]")
                    for b in blocks:
                        print(f"  block [{b.color:9s}] center=({b.cx:4d}, {b.cy:4d})  r={b.radius:3d}")
                else:
                    print("  [Vision BLOCK] No blocks detected in current frame.")

    if sender:
        sender.close()
    if cam_qr and qr_active:
        cam_qr.stop()
    cam_block.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
