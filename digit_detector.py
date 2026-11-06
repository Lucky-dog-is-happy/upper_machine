#!/usr/bin/env python3
"""Standalone Multi-Target (up to 3) Ring & Digit (1, 2, 3) Fusion Detector.

Features:
  - Multi-Target Detection: Detects up to 3 black rings and digits simultaneously.
  - Spatial Association: Matches each black ring with the digit (1, 2, 3) at its center.
  - Weighted Center Fusion:
      • Both detected: Center = (w_ring * Ring_Center) + (w_digit * Digit_Center)
      • Only Ring detected: Center = Ring_Center
      • Only Digit detected: Center = Digit_Center
  - Serial Protocol Simulation: Formats and prints exact RDLC payload bytes for lower board.
  - Optional Real Serial: Pass --serial-port /dev/ttyUSB0 to transmit live data to STM32.

Usage:
  # Standalone test on block camera (/dev/video20, simulated serial preview):
  python digit_detector.py

  # Adjust fusion weights (e.g., 60% ring, 40% digit):
  python digit_detector.py --weight-ring 0.6

  # Real serial output test:
  python digit_detector.py --serial-port /dev/ttyUSB0

  # Test on webcam 0 / 2:
  python digit_detector.py --camera 0

Controls:
  'm'     : Toggle between Ring Mode (ring ROI search) and Full Frame Mode
  'w'/'W' : Increase / decrease ring weight (0.0 ~ 1.0)
  'p'     : Print current target coordinates and serial frame payload
  's'     : Save snapshot image
  '+'/'-' : Adjust min contour area threshold
  'q'/ESC : Quit
"""

import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add rcs_comm to sys.path for optional real serial testing
BASE_COMM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RCS_Robot_Upper", "src", "rcs_comm")
for pkg in ["rcs_serial", "rcs_ptl"]:
    p = os.path.join(BASE_COMM, pkg)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from rcs_serial import RdlcSerialLink, SerialConfig
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ── Configuration & Thresholds ────────────────────────────────────────────────

BLACK_RING_HSV = {
    "lower": np.array([0, 0, 0]),
    "upper": np.array([179, 255, 96]),
}

RING_BAND_INNER = 0.70  # Annulus inner ratio to isolate the black ring
RING_INNER_ROI = 0.75   # Inner fraction to search for digits
NORM_SIZE = 32          # Standard normalized size (32x32) for classifier

MSG_RING = 0x02         # RDLC Message ID for Ring / Target info


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RingCandidate:
    cx: int
    cy: int
    radius: int
    black_area: int


@dataclass
class DigitResult:
    digit: int           # 1, 2, or 3
    confidence: float   # 0.0 ~ 1.0
    x: int              # Bounding box x
    y: int              # Bounding box y
    w: int              # Bounding box width
    h: int              # Bounding box height
    cx: int             # Center x
    cy: int             # Center y
    binary_roi: np.ndarray | None = None


@dataclass
class FusedTarget:
    target_id: int            # 1, 2, 3 (or 0 if only ring without digit)
    fused_cx: float           # Weighted center X
    fused_cy: float           # Weighted center Y
    radius: float             # Ring radius
    status: str               # "FUSED", "RING_ONLY", "DIGIT_ONLY"
    has_ring: bool = False
    has_digit: bool = False
    ring_cx: float | None = None
    ring_cy: float | None = None
    digit_cx: float | None = None
    digit_cy: float | None = None
    confidence: float = 1.0


# ── Template Generator & Classifier ───────────────────────────────────────────

def create_digit_templates(size: int = NORM_SIZE) -> dict[int, list[np.ndarray]]:
    """Generate high-contrast normalized reference templates for 1, 2, 3."""
    templates: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    font_candidates = [
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/wps-fonts/timesbd.ttf",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    try:
        font = ImageFont.truetype(font_path, 52) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for digit in [1, 2, 3]:
        text = str(digit)
        img = Image.new("L", (80, 80), 0)
        draw = ImageDraw.Draw(img)

        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((80 - bw) // 2 - bbox[0], (80 - bh) // 2 - bbox[1]), text, fill=255, font=font)
        except Exception:
            draw.text((25, 15), text, fill=255)

        arr = np.array(img)
        cnts, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            cropped = arr[by:by + bh, bx:bx + bw]
            padded = cv2.copyMakeBorder(cropped, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
            norm = cv2.resize(padded, (size, size))
            templates[digit].append(norm)

    return templates


class DigitClassifier:
    """Classifies candidate binary contours into 1, 2, or 3 using template matching + geometry."""

    def __init__(self):
        self.templates = create_digit_templates(size=NORM_SIZE)

    def classify(self, binary_crop: np.ndarray) -> tuple[int | None, float]:
        h, w = binary_crop.shape[:2]
        if h < 10 or w < 5:
            return None, 0.0

        aspect = h / float(w)

        # Pad to square and resize
        pad = max(2, int(max(h, w) * 0.1))
        padded = cv2.copyMakeBorder(binary_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        norm = cv2.resize(padded, (NORM_SIZE, NORM_SIZE))

        # 1. Template Matching Correlation
        scores = {}
        for d, tpl_list in self.templates.items():
            best_s = -1.0
            for tpl in tpl_list:
                res = cv2.matchTemplate(norm, tpl, cv2.TM_CCOEFF_NORMED)
                s = float(res[0][0])
                best_s = max(best_s, s)
            scores[d] = max(0.0, best_s)

        # 2. Geometric & Topological Rules
        # Digit 1: Slender vertical stroke
        if aspect >= 1.75:
            scores[1] += 0.35
            scores[2] -= 0.30
            scores[3] -= 0.30
        elif aspect < 1.15:
            scores[1] -= 0.50

        # Digit 2: Solid baseline stroke at the bottom
        bot_density = np.mean(norm[int(NORM_SIZE * 0.80):, :]) / 255.0
        if bot_density > 0.55:
            scores[2] += 0.25

        # Digit 3: Waist indentation on middle-left
        mid_left_density = np.mean(norm[int(NORM_SIZE * 0.35):int(NORM_SIZE * 0.65), :int(NORM_SIZE * 0.30)]) / 255.0
        if mid_left_density < 0.15 and aspect < 1.75:
            scores[3] += 0.20
            scores[1] -= 0.10

        best_digit = max(scores, key=scores.get)
        confidence = scores[best_digit]

        if confidence < 0.55:
            return None, 0.0

        return best_digit, min(1.0, confidence)


# ── Multi-Ring Detection ──────────────────────────────────────────────────────

def dedup_concentric_circles(circles: list[tuple[int, int, int]], tolerance: int = 25) -> list[tuple[int, int, int]]:
    """Deduplicate concentric detections, keeping the most stable radius."""
    remaining = list(circles)
    result = []
    while remaining:
        x0, y0, _ = remaining[0]
        group = [c for c in remaining if abs(c[0] - x0) <= tolerance and abs(c[1] - y0) <= tolerance]
        result.append(max(group, key=lambda c: c[2]))
        remaining = [c for c in remaining if abs(c[0] - x0) > tolerance or abs(c[1] - y0) > tolerance]
    return result


def detect_black_rings(frame: np.ndarray,
                       hsv: np.ndarray,
                       min_radius: int = 20,
                       max_radius: int = 220,
                       min_color_area: int = 350,
                       param2: float = 0.88,
                       max_rings: int = 3) -> list[RingCandidate]:
    """Detect up to max_rings black circular rings on the ground."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gauss, cv2.HOUGH_GRADIENT_ALT, 0.5, minDist=70,
        param1=100, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []

    circles = np.round(circles.reshape((-1, 3))).astype(int)
    circles = dedup_concentric_circles([(int(c[0]), int(c[1]), int(c[2])) for c in circles])

    # Black mask
    black_cfg = BLACK_RING_HSV
    mask = cv2.inRange(hsv, black_cfg["lower"], black_cfg["upper"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    h, w = gray.shape
    candidates: list[RingCandidate] = []

    for cx, cy, r in circles:
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            continue

        band = np.zeros(gray.shape, np.uint8)
        cv2.circle(band, (cx, cy), r, 255, -1)
        cv2.circle(band, (cx, cy), max(1, int(r * RING_BAND_INNER)), 0, -1)
        band_roi = band[y1:y2, x1:x2]

        area = int(cv2.countNonZero(cv2.bitwise_and(mask[y1:y2, x1:x2], band_roi)))
        if area >= min_color_area:
            candidates.append(RingCandidate(cx=cx, cy=cy, radius=r, black_area=area))

    # Sort by black area and keep top max_rings
    candidates.sort(key=lambda c: c.black_area, reverse=True)
    return candidates[:max_rings]


# ── Digit Extraction ──────────────────────────────────────────────────────────

def find_digits_in_roi(roi_bgr: np.ndarray,
                       classifier: DigitClassifier,
                       offset_x: int = 0,
                       offset_y: int = 0,
                       min_area: int = 60,
                       max_area: int = 15000,
                       center_constraint: tuple[int, int, float] | None = None) -> list[DigitResult]:
    """Find digits 1, 2, 3 inside an ROI with optional center distance constraint."""
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Inverse Otsu (dark digit on light background) + regular Otsu
    _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_std = cv2.bitwise_not(thresh_inv)

    results: list[DigitResult] = []

    for thresh in [thresh_inv, thresh_std]:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh < 12 or bw < 4 or bw > roi_bgr.shape[1] * 0.9 or bh > roi_bgr.shape[0] * 0.9:
                continue

            cx = offset_x + bx + bw // 2
            cy = offset_y + by + bh // 2

            if center_constraint is not None:
                rcx, rcy, max_dist = center_constraint
                if np.hypot(cx - rcx, cy - rcy) > max_dist:
                    continue

            crop = thresh[by:by + bh, bx:bx + bw]
            digit, conf = classifier.classify(crop)
            if digit is not None:
                results.append(DigitResult(
                    digit=digit,
                    confidence=conf,
                    x=offset_x + bx,
                    y=offset_y + by,
                    w=bw,
                    h=bh,
                    cx=cx,
                    cy=cy,
                    binary_roi=crop,
                ))

        if results:
            break

    results.sort(key=lambda r: r.confidence, reverse=True)
    deduped: list[DigitResult] = []
    for r in results:
        if not any(abs(r.cx - d.cx) < 20 and abs(r.cy - d.cy) < 20 for d in deduped):
            deduped.append(r)

    return deduped


# ── Ring & Digit Spatial Association & Center Fusion ──────────────────────────

def fuse_rings_and_digits(rings: list[RingCandidate],
                          all_digits: list[DigitResult],
                          w_ring: float = 0.5,
                          max_targets: int = 3) -> list[FusedTarget]:
    """Fuse ring and digit coordinates with weighted center averaging.

    Rules:
      1. If a ring and digit match spatially (dist < ring.radius * 0.65):
         Fused_Center = (w_ring * Ring_Center) + ((1 - w_ring) * Digit_Center)
      2. If ring has no digit:
         Fused_Center = Ring_Center (Status: RING_ONLY)
      3. If digit has no ring:
         Fused_Center = Digit_Center (Status: DIGIT_ONLY)
      4. Returns up to max_targets (default 3), sorted by digit ID (1,2,3) if present.
    """
    fused_targets: list[FusedTarget] = []
    w_digit = 1.0 - w_ring

    matched_ring_indices = set()
    matched_digit_indices = set()

    # Step 1: Match Rings to Digits
    for r_idx, ring in enumerate(rings):
        best_d_idx = None
        best_dist = float("inf")
        match_thresh = ring.radius * 0.65

        for d_idx, digit in enumerate(all_digits):
            if d_idx in matched_digit_indices:
                continue
            dist = np.hypot(ring.cx - digit.cx, ring.cy - digit.cy)
            if dist < match_thresh and dist < best_dist:
                best_dist = dist
                best_d_idx = d_idx

        if best_d_idx is not None:
            # Pair found! Compute weighted center
            d = all_digits[best_d_idx]
            fx = w_ring * ring.cx + w_digit * d.cx
            fy = w_ring * ring.cy + w_digit * d.cy

            fused_targets.append(FusedTarget(
                target_id=d.digit,
                fused_cx=round(fx, 1),
                fused_cy=round(fy, 1),
                radius=float(ring.radius),
                status="FUSED",
                has_ring=True,
                has_digit=True,
                ring_cx=float(ring.cx),
                ring_cy=float(ring.cy),
                digit_cx=float(d.cx),
                digit_cy=float(d.cy),
                confidence=d.confidence,
            ))
            matched_ring_indices.add(r_idx)
            matched_digit_indices.add(best_d_idx)

    # Step 2: Unmatched Rings -> RING_ONLY (Use 5=black according to protocol color table)
    for r_idx, ring in enumerate(rings):
        if r_idx not in matched_ring_indices:
            fused_targets.append(FusedTarget(
                target_id=5,  # 5 indicates black ring (without recognized digit)
                fused_cx=float(ring.cx),
                fused_cy=float(ring.cy),
                radius=float(ring.radius),
                status="RING_ONLY",
                has_ring=True,
                has_digit=False,
                ring_cx=float(ring.cx),
                ring_cy=float(ring.cy),
                confidence=0.75,
            ))

    # Step 3: Unmatched Digits -> DIGIT_ONLY
    for d_idx, d in enumerate(all_digits):
        if d_idx not in matched_digit_indices:
            fused_targets.append(FusedTarget(
                target_id=d.digit,
                fused_cx=float(d.cx),
                fused_cy=float(d.cy),
                radius=float(max(d.w, d.h) * 1.5),  # estimated radius
                status="DIGIT_ONLY",
                has_ring=False,
                has_digit=True,
                digit_cx=float(d.cx),
                digit_cy=float(d.cy),
                confidence=d.confidence,
            ))

    # Sort: Identified digits 1, 2, 3 first, then by X position
    fused_targets.sort(key=lambda t: (0 if t.target_id > 0 else 1, t.target_id, t.fused_cx))
    return fused_targets[:max_targets]


# ── Serial Packet Formatter ───────────────────────────────────────────────────

def format_serial_payload(targets: list[FusedTarget]) -> bytes:
    """Format targets into RDLC MSG_RING (0x02) byte payload.

    Layout:
      [msg_id(1B)=0x02] [count(1B)] + [target_id(1B) cx(f32) cy(f32) radius(f32)] * count
    """
    count = min(len(targets), 3)
    data = b""
    for t in targets[:count]:
        data += struct.pack("<Bfff", t.target_id, float(t.fused_cx), float(t.fused_cy), float(t.radius))
    return struct.pack("<BB", MSG_RING, count) + data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Standalone Multi-Target Ring & Digit Weighted Fusion")
    parser.add_argument("--camera", type=str, default="/dev/video2",
                        help="Camera index or path (default: /dev/video20)")
    parser.add_argument("--width", type=int, default=640, help="Width")
    parser.add_argument("--height", type=int, default=480, help="Height")
    parser.add_argument("--weight-ring", type=float, default=0.50,
                        help="Fusion weight for Ring center (0.0=digit only, 1.0=ring only, default 0.50)")
    parser.add_argument("--serial-port", type=str, default=None,
                        help="Optional serial port to send real RDLC frames, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--min-area", type=int, default=70,
                        help="Min contour area for digit candidate")
    args = parser.parse_args()

    cam_src = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {cam_src}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Optional Serial
    link = None
    if args.serial_port and HAS_SERIAL:
        try:
            cfg = SerialConfig(port=args.serial_port, baudrate=args.baudrate, local_address=0x01, remote_address=0x02)
            link = RdlcSerialLink(cfg)
            link.open()
            print(f"[Serial] Opened {args.serial_port} @ {args.baudrate}")
        except Exception as e:
            print(f"[Serial] Failed to open {args.serial_port}: {e}", file=sys.stderr)

    classifier = DigitClassifier()
    w_ring = max(0.0, min(1.0, args.weight_ring))
    min_area = args.min_area
    ring_mode = True

    win_name = "Ring & Digit Fusion [Up to 3 Targets] (ESC=quit, w/W=weight, p=print, m=mode)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    print("\n=== Ring & Digit Fusion Detector Running ===")
    print(f"Camera: {cam_src} | Fusion Weight: Ring={w_ring:.2f}, Digit={1.0 - w_ring:.2f}")
    print(f"Serial: {args.serial_port or 'Simulated preview (offline)'}")
    print("Controls: [w/W] Adjust Weight  [p] Print Payload  [m] Toggle Mode  [s] Snap  [ESC] Quit\n")

    frame_count = 0
    fps_start = time.time()
    fps = 0.0
    last_print_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        now = time.time()
        if frame_count % 15 == 0:
            fps = 15.0 / max(0.001, now - fps_start)
            fps_start = now

        display = frame.copy()
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── 1. Detect up to 3 Black Rings ─────────────────────────────────────
        rings = detect_black_rings(frame, hsv, min_radius=20, max_radius=220, max_rings=3)

        # ── 2. Detect Digits (Inside Rings or Full Frame) ──────────────────────
        all_digits: list[DigitResult] = []
        if ring_mode and rings:
            # Search digits inside each detected ring
            for ring in rings:
                inner_r = max(10, int(ring.radius * RING_INNER_ROI))
                rx1, ry1 = max(0, ring.cx - inner_r), max(0, ring.cy - inner_r)
                rx2, ry2 = min(w, ring.cx + inner_r), min(h, ring.cy + inner_r)
                if rx2 > rx1 and ry2 > ry1:
                    roi = frame[ry1:ry2, rx1:rx2]
                    d_list = find_digits_in_roi(
                        roi, classifier,
                        offset_x=rx1, offset_y=ry1,
                        min_area=min_area,
                        center_constraint=(ring.cx, ring.cy, ring.radius * 0.60),
                    )
                    all_digits.extend(d_list)
        else:
            # Full frame search
            all_digits = find_digits_in_roi(frame, classifier, min_area=min_area)

        # ── 3. Weighted Fusion (Up to 3 Targets) ──────────────────────────────
        fused_targets = fuse_rings_and_digits(rings, all_digits, w_ring=w_ring, max_targets=3)

        # ── 4. Visualization ──────────────────────────────────────────────────
        # Draw all detected rings (Cyan circles)
        for ring in rings:
            cv2.circle(display, (ring.cx, ring.cy), ring.radius, (255, 255, 0), 2)
            cv2.circle(display, (ring.cx, ring.cy), 3, (255, 255, 0), -1)

        # Draw all detected digits (Magenta boxes)
        for d in all_digits:
            cv2.rectangle(display, (d.x, d.y), (d.x + d.w, d.y + d.h), (255, 0, 255), 2)
            cv2.circle(display, (d.cx, d.cy), 3, (255, 0, 255), -1)

        # Draw Fused Targets (Red Crosshair + Target Card)
        colors_status = {
            "FUSED": (0, 0, 255),        # Red
            "RING_ONLY": (0, 165, 255),   # Orange
            "DIGIT_ONLY": (255, 0, 150),  # Purple
        }

        for idx, t in enumerate(fused_targets):
            fcx, fcy = int(t.fused_cx), int(t.fused_cy)
            color = colors_status.get(t.status, (0, 0, 255))

            # Crosshair at Fused Center
            arm = 12
            cv2.line(display, (fcx - arm, fcy), (fcx + arm, fcy), color, 2)
            cv2.line(display, (fcx, fcy - arm), (fcx, fcy + arm), color, 2)
            cv2.circle(display, (fcx, fcy), 5, color, 1)

            # If FUSED: draw connecting line between Ring Center & Digit Center
            if t.status == "FUSED" and t.ring_cx is not None and t.digit_cx is not None:
                rc = (int(t.ring_cx), int(t.ring_cy))
                dc = (int(t.digit_cx), int(t.digit_cy))
                cv2.line(display, rc, dc, (0, 255, 255), 1)

            # Target label
            id_name = f"DIGIT {t.target_id}" if t.target_id > 0 else "RING"
            tag = f"#{idx+1} [{id_name}] ({t.status})"
            coord = f"({t.fused_cx:.1f}, {t.fused_cy:.1f}) r={t.radius:.0f}"
            cv2.putText(display, tag, (fcx + 8, fcy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
            cv2.putText(display, coord, (fcx + 8, fcy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── 5. HUD Info Overlay ───────────────────────────────────────────────
        mode_text = "RING ROI" if ring_mode else "FULL FRAME"
        header = f"Targets: {len(fused_targets)}/3 | Mode: {mode_text} | FPS: {fps:.1f} | W(Ring)={w_ring:.2f} W(Digit)={1-w_ring:.2f}"
        cv2.putText(display, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Target Summary Table on top left
        y_offset = 48
        for i, t in enumerate(fused_targets):
            info = f" T{i+1}: ID={t.target_id} Fused=({t.fused_cx:.1f},{t.fused_cy:.1f}) [{t.status}]"
            cv2.putText(display, info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colors_status.get(t.status, (200, 200, 200)), 1)
            y_offset += 22

        cv2.putText(display, "[w/W] Weight  [p] Print Payload  [m] Mode  [s] Snap  [ESC] Quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow(win_name, display)

        # ── 6. Serial Send / Simulated Output ─────────────────────────────────
        payload = format_serial_payload(fused_targets)
        if link and len(fused_targets) > 0 and now - last_print_time >= 0.5:
            try:
                link.send(payload, dst=0x02, src=0x01)
                last_print_time = now
            except Exception:
                pass

        # ── 7. Keyboard Controls ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('w'):
            w_ring = min(1.0, round(w_ring + 0.05, 2))
            print(f"[Weight] Ring={w_ring:.2f}, Digit={1.0 - w_ring:.2f}")
        elif key == ord('W'):
            w_ring = max(0.0, round(w_ring - 0.05, 2))
            print(f"[Weight] Ring={w_ring:.2f}, Digit={1.0 - w_ring:.2f}")
        elif key == ord('m'):
            ring_mode = not ring_mode
            print(f"[Mode] Switched to: {'RING ROI' if ring_mode else 'FULL FRAME'}")
        elif key == ord('p'):
            print("\n--- Current Serial Payload (MSG_RING 0x02) ---")
            print(f"Target count: {len(fused_targets)}")
            for i, t in enumerate(fused_targets):
                print(f"  Target #{i+1}: ID={t.target_id} ({t.status}) -> Fused=({t.fused_cx:.1f}, {t.fused_cy:.1f}), r={t.radius:.1f}")
                if t.status == "FUSED":
                    print(f"             Ring=({t.ring_cx}, {t.ring_cy}) Digit=({t.digit_cx}, {t.digit_cy})")
            print(f"Raw Hex: {' '.join(f'{b:02X}' for b in payload)}")
            print("----------------------------------------------\n")
        elif key == ord('s'):
            path = f"fusion_snapshot_{int(time.time())}.png"
            cv2.imwrite(path, display)
            print(f"[Snap] Saved snapshot: {path}")

    if link:
        link.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()
