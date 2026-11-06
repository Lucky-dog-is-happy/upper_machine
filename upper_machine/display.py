"""OpenCV visualization helpers for blocks, rings, and task code."""

import cv2
import numpy as np

from .config import BlockInfo, RingInfo, TaskCode, COLOR_MAP, COLOR_HSV, RING_HSV


def draw_blocks(display: np.ndarray, blocks: list[BlockInfo]) -> None:
    for b in blocks:
        cfg = COLOR_HSV.get(b.color, COLOR_HSV["red"])
        bgr = cfg["bgr"]
        cv2.circle(display, (b.cx, b.cy), b.radius, bgr, 2)
        cv2.circle(display, (b.cx, b.cy), 3, bgr, -1)
        label = f"{b.color} ({b.cx},{b.cy}) r={b.radius}"
        cv2.putText(display, label, (b.cx - b.radius, b.cy - b.radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)


def draw_rings(display: np.ndarray, rings: list[RingInfo]) -> None:
    for r in rings:
        digit = getattr(r, "digit_id", 5)
        conf = getattr(r, "confidence", 1.0)

        if digit in (1, 2, 3):
            outline_color = (0, 255, 0)   # Green outline for confirmed target
            label = f"Target #{digit} ({r.cx},{r.cy}) r={r.radius} [{conf:.2f}]"
            center_text = str(digit)
        else:
            outline_color = (0, 255, 255) # Yellow outline for unknown digit ring
            label = f"Ring (unknown) ({r.cx},{r.cy}) r={r.radius}"
            center_text = "?"

        # Outer circle outline
        cv2.circle(display, (r.cx, r.cy), r.radius, outline_color, 2)
        # Inner search core boundary
        cv2.circle(display, (r.cx, r.cy), max(1, int(r.radius * 0.65)), (100, 100, 100), 1)
        # Center dot
        cv2.circle(display, (r.cx, r.cy), 4, (0, 0, 255), -1)

        # Label above ring
        cv2.putText(display, label, (r.cx - r.radius, max(15, r.cy - r.radius - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 1)

        # Center digit label
        cv2.putText(display, center_text, (r.cx - 6, r.cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_task_code(display: np.ndarray, tc: TaskCode) -> None:
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


def draw_status(display: np.ndarray, status: str) -> None:
    cv2.putText(display, status, (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
