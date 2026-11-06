"""Color block and black ring detection algorithms."""

import cv2
import numpy as np

from .config import (
    COLOR_HSV, COLOR_NAME_TO_ID, RING_HSV, RING_BAND_INNER,
    BlockInfo, RingInfo,
)


def get_mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             kernel_size: tuple[int, int] = (5, 5)) -> np.ndarray:
    """Create a cleaned binary mask from HSV range."""
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
    """Detect colored circular blocks via HSV thresholding + contour analysis."""
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

            x, y, w, h = cv2.boundingRect(cnt)
            if max(w, h) == 0:
                continue
            aspect_ratio = min(w, h) / max(w, h)
            if aspect_ratio < min_aspect_ratio:
                continue

            circle_area = np.pi * (r ** 2)
            fill_ratio = area / circle_area if circle_area > 0 else 0
            if fill_ratio < min_fill_ratio:
                continue

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
                 param2: float = 0.8) -> list[RingInfo]:
    """Detect black circular rings on the ground.

    Runs HoughCircles on grayscale frame, dedups concentric circles,
    and checks if the ring annulus contains sufficient black pixels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    circles = cv2.HoughCircles(
        gauss, cv2.HOUGH_GRADIENT_ALT, 1.2, minDist=max(30, min_radius),
        param1=100, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []

    circles = np.round(circles.reshape((-1, 3))).astype(int)
    circles = dedup_concentric_circles([(int(c[0]), int(c[1]), int(c[2])) for c in circles])

    # Direct black mask for ring verification (use gentler (3,3) kernel to protect thin lines)
    black_cfg = RING_HSV["black"]
    black_mask = get_mask(hsv, black_cfg["lower"], black_cfg["upper"], kernel_size=(3, 3))

    h, w = gray.shape
    rings = []
    for cx, cy, r in circles:
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            continue

        # Annulus band mask between RING_BAND_INNER*r and r
        band = np.zeros(gray.shape, np.uint8)
        cv2.circle(band, (cx, cy), r, 255, -1)
        cv2.circle(band, (cx, cy), max(1, int(r * RING_BAND_INNER)), 0, -1)
        band_roi = band[y1:y2, x1:x2]

        black_area = int(cv2.countNonZero(cv2.bitwise_and(black_mask[y1:y2, x1:x2], band_roi)))

        # Adaptive area requirement: thin ring lines scale with circumference (~2*pi*r).
        # Capped by min_color_area so small circles (r < 90) are not unfairly dropped.
        adaptive_threshold = max(30, int(2 * np.pi * r * 0.35))
        effective_min_area = min(min_color_area, adaptive_threshold)
        if black_area < effective_min_area:
            continue

        rings.append(RingInfo(
            color="black",
            color_id=COLOR_NAME_TO_ID.get("black", 5),
            cx=int(cx),
            cy=int(cy),
            radius=int(r),
        ))

    return rings
