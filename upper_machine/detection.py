"""Color block and black ring detection algorithms."""

import cv2
import numpy as np

from .config import (
    COLOR_HSV, COLOR_NAME_TO_ID, RING_HSV, RING_BAND_INNER,
    BlockInfo, RingInfo,
)
from .digit import (
    DigitClassifier, extract_digit_scores_from_ring, match_rings_to_digits,
)


def clean_mask(mask: np.ndarray, kernel_size: tuple[int, int] = (5, 5)) -> np.ndarray:
    """Apply morphological open and close to a binary mask."""
    if kernel_size[0] > 1 and kernel_size[1] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def get_mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             kernel_size: tuple[int, int] = (5, 5)) -> np.ndarray:
    """Create a cleaned binary mask from HSV range."""
    mask = cv2.inRange(hsv, lower, upper)
    return clean_mask(mask, kernel_size)


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
        if name == "red":
            mask1 = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
            mask2 = cv2.inRange(hsv, COLOR_HSV["red2"]["lower"], COLOR_HSV["red2"]["upper"])
            mask = cv2.bitwise_or(mask1, mask2)
            mask = clean_mask(mask)
        else:
            mask = get_mask(hsv, cfg["lower"], cfg["upper"])

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
                             tolerance: int = 25) -> list[tuple[int, int, int]]:
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
                 param2: float = 0.72,
                 digit_classifier: DigitClassifier | None = None,
                 max_rings: int = 3,
                 digit_min_conf: float = 0.65) -> list[RingInfo]:
    """Detect black circular rings on the ground and classify inner digits (1, 2, 3).

    Pipeline:
      1. Detect black circular rings using HoughCircles ALT + black annulus verification.
      2. Keep up to max_rings valid ring candidates (sorted by detected radius/area).
      3. For each ring candidate, inspect the inner core region for digits 1, 2, 3.
      4. Globally match digits 1, 2, 3 to rings without conflicts (each digit unique).
    """
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

    # Direct black mask for ring verification (use gentler (3,3) kernel to protect thin lines)
    black_cfg = RING_HSV["black"]
    black_mask = get_mask(hsv, black_cfg["lower"], black_cfg["upper"], kernel_size=(3, 3))

    h, w = gray.shape
    candidates: list[tuple[int, int, int, int]] = []  # (cx, cy, r, black_area)
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

        candidates.append((int(cx), int(cy), int(r), black_area))

    if not candidates:
        return []

    # Sort candidates by radius (or black area) and keep up to max_rings
    candidates.sort(key=lambda c: c[2], reverse=True)
    selected = candidates[:max_rings]

    # Inspect inner core for digits 1, 2, 3 if classifier provided
    if digit_classifier is not None:
        scores_list = [
            extract_digit_scores_from_ring(frame, cx, cy, r, digit_classifier)
            for cx, cy, r, _ in selected
        ]
        assigned_ids = match_rings_to_digits(scores_list, min_confidence=digit_min_conf)
    else:
        assigned_ids = [5] * len(selected)

    rings = []
    for idx, (cx, cy, r, _) in enumerate(selected):
        target_id = assigned_ids[idx]
        conf = 1.0
        if digit_classifier is not None and target_id in (1, 2, 3):
            conf = scores_list[idx].get(target_id, 1.0)

        rings.append(RingInfo(
            color="black",
            color_id=target_id,  # 1, 2, 3 for digit, 5 for unknown black ring
            cx=cx,
            cy=cy,
            radius=r,
            digit_id=target_id,
            confidence=conf,
        ))

    return rings
