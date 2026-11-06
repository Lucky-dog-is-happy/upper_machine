#!/usr/bin/env python3
"""Real-time color block detector using OpenCV.

Detects 6 colors (yellow, blue, red, green, sky-blue, black) via HSV thresholding,
finds circular shapes using HoughCircles, and outputs center coordinates.
"""

import argparse
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

COLORS = {
    "yellow": {
        "lower": np.array([17, 75, 70]),
        "upper": np.array([38, 255, 255]),
        "bgr": (0, 255, 255),
    },
    "blue": {
        "lower": np.array([95, 165, 50]),
        "upper": np.array([125, 255, 255]),
        "bgr": (255, 0, 0),
    },
    "red": {
        "lower": np.array([0, 55, 50]),
        "upper": np.array([13, 255, 255]),
        "bgr": (0, 0, 255),
    },
    "green": {
        "lower": np.array([46, 55, 35]),
        "upper": np.array([78, 255, 255]),
        "bgr": (0, 255, 0),
    },
    "sky-blue": {
        "lower": np.array([90, 45, 60]),
        "upper": np.array([125, 160, 255]),
        "bgr": (255, 255, 0),
    },
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([179, 110, 90]),
        "bgr": (50, 50, 50),
    },
}

RED_WRAP_LOWER = np.array([165, 55, 50])
RED_WRAP_UPPER = np.array([180, 255, 255])


def build_color_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    """Build binary masks for each color from an HSV frame."""
    masks = {}
    for name, cfg in COLORS.items():
        mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        if name == "red":
            wrap_mask = cv2.inRange(hsv, RED_WRAP_LOWER, RED_WRAP_UPPER)
            mask = cv2.bitwise_or(mask, wrap_mask)
        masks[name] = mask
    return masks


def merge_and_clean(
    masks: dict[str, np.ndarray], kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge all masks and apply morphological open+close.

    Returns:
        combined: merged mask after morphology
        label_map: per-pixel color index (0=bg, 1..N=colors)
    """
    combined = np.zeros(list(masks.values())[0].shape, dtype=np.uint8)
    label_map = np.zeros(combined.shape, dtype=np.int32)

    for i, (name, mask) in enumerate(masks.items(), start=1):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined = cv2.bitwise_or(combined, cleaned)
        label_map[cleaned > 0] = i

    return combined, label_map


def detect_circles(
    mask: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> np.ndarray | None:
    """Run HoughCircles on a binary mask. Returns circles or None."""
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        return np.uint16(np.around(circles[0]))
    return None


def match_color(
    cx: int, cy: int, r: int, label_map: np.ndarray, color_names: list[str]
) -> str:
    """Determine the color of a detected circle by sampling its region."""
    h, w = label_map.shape
    x1 = max(0, cx - r)
    x2 = min(w, cx + r)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r)

    region = label_map[y1:y2, x1:x2]
    if region.size == 0:
        return "unknown"

    counts = {}
    for i, name in enumerate(color_names, start=1):
        cnt = np.count_nonzero(region == i)
        if cnt > 0:
            counts[name] = cnt

    if not counts:
        return "unknown"
    return max(counts, key=counts.get)


def draw_results(
    display: np.ndarray,
    circles: np.ndarray | None,
    label_map: np.ndarray,
    color_names: list[str],
) -> list[tuple[str, int, int, int]]:
    """Draw circles, centers, and labels on the display frame.

    Returns list of (color_name, cx, cy, radius).
    """
    results = []
    if circles is None:
        return results

    for cx, cy, r in circles:
        color_name = match_color(int(cx), int(cy), int(r), label_map, color_names)
        bgr = COLORS.get(color_name, {}).get("bgr", (255, 255, 255))

        cv2.circle(display, (cx, cy), r, bgr, 2)
        cv2.line(display, (cx - 5, cy), (cx + 5, cy), bgr, 2)
        cv2.line(display, (cx, cy - 5), (cx, cy + 5), bgr, 2)

        label = f"{color_name} ({cx},{cy})"
        cv2.putText(
            display, label, (cx - r, cy - r - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2,
        )
        results.append((color_name, int(cx), int(cy), int(r)))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Color block circle detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--min-area", type=int, default=500, help="Min contour area")
    parser.add_argument("--min-radius", type=int, default=20, help="Min circle radius")
    parser.add_argument("--max-radius", type=int, default=200, help="Max circle radius")
    parser.add_argument("--show-mask", action="store_true", help="Show debug mask windows")
    args = parser.parse_args()

    color_names = list(COLORS.keys())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print("ESC to quit.\n")

    window_name = "Color Detector (ESC quit)"
    cv2.namedWindow(window_name)

    if args.show_mask:
        cv2.namedWindow("Combined Mask")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame", file=sys.stderr)
            break

        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = build_color_masks(hsv)
        combined, label_map = merge_and_clean(masks, kernel)

        if args.show_mask:
            cv2.imshow("Combined Mask", combined)

        circles = detect_circles(combined, args.min_radius, args.max_radius)

        results = draw_results(display, circles, label_map, color_names)

        frame_count += 1
        if frame_count % 30 == 0:
            for name, cx, cy, r in results:
                print(f"  [{name:9s}] center=({cx:4d}, {cy:4d})  r={r:3d}")
            if not results:
                print("  (no blocks detected)")
            print()

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
