#!/usr/bin/env python3
"""Simple HSV color block circle detector."""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

COLORS = {
    "yellow": {
        "lower": np.array([10, 85, 50]),
        "upper": np.array([35, 255, 255]),
        "bgr": (0, 255, 255),
    },
    "blue": {
        "lower": np.array([98, 179, 0]),
        "upper": np.array([121, 255, 255]),
        "bgr": (255, 0, 0),
    },
    "red": {
        "lower": np.array([0, 10, 10]),
        "upper": np.array([30, 255, 255]),
        "bgr": (0, 0, 255),
    },
    "red2": {
        "lower": np.array([150, 10, 10]),
        "upper": np.array([180, 255, 255]),
        "bgr": (0, 0, 255),
    },
    "green": {
        "lower": np.array([58, 73, 0]),
        "upper": np.array([74, 255, 186]),
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

MIN_RADIUS = 40


def get_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def detect(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * MIN_RADIUS * MIN_RADIUS:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < MIN_RADIUS:
            continue
        results.append((int(cx), int(cy), int(r)))
    return results


def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Detecting color blocks. ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for name, cfg in COLORS.items():
            if name == "red2":
                continue
            mask = get_mask(hsv, cfg["lower"], cfg["upper"])
            if name == "red":
                mask2 = get_mask(hsv, COLORS["red2"]["lower"], COLORS["red2"]["upper"])
                mask = cv2.bitwise_or(mask, mask2)

            circles = detect(mask)
            for cx, cy, r in circles:
                cv2.circle(display, (cx, cy), r, cfg["bgr"], 2)
                cv2.circle(display, (cx, cy), 3, cfg["bgr"], -1)
                cv2.putText(display, f"{name} ({cx},{cy}) r={r}",
                            (cx - r, cy - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg["bgr"], 1)
                print(f"[{name:9s}] center=({cx:4d}, {cy:4d})  r={r:3d}")

        cv2.imshow("Detect", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
