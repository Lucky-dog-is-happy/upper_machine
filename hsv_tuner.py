#!/usr/bin/env python3
"""HSV threshold tuner with trackbars for finding optimal color ranges."""

import argparse
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

WIN_ORIG = "Original"
WIN_HSV = "HSV"
WIN_MASK = "Mask"
WIN_RESULT = "Result"


def nothing(x):
    pass


def main():
    parser = argparse.ArgumentParser(description="HSV threshold tuner")
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--image", type=str, help="Use image file instead of camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read image: {args.image}", file=sys.stderr)
            sys.exit(1)
        use_camera = False
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Cannot open camera {args.camera}", file=sys.stderr)
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        use_camera = True

    cv2.namedWindow(WIN_HSV)
    cv2.namedWindow(WIN_MASK)
    cv2.namedWindow(WIN_RESULT)

    cv2.createTrackbar("H min", WIN_HSV, 0, 179, nothing)
    cv2.createTrackbar("H max", WIN_HSV, 179, 179, nothing)
    cv2.createTrackbar("S min", WIN_HSV, 0, 255, nothing)
    cv2.createTrackbar("S max", WIN_HSV, 255, 255, nothing)
    cv2.createTrackbar("V min", WIN_HSV, 0, 255, nothing)
    cv2.createTrackbar("V max", WIN_HSV, 255, 255, nothing)

    # morphology
    cv2.namedWindow(WIN_ORIG)
    cv2.createTrackbar("Open ite", WIN_ORIG, 2, 10, nothing)
    cv2.createTrackbar("Close ite", WIN_ORIG, 2, 10, nothing)
    cv2.createTrackbar("Kernel", WIN_ORIG, 5, 20, nothing)

    print("Adjust trackbars to find HSV range. Press ESC to quit.")
    print("After tuning, copy the values:\n")

    while True:
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H min", WIN_HSV)
        h_max = cv2.getTrackbarPos("H max", WIN_HSV)
        s_min = cv2.getTrackbarPos("S min", WIN_HSV)
        s_max = cv2.getTrackbarPos("S max", WIN_HSV)
        v_min = cv2.getTrackbarPos("V min", WIN_HSV)
        v_max = cv2.getTrackbarPos("V max", WIN_HSV)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)

        # morphology
        k = max(1, cv2.getTrackbarPos("Kernel", WIN_ORIG))
        k = k if k % 2 == 1 else k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        open_ite = cv2.getTrackbarPos("Open ite", WIN_ORIG)
        close_ite = cv2.getTrackbarPos("Close ite", WIN_ORIG)
        if open_ite > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_ite)
        if close_ite > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_ite)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        info = f"H:{h_min}-{h_max} S:{s_min}-{s_max} V:{v_min}-{v_max}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Kernel:{k} Open:{open_ite} Close:{close_ite}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WIN_ORIG, frame)
        cv2.imshow(WIN_HSV, hsv)
        cv2.imshow(WIN_MASK, mask)
        cv2.imshow(WIN_RESULT, result)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("p"):
            print(f'lower = np.array([{h_min}, {s_min}, {v_min}])')
            print(f'upper = np.array([{h_max}, {s_max}, {v_max}])')
            print()

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
