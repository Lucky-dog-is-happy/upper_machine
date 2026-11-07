#!/usr/bin/env python3
"""Interactive Camera Photo Capture Tool.

Captures real-time frames from specified USB / UVC / onboard cameras,
displays a live preview window, and saves photos with auto-incrementing
indices upon pressing SPACE or 's'.

Usage Examples:
  # 1. Capture from default camera 0:
  python capture_photo.py

  # 2. Capture from camera 2 with prefix 'digits' (saved as digits_8.jpeg, digits_9.jpeg...):
  python capture_photo.py --camera 2 --prefix digits

  # 3. Capture from specific V4L2 device (e.g. /dev/video20) at 1280x720:
  python capture_photo.py --camera /dev/video20 --width 1280 --height 720

  # 4. Save photos into a specific folder:
  python capture_photo.py --output-dir ./captured_dataset --prefix sample

Keyboard Controls:
  SPACE / 's' : Take photo and save
  'c'         : Switch camera (if multiple cameras available)
  'f'         : Toggle fullscreen / windowed
  'q' / ESC   : Exit program
"""

import argparse
import glob
import os
import re
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

import cv2
import numpy as np


def parse_camera_src(val: str) -> int | str:
    """Parse camera argument: return integer if purely digits, else device path string."""
    val = str(val).strip()
    if val.isdigit():
        return int(val)
    return val


def get_next_index(directory: str, prefix: str, ext: str) -> int:
    """Find the next auto-incrementing file index that avoids overwriting existing files."""
    pattern = os.path.join(directory, f"{prefix}_*.{ext}")
    existing_files = glob.glob(pattern)
    max_idx = 0
    regex = re.compile(rf"^{re.escape(prefix)}_(\d+)\.{re.escape(ext)}$")
    for f in existing_files:
        basename = os.path.basename(f)
        match = regex.match(basename)
        if match:
            try:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return max_idx + 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Camera Photo Capture Tool")
    parser.add_argument("--camera", type=str, default="0",
                        help="Camera index (e.g. 0, 2) or device path (e.g. /dev/video20). Default: 0")
    parser.add_argument("--width", type=int, default=640,
                        help="Frame capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Frame capture height (default: 480)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save captured photos (default: current directory '.')")
    parser.add_argument("--prefix", type=str, default="digits",
                        help="File prefix for saved photos (default: 'digits')")
    parser.add_argument("--ext", type=str, choices=["jpeg", "jpg", "png"], default="jpeg",
                        help="Image file extension (default: jpeg)")
    parser.add_argument("--flip-h", action="store_true",
                        help="Horizontally flip camera frame (for mirrored cameras)")
    parser.add_argument("--flip-v", action="store_true",
                        help="Vertically flip camera frame (for inverted mounting)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cam_src = parse_camera_src(args.camera)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening camera: {cam_src} ({args.width}x{args.height})...")
    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {cam_src}", file=sys.stderr)
        print("Please check camera connection or try another index (e.g. --camera 0, --camera 2).", file=sys.stderr)
        sys.exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read back actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened successfully! Actual Resolution: {actual_w}x{actual_h}")
    print(f"Photos will be saved to: {os.path.abspath(args.output_dir)}/{args.prefix}_X.{args.ext}")
    print("\nControls:")
    print("  [SPACE] / 's' : Take photo")
    print("  'f'           : Toggle Fullscreen")
    print("  [ESC] / 'q'   : Quit\n")

    win_name = "Camera Capture Tool (SPACE=Save, ESC=Quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, actual_w, actual_h)

    shutter_timer = 0.0
    last_saved_path = ""
    next_idx = get_next_index(args.output_dir, args.prefix, args.ext)
    save_count = 0

    fps = 0.0
    fps_time = time.time()
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to read frame from camera. Retrying...", file=sys.stderr)
            time.sleep(0.01)
            continue

        # Optional hardware flips
        if args.flip_h and args.flip_v:
            frame = cv2.flip(frame, -1)
        elif args.flip_h:
            frame = cv2.flip(frame, 1)
        elif args.flip_v:
            frame = cv2.flip(frame, 0)

        # FPS calculation
        frame_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_counter / (now - fps_time)
            fps_time = now
            frame_counter = 0

        # Keep a clean copy for pristine saving without overlay text
        raw_frame = frame.copy()
        display = frame.copy()

        # Shutter visual effect (flash white briefly)
        if now < shutter_timer:
            flash_intensity = int(180 * ((shutter_timer - now) / 0.15))
            overlay = np.full(display.shape, flash_intensity, dtype=np.uint8)
            display = cv2.add(display, overlay)

        # Draw UI Overlay
        h, w = display.shape[:2]
        info_bar = f"Cam: {cam_src} | {w}x{h} | FPS: {fps:.1f} | Saved: {save_count}"
        cv2.putText(display, info_bar, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(display, "[SPACE / 's'] Capture   [ESC / 'q'] Exit", (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

        # If recently saved, display filename feedback
        if last_saved_path and now - (shutter_timer - 0.15) < 2.0:
            saved_msg = f"Saved: {os.path.basename(last_saved_path)}"
            cv2.putText(display, saved_msg, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            print("Exiting...")
            break
        elif key in (32, ord("s"), ord("S")):  # SPACE or 's'
            filename = f"{args.prefix}_{next_idx}.{args.ext}"
            filepath = os.path.join(args.output_dir, filename)

            # Save unadorned raw image
            cv2.imwrite(filepath, raw_frame)
            last_saved_path = filepath
            shutter_timer = now + 0.15  # 150ms flash effect
            save_count += 1
            print(f"[CAPTURE #{save_count}] Saved: {filepath} ({actual_w}x{actual_h})")
            next_idx += 1
        elif key in (ord("f"), ord("F")):
            prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
            new_prop = cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, new_prop)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total {save_count} photo(s) captured.")


if __name__ == "__main__":
    main()
