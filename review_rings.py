#!/usr/bin/env python3
"""Batch Ring & Digit Detection Review Tool.

Detects black circular rings and classifies inner digits 1, 2, 3 across
images in a specified directory (e.g. raw_images/), saves annotated review
images with bounding circles and confidence scores, and produces a summary.

Usage:
  # Review all images in raw_images/ and save output to ring_review_output/
  python review_rings.py

  # Custom directory and confidence:
  python review_rings.py --input-dir raw_images --output-dir ring_review_output --conf 0.65
"""

import argparse
import glob
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

from upper_machine.config import RingInfo
from upper_machine.detection import detect_rings
from upper_machine.digit import DigitClassifier
from upper_machine.display import draw_rings


def parse_args():
    parser = argparse.ArgumentParser(description="Ring & Digit Detection Review Tool")
    parser.add_argument("--input-dir", type=str, default="raw_images",
                        help="Input directory containing photos (default: raw_images)")
    parser.add_argument("--output-dir", type=str, default="ring_review_output",
                        help="Directory to save annotated review photos (default: ring_review_output)")
    parser.add_argument("--ring-param2", type=float, default=0.70,
                        help="HoughCircles param2 (circle perfectness, default 0.70)")
    parser.add_argument("--conf", type=float, default=0.55,
                        help="Digit minimum confidence threshold (default: 0.55)")
    parser.add_argument("--min-radius", type=int, default=25,
                        help="Min ring radius (default: 25)")
    parser.add_argument("--max-radius", type=int, default=220,
                        help="Max ring radius (default: 220)")
    parser.add_argument("--min-area", type=int, default=40,
                        help="Min ring black area (default: 40)")
    parser.add_argument("--max-rings", type=int, default=3,
                        help="Max rings to retain (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
    image_files = [f for f in input_files if f.lower().endswith((".jpeg", ".jpg", ".png", ".bmp"))]

    if not image_files:
        print(f"Error: No images found in '{args.input_dir}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    classifier = DigitClassifier()

    print(f"================================================================")
    print(f"  Starting Ring & Digit Detection Review")
    print(f"  Input: {args.input_dir} ({len(image_files)} images)")
    print(f"  Output: {args.output_dir}")
    print(f"  Confidence Threshold: {args.conf}")
    print(f"  (Note: ID 5 cancelled, empty rings will not be recognized/sent)")
    print(f"================================================================\n")

    total_images = len(image_files)
    images_with_targets = 0
    digit_counts = {1: 0, 2: 0, 3: 0}
    detailed_records = []

    start_time = time.time()

    for idx, p in enumerate(image_files, 1):
        filename = os.path.basename(p)
        frame = cv2.imread(p)
        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rings = detect_rings(
            frame=frame,
            hsv=hsv,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_color_area=args.min_area,
            param2=args.ring_param2,
            digit_classifier=classifier,
            max_rings=args.max_rings,
            digit_min_conf=args.conf,
        )

        display = frame.copy()
        if rings:
            images_with_targets += 1
            draw_rings(display, rings)
            target_descs = []
            for r in rings:
                d = getattr(r, "digit_id", r.color_id)
                digit_counts[d] = digit_counts.get(d, 0) + 1
                target_descs.append(f"#{d}({r.cx},{r.cy},conf={r.confidence:.2f})")
            detailed_records.append((filename, len(rings), ", ".join(target_descs)))
        else:
            detailed_records.append((filename, 0, "No valid digit targets (Blank / rejected)"))

        out_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(out_path, display)

    elapsed = time.time() - start_time

    # Print summary table
    print(f"{'Image File':<36} | {'Targets':<7} | {'Detected Details'}")
    print("-" * 80)
    for fn, count, desc in detailed_records:
        status_symbol = "✓" if count > 0 else "·"
        print(f"{status_symbol} {fn:<34} | {count:<7} | {desc}")
    print("-" * 80)

    print("\nSummary Statistics:")
    print(f"  • Total Images Evaluated   : {total_images}")
    print(f"  • Images With Valid Targets: {images_with_targets} ({images_with_targets/total_images*100:.1f}%)")
    print(f"  • Blank/Filtered Ring Images: {total_images - images_with_targets} (Safe 0 False-Positives)")
    print(f"  • Target Counts Recognized : Digit #1: {digit_counts[1]}, Digit #2: {digit_counts[2]}, Digit #3: {digit_counts[3]}")
    print(f"  • Processing Speed         : {elapsed / total_images * 1000:.1f} ms/frame")
    print(f"  • Annotated Images Saved to: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
