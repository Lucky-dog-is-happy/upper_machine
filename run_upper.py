#!/usr/bin/env python3
"""One-shot entry point for the upper machine vision system.

Default configuration:
  - QR camera:        0
  - Block camera:     /dev/video20
  - Serial port:      /dev/ttyUSB0 (115200)
  - Ring detection:   Black ring only
  - Status display:   Enabled by default (Fullscreen 12-digit 4-group & Gripper stats)
  - QR send duration: 5.0 seconds after confirmed

Usage examples
──────────────
# 1. Default simplest start (QR cam0 + Block /dev/video20 + Serial /dev/ttyUSB0 + Fullscreen Display):
python run_upper.py

# 2. Disable full-screen display (headless / performance mode):
python run_upper.py --no-display

# 3. No QR mode (only block camera /dev/video20 + Serial /dev/ttyUSB0):
python run_upper.py --no-qr

# 3. Custom cameras or ports:
python run_upper.py --qr-camera 0 --block-camera /dev/video20 --serial-port /dev/ttyUSB0

# 4. List available serial ports:
python run_upper.py --list-ports

# 5. Disable serial (for local camera testing without hardware):
python run_upper.py --no-serial
"""

import os
import sys

# Ensure the project root is on sys.path so "upper_machine" package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from upper_machine.main import main

if __name__ == "__main__":
    main()
