"""QR code detection and task code parsing."""

import os
import sys

import cv2

from .config import TaskCode


def init_qr_detector():
    """Initialize WeChat QR code detector (auto-detect OpenCV version)."""
    cv_version = tuple(int(x) for x in cv2.__version__.split(".")[:2])
    if cv_version >= (5, 0):
        return cv2.wechat_qrcode_WeChatQRCode()
    models_dir = os.path.expanduser("~/.opencv/wechat_qrcode")
    model_names = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    model_paths = [os.path.join(models_dir, n) for n in model_names]
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing QR model files: {missing}", file=sys.stderr)
        print("Run: python qr_scanner.py --download-models", file=sys.stderr)
        sys.exit(1)
    return cv2.wechat_qrcode_WeChatQRCode(*model_paths)


def parse_task_code(text: str) -> TaskCode:
    """Parse task code string like '156+123+516+231' into TaskCode.

    Format: group1_colors + group1_positions + group2_colors + group2_positions
    Each group is a 3-digit number, each digit 1-6.
    """
    tc = TaskCode(raw=text)
    text = text.strip()
    parts = text.split("+")
    if len(parts) != 4:
        return tc
    try:
        nums = [int(p) for p in parts]
        if not all(100 <= n <= 999 for n in nums):
            return tc
        tc.batch1_colors = [int(d) for d in str(nums[0])]
        tc.batch1_positions = [int(d) for d in str(nums[1])]
        tc.batch2_colors = [int(d) for d in str(nums[2])]
        tc.batch2_positions = [int(d) for d in str(nums[3])]
        if not all(1 <= c <= 6 for c in tc.batch1_colors + tc.batch2_colors):
            return tc
        if not all(1 <= p <= 6 for p in tc.batch1_positions + tc.batch2_positions):
            return tc
        tc.valid = True
    except (ValueError, IndexError):
        pass
    return tc
