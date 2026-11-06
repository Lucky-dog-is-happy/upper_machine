#!/usr/bin/env python3
"""Standalone runner for Full-Screen Task & Gripper Status Display."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from upper_machine.status_display import TaskDisplay


def main():
    parser = argparse.ArgumentParser(description="Full-Screen 12-Digit 4-Group Task Code & Gripper Display")
    parser.add_argument("--task-code", type=str, default="156+123+516+231",
                        help="Initial 12-digit task code (default: 156+123+516+231)")
    parser.add_argument("--waiting", action="store_true",
                        help="Start in waiting-for-QR state (hide digits)")
    parser.add_argument("--pick", type=int, default=0,
                        help="Initial successful pick count X (default: 0)")
    parser.add_argument("--place", type=int, default=0,
                        help="Initial successful place count Y (default: 0)")
    parser.add_argument("--width", type=int, default=None,
                        help="Custom display width (default: auto-detect screen width)")
    parser.add_argument("--height", type=int, default=None,
                        help="Custom display height (default: auto-detect screen height)")
    parser.add_argument("--no-fullscreen", action="store_true",
                        help="Run in windowed mode instead of fullscreen")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide the 4-group Chinese annotations under each card")

    args = parser.parse_args()

    disp = TaskDisplay(
        width=args.width,
        height=args.height,
        fullscreen=not args.no_fullscreen,
        show_group_labels=not args.no_labels,
    )
    if not args.waiting:
        disp.set_task_code(args.task_code)
    else:
        disp.set_task_code(None)

    disp.set_counts(pick=args.pick, place=args.place)

    print("=== Full-Screen Task Display Launched (12 Digits, 4 Groups) ===")
    print(f"Task Code   : {args.task_code if not args.waiting else '(waiting for QR)'}")
    print(f"Pick Count X: {args.pick}")
    print(f"Place Count Y: {args.place}")
    print("Interactive keys in window:")
    print("  [1] / [+] : 抓取成功次数 +1")
    print("  [2] / []] : 放置成功次数 +1")
    print("  [3] / [-] : 抓取成功次数 -1")
    print("  [4] / [[] : 放置成功次数 -1")
    print("  [t]       : 切换 12 位四组示例任务码 / 等待状态")
    print("  [f]       : 全屏/窗口切换")
    print("  [q] / ESC : 退出程序\n")

    disp.run()


if __name__ == "__main__":
    main()
