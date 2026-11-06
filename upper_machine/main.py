"""Main loop: argument parsing + state machine.

State machine flow
──────────────────
PHASE 1  QR scanning (qr_active=True, when not --no-qr)
  - cam_qr (default: 0) scans for task code (3 consecutive matches to confirm)
  - cam_block (default: /dev/video20) detects blocks simultaneously in MODE_BLOCK
  - On QR confirmed:
      • send_task_code() immediately
      • record qr_send_until = now + qr_send_duration
      • close QR camera → PHASE 2

PHASE 2  Task-code timed broadcast + block detection
  - Every send_interval: send_task_code() while now < qr_send_until
  - After qr_send_until passes: task-code serial sending stops permanently → PHASE 3

PHASE 3  Block detection (MODE_BLOCK)
  - Continuously send_blocks()
  - Lower board sends MSG_MODE_CMD(MODE_RING=1) → PHASE 4

PHASE 4  Black Ring detection (MODE_RING)
  - Continuously detect & send black rings
  - Lower board sends MSG_MODE_CMD(MODE_BLOCK=0) → back to PHASE 3
"""

import argparse
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

import cv2

from .camera import ThreadedCamera
from .config import TaskCode
from .detection import detect_blocks, detect_rings
from .digit import DigitClassifier
from .display import draw_blocks, draw_rings, draw_task_code, draw_status
from .qr_code import init_qr_detector, parse_task_code
from .serial_comm import (
    SerialSender, MODE_BLOCK, MODE_RING,
    MSG_GRIPPER_COUNT, GRIPPER_PICK_INC, GRIPPER_PICK_DEC,
    GRIPPER_PLACE_INC, GRIPPER_PLACE_DEC, GRIPPER_RESET, GRIPPER_SET
)
from .status_display import TaskDisplay


def parse_camera_src(val: str) -> int | str:
    """Parse camera argument: return integer if purely digits, else device path string."""
    val = str(val).strip()
    if val.isdigit():
        return int(val)
    return val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upper machine: QR + color block/black-ring detector")

    # Camera defaults: QR=0, Block=/dev/video20
    parser.add_argument("--qr-camera", type=str, default="0",
                        help="Camera index or path for QR scanning (default '0')")
    parser.add_argument("--block-camera", type=str, default="/dev/video20",
                        help="Camera index or path for block/ring detection (default '/dev/video20')")
    parser.add_argument("--no-qr", action="store_true",
                        help="Disable QR camera (single-camera mode: only serial + block camera)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")

    # Serial defaults: /dev/ttyUSB0, enabled by default
    parser.add_argument("--serial-port", type=str, default="/dev/ttyUSB0",
                        help="Serial port (default '/dev/ttyUSB0')")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--no-serial", action="store_true", help="Disable serial output")
    parser.add_argument("--list-ports", action="store_true",
                        help="List available serial ports and exit")
    parser.add_argument("--send-interval", type=float, default=0.5,
                        help="Seconds between serial sends (default 0.5)")

    # QR send duration
    parser.add_argument("--qr-send-duration", type=float, default=5.0,
                        help="Seconds to keep sending task-code after QR confirmed (default 5.0). "
                             "Set 0 to send only once immediately.")

    # Initial mode
    parser.add_argument("--mode", choices=["block", "ring"], default="block",
                        help="Initial detection mode (lower board overrides via serial)")

    # Block detection params
    parser.add_argument("--block-min-radius", type=int, default=20)
    parser.add_argument("--block-max-radius", type=int, default=150)
    parser.add_argument("--block-min-circularity", type=float, default=0.60)
    parser.add_argument("--block-min-fill", type=float, default=0.60)
    parser.add_argument("--block-min-aspect", type=float, default=0.65)

    # Black Ring detection params
    parser.add_argument("--ring-min-radius", type=int, default=60,
                        help="Min ring radius (px, default 60)")
    parser.add_argument("--ring-max-radius", type=int, default=200)
    parser.add_argument("--ring-min-area", type=int, default=100,
                        help="Min black pixel count inside ring annulus (default 100, adaptive for small rings)")
    parser.add_argument("--ring-param2", type=float, default=0.72,
                        help="HoughCircles param2 (circle perfectness, 0~1, default 0.72)")

    parser.add_argument("--flip-h", action="store_true",
                        help="Horizontally flip camera frame (for mirrored cameras)")
    parser.add_argument("--flip-v", action="store_true",
                        help="Vertically flip camera frame (for inverted mounting)")

    # Full-screen Status Display (Enabled by default)
    parser.add_argument("--no-display", action="store_true",
                        help="Disable full-screen task code & gripper status display (enabled by default)")
    parser.add_argument("--no-fullscreen", action="store_true",
                        help="Run display in windowed mode instead of fullscreen")

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── List ports and exit ──────────────────────────────────────────────────
    if args.list_ports:
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            ports = []
        print("Available serial ports:")
        for p in ports:
            print(f"  {p}")
        if not ports:
            print("  (none found)")
        return

    # Parse camera device specifiers (int or path string)
    qr_cam_src = parse_camera_src(args.qr_camera)
    block_cam_src = parse_camera_src(args.block_camera)

    # ── QR camera setup ──────────────────────────────────────────────────────
    qr_detector = None
    cam_qr: ThreadedCamera | None = None
    qr_active = False

    if not args.no_qr:
        if qr_cam_src == block_cam_src:
            print(f"[WARN] --qr-camera and --block-camera are both {qr_cam_src}. "
                  "OpenCV cannot open the same camera twice. Auto-disabling QR.",
                  file=sys.stderr)
            args.no_qr = True
        else:
            qr_detector = init_qr_detector()
            cam_qr = ThreadedCamera(qr_cam_src, args.width, args.height)
            qr_active = cam_qr.start()
            if not qr_active:
                print(f"[WARN] Cannot open QR camera {qr_cam_src}, QR disabled.",
                      file=sys.stderr)
    else:
        print("QR scan disabled (single-camera mode: serial + block camera only).")

    # ── Block camera setup ───────────────────────────────────────────────────
    cam_block = ThreadedCamera(block_cam_src, args.width, args.height)
    if not cam_block.start():
        print(f"[ERROR] Cannot open block camera {block_cam_src}", file=sys.stderr)
        if cam_qr and qr_active:
            cam_qr.stop()
        sys.exit(1)

    qr_info_str = f"{qr_cam_src} (active)" if qr_active else "disabled"
    print(f"Cameras: QR={qr_info_str} | Block={block_cam_src}")

    # ── Serial setup (default: /dev/ttyUSB0) ──────────────────────────────────
    sender: SerialSender | None = None
    if not args.no_serial and args.serial_port:
        sender = SerialSender(args.serial_port, args.baudrate)
        sender.open()
    else:
        print("Serial disabled.")

    # ── Digit classifier for ring-digit association ─────────────────────────
    digit_classifier = DigitClassifier()

    # ── Windows & Display Setup ──────────────────────────────────────────────
    WIN_QR = "QR Camera (ESC=quit, m=toggle mode)"
    WIN_BLOCK = "Block/Ring Camera (ESC=quit, m=toggle mode, p=print)"
    WIN_DISPLAY = "Task & Gripper Status Monitor"

    if qr_active:
        cv2.namedWindow(WIN_QR)
    cv2.namedWindow(WIN_BLOCK)

    task_display: TaskDisplay | None = None
    if not args.no_display:
        task_display = TaskDisplay(fullscreen=not args.no_fullscreen, window_name=WIN_DISPLAY)
        # Requirement: Before QR is detected, do not display numbers
        task_display.set_task_code(None)
        cv2.namedWindow(WIN_DISPLAY, cv2.WINDOW_NORMAL)
        if not args.no_fullscreen:
            cv2.setWindowProperty(WIN_DISPLAY, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WIN_DISPLAY, 1280, 720)
        print(f"Status Display: ENABLED ({'FULLSCREEN' if not args.no_fullscreen else 'WINDOWED'})")
    else:
        print("Status Display: DISABLED (--no-display)")

    # ── State ────────────────────────────────────────────────────────────────
    last_task_code = TaskCode()
    qr_candidate = ""
    qr_match_count = 0
    qr_send_until = 0.0          # timestamp after which task-code sending stops
    task_code_sent = False

    last_send_time = 0.0
    consecutive_block_fails = 0
    mode = MODE_RING if args.mode == "ring" else MODE_BLOCK

    print(f"Running. Initial mode={'RING' if mode == MODE_RING else 'BLOCK'}. "
          f"QR send duration={args.qr_send_duration}s.\n"
          "ESC=quit  m=toggle mode  p=print detections")

    # ════════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ════════════════════════════════════════════════════════════════════════
    while True:
        now = time.time()

        # ── Poll lower-board commands (mode switch and gripper stats) ───────
        if sender:
            events = sender.poll_events()
            if events.new_mode is not None and events.new_mode != mode:
                mode = events.new_mode
                print(f"[RX] Mode -> {'RING' if mode == MODE_RING else 'BLOCK'}")

            if task_display:
                if events.set_pick is not None:
                    task_display.set_counts(pick=events.set_pick)
                    print(f"[RX Gripper] Set Pick: {events.set_pick}")
                elif events.pick_delta != 0:
                    task_display.increment_pick(events.pick_delta)
                    print(f"[RX Gripper] Pick {events.pick_delta:+d} -> Total: {task_display.pick_count}")

                if events.set_place is not None:
                    task_display.set_counts(place=events.set_place)
                    print(f"[RX Gripper] Set Place: {events.set_place}")
                elif events.place_delta != 0:
                    task_display.increment_place(events.place_delta)
                    print(f"[RX Gripper] Place {events.place_delta:+d} -> Total: {task_display.place_count}")

        # ── PHASE 1 / 2: QR camera (active until confirmed) ─────────────────
        if qr_active and cam_qr:
            ret_qr, frame_qr = cam_qr.read()
            if ret_qr and frame_qr is not None:
                display_qr = frame_qr.copy()
                results, points = qr_detector.detectAndDecode(frame_qr)
                if results:
                    for text, pt in zip(results, points):
                        tc = parse_task_code(text)
                        if tc.valid:
                            if tc.raw == qr_candidate:
                                qr_match_count += 1
                            else:
                                qr_candidate = tc.raw
                                qr_match_count = 1
                            print(f"[QR] Candidate: {tc.raw} ({qr_match_count}/3)")
                            if qr_match_count >= 3:
                                last_task_code = tc
                                print(f"[QR] Confirmed: {tc.raw}. "
                                      f"Will broadcast for {args.qr_send_duration}s.")
                                if task_display:
                                    task_display.set_task_code(tc.raw)
                                if sender:
                                    sender.send_task_code(tc)  # immediate send
                                qr_send_until = now + args.qr_send_duration
                                task_code_sent = True
                                qr_active = False
                                cam_qr.stop()
                                cv2.destroyWindow(WIN_QR)
                                break

                        pts = pt.astype(int).reshape((-1, 1, 2))
                        cv2.polylines(display_qr, [pts], True, (0, 255, 0), 2)
                        cx, cy = pts[:, 0, :].mean(axis=0).astype(int)
                        cv2.putText(display_qr, f"QR: {text}", (cx, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if qr_active:
                    draw_task_code(display_qr, last_task_code)
                    cv2.imshow(WIN_QR, display_qr)

        # ── Block / Ring camera ──────────────────────────────────────────────
        ret_block, frame_block = cam_block.read()
        if not ret_block or frame_block is None:
            consecutive_block_fails += 1
            if consecutive_block_fails > 100:
                print("[ERROR] Block camera failed continuously. Exiting.", file=sys.stderr)
                break
            time.sleep(0.005)
            continue
        consecutive_block_fails = 0

        # Optional hardware frame flips
        if args.flip_h and args.flip_v:
            frame_block = cv2.flip(frame_block, -1)
        elif args.flip_h:
            frame_block = cv2.flip(frame_block, 1)
        elif args.flip_v:
            frame_block = cv2.flip(frame_block, 0)

        hsv = cv2.cvtColor(frame_block, cv2.COLOR_BGR2HSV)
        display_block = frame_block.copy()
        blocks = []
        rings = []

        if mode == MODE_RING:
            rings = detect_rings(
                frame_block, hsv,
                min_radius=args.ring_min_radius,
                max_radius=args.ring_max_radius,
                min_color_area=args.ring_min_area,
                param2=args.ring_param2,
                digit_classifier=digit_classifier,
                max_rings=3,
            )
            draw_rings(display_block, rings)
            target_ids = [str(r.digit_id) if r.digit_id in (1, 2, 3) else "?" for r in rings]
            ring_desc = f" [{', '.join(target_ids)}]" if target_ids else ""
            status = f"Mode: RING (black){ring_desc} | Rings: {len(rings)}"
        else:
            blocks = detect_blocks(
                hsv,
                min_radius=args.block_min_radius,
                max_radius=args.block_max_radius,
                min_fill_ratio=args.block_min_fill,
                min_circularity=args.block_min_circularity,
                min_aspect_ratio=args.block_min_aspect,
            )
            draw_blocks(display_block, blocks)
            status = f"Mode: BLOCK | Blocks: {len(blocks)}"

        # Serial status indicator
        if sender and sender.link:
            status += f" | Serial: {args.serial_port}"
        elif not args.no_serial:
            status += " | Serial: disconnected"
        else:
            status += " | Serial: off"

        # QR / task-code status indicator
        if qr_active:
            status += f" | QR: scanning ({qr_match_count}/3)"
        elif last_task_code.valid and now < qr_send_until:
            remaining = qr_send_until - now
            status += f" | TaskCode: broadcasting ({remaining:.1f}s left)"
        elif last_task_code.valid:
            status += f" | TaskCode: done ({last_task_code.raw})"
        else:
            status += " | QR: off"

        draw_status(display_block, status)
        cv2.imshow(WIN_BLOCK, display_block)

        if task_display:
            disp_frame = task_display.render_frame()
            cv2.imshow(WIN_DISPLAY, disp_frame)

        # ── Periodic send or console log (rate-limited by send_interval) ────
        if now - last_send_time >= args.send_interval:
            if sender:
                # Broadcast task code while within timed window
                if last_task_code.valid and now < qr_send_until:
                    sender.send_task_code(last_task_code)

                if mode == MODE_RING:
                    if rings:
                        sender.send_rings(rings)
                else:
                    if blocks:
                        sender.send_blocks(blocks)
            else:
                # No-serial mode: print live detections to terminal
                if mode == MODE_RING and rings:
                    desc_list = [
                        f"Target #{r.digit_id}(conf={r.confidence:.2f})@({r.cx},{r.cy},r={r.radius})"
                        if r.digit_id in (1, 2, 3) else f"Ring unknown@({r.cx},{r.cy},r={r.radius})"
                        for r in rings
                    ]
                    print(f"[Vision RING] {len(rings)} ring(s): {', '.join(desc_list)}")
                elif mode == MODE_BLOCK and blocks:
                    desc_list = [f"{b.color}@({b.cx},{b.cy},r={b.radius})" for b in blocks]
                    print(f"[Vision BLOCK] {len(blocks)} block(s): {', '.join(desc_list)}")
            last_send_time = now

        # ── Keyboard ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC
            break
        elif key == ord("m"):  # toggle mode manually
            mode = MODE_BLOCK if mode == MODE_RING else MODE_RING
            print(f"[key] Mode -> {'RING' if mode == MODE_RING else 'BLOCK'}")
        elif key == ord("p"):  # print current detections
            if mode == MODE_RING:
                if rings:
                    print(f"[Current Detections: {len(rings)} Ring(s)]")
                    for r in rings:
                        target_str = f"Target #{r.digit_id} (conf={r.confidence:.2f})" if r.digit_id in (1, 2, 3) else "unknown digit"
                        print(f"  ring [{r.color:5s}] {target_str:25s} center=({r.cx:4d},{r.cy:4d}) r={r.radius:3d}")
                else:
                    print("  [Vision RING] No rings detected in current frame.")
            else:
                if blocks:
                    print(f"[Current Detections: {len(blocks)} Block(s)]")
                    for b in blocks:
                        print(f"  block [{b.color:9s}] center=({b.cx:4d},{b.cy:4d}) r={b.radius:3d}")
                else:
                    print("  [Vision BLOCK] No blocks detected in current frame.")
        elif key in (ord("1"), ord("+"), ord("=")):
            if task_display:
                task_display.increment_pick(1)
                print(f"[Key] Pick +1 -> Total: {task_display.pick_count}")
        elif key in (ord("2"), ord("]")):
            if task_display:
                task_display.increment_place(1)
                print(f"[Key] Place +1 -> Total: {task_display.place_count}")
        elif key in (ord("3"), ord("-"), ord("_")):
            if task_display:
                task_display.increment_pick(-1)
                print(f"[Key] Pick -1 -> Total: {task_display.pick_count}")
        elif key in (ord("4"), ord("[")):
            if task_display:
                task_display.increment_place(-1)
                print(f"[Key] Place -1 -> Total: {task_display.place_count}")
        elif key == ord("f"):
            if task_display:
                task_display.fullscreen = not task_display.fullscreen
                prop = cv2.WINDOW_FULLSCREEN if task_display.fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WIN_DISPLAY, cv2.WND_PROP_FULLSCREEN, prop)
                if not task_display.fullscreen:
                    cv2.resizeWindow(WIN_DISPLAY, 1280, 720)
                print(f"[Key] Display Fullscreen -> {task_display.fullscreen}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if sender:
        sender.close()
    if cam_qr and qr_active:
        cam_qr.stop()
    cam_block.stop()
    cv2.destroyAllWindows()
    print("Exited.")
