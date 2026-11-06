#!/usr/bin/env python3
"""WeChat QR code scanner using OpenCV and camera feed.

Features:
  - Auto-download of WeChat QR models (--download-models)
  - Mouse click to save frames
  - Chinese text rendering via PIL
  - Timestamped save directories
  - Toast notifications on save
  - Configurable resolution (--width, --height)
  - Camera tuning (exposure, focus, brightness)
  - Fallback camera retry
  - Optional FPS counter (--show-fps)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def download_models(models_dir: Path) -> None:
    import requests

    base_url = "https://github.com/WeChatCV/opencv_3rdparty/raw/wechat_qrcode"
    files = {
        "detect.prototxt": f"{base_url}/detect.prototxt",
        "detect.caffemodel": f"{base_url}/detect.caffemodel",
        "sr.prototxt": f"{base_url}/sr.prototxt",
        "sr.caffemodel": f"{base_url}/sr.caffemodel",
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in files.items():
        dest = models_dir / name
        if not dest.exists():
            print(f"Downloading {name}...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"Saved {dest}")


def get_chinese_font(size: int = 32) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_chinese_text(
    img_bgr: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def adjust_camera(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -3.0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    cap.set(cv2.CAP_PROP_GAIN, 0.5)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 100)


class SaveState:
    def __init__(self) -> None:
        self.save_dir_base = Path("images_videos")
        self.save_dir_base.mkdir(exist_ok=True)
        self.counter = 0
        self.toast_text = ""
        self.toast_until = 0.0
        self.current_frame: np.ndarray | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="WeChat QR code scanner")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--models",
        type=Path,
        default=Path.home() / ".opencv" / "wechat_qrcode",
        help="Directory for WeChat QR model files",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download model files if missing",
    )
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--show-fps", action="store_true", help="Display FPS overlay")
    args = parser.parse_args()

    models_dir = args.models
    model_names = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    model_paths = [models_dir / n for n in model_names]

    cv_version = tuple(int(x) for x in cv2.__version__.split(".")[:2])
    use_builtin = cv_version >= (5, 0)

    if not use_builtin:
        if args.download_models:
            download_models(models_dir)

        missing = [p for p in model_paths if not p.exists()]
        if missing:
            print(
                "Missing model files:\n  " + "\n  ".join(str(p) for p in missing),
                file=sys.stderr,
            )
            print("Run with --download-models or place them manually.", file=sys.stderr)
            sys.exit(1)

    if use_builtin:
        detector = cv2.wechat_qrcode_WeChatQRCode()
    else:
        detector = cv2.wechat_qrcode_WeChatQRCode(
            str(model_paths[0]), str(model_paths[1]),
            str(model_paths[2]), str(model_paths[3]),
        )

    #cap = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}, trying index 0...", file=sys.stderr)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open any camera", file=sys.stderr)
            sys.exit(1)

    adjust_camera(cap)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    font = get_chinese_font()
    save_state = SaveState()

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and save_state.current_frame is not None:
            save_state.counter += 1
            subdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = save_state.save_dir_base / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.png"
            cv2.imwrite(str(path), save_state.current_frame)
            print(f"Saved {path}")
            save_state.toast_text = f"Saved #{save_state.counter}"
            save_state.toast_until = time.time() + 1.5

    window_name = "WeChat QR Scanner (Left-click save, ESC quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Left-click to save frame, ESC to quit.")
    fps_history: list[float] = []

    while True:
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame", file=sys.stderr)
            break

        save_state.current_frame = frame.copy()
        display = frame.copy()

        results, points = detector.detectAndDecode(frame)

        if results:
            for text, pts in zip(results, points):
                pts = pts.astype(int).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                cx, cy = pts[:, 0, :].mean(axis=0).astype(int)
                display = draw_chinese_text(
                    display, f"QR: {text}", (cx, cy - 10), font, (0, 255, 0),
                )
                print(f"[QR] {text}")

        if save_state.toast_text and time.time() < save_state.toast_until:
            display = draw_chinese_text(
                display, save_state.toast_text, (20, 50), font, (0, 255, 0),
            )

        if args.show_fps:
            fps_history.append(loop_start)
            cutoff = loop_start - 2.0
            fps_history = [t for t in fps_history if t > cutoff]
            fps_val = len(fps_history) / 2.0 if fps_history else 0.0
            cv2.putText(
                display, f"FPS: {fps_val:.1f}",
                (display.shape[1] - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

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
