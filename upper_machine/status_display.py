"""Full-Screen Task Code & Gripper Statistics Display Component (3-Row Layout).

Layout:
  - Row 1: Batch 1 task code 'xxx+xxx' (Large bold font)
  - Row 2: Batch 2 task code 'xxx+xxx' (Large bold font)
  - Row 3: Progress display (Pick & Place statistics, smaller font)
  - Full-screen auto-adaptation: Scales fonts and paddings dynamically to fit any display.
"""

import ctypes
import glob
import os
import sys
import threading
from typing import List, Tuple

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def detect_screen_resolution() -> Tuple[int, int]:
    """Auto-detect current monitor resolution across Linux / X11 / Framebuffer."""
    try:
        x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        d = x11.XOpenDisplay(None)
        if d:
            d_ptr = ctypes.c_void_p(d)
            s = x11.XDefaultScreen(d_ptr)
            w = x11.XDisplayWidth(d_ptr, s)
            h = x11.XDisplayHeight(d_ptr, s)
            x11.XCloseDisplay(d_ptr)
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass

    try:
        fb_path = "/sys/class/graphics/fb0/virtual_size"
        if os.path.exists(fb_path):
            with open(fb_path, "r") as f:
                parts = f.read().strip().split(",")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
    except Exception:
        pass

    return 1920, 1080


def find_chinese_font() -> str | None:
    """Find a system TrueType font supporting Chinese characters."""
    preferred = [
        "/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/wps-fonts/timesbd.ttf",
    ]
    for p in preferred:
        if os.path.exists(p):
            return p

    for p in glob.glob("/usr/share/fonts/**/*.ttf", recursive=True):
        try:
            f = ImageFont.truetype(p, 24)
            img = Image.new("L", (40, 40))
            d = ImageDraw.Draw(img)
            bbox = d.textbbox((0, 0), "抓取", font=f)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                return p
        except Exception:
            continue

    return None


class TaskDisplay:
    """Thread-safe Full-screen Display Component for 12-Digit 4-Group Task Code & Gripper Status."""

    GROUP_LABELS = ["第一批物料颜色", "第一批放置环号", "第二批物料颜色", "第二批放置环号"]

    def __init__(self,
                 width: int | None = None,
                 height: int | None = None,
                 fullscreen: bool = True,
                 show_group_labels: bool = True,
                 window_name: str = "Robot Task & Gripper Monitor",
                 task_code: str | None = None):
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.show_group_labels = show_group_labels

        detected_w, detected_h = detect_screen_resolution()
        self.width = width or detected_w
        self.height = height or detected_h

        # Data states: task_code is None until QR is detected
        self.task_code: str | None = str(task_code).strip() if task_code else None
        self.pick_count: int = 0
        self.place_count: int = 0

        # Colors (RGB format for PIL)
        self.bg_color = (15, 23, 42)          # Deep Navy Slate (#0F172A)
        self.card_bg = (30, 41, 59)           # Card Slate (#1E293B)
        self.card_border = (56, 189, 248)     # Vivid Cyan Border (#38BDF8)
        self.card_border_waiting = (71, 85, 105) # Muted border when waiting
        self.task_color = (56, 189, 248)      # Vivid Cyan Digits
        self.placeholder_color = (100, 116, 139) # Grey for '---'
        self.plus_color = (148, 163, 184)     # Muted Slate Grey for '+'
        self.plus_color_waiting = (71, 85, 105)
        self.label_color = (203, 213, 225)    # Light Slate Text
        self.sub_label_color = (148, 163, 184)# Sub-label Grey
        self.pick_color = (74, 222, 128)      # Emerald Green for pick (#4ADE80)
        self.place_color = (251, 146, 60)     # Amber Orange for place (#FB923C)
        self.divider_color = (51, 65, 85)     # Divider Line
        self.warn_color = (251, 191, 36)      # Amber yellow for waiting status

        self.font_path = find_chinese_font()

    def set_task_code(self, code: str | None):
        """Update the task code string. Pass None or empty string to hide numbers."""
        with self.lock:
            if code is None or not str(code).strip():
                self.task_code = None
            else:
                self.task_code = str(code).strip()

    def set_counts(self, pick: int | None = None, place: int | None = None):
        """Update pick and/or place counts directly."""
        with self.lock:
            if pick is not None:
                self.pick_count = max(0, int(pick))
            if place is not None:
                self.place_count = max(0, int(place))

    def increment_pick(self, delta: int = 1):
        """Increment (or decrement) successful pick count."""
        with self.lock:
            self.pick_count = max(0, self.pick_count + delta)

    def increment_place(self, delta: int = 1):
        """Increment (or decrement) successful place count."""
        with self.lock:
            self.place_count = max(0, self.place_count + delta)

    def _parse_groups(self, task_str: str | None) -> Tuple[List[str], bool]:
        """Parse task code into 4 groups. Returns (groups, is_valid)."""
        if not task_str:
            return ["---", "---", "---", "---"], False

        clean = task_str.strip()
        if "+" in clean:
            parts = [p.strip() for p in clean.split("+") if p.strip()]
            if len(parts) == 4:
                return parts, True
            elif parts:
                return parts, True

        digits = "".join(filter(str.isdigit, clean))
        if len(digits) >= 12:
            return [digits[0:3], digits[3:6], digits[6:9], digits[9:12]], True
        elif len(digits) >= 3:
            return [digits[i:i + 3] for i in range(0, len(digits), 3)], True

        return [clean], True

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if self.font_path:
            return ImageFont.truetype(self.font_path, size)
        return ImageFont.load_default()

    def render_frame(self) -> np.ndarray:
        """Render the 3-row clean UI frame as a BGR numpy array for OpenCV.

        Row 1: Batch 1 task code 'xxx + xxx' (Large bold font, perfectly centered)
        Row 2: Batch 2 task code 'xxx + xxx' (Large bold font, perfectly centered)
        Row 3: Pick & Place statistics '抓取：X   放置：Y' (Smaller font, no extra text)
        """
        with self.lock:
            raw_task = self.task_code
            pick_cnt = self.pick_count
            place_cnt = self.place_count

        W, H = self.width, self.height

        img = Image.new("RGB", (W, H), self.bg_color)
        draw = ImageDraw.Draw(img)

        # ── Parse Task Code into Batch 1 and Batch 2 ────────────────────────
        is_valid = bool(raw_task and str(raw_task).strip())
        if is_valid:
            clean = str(raw_task).strip()
            if "+" in clean:
                parts = [p.strip() for p in clean.split("+") if p.strip()]
                if len(parts) >= 4:
                    r1_1, r1_2 = parts[0], parts[1]
                    r2_1, r2_2 = parts[2], parts[3]
                elif len(parts) >= 2:
                    r1_1, r1_2 = parts[0], parts[1]
                    r2_1, r2_2 = "---", "---"
                else:
                    r1_1, r1_2 = parts[0], "---"
                    r2_1, r2_2 = "---", "---"
            else:
                digits = "".join(filter(str.isdigit, clean))
                if len(digits) >= 12:
                    r1_1, r1_2 = digits[0:3], digits[3:6]
                    r2_1, r2_2 = digits[6:9], digits[9:12]
                else:
                    r1_1, r1_2 = clean, "---"
                    r2_1, r2_2 = "---", "---"
        else:
            r1_1, r1_2 = "---", "---"
            r2_1, r2_2 = "---", "---"

        # ── Layout Geometry ─────────────────────────────────────────────────
        margin_x = int(W * 0.04)
        margin_y = int(H * 0.03)
        gap_y = int(H * 0.025)
        card_w = W - margin_x * 2

        # 3 Rows height allocation:
        # Row 1 (Batch 1): ~39% of available height
        # Row 2 (Batch 2): ~39% of available height
        # Row 3 (Stats): ~22% of available height
        avail_h = H - margin_y * 2 - gap_y * 2
        row_h_1 = int(avail_h * 0.39)
        row_h_2 = int(avail_h * 0.39)
        row_h_3 = avail_h - row_h_1 - row_h_2

        y1 = margin_y
        y2 = y1 + row_h_1 + gap_y
        y3 = y2 + row_h_2 + gap_y

        card_radius = max(6, int(row_h_1 * 0.10))

        # Fonts for Row 1 & Row 2 (Ultra-large font, high visual impact)
        font_size_large = int(row_h_1 * 0.78)
        font_large = self._get_font(font_size_large)
        font_plus = self._get_font(int(font_size_large * 0.85))

        # Height safety guard: glyph height should not exceed 75% of row height
        sample_h = font_large.getbbox("888")[3] - font_large.getbbox("888")[1]
        if sample_h > row_h_1 * 0.75:
            scale_h = (row_h_1 * 0.75) / sample_h
            font_size_large = max(20, int(font_size_large * scale_h))
            font_large = self._get_font(font_size_large)
            font_plus = self._get_font(int(font_size_large * 0.85))

        # Width safety guard: ensure text never exceeds card width (max 88% of card width)
        test_w = (font_large.getbbox("888")[2] - font_large.getbbox("888")[0]) * 2 + (font_plus.getbbox(" + ")[2] - font_plus.getbbox(" + ")[0])
        if test_w > card_w * 0.88:
            scale_w = (card_w * 0.88) / test_w
            font_size_large = max(20, int(font_size_large * scale_w))
            font_large = self._get_font(font_size_large)
            font_plus = self._get_font(int(font_size_large * 0.85))

        card_border_active = self.card_border
        card_border_dim = self.card_border_waiting
        digit_color = self.task_color if is_valid else self.placeholder_color
        plus_color = self.plus_color if is_valid else (71, 85, 105)

        # ── Draw Code Row 1 & Row 2 (Centered 'xxx + xxx' only) ─────────────
        def draw_code_card(y: int, h: int, p1: str, p2: str):
            border_col = card_border_active if is_valid else card_border_dim

            # Card background
            draw.rounded_rectangle(
                [(margin_x, y), (margin_x + card_w, y + h)],
                radius=card_radius,
                fill=self.card_bg,
                outline=border_col,
                width=2
            )

            # Centered Text: p1 + " + " + p2
            cur_font_large = font_large
            cur_font_plus = font_plus

            bbox1 = cur_font_large.getbbox(p1)
            w1 = bbox1[2] - bbox1[0]
            bbox_p = cur_font_plus.getbbox(" + ")
            wp = bbox_p[2] - bbox_p[0]
            bbox2 = cur_font_large.getbbox(p2)
            w2 = bbox2[2] - bbox2[0]

            total_w = w1 + wp + w2
            if total_w > card_w * 0.88:
                scale = (card_w * 0.88) / total_w
                cur_font_large = self._get_font(max(14, int(font_size_large * scale)))
                cur_font_plus = self._get_font(max(12, int(font_size_large * 0.85 * scale)))
                bbox1 = cur_font_large.getbbox(p1)
                w1 = bbox1[2] - bbox1[0]
                bbox_p = cur_font_plus.getbbox(" + ")
                wp = bbox_p[2] - bbox_p[0]
                bbox2 = cur_font_large.getbbox(p2)
                w2 = bbox2[2] - bbox2[0]
                total_w = w1 + wp + w2

            sx = margin_x + (card_w - total_w) // 2

            # Exact vertical centering within card
            ty1 = y + (h - (bbox1[3] - bbox1[1])) // 2 - bbox1[1]
            typ = y + (h - (bbox_p[3] - bbox_p[1])) // 2 - bbox_p[1]
            ty2 = y + (h - (bbox2[3] - bbox2[1])) // 2 - bbox2[1]

            draw.text((sx - bbox1[0], ty1), p1, fill=digit_color, font=cur_font_large)
            draw.text((sx + w1 - bbox_p[0], typ), " + ", fill=plus_color, font=cur_font_plus)
            draw.text((sx + w1 + wp - bbox2[0], ty2), p2, fill=digit_color, font=cur_font_large)

        # Draw Row 1 & Row 2
        draw_code_card(y1, row_h_1, r1_1, r1_2)
        draw_code_card(y2, row_h_2, r2_1, r2_2)

        # ── Row 3: Only '抓取：X          放置：Y' (Smaller Font) ────────────
        draw.rounded_rectangle(
            [(margin_x, y3), (margin_x + card_w, y3 + row_h_3)],
            radius=card_radius,
            fill=self.card_bg,
            outline=card_border_dim,
            width=2
        )

        stat_font_size = int(row_h_3 * 0.42)
        font_stat = self._get_font(stat_font_size)

        segments = [
            ("抓取：", self.label_color),
            (f" {pick_cnt} ", self.pick_color),
            ("          放置：", self.label_color),
            (f" {place_cnt} ", self.place_color),
        ]

        total_stat_w = sum(font_stat.getbbox(t)[2] - font_stat.getbbox(t)[0] for t, _ in segments)
        if total_stat_w > card_w * 0.85:
            scale = (card_w * 0.85) / total_stat_w
            stat_font_size = max(16, int(stat_font_size * scale))
            font_stat = self._get_font(stat_font_size)
            total_stat_w = sum(font_stat.getbbox(t)[2] - font_stat.getbbox(t)[0] for t, _ in segments)

        sx = margin_x + (card_w - total_stat_w) // 2
        sy = y3 + (row_h_3 - stat_font_size) // 2

        cur_x = sx
        for text, col in segments:
            bbox = font_stat.getbbox(text)
            draw.text((cur_x - bbox[0], sy - bbox[1]), text, fill=col, font=font_stat)
            cur_x += bbox[2] - bbox[0]

        rgb_arr = np.array(img)
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    def run(self):
        """Run the display window loop (blocking)."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, self.width, self.height)

        self.running = True
        print(f"[Display] Window active: {self.width}x{self.height} (Fullscreen={self.fullscreen})")

        while self.running:
            frame = self.render_frame()
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)
                if not self.fullscreen:
                    cv2.resizeWindow(self.window_name, self.width, self.height)
            elif key in (ord("1"), ord("+"), ord("=")):
                self.increment_pick(1)
            elif key in (ord("2"), ord("]")):
                self.increment_place(1)
            elif key in (ord("3"), ord("-"), ord("_")):
                self.increment_pick(-1)
            elif key in (ord("4"), ord("[")):
                self.increment_place(-1)
            elif key == ord("t"):
                demo_codes = [
                    "156+123+516+231",
                    "123+231+312+123",
                    "642+135+246+531",
                    None,
                ]
                cur_idx = demo_codes.index(self.task_code) if self.task_code in demo_codes else -1
                self.set_task_code(demo_codes[(cur_idx + 1) % len(demo_codes)])

        self.running = False
        cv2.destroyWindow(self.window_name)
        print("[Display] Window closed.")

    def start(self):
        """Start display loop in a background thread."""
        if self.running:
            return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop display loop."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TaskDisplay Runner")
    parser.add_argument("--width", type=int, default=1024, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    parser.add_argument("--code", type=str, default="156+123+516+231", help="Task code string")
    parser.add_argument("--pick", type=int, default=0, help="Initial pick count")
    parser.add_argument("--place", type=int, default=0, help="Initial place count")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen")
    args = parser.parse_args()

    disp = TaskDisplay(width=args.width, height=args.height, fullscreen=args.fullscreen, task_code=args.code)
    disp.set_counts(pick=args.pick, place=args.place)
    disp.run()

