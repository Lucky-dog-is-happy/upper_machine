"""Digit (1, 2, 3) classification and ring-digit association module.

Features:
  - Concentric target adaptation: extracts core digit region (inner_ratio=0.42)
    to completely avoid the inner concentric ring border.
  - Multi-orientation dual evaluation (0 deg & 180 deg) to support both forward
    and opposite-side robot viewing angles, plus horizontal mirroring.
  - Robust topological separation between 2 and 3:
      • Lower-mid (Y 65%~82%) horizontal centroid:
          - Digit 2 is heavily on the left (X_cg <= 0.40) due to diagonal stroke.
          - Digit 3 is heavily on the right (X_cg >= 0.60) due to lower curve hollow.
      • Digit 1: slender vertical stroke (aspect >= 1.75).
  - Multi-font normalized template matching fused with topological scores.
  - Global mutual exclusion assignment: uniquely maps detected rings to digits 1, 2, 3.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NORM_SIZE = 32


@dataclass
class DigitMatch:
    digit: int           # 1, 2, 3
    confidence: float   # 0.0 ~ 1.0
    cx: int
    cy: int
    w: int
    h: int


def _render_digit_with_cv2(digit: int, size: int = 80) -> np.ndarray:
    """Fallback generator for reference digit using OpenCV Hershey fonts."""
    img = np.zeros((size, size), dtype=np.uint8)
    text = str(digit)
    font_scale = 2.0
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (size - tw) // 2
    ty = (size + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
    return img


# ── Golden exemplar templates extracted from high-fidelity contest images ──────────
# Standard sans-serif contest font (Arial/Helvetica style) 32x32 packed into 128 bytes each
GOLDEN_TEMPLATES_HEX = {
    1: "000000000000000000000000000018000000f8000003f8000007b800000e3800000038000000380000003800000038000000380000003800000038000000380000003800000038000000380000003800000038000000380000003800000038000000380000003800000038000000380000003800000000000000000000000000",
    2: "00000000000000000000000000000000001ff00000703c0000601e0000400e0000000e0000000e0000000e0000000e0000001e0000001e0000003c00000078000000f0000001f0000003c000000f8000001f0000003e0000003c0000007800000070000000700000007fff00007fff00007fff00000000000000000000000000",
    3: "00000000000000000000000000030000001ff000007ffc0000603c0000003c0000001c0000001c0000003c0000003800000038000000f0000007e000001ff000001ff80000003e0000001e0000000e0000000e0000000e0000000e0000000e0000001e0000003c000073f800007ff000007fc000000000000000000000000000",
}


def make_square_aspect_preserved(crop: np.ndarray, target_size: int = NORM_SIZE, max_inner: int = 26) -> np.ndarray:
    """Resize binary crop while strictly preserving aspect ratio, preventing slender digits from squashing."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    scale = max_inner / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - nw) // 2
    y_offset = (target_size - nh) // 2
    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
    return canvas


def create_digit_templates(size: int = NORM_SIZE) -> dict[int, list[np.ndarray]]:
    """Generate normalized multi-font templates for 1, 2, 3 with aspect ratio preservation."""
    templates: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    # Style 0: High-fidelity Golden Exemplar from actual contest targets (highest priority)
    for digit in [1, 2, 3]:
        hex_data = GOLDEN_TEMPLATES_HEX.get(digit)
        if hex_data:
            raw_bytes = bytes.fromhex(hex_data)
            unpacked = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).reshape((32, 32))
            golden_img = (unpacked * 255).astype(np.uint8)
            templates[digit].append(golden_img)

    font_candidates = [
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    for digit in [1, 2, 3]:
        # Style 1: System font
        if font_path:
            try:
                font = ImageFont.truetype(font_path, 52)
                img = Image.new("L", (80, 80), 0)
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), str(digit), font=font)
                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((80 - bw) // 2 - bbox[0], (80 - bh) // 2 - bbox[1]), str(digit), fill=255, font=font)
                arr = np.array(img)
                cnts, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    bx, by, cbw, cbh = cv2.boundingRect(c)
                    cropped = arr[by:by + cbh, bx:bx + cbw]
                    templates[digit].append(make_square_aspect_preserved(cropped, target_size=size, max_inner=26))
            except Exception:
                pass

        # Style 2: OpenCV standard vector font
        for font_face in [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]:
            canvas = np.zeros((80, 80), dtype=np.uint8)
            cv2.putText(canvas, str(digit), (20, 60), font_face, 1.8, 255, 4, cv2.LINE_AA)
            cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                bx, by, cbw, cbh = cv2.boundingRect(c)
                cropped = canvas[by:by + cbh, bx:bx + cbw]
                templates[digit].append(make_square_aspect_preserved(cropped, target_size=size, max_inner=26))

    return templates


class DigitClassifier:
    """Classifies binary candidate crop into 1, 2, or 3 using geometric topology + template correlation."""

    def __init__(self):
        self.templates = create_digit_templates(size=NORM_SIZE)

    def _evaluate_single_view(self, t: np.ndarray) -> dict[int, float]:
        """Evaluate a binary crop in a specific orientation using multi-font normalized template matching and soft priors."""
        th, tw = t.shape[:2]
        if th < 8 or tw < 3:
            return {1: 0.0, 2: 0.0, 3: 0.0}

        aspect = th / float(tw)

        # 1. Multi-font aspect-preserved template matching correlation
        norm = make_square_aspect_preserved(t, target_size=NORM_SIZE, max_inner=26)

        tpl_s = {}
        for d in [1, 2, 3]:
            s = 0.0
            for tpl in self.templates[d]:
                res = cv2.matchTemplate(norm, tpl, cv2.TM_CCOEFF_NORMED)
                s = max(s, float(res[0][0]))
            tpl_s[d] = max(0.0, s)

        # 2. Strict Aspect Ratio separation between 1 and (2, 3)
        # Digits 2 and 3 have aspect ratio around 1.35 ~ 1.75.
        # Only truly slender strokes (aspect >= 1.95) should be favored as digit 1.
        s_final = dict(tpl_s)
        if aspect >= 1.95:
            s_final[1] *= 1.25
            s_final[2] *= 0.70
            s_final[3] *= 0.50
        elif aspect <= 1.35:
            s_final[1] *= 0.40

        # 3. Mid-Waist Beam Span Discrimination (Y 40% ~ 60%)
        # Digit 3 has a prominent horizontal middle beam spanning >= 48% of total width.
        # Digit 1 is a uniform vertical column strictly < 35% with no cross beam.
        # 3. Mid-Waist Beam Span Discrimination (Y 40% ~ 60%)
        # Digit 3 has a prominent horizontal middle beam spanning >= 46% of total width.
        # Digit 1 is a uniform vertical column strictly < 38% with no cross beam.
        mid_strip = t[int(th * 0.40):int(th * 0.60), :]
        mid_spans = []
        for row in mid_strip:
            idx = np.where(row > 100)[0]
            if len(idx) > 0:
                mid_spans.append((idx[-1] - idx[0] + 1) / float(tw))
        max_mid_span = max(mid_spans) if mid_spans else 0.0

        if max_mid_span >= 0.46:
            # Prominent horizontal middle beam: definitively NOT 1, cut 1 off
            s_final[1] *= 0.20
        elif max_mid_span < 0.38 and aspect >= 1.60:
            # Slender uniform column without middle beam: definitively NOT 3, cut 3 off
            s_final[3] *= 0.20

        # 4. Double Cavity Topology for Digit 3 vs Single Base for Digit 2
        # Digit 3 has two deep indents on the left side (upper and lower openings).
        # Digit 2 has a flat solid left-extended base (lower indent is negative or zero).
        norm_h, norm_w = norm.shape[:2]
        left_xs = []
        for y_idx in range(norm_h):
            cols = np.where(norm[y_idx] > 100)[0]
            left_xs.append(cols[0] / float(norm_w) if len(cols) > 0 else 1.0)

        upper_strip = [left_xs[y] for y in range(int(norm_h * 0.22), int(norm_h * 0.42))]
        mid_strip_l = [left_xs[y] for y in range(int(norm_h * 0.44), int(norm_h * 0.58))]
        lower_strip = [left_xs[y] for y in range(int(norm_h * 0.62), int(norm_h * 0.80))]
        if upper_strip and mid_strip_l and lower_strip:
            min_mid = min(mid_strip_l)
            up_indent = max(upper_strip) - min_mid
            low_indent = max(lower_strip) - min_mid
            if up_indent >= 0.12 and low_indent >= 0.12:
                # Confirmed double cavity: definitively 3, boost 3, suppress 2 & 1
                s_final[3] *= 1.20
                s_final[2] *= 0.70
                s_final[1] *= 0.50
            elif low_indent <= 0.02 and up_indent >= 0.10:
                # Solid bottom-left base with upper curve: definitively 2
                s_final[2] *= 1.15
                s_final[3] *= 0.70

        # 5. Lower-mid centroid discrimination (Y 65% ~ 82% horizontal centroid X)
        # Digit 2 has diagonal stroke leaning left (cg_x <= 0.38)
        # Digit 3 has lower circle curve leaning right (cg_x >= 0.60)
        lower_mid = t[int(th * 0.65):int(th * 0.82), :]
        ys, xs = np.where(lower_mid > 100)
        cg_x = np.mean(xs) / float(tw) if len(xs) > 0 else 0.50

        if aspect < 1.85:
            if cg_x <= 0.38:
                s_final[2] *= 1.10
                s_final[3] *= 0.85
            elif cg_x >= 0.60:
                s_final[3] *= 1.10
                s_final[2] *= 0.85

        return {d: max(0.0, min(1.0, float(s))) for d, s in s_final.items()}

    def evaluate(self, binary_crop: np.ndarray) -> dict[int, float]:
        """Return confidence scores (0.0 ~ 1.0) by selecting the most confident view orientation.

        Supports full 360-degree rotation invariance (0 deg, 90 deg CW, 180 deg, 270 deg CCW, and mirrored).
        """
        h, w = binary_crop.shape[:2]
        if h < 8 or w < 3:
            return {1: 0.0, 2: 0.0, 3: 0.0}

        orientations = [
            binary_crop,
            cv2.rotate(binary_crop, cv2.ROTATE_180),
            cv2.rotate(binary_crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(binary_crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.flip(binary_crop, 1),
        ]

        best_view_scores = {1: 0.0, 2: 0.0, 3: 0.0}
        best_peak = -1.0

        for view in orientations:
            s = self._evaluate_single_view(view)
            sorted_s = sorted(s.values(), reverse=True)
            peak = sorted_s[0]
            if peak > best_peak:
                best_peak = peak
                best_view_scores = s

        return best_view_scores


def deskew_and_crop(thresh: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Deskew angled character using minimum area bounding rectangle and return upright crop."""
    h_roi, w_roi = thresh.shape[:2]
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect

    # Digits 1, 2, 3 are taller than wide. Ensure the long axis is vertical.
    if w < h:
        deskew_angle = angle
    else:
        deskew_angle = angle + 90

    M = cv2.getRotationMatrix2D((cx, cy), deskew_angle, 1.0)
    rotated = cv2.warpAffine(
        thresh, M, (w_roi, h_roi),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    cnts, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        return thresh[by:by + bh, bx:bx + bw]

    # Find the contour closest to the original center
    best_c = min(
        cnts,
        key=lambda c: np.hypot(
            cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2.0 - cx,
            cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2.0 - cy
        )
    )
    bx, by, bw, bh = cv2.boundingRect(best_c)
    return rotated[by:by + bh, bx:bx + bw]


def extract_digit_scores_from_ring(frame: np.ndarray,
                                    cx: int, cy: int, radius: int,
                                    classifier: DigitClassifier,
                                    inner_ratio: float = 0.42) -> dict[int, float]:
    """Inspect inner core of the ring to find digits and return scores for 1, 2, 3.

    Default inner_ratio=0.42 strictly confines the search to the central white region,
    completely avoiding the inner concentric rings of the contest target.
    Automatically deskews angled digits before classification.
    """
    h_img, w_img = frame.shape[:2]
    inner_r = max(8, int(radius * inner_ratio))

    x1, y1 = max(0, cx - inner_r), max(0, cy - inner_r)
    x2, y2 = min(w_img, cx + inner_r), min(h_img, cy + inner_r)
    if x2 <= x1 or y2 <= y1:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    local_cx = cx - x1
    local_cy = cy - y1

    # Circular mask strictly inside inner_r
    circular_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(circular_mask, (local_cx, local_cy), inner_r, 255, -1)

    inner_pixels = gray[circular_mask > 0]
    if len(inner_pixels) == 0:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    # 1. 核心灰度动态范围与对比度门限校验
    # 抬高摄像头后距离较远，细笔画受亚像素混叠影响 rng 会自然收缩至 40~55，结合标准差自适应放宽
    gray_rng = int(np.ptp(inner_pixels))
    gray_std = float(np.std(inner_pixels))
    if gray_rng < 38 or (gray_rng < 52 and gray_std < 11.0) or gray_std < 9.0:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Inverse Otsu for dark digit on light background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, circular_mask)

    # 2. 前景黑色像素占空比校验 (数字笔画在内圆占比通常在 4% ~ 42%，排除极碎杂点与大片阴影/环内壁)
    fg_count = cv2.countNonZero(thresh)
    circle_area = cv2.countNonZero(circular_mask)
    fg_ratio = fg_count / max(1, circle_area)
    if fg_ratio < 0.04 or fg_ratio > 0.42:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    # Light morphological close to bridge fine stroke gaps without bridging to borders
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    if inner_r <= 28:
        # 相机抬高小半径场景追加水平定向闭运算，保护数字 3 中腰细横梁连通
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_horiz, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    min_area = max(8, int(np.pi * (inner_r ** 2) * 0.025))
    max_area = int(np.pi * (inner_r ** 2) * 0.45)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # 使用图像矩 (Image Moments) 计算像素物理质心，杜绝斜角/楔形阴影利用外接矩形中心逃逸
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cnt_cx = M["m10"] / M["m00"]
            cnt_cy = M["m01"] / M["m00"]
        else:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cnt_cx = bx + bw / 2.0
            cnt_cy = by + bh / 2.0

        dist_to_center = np.hypot(cnt_cx - local_cx, cnt_cy - local_cy)
        # 真实数字质心严格位于内圆核心区域内，小尺寸下预留离散化容差
        if dist_to_center > max(6.0, inner_r * 0.50):
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        # 3. 直立宽高比校验: 数字 1/2/3 均为竖向直立字符 (H/W >= 1.10)，排除横向扁平噪点块
        aspect = bh / max(1.0, float(bw))
        if aspect < 1.10:
            continue

        # 4. 实心度/凸包镂空校验: 数字 1/2/3 存在笔画弯折和凹陷镂空，solidity 必须在 0.20 ~ 0.70 之间，排除实心团块
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)
        if solidity > 0.70 or solidity < 0.20:
            continue

        candidates.append((cnt, area, dist_to_center, bx, by, bw, bh))

    if not candidates:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    # Evaluate the largest clean central contour with automatic deskewing
    largest = max(candidates, key=lambda c: c[1])
    cnt_largest = largest[0]
    crop_largest = deskew_and_crop(thresh, cnt_largest)

    # 5. 笔画均匀度校验: 数字 1 是一根均匀立柱，顶部与底部笔画宽度比绝不超 2.0 (排除斜三角形/楔形阴影)
    hc, wc = crop_largest.shape[:2]
    top_w = np.mean([np.count_nonzero(row) for row in crop_largest[:max(1, hc // 3)] if np.count_nonzero(row) > 0]) or 0.0
    bot_w = np.mean([np.count_nonzero(row) for row in crop_largest[2 * hc // 3:] if np.count_nonzero(row) > 0]) or 0.0
    taper_ratio = max(top_w, bot_w) / max(0.1, min(top_w, bot_w)) if (top_w > 0 and bot_w > 0) else 1.0

    best_scores = classifier.evaluate(crop_largest)
    if taper_ratio > 1.85:
        # 明显楔形梯形阴影，坚决惩罚 1
        best_scores[1] *= 0.15

    # If multiple fragments exist, evaluate deskewed merged cluster
    if len(candidates) > 1:
        merged_pts = np.vstack([c[0] for c in candidates])
        crop_merged = deskew_and_crop(thresh, merged_pts)
        scores_merged = classifier.evaluate(crop_merged)
        if max(scores_merged.values()) > max(best_scores.values()):
            best_scores = scores_merged

    return best_scores


def match_rings_to_digits(ring_scores_list: list[dict[int, float]],
                          min_confidence: float = 0.55,
                          min_margin: float = 0.06) -> list[int]:
    """Globally assign unique digits (1, 2, 3) to up to 3 rings without duplicate conflicts.

    Includes deadband protection: if top score does not beat 2nd place by at least
    min_margin, the candidate is safely demoted to 5 (unknown) to prevent false positives.
    """
    n_rings = len(ring_scores_list)
    if n_rings == 0:
        return []

    all_digits = [1, 2, 3]

    import itertools
    best_assignment = [5] * n_rings
    best_total_score = -1.0

    possible_permutations = list(itertools.permutations(all_digits, min(n_rings, 3)))
    if n_rings > 3:
        base_perms = possible_permutations
        possible_permutations = []
        for p in base_perms:
            possible_permutations.append(p + (5,) * (n_rings - 3))

    for perm in possible_permutations:
        current_score = 0.0
        current_valid_count = 0
        valid_perm = list(perm)

        for ring_idx, digit in enumerate(valid_perm):
            score = ring_scores_list[ring_idx].get(digit, 0.0)
            if score >= min_confidence:
                current_score += score
                current_valid_count += 1
            else:
                valid_perm[ring_idx] = 5

        composite_score = current_valid_count * 10.0 + current_score
        if composite_score > best_total_score:
            best_total_score = composite_score
            best_assignment = valid_perm

    # Deadband protection: safely demote borderline / ambiguous candidates to 5 (unknown)
    # Strictly enforces that target score must be >= min_confidence, and must beat 2nd place by min_margin
    safe_assignment = list(best_assignment)
    for ring_idx, digit in enumerate(safe_assignment):
        if digit in (1, 2, 3):
            scores = ring_scores_list[ring_idx]
            target_score = scores.get(digit, 0.0)
            sorted_vals = sorted(scores.values(), reverse=True)
            margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]
            if target_score < min_confidence or margin < min_margin:
                safe_assignment[ring_idx] = 5

    return safe_assignment
