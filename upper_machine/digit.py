"""Digit (1, 2, 3) classification and ring-digit association module.

Features:
  - Adaptive dual-thresholding with morphological closing (avoids broken strokes).
  - Centroid-constrained connected components clustering to recover fragmented digits.
  - Multi-feature topological classifier:
      • Aspect ratio (distinguishes 1 from 2, 3)
      • Bottom-left baseline density (solid for 2, hollow for 3)
      • Middle-left waist indentation (deep inset for 3)
      • Profile horizontal spans (top, mid, bottom)
      • Multi-font normalized template matching
  - Global mutual exclusion assignment (allocates 1, 2, 3 uniquely across up to 3 rings).
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


def create_digit_templates(size: int = NORM_SIZE) -> dict[int, list[np.ndarray]]:
    """Generate multi-font normalized reference templates for 1, 2, 3.

    Includes modern sans-serif (Noto/DejaVu) and serif/standard styles,
    plus aspect-squashed variants to handle perspective compression.
    """
    templates: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    font_candidates = [
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    for digit in [1, 2, 3]:
        # Style 1: System TrueType font
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
                    pad = max(2, int(max(cbw, cbh) * 0.1))
                    padded = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                    norm = cv2.resize(padded, (size, size))
                    templates[digit].append(norm)

                    # Also add a squashed template (pitch perspective compression)
                    squashed = cv2.resize(cropped, (cbw, int(cbh * 0.80)))
                    s_pad = max(2, int(max(squashed.shape) * 0.1))
                    templates[digit].append(cv2.resize(
                        cv2.copyMakeBorder(squashed, s_pad, s_pad, s_pad, s_pad, cv2.BORDER_CONSTANT, value=0),
                        (size, size),
                    ))
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
                pad = max(2, int(max(cbw, cbh) * 0.1))
                templates[digit].append(cv2.resize(
                    cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0),
                    (size, size),
                ))

    return templates


class DigitClassifier:
    """Classifies binary candidate crop into 1, 2, or 3 using geometric topology + template correlation."""

    def __init__(self):
        self.templates = create_digit_templates(size=NORM_SIZE)

    def evaluate(self, binary_crop: np.ndarray) -> dict[int, float]:
        """Return confidence scores (0.0 ~ 1.0) for each digit 1, 2, 3."""
        h, w = binary_crop.shape[:2]
        if h < 8 or w < 3:
            return {1: 0.0, 2: 0.0, 3: 0.0}

        aspect = h / float(w)

        # 1. Template Matching Correlation
        pad = max(2, int(max(h, w) * 0.1))
        padded = cv2.copyMakeBorder(binary_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        norm = cv2.resize(padded, (NORM_SIZE, NORM_SIZE))

        tpl_scores = {}
        for d, tpl_list in self.templates.items():
            best_s = 0.0
            for tpl in tpl_list:
                res = cv2.matchTemplate(norm, tpl, cv2.TM_CCOEFF_NORMED)
                best_s = max(best_s, float(res[0][0]))
            tpl_scores[d] = max(0.0, best_s)

        # 2. Geometric & Topological Feature Extraction
        # a. Bottom-left density (lower 25% height, left 35% width)
        #    Digit 2 has a solid bottom horizontal baseline -> high density
        #    Digit 3 has a hollow bottom-left space -> low/medium density
        bot_left = binary_crop[int(h * 0.75):, :max(1, int(w * 0.35))]
        bot_left_density = (np.mean(bot_left) / 255.0) if bot_left.size > 0 else 0.0

        # b. Mid-left waist density (height 35% ~ 65%, left 35% width)
        #    Digit 3 has a deep inset/waist hollow on middle-left -> very low density
        mid_left = binary_crop[int(h * 0.35):int(h * 0.65), :max(1, int(w * 0.35))]
        mid_left_density = (np.mean(mid_left) / 255.0) if mid_left.size > 0 else 0.0

        # c. Bottom horizontal stroke span
        bot_strip = binary_crop[int(h * 0.85):, :]
        if bot_strip.size > 0:
            cols_with_pixels = np.where(np.any(bot_strip > 100, axis=0))[0]
            bot_span = (cols_with_pixels[-1] - cols_with_pixels[0] + 1) / float(w) if len(cols_with_pixels) > 0 else 0.0
        else:
            bot_span = 0.0

        # 3. Fuse Template Scores and Topological Heuristics
        final_scores = {1: tpl_scores.get(1, 0.0), 2: tpl_scores.get(2, 0.0), 3: tpl_scores.get(3, 0.0)}

        # ── Rules for Digit 1 ──
        if aspect >= 1.60:
            final_scores[1] += 0.35
            final_scores[2] -= 0.25
            final_scores[3] -= 0.25
        elif aspect < 1.10:
            final_scores[1] -= 0.40

        # ── Rules for Digit 2 (Solid horizontal base line) ──
        if bot_left_density >= 0.50 and bot_span >= 0.70:
            final_scores[2] += 0.40
            final_scores[3] -= 0.30
        elif bot_left_density >= 0.35 and bot_span >= 0.60:
            final_scores[2] += 0.20

        # ── Rules for Digit 3 (Hollow mid-left waist, no solid bottom-left line) ──
        if mid_left_density <= 0.15 and bot_left_density < 0.65 and aspect < 1.65:
            final_scores[3] += 0.35
            final_scores[1] -= 0.15
        elif mid_left_density <= 0.25 and aspect < 1.65:
            final_scores[3] += 0.20

        # Clamp all scores into [0.0, 1.0]
        return {d: max(0.0, min(1.0, float(s))) for d, s in final_scores.items()}


def extract_digit_scores_from_ring(frame: np.ndarray,
                                    cx: int, cy: int, radius: int,
                                    classifier: DigitClassifier,
                                    inner_ratio: float = 0.65) -> dict[int, float]:
    """Inspect inner core of the ring to find digits and return scores for 1, 2, 3.

    Uses adaptive local binarization, morphological closing to bridge gaps,
    and clusters central connected components to recover complete characters.
    """
    h_img, w_img = frame.shape[:2]
    inner_r = max(8, int(radius * inner_ratio))

    x1, y1 = max(0, cx - inner_r), max(0, cy - inner_r)
    x2, y2 = min(w_img, cx + inner_r), min(h_img, cy + inner_r)
    if x2 <= x1 or y2 <= y1:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Circular mask strictly inside inner_r to avoid outer black ring interference
    circular_mask = np.zeros(gray.shape, dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(circular_mask, (local_cx, local_cy), inner_r, 255, -1)

    # Combine Otsu + Adaptive Gaussian Thresholding for high-contrast foreground
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 8,
    )
    # Bitwise OR captures thin strokes while rejecting noise via circular mask
    thresh = cv2.bitwise_or(thresh_otsu, thresh_adapt)
    thresh = cv2.bitwise_and(thresh, circular_mask)

    # Use MORPH_CLOSE (dilation then erosion) to bridge thin gaps/creases
    # (Never use MORPH_OPEN here which erodes and breaks strokes 2 and 3)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Find connected components in the inner core
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    min_area = max(10, int(np.pi * (inner_r ** 2) * 0.015))
    max_area = int(np.pi * (inner_r ** 2) * 0.55)

    # Collect valid components reasonably close to the center
    central_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cnt_cx = bx + bw / 2.0
        cnt_cy = by + bh / 2.0
        if np.hypot(cnt_cx - local_cx, cnt_cy - local_cy) <= inner_r * 0.65:
            central_boxes.append((bx, by, bw, bh, area))

    if not central_boxes:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    # Cluster components: merge nearby boxes in case strokes are fractured
    # Merge all qualifying boxes into a unified bounding box
    min_x = min(b[0] for b in central_boxes)
    min_y = min(b[1] for b in central_boxes)
    max_x = max(b[0] + b[2] for b in central_boxes)
    max_y = max(b[1] + b[3] for b in central_boxes)

    merged_w = max_x - min_x
    merged_h = max_y - min_y

    # Evaluate the merged character region
    merged_crop = thresh[min_y:max_y, min_x:max_x]
    scores_merged = classifier.evaluate(merged_crop)

    # Also evaluate the single largest contour individually (if clean)
    largest_cnt = max(central_boxes, key=lambda b: b[4])
    lbx, lby, lbw, lbh, _ = largest_cnt
    single_crop = thresh[lby:lby + lbh, lbx:lbx + lbw]
    scores_single = classifier.evaluate(single_crop)

    # Return whichever gives higher peak confidence
    if max(scores_merged.values()) >= max(scores_single.values()):
        return scores_merged
    return scores_single


def match_rings_to_digits(ring_scores_list: list[dict[int, float]],
                          min_confidence: float = 0.35) -> list[int]:
    """Globally assign unique digits (1, 2, 3) to up to 3 rings without duplicate conflicts.

    Returns a list of target_id (1, 2, 3, or 5 for unassigned/unconfident) of the same length.
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

    return best_assignment
