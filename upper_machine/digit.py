"""Digit (1, 2, 3) classification and ring-digit association module.

Features:
  - Concentric target adaptation: extracts core digit region (inner_ratio=0.42)
    to completely avoid the inner concentric ring border.
  - Automatic mirror invariance: evaluates characters in both standard and horizontally
    flipped orientations to seamlessly support mirrored cameras or inverted target papers.
  - Multi-feature topological classifier:
      • Aspect ratio: distinguishes slender 1 (aspect >= 1.75)
      • Bottom baseline solid density: digit 2 has a solid, continuous horizontal base
      • Middle-left waist indentation: digit 3 has deep inward hollows on its middle-left
      • Multi-font normalized template correlation
  - Global mutual exclusion assignment: uniquely maps up to 3 detected rings to digits 1, 2, 3.
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
    """Generate normalized multi-font templates for 1, 2, 3."""
    templates: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

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
                    pad = max(2, int(max(cbw, cbh) * 0.1))
                    padded = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                    templates[digit].append(cv2.resize(padded, (size, size)))
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
    """Classifies binary candidate crop into 1, 2, or 3 with automatic mirror-invariance."""

    def __init__(self):
        self.templates = create_digit_templates(size=NORM_SIZE)

    def _evaluate_single_orientation(self, binary_crop: np.ndarray) -> dict[int, float]:
        """Evaluate a binary crop in one orientation."""
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

        # 2. Geometric & Topological Metrics
        bot_left = binary_crop[int(h * 0.75):, :max(1, int(w * 0.35))]
        bot_left_density = (np.mean(bot_left) / 255.0) if bot_left.size > 0 else 0.0

        mid_left = binary_crop[int(h * 0.35):int(h * 0.65), :max(1, int(w * 0.35))]
        mid_left_density = (np.mean(mid_left) / 255.0) if mid_left.size > 0 else 0.0

        bot_strip = binary_crop[int(h * 0.85):, :]
        if bot_strip.size > 0:
            cols = np.where(np.any(bot_strip > 100, axis=0))[0]
            bot_span = (cols[-1] - cols[0] + 1) / float(w) if len(cols) > 0 else 0.0
        else:
            bot_span = 0.0

        # Base scores from template matching
        scores = {d: tpl_scores.get(d, 0.0) * 0.40 for d in [1, 2, 3]}

        # ── Rules for Digit 1 ──
        if aspect >= 1.75:
            scores[1] += 0.60
            scores[2] -= 0.30
            scores[3] -= 0.30
        elif aspect >= 1.35 and mid_left_density < 0.12 and bot_span < 0.65:
            scores[1] += 0.35

        # ── Rules for Digit 2 (Solid horizontal base line) ──
        if bot_left_density >= 0.45 and bot_span >= 0.70 and aspect < 1.75:
            scores[2] += 0.60
            scores[1] -= 0.20
            scores[3] -= 0.35
        elif bot_span >= 0.65 and aspect < 1.75:
            scores[2] += 0.40

        # ── Rules for Digit 3 (Hollow mid-left waist, non-solid left bottom) ──
        if mid_left_density <= 0.20 and bot_left_density < 0.45 and aspect < 1.75:
            scores[3] += 0.60
            scores[1] -= 0.20
            scores[2] -= 0.30
        elif mid_left_density <= 0.28 and aspect < 1.75:
            scores[3] += 0.35

        return {d: max(0.0, min(1.0, float(s))) for d, s in scores.items()}

    def evaluate(self, binary_crop: np.ndarray) -> dict[int, float]:
        """Return confidence scores (0.0 ~ 1.0) with automatic mirror-invariance."""
        # Evaluate standard orientation
        scores_orig = self._evaluate_single_orientation(binary_crop)

        # Evaluate horizontally flipped orientation (handles mirrored camera feeds)
        flipped_crop = cv2.flip(binary_crop, 1)
        scores_flipped = self._evaluate_single_orientation(flipped_crop)

        # Merge by taking the best score for each digit
        return {
            d: max(scores_orig.get(d, 0.0), scores_flipped.get(d, 0.0))
            for d in [1, 2, 3]
        }


def extract_digit_scores_from_ring(frame: np.ndarray,
                                    cx: int, cy: int, radius: int,
                                    classifier: DigitClassifier,
                                    inner_ratio: float = 0.42) -> dict[int, float]:
    """Inspect inner core of the ring to find digits and return scores for 1, 2, 3.

    Default inner_ratio=0.42 strictly confines the search to the central white region,
    completely avoiding the inner concentric rings of the contest target.
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

    # Circular mask strictly inside inner_r
    circular_mask = np.zeros(gray.shape, dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(circular_mask, (local_cx, local_cy), inner_r, 255, -1)

    # Inverse Otsu for dark digit on light background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, circular_mask)

    # Light morphological close to bridge fine stroke gaps without bridging to borders
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    min_area = max(15, int(np.pi * (inner_r ** 2) * 0.02))
    max_area = int(np.pi * (inner_r ** 2) * 0.85)

    # Select candidate contours that are centrally located inside the circle
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cnt_cx = bx + bw / 2.0
        cnt_cy = by + bh / 2.0
        dist_to_center = np.hypot(cnt_cx - local_cx, cnt_cy - local_cy)
        if dist_to_center <= inner_r * 0.65:
            candidates.append((cnt, area, dist_to_center, bx, by, bw, bh))

    if not candidates:
        return {1: 0.0, 2: 0.0, 3: 0.0}

    # If multiple fragments exist near the center, also consider merged bounding box
    best_scores = {1: 0.0, 2: 0.0, 3: 0.0}

    # Evaluate the largest clean central contour
    largest = max(candidates, key=lambda c: c[1])
    _, _, _, lbx, lby, lbw, lbh = largest
    crop_largest = thresh[lby:lby + lbh, lbx:lbx + lbw]
    best_scores = classifier.evaluate(crop_largest)

    # If multiple central pieces exist (e.g. broken stroke), evaluate merged bbox
    if len(candidates) > 1:
        min_bx = min(c[3] for c in candidates)
        min_by = min(c[4] for c in candidates)
        max_bx = max(c[3] + c[5] for c in candidates)
        max_by = max(c[4] + c[6] for c in candidates)
        crop_merged = thresh[min_by:max_by, min_bx:max_bx]
        scores_merged = classifier.evaluate(crop_merged)
        if max(scores_merged.values()) > max(best_scores.values()):
            best_scores = scores_merged

    return best_scores


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
