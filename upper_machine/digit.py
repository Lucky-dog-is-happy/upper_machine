"""Digit (1, 2, 3) classification and ring-digit association module."""

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
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (size - tw) // 2
    ty = (size + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
    return img


def create_digit_templates(size: int = NORM_SIZE) -> dict[int, list[np.ndarray]]:
    """Generate multi-font normalized reference templates for 1, 2, 3.

    Includes both modern sans-serif (Noto/DejaVu) and serif/standard styles
    to accommodate different contest printing fonts.
    """
    templates: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    font_candidates = [
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    for digit in [1, 2, 3]:
        # Style 1: System TrueType font (if available)
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
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cropped = arr[by:by + bh, bx:bx + bw]
                    pad = max(2, int(max(bw, bh) * 0.1))
                    templates[digit].append(cv2.resize(
                        cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0),
                        (size, size),
                    ))
            except Exception:
                pass

        # Style 2: OpenCV standard vector font (serif/standard stroke)
        for font_face in [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]:
            canvas = np.zeros((80, 80), dtype=np.uint8)
            cv2.putText(canvas, str(digit), (20, 60), font_face, 1.8, 255, 4, cv2.LINE_AA)
            cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                cropped = canvas[by:by + bh, bx:bx + bw]
                pad = max(2, int(max(bw, bh) * 0.1))
                templates[digit].append(cv2.resize(
                    cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0),
                    (size, size),
                ))

    return templates


class DigitClassifier:
    """Classifies binary candidate crop into 1, 2, or 3 using template matching and geometric rules."""

    def __init__(self):
        self.templates = create_digit_templates(size=NORM_SIZE)

    def evaluate(self, binary_crop: np.ndarray) -> dict[int, float]:
        """Return confidence scores (0.0 ~ 1.0) for each digit 1, 2, 3."""
        h, w = binary_crop.shape[:2]
        if h < 8 or w < 3:
            return {1: 0.0, 2: 0.0, 3: 0.0}

        aspect = h / float(w)

        # Pad to square and resize
        pad = max(2, int(max(h, w) * 0.1))
        padded = cv2.copyMakeBorder(binary_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        norm = cv2.resize(padded, (NORM_SIZE, NORM_SIZE))

        # 1. Template Matching Correlation
        scores = {}
        for d, tpl_list in self.templates.items():
            best_s = 0.0
            for tpl in tpl_list:
                res = cv2.matchTemplate(norm, tpl, cv2.TM_CCOEFF_NORMED)
                best_s = max(best_s, float(res[0][0]))
            scores[d] = max(0.0, best_s)

        # 2. Geometric & Topological Enhancements
        # Digit 1: Slender vertical stroke
        if aspect >= 1.65:
            scores[1] += 0.35
            scores[2] -= 0.30
            scores[3] -= 0.30
        elif aspect < 1.15:
            scores[1] -= 0.50

        # Digit 2: Solid baseline stroke at the bottom
        bot_density = np.mean(norm[int(NORM_SIZE * 0.80):, :]) / 255.0
        if bot_density > 0.45 and aspect < 1.70:
            scores[2] += 0.25

        # Digit 3: Waist indentation on middle-left
        mid_left_density = np.mean(
            norm[int(NORM_SIZE * 0.35):int(NORM_SIZE * 0.65), :int(NORM_SIZE * 0.30)]
        ) / 255.0
        if mid_left_density < 0.20 and aspect < 1.70:
            scores[3] += 0.25
            scores[1] -= 0.15

        # Normalize scores to [0.0, 1.0]
        return {d: max(0.0, min(1.0, float(s))) for d, s in scores.items()}


def extract_digit_scores_from_ring(frame: np.ndarray,
                                    cx: int, cy: int, radius: int,
                                    classifier: DigitClassifier,
                                    inner_ratio: float = 0.65) -> dict[int, float]:
    """Inspect inner core of the ring to find digits and return scores for 1, 2, 3.

    Only searches within inner_ratio * radius to completely avoid the black ring border.
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

    # Mask out the 4 corners of the bounding box outside the inner circle
    circular_mask = np.zeros(gray.shape, dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(circular_mask, (local_cx, local_cy), inner_r, 255, -1)

    # Dark digits on light floor -> THRESH_BINARY_INV with Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, circular_mask)

    # Clean small noise while preserving strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(15, int(np.pi * (inner_r ** 2) * 0.02))
    max_area = int(np.pi * (inner_r ** 2) * 0.50)

    best_scores = {1: 0.0, 2: 0.0, 3: 0.0}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        cnt_cx = bx + bw / 2.0
        cnt_cy = by + bh / 2.0

        # Digit must be reasonably centered inside the ring
        dist_to_center = np.hypot(cnt_cx - local_cx, cnt_cy - local_cy)
        if dist_to_center > inner_r * 0.70:
            continue

        crop = thresh[by:by + bh, bx:bx + bw]
        scores = classifier.evaluate(crop)

        # Update best scores if this contour provides higher confidence
        if max(scores.values()) > max(best_scores.values()):
            best_scores = scores

    return best_scores


def match_rings_to_digits(ring_scores_list: list[dict[int, float]],
                          min_confidence: float = 0.50) -> list[int]:
    """Globally assign unique digits (1, 2, 3) to up to 3 rings without duplicate conflicts.

    Returns a list of target_id (1, 2, 3, or 5 for unassigned/unconfident) of the same length.
    """
    n_rings = len(ring_scores_list)
    if n_rings == 0:
        return []

    # Candidates for digits
    all_digits = [1, 2, 3]

    # Generate all possible non-repeating assignments of length n_rings
    # E.g. for 3 rings: permutations of [1, 2, 3] (6 permutations)
    import itertools
    best_assignment = [5] * n_rings
    best_total_score = -1.0

    possible_permutations = list(itertools.permutations(all_digits, min(n_rings, 3)))
    if n_rings > 3:
        # If somehow more than 3 rings, remaining get 5
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
                # Mark as 5 (unknown) if confidence below threshold
                valid_perm[ring_idx] = 5

        # Prefer assignment with more valid digits and higher total confidence
        composite_score = current_valid_count * 10.0 + current_score
        if composite_score > best_total_score:
            best_total_score = composite_score
            best_assignment = valid_perm

    return best_assignment
