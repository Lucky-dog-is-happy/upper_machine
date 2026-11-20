"""Color configuration, HSV thresholds, and data structures."""

from dataclasses import dataclass, field

import numpy as np


# ── Color ID mapping ──────────────────────────────────────────────────────────

COLOR_MAP = {
    1: "red",
    2: "yellow",
    3: "blue",
    4: "green",
    5: "black",
    6: "sky-blue",
}

COLOR_NAME_TO_ID = {
    "red": 1, "yellow": 2, "blue": 3, "green": 4, "black": 5, "sky-blue": 6,
}


# ── Block HSV thresholds ─────────────────────────────────────────────────────

COLOR_HSV = {
    # 红色：两段式覆盖，H1 放宽到 13 容忍暖光，S/V 下限降至 55/50 解决反光穿孔与阴影暗角
    "red": {
        "lower": np.array([0, 55, 50]),
        "upper": np.array([13, 255, 255]),
        "bgr": (0, 0, 255),
    },
    "red2": {
        "lower": np.array([165, 55, 50]),
        "upper": np.array([180, 255, 255]),
        "bgr": (0, 0, 255),
    },
    # 黄色：H 上限收至 38，与绿色 46 拉开 8 个单位的缓冲，杜绝互踩
    "yellow": {
        "lower": np.array([17, 75, 70]),
        "upper": np.array([38, 255, 255]),
        "bgr": (0, 255, 255),
    },
    # 绿色：H 下限从 46 起步避开黄光，上限 78 避开青蓝，V 下限放宽至 35 彻底解决墨绿吃光
    "green": {
        "lower": np.array([46, 55, 35]),
        "upper": np.array([78, 255, 255]),
        "bgr": (0, 255, 0),
    },
    # 浅蓝/天蓝：实测色相中心位于 104 [102~112]
    "sky-blue": {
        "lower": np.array([82, 45, 50]),
        "upper": np.array([115, 255, 255]),
        "bgr": (255, 255, 0),
    },
    # 深蓝：实测色相中心位于 117 [115~124]
    "blue": {
        "lower": np.array([116, 60, 45]),
        "upper": np.array([130, 255, 255]),
        "bgr": (255, 0, 0),
    },
    # 黑色：限制 S 必须在 110 以下，防止把阴影下的暗色物料误判为黑
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 255, 80]),
        "bgr": (50, 50, 50),
    },
}

# ── Blue family combined threshold & secondary classifier ────────────────────
# 蓝色系统一检测区间（覆盖天蓝与深蓝全部范围，杜绝分界生硬切断轮廓）
BLUE_COMBINED_HSV = {
    "lower": np.array([82, 45, 45]),
    "upper": np.array([130, 255, 255]),
}
# 纯蓝偏向指数 (Blue Preference Index) 与判定阈值：
# BPI = (B / (G + 1e-5)) * (1.0 - R / (B + 1e-5))
BLUE_BPI_THRESH = 1.3
BLUE_BG_RATIO_THRESH = 2.0
BLUE_MAX_GREEN_THRESH = 50


# ── Ring HSV thresholds — black ring only ────────────────────────────────────

RING_BAND_INNER = 0.7  # annulus inner fraction

RING_HSV = {
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([179, 255, 96]),
        "bgr": (50, 50, 50),
    },
}

MIN_BLOCK_RADIUS = 20


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BlockInfo:
    color: str
    color_id: int
    cx: int
    cy: int
    radius: int


@dataclass
class RingInfo:
    color: str
    color_id: int          # target_id: 1, 2, 3 (digit) or 5 (black ring unknown)
    cx: int
    cy: int
    radius: int
    digit_id: int = 5      # 1, 2, 3 (digit) or 5 (black ring without digit)
    confidence: float = 1.0


@dataclass
class TaskCode:
    raw: str = ""
    batch1_colors: list[int] = field(default_factory=list)
    batch1_positions: list[int] = field(default_factory=list)
    batch2_colors: list[int] = field(default_factory=list)
    batch2_positions: list[int] = field(default_factory=list)
    valid: bool = False
