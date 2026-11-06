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
    # 浅蓝：提升 S 下限至 45 杜绝白地误检，S 上限设为 160 彻底与深蓝隔离
    "sky-blue": {
        "lower": np.array([90, 45, 60]),
        "upper": np.array([125, 160, 255]),
        "bgr": (255, 255, 0),
    },
    # 深蓝：S 从 165 起步，与浅蓝产生 5 像素的安全隔离带
    "blue": {
        "lower": np.array([95, 165, 50]),
        "upper": np.array([125, 255, 255]),
        "bgr": (255, 0, 0),
    },
    # 黑色：限制 S 必须在 110 以下，防止把阴影下的暗色物料误判为黑
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([179, 110, 90]),
        "bgr": (50, 50, 50),
    },
}


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
