"""Đo kích thước lá theo định nghĩa trong faq.MD.

* Chiều dài = chiều dài gân chính  -> xấp xỉ bằng trục chính (PCA) của mask lá,
  đo từ gốc phiến lá tới chóp lá.
* Chiều rộng = khoảng rộng nhất VUÔNG GÓC với gân chính -> quét dọc trục chính,
  lấy dây cung vuông góc dài nhất.

Lưu ý: đo trên ảnh 2D chỉ chính xác khi phiến lá gần song song với mặt phẳng ảnh
và vật chuẩn (marker/thước) nằm cùng khoảng cách tới camera với lá.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LeafMeasurement:
    length_px: float
    width_px: float
    length_mm: float | None
    width_mm: float | None
    angle_deg: float          # hướng gân chính trong ảnh
    center_xy: tuple[float, float]
    axis_p1: tuple[float, float]  # 2 đầu gân chính (để vẽ)
    axis_p2: tuple[float, float]
    width_p1: tuple[float, float]
    width_p2: tuple[float, float]
    reliable: bool            # False nếu lá bị cắt ở mép ảnh / bị che nhiều

    def to_dict(self):
        return asdict(self)


def measure_mask(mask: np.ndarray, mm_per_px: float | None = None, solidity: float | None = None,
                 min_solidity_reliable: float = 0.85, border_margin: int = 3) -> LeafMeasurement:
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    minor = np.array([-major[1], major[0]])
    u = (pts - c) @ major
    v = (pts - c) @ minor

    u_min, u_max = u.min(), u.max()
    length = u_max - u_min + 1

    # Quét theo từng lát 1px dọc gân chính, lấy dây cung vuông góc lớn nhất
    bins = np.floor(u - u_min).astype(int)
    nb = bins.max() + 1
    vmin = np.full(nb, np.inf)
    vmax = np.full(nb, -np.inf)
    np.minimum.at(vmin, bins, v)
    np.maximum.at(vmax, bins, v)
    widths = np.where(np.isfinite(vmin), vmax - vmin + 1, 0)
    # làm mượt nhẹ để tránh nhiễu 1 lát pixel
    k = max(1, nb // 50)
    widths_s = np.convolve(widths, np.ones(k) / k, mode="same") if k > 1 else widths
    bi = int(np.argmax(widths_s))
    width = float(widths[bi])
    u_w = u_min + bi + 0.5

    h, w = mask.shape
    touches = (xs.min() <= border_margin or ys.min() <= border_margin or
               xs.max() >= w - 1 - border_margin or ys.max() >= h - 1 - border_margin)
    reliable = not touches and (solidity is None or solidity >= min_solidity_reliable)

    to_xy = lambda uu, vv: tuple(map(float, c + uu * major + vv * minor))
    s = mm_per_px
    return LeafMeasurement(
        length_px=float(length), width_px=width,
        length_mm=float(length * s) if s else None, width_mm=float(width * s) if s else None,
        angle_deg=float(np.degrees(np.arctan2(major[1], major[0]))),
        center_xy=tuple(map(float, c)),
        axis_p1=to_xy(u_min, 0), axis_p2=to_xy(u_max, 0),
        width_p1=to_xy(u_w, vmin[bi]), width_p2=to_xy(u_w, vmax[bi]),
        reliable=bool(reliable),
    )
