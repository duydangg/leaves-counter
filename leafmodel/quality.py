"""Phán đoán chất lượng ảnh (task phụ theo faq.MD: sai < 5/100 ảnh là pass).

Các tiêu chí: độ nét, phơi sáng, độ phủ lá, độ phân giải. Ngưỡng mặc định là
giá trị khởi đầu — cần hiệu chỉnh trên tập ảnh đã gán nhãn tốt/xấu
(xem `evaluate.tune_quality_thresholds`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .segmentation import vegetation_mask


@dataclass
class QualityThresholds:
    min_sharpness: float = 60.0        # variance of Laplacian (trên ảnh chuẩn hoá rộng 1024px)
    max_overexposed: float = 0.25      # tỉ lệ pixel cháy sáng
    max_underexposed: float = 0.25     # tỉ lệ pixel quá tối
    min_brightness: float = 50.0
    max_brightness: float = 220.0
    min_vegetation: float = 0.10       # tỉ lệ diện tích lá trong ảnh
    min_side_px: int = 1000            # cạnh ngắn của ảnh gốc


@dataclass
class QualityReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _normalize(bgr: np.ndarray, width: int = 1024) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = width / w
    return cv2.resize(bgr, (width, round(h * s)), interpolation=cv2.INTER_AREA)


def image_metrics(bgr: np.ndarray) -> dict[str, float]:
    small = _normalize(bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    veg = vegetation_mask(small)
    # Độ nét đo trên vùng lá (nền lưới/nilon dễ làm sai số), fallback toàn ảnh
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp_all = float(lap.var())
    sharp_veg = float(lap[veg].var()) if veg.sum() > 1000 else sharp_all
    # Tenengrad (gradient Sobel) — bổ sung cho Laplacian
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return {
        "sharpness": sharp_veg,
        "sharpness_global": sharp_all,
        "tenengrad": float(np.mean(gx**2 + gy**2)),
        "brightness": float(gray.mean()),
        "overexposed": float((gray >= 250).mean()),
        "underexposed": float((gray <= 10).mean()),
        "vegetation": float(veg.mean()),
    }


def assess_quality(bgr: np.ndarray, orig_size: tuple[int, int] | None = None,
                   th: QualityThresholds | None = None) -> QualityReport:
    th = th or QualityThresholds()
    m = image_metrics(bgr)
    reasons = []
    if m["sharpness"] < th.min_sharpness:
        reasons.append("blurry")
    if m["overexposed"] > th.max_overexposed:
        reasons.append("overexposed")
    if m["underexposed"] > th.max_underexposed:
        reasons.append("underexposed")
    if not th.min_brightness <= m["brightness"] <= th.max_brightness:
        reasons.append("bad_brightness")
    if m["vegetation"] < th.min_vegetation:
        reasons.append("too_little_leaf")
    if orig_size is not None:
        m["min_side_px"] = float(min(orig_size))
        if min(orig_size) < th.min_side_px:
            reasons.append("low_resolution")
    return QualityReport(passed=not reasons, reasons=reasons, metrics=m)
