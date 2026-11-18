"""Tách từng lá (instance segmentation).

3 backend, cùng interface `segment(bgr) -> list[LeafInstance]`:

* ``YoloSegmenter``     — YOLOv8/11-seg huấn luyện trên ảnh đã gán nhãn (khuyến nghị
                          cho production; là cách duy nhất có thể đạt tiêu chí 95%).
* ``FastSAMSegmenter``  — zero-shot (không cần nhãn), lọc mask theo màu xanh + hình dạng.
* ``ClassicalSegmenter`` — OpenCV thuần (ExG + watershed), baseline không cần GPU/torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------- utils


def vegetation_mask(bgr: np.ndarray) -> np.ndarray:
    """Mask pixel lá xanh: HSV + chỉ số Excess-Green. Loại nền nilon/lưới trắng, trời, thước."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    f = bgr.astype(np.float32) + 1e-6
    tot = f.sum(axis=2)
    b, g, r = f[..., 0] / tot, f[..., 1] / tot, f[..., 2] / tot
    exg = 2 * g - r - b
    m = (h >= 25) & (h <= 95) & (s >= 35) & (v >= 25) & (exg > 0.03)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


@dataclass
class LeafInstance:
    mask: np.ndarray                 # bool HxW (kích thước ảnh làm việc)
    score: float = 1.0
    label: str = "leaf"
    meta: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def shape_stats(mask: np.ndarray) -> dict[str, float]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return {"solidity": 0.0, "aspect": 0.0}
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    aspect = max(w, h) / max(min(w, h), 1e-6)
    return {"solidity": area / hull if hull > 0 else 0.0, "aspect": float(aspect)}


@dataclass
class FilterConfig:
    min_area_frac: float = 0.002     # diện tích lá tối thiểu / diện tích ảnh
    max_area_frac: float = 0.40
    min_green_frac: float = 0.65     # tỉ lệ pixel xanh trong mask
    min_solidity: float = 0.75
    max_aspect: float = 2.6          # lá non chưa xoè (cuộn/gập) thường dài-hẹp -> không đếm


def filter_instances(insts: list[LeafInstance], veg: np.ndarray, cfg: FilterConfig) -> list[LeafInstance]:
    """Lọc mask không phải lá, gắn nhãn 'young_leaf' cho lá chưa xoè, loại mask trùng/gộp."""
    img_area = veg.size
    kept: list[LeafInstance] = []
    for ins in insts:
        a0 = ins.area
        if a0 == 0:
            continue
        green = (ins.mask & veg).sum() / a0
        if green < cfg.min_green_frac:
            continue
        ins.mask = _largest_component(ins.mask & veg)  # bỏ phần nền lẫn vào mask
        a = ins.area
        if not cfg.min_area_frac * img_area <= a <= cfg.max_area_frac * img_area:
            continue
        st = shape_stats(ins.mask)
        if st["solidity"] < cfg.min_solidity:
            continue
        ins.meta.update(st, green_frac=float(green))
        if st["aspect"] > cfg.max_aspect:
            ins.label = "young_leaf"
        kept.append(ins)
    return deduplicate(kept)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def deduplicate(insts: list[LeafInstance], contain_th: float = 0.6) -> list[LeafInstance]:
    """Bỏ mask 'gộp' (chứa >=2 lá con) và mask trùng lặp. Tính trên mask thu nhỏ cho nhanh."""
    if len(insts) <= 1:
        return insts
    h, w = insts[0].mask.shape
    s = min(1.0, 512 / max(h, w))
    small = [cv2.resize(i.mask.astype(np.uint8), (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_NEAREST).astype(bool) for i in insts]
    areas = np.array([m.sum() for m in small], dtype=float) + 1e-6
    n = len(insts)
    is_group = np.zeros(n, bool)
    for i in range(n):
        children = 0
        for j in range(n):
            if j != i and areas[j] < 0.8 * areas[i] and areas[j] > 0.08 * areas[i]:
                if (small[i] & small[j]).sum() / areas[j] > 0.6:
                    children += 1
        is_group[i] = children >= 2
    order = [i for i in np.argsort(-areas) if not is_group[i]]
    keep: list[int] = []
    for i in order:
        if all((small[i] & small[k]).sum() / min(areas[i], areas[k]) < contain_th for k in keep):
            keep.append(i)
    return [insts[i] for i in keep]


# --------------------------------------------------------------------------- backends


class Segmenter:
    def segment(self, bgr: np.ndarray) -> list[LeafInstance]:  # pragma: no cover
        raise NotImplementedError


class YoloSegmenter(Segmenter):
    """Model ultralytics *-seg đã fine-tune. Class gợi ý: 0=leaf, 1=young_leaf (chưa xoè)."""

    def __init__(self, weights: str, conf: float = 0.35, imgsz: int = 1024, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def segment(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, retina_masks=True,
                               device=self.device, verbose=False)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        out = []
        for m, c, s in zip(masks, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if m.shape != bgr.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            ins = LeafInstance(m, float(s), self.model.names[int(c)])
            ins.meta.update(shape_stats(m))
            out.append(ins)
        return out


class FastSAMSegmenter(Segmenter):
    """Zero-shot: FastSAM 'segment everything' rồi lọc ra lá. Không cần dữ liệu gán nhãn."""

    def __init__(self, weights: str = "FastSAM-s.pt", conf: float = 0.4, iou: float = 0.9,
                 imgsz: int = 1024, filt: FilterConfig | None = None, device: str | None = None):
        from ultralytics import FastSAM

        self.model = FastSAM(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.filt = filt or FilterConfig()

    def segment(self, bgr):
        r = self.model(bgr, retina_masks=True, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                       device=self.device, verbose=False)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        scores = r.boxes.conf.cpu().numpy()
        insts = [LeafInstance(m, float(s)) for m, s in zip(masks, scores)]
        return filter_instances(insts, vegetation_mask(bgr), self.filt)


class ClassicalSegmenter(Segmenter):
    """Baseline: mask lá xanh + tách lá dính nhau bằng distance-transform watershed."""

    def __init__(self, min_distance_frac: float = 0.03, filt: FilterConfig | None = None):
        self.min_distance_frac = min_distance_frac
        self.filt = filt or FilterConfig()

    def segment(self, bgr):
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        veg = vegetation_mask(bgr)
        # Gân lá & mép tối cắt mask thành lá riêng: dùng edge để tách bớt
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        work = veg & ~cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(bool)
        work = cv2.morphologyEx(work.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        md = max(5, int(self.min_distance_frac * max(bgr.shape[:2])))
        peaks = peak_local_max(dist, min_distance=md, threshold_abs=md * 0.5, labels=work)
        markers = np.zeros(dist.shape, np.int32)
        for k, (y, x) in enumerate(peaks, 1):
            markers[y, x] = k
        labels = watershed(-dist, markers, mask=veg)
        insts = [LeafInstance(labels == k, 1.0) for k in range(1, len(peaks) + 1)]
        return filter_instances(insts, veg, self.filt)


def build_segmenter(backend: str, weights: str | None = None, **kw) -> Segmenter:
    backend = backend.lower()
    if backend == "yolo":
        if not weights:
            raise ValueError("backend 'yolo' cần --weights (file .pt đã train)")
        return YoloSegmenter(weights, **kw)
    if backend == "fastsam":
        return FastSAMSegmenter(weights or "FastSAM-s.pt", **kw)
    if backend == "classical":
        return ClassicalSegmenter(**kw)
    raise ValueError(f"backend không hỗ trợ: {backend}")
