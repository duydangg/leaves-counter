"""LeafModel: kiểm tra chất lượng ảnh -> tách lá -> đếm lá -> đo dài/rộng."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import calibration
from .io import LoadedImage, load_image
from .measure import LeafMeasurement, measure_mask
from .quality import QualityReport, QualityThresholds, assess_quality
from .segmentation import LeafInstance, Segmenter


@dataclass
class LeafResult:
    id: int
    label: str
    score: float
    counted: bool
    measurement: LeafMeasurement | None

    def to_dict(self):
        d = {"id": self.id, "label": self.label, "score": round(self.score, 3), "counted": self.counted}
        if self.measurement:
            m = self.measurement
            d.update(length_mm=_r(m.length_mm), width_mm=_r(m.width_mm), length_px=_r(m.length_px),
                     width_px=_r(m.width_px), reliable=m.reliable,
                     center_xy=[_r(v) for v in m.center_xy])
        return d


@dataclass
class ImageResult:
    path: str
    quality: QualityReport
    leaf_count: int | None
    leaves: list[LeafResult] = field(default_factory=list)
    mm_per_px: float | None = None
    scale_source: str | None = None
    image_scale: float = 1.0
    _image: LoadedImage | None = field(default=None, repr=False)
    _instances: list[LeafInstance] = field(default_factory=list, repr=False)

    def to_dict(self):
        return {
            "path": self.path, "quality_ok": self.quality.passed, "quality_reasons": self.quality.reasons,
            "quality_metrics": {k: _r(v) for k, v in self.quality.metrics.items()},
            "leaf_count": self.leaf_count, "mm_per_px": self.mm_per_px, "scale_source": self.scale_source,
            "leaves": [l.to_dict() for l in self.leaves],
        }


@dataclass
class PlantResult:
    plant: str
    images: list[ImageResult]
    leaf_count: int | None
    aggregation: str

    def to_dict(self):
        good = [i for i in self.images if i.leaf_count is not None]
        return {"plant": self.plant, "leaf_count": self.leaf_count, "aggregation": self.aggregation,
                "per_image_counts": {Path(i.path).name: i.leaf_count for i in self.images},
                "n_images": len(self.images), "n_good_images": len(good),
                "images": [i.to_dict() for i in self.images]}


def _r(v, n=2):
    return None if v is None else round(float(v), n)


class LeafModel:
    """
    Parameters
    ----------
    segmenter : backend tách lá (xem segmentation.build_segmenter)
    count_labels : nhãn được đếm. Lá non chưa xoè ('young_leaf') KHÔNG đếm (faq.MD).
    measure_labels : nhãn được đo (vd chỉ 'leaflet' trung tâm nếu model có class đó).
    leaflets_per_leaf : >1 nếu segmenter trả về lá chét (lá dâu tây = 3 lá chét) mà
        muốn quy ra số lá kép. Mặc định 1 = đếm đúng đơn vị model trả về.
    max_side : cạnh dài của ảnh làm việc.
    skip_bad_quality : ảnh không đạt chất lượng thì không đếm/đo.
    """

    def __init__(self, segmenter: Segmenter, count_labels=("leaf",), measure_labels=("leaf", "leaflet"),
                 leaflets_per_leaf: int = 1, quality_thresholds: QualityThresholds | None = None,
                 max_side: int = 2048, skip_bad_quality: bool = True,
                 aruco_mm: float | None = None, aruco_dict: str = "4x4_50", mm_per_px: float | None = None):
        self.segmenter = segmenter
        self.count_labels = set(count_labels)
        self.measure_labels = set(measure_labels)
        self.leaflets_per_leaf = leaflets_per_leaf
        self.qth = quality_thresholds or QualityThresholds()
        self.max_side = max_side
        self.skip_bad_quality = skip_bad_quality
        self.aruco_mm, self.aruco_dict = aruco_mm, aruco_dict
        self.mm_per_px_original = mm_per_px

    # ------------------------------------------------------------------ public API
    def check_quality(self, path) -> QualityReport:
        img = load_image(path, self.max_side)
        return assess_quality(img.bgr, img.orig_size, self.qth)

    def analyze_image(self, path, scale: calibration.Scale | None = None) -> ImageResult:
        img = load_image(path, self.max_side)
        q = assess_quality(img.bgr, img.orig_size, self.qth)
        res = ImageResult(str(path), q, None, image_scale=img.scale, _image=img)
        if not q.passed and self.skip_bad_quality:
            return res

        scale = scale or self._auto_scale(img)
        if scale:
            res.mm_per_px, res.scale_source = scale.mm_per_px, scale.source

        insts = self.segmenter.segment(img.bgr)
        res._instances = insts
        n_counted = 0
        for k, ins in enumerate(insts):
            counted = ins.label in self.count_labels
            n_counted += counted
            meas = None
            if ins.label in self.measure_labels:
                meas = measure_mask(ins.mask, res.mm_per_px, ins.meta.get("solidity"))
            res.leaves.append(LeafResult(k, ins.label, ins.score, counted, meas))
        res.leaf_count = int(round(n_counted / self.leaflets_per_leaf))
        return res

    def analyze_plant(self, paths, name: str | None = None, aggregation: str = "max") -> PlantResult:
        """Mỗi cây chụp nhiều góc. Lá bị che ở góc này có thể lộ ra ở góc khác, nên mặc định
        lấy MAX số lá trong các ảnh đạt chất lượng ('median' ổn định hơn nếu hay đếm thừa)."""
        imgs = [self.analyze_image(p) for p in paths]
        counts = [i.leaf_count for i in imgs if i.leaf_count is not None]
        if not counts:
            total = None
        elif aggregation == "max":
            total = max(counts)
        elif aggregation == "median":
            total = int(round(float(np.median(counts))))
        else:
            raise ValueError(aggregation)
        return PlantResult(name or str(Path(paths[0]).parent), imgs, total, aggregation)

    # ------------------------------------------------------------------ helpers
    def _auto_scale(self, img: LoadedImage) -> calibration.Scale | None:
        if self.aruco_mm:
            s = calibration.from_aruco(img.bgr, self.aruco_mm, self.aruco_dict)
            if s:
                return s
        if self.mm_per_px_original:
            return calibration.from_manual(self.mm_per_px_original, img.scale)
        return None

    @staticmethod
    def visualize(res: ImageResult, out_path: str | Path) -> None:
        if res._image is None:
            return
        vis = res._image.bgr.copy()
        overlay = vis.copy()
        rng = np.random.default_rng(0)
        for leaf, ins in zip(res.leaves, res._instances):
            color = (0, 0, 255) if not leaf.counted else tuple(int(c) for c in rng.integers(60, 255, 3))
            overlay[ins.mask] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        for leaf, ins in zip(res.leaves, res._instances):
            cnts, _ = cv2.findContours(ins.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 1)
            m = leaf.measurement
            if m:
                p = lambda xy: tuple(int(round(v)) for v in xy)
                cv2.line(vis, p(m.axis_p1), p(m.axis_p2), (0, 255, 255), 2)
                cv2.line(vis, p(m.width_p1), p(m.width_p2), (255, 0, 255), 2)
                txt = f"#{leaf.id}"
                if m.length_mm:
                    txt += f" {m.length_mm:.0f}x{m.width_mm:.0f}mm"
                cv2.putText(vis, txt, p(m.center_xy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(vis, txt, p(m.center_xy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        head = f"leaves: {res.leaf_count}" if res.leaf_count is not None else f"BAD QUALITY: {res.quality.reasons}"
        cv2.putText(vis, head, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
        cv2.putText(vis, head, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
