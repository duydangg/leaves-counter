"""Đánh giá theo tiêu chí nghiệm thu trong faq.MD.

* Chất lượng ảnh : sai < 5/100 ảnh            -> accuracy >= 95%
* Đếm lá         : đếm ĐÚNG số lá >= 95/100 ảnh -> exact-match >= 95%
* Đo lá          : sai số trong ±5% so với thực tế (pass nếu >= `pass_rate` số lá đạt)

File ground-truth (CSV, cột `image` là tên file, vd IMG_1423.HEIC):
    quality.csv  : image,good                    (good = 1/0)
    counts.csv   : image,leaf_count
    measures.csv : image,x,y,length_mm,width_mm  (x,y = 1 điểm nằm trên lá, toạ độ ẢNH GỐC)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .pipeline import ImageResult, LeafModel
from .quality import QualityThresholds, assess_quality, image_metrics
from .io import load_image


def _read_csv(path, idx: dict | None = None) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if idx is not None:
        missing = [r["image"] for r in rows if r["image"] not in idx]
        if missing:
            print(f"[warn] {path}: bỏ qua {len(missing)} ảnh không tìm thấy: {missing[:5]}")
        rows = [r for r in rows if r["image"] in idx]
    return rows


def _index(paths: list[Path]) -> dict[str, Path]:
    return {p.name: p for p in paths}


def eval_quality(model: LeafModel, gt_csv, images: list[Path]) -> dict:
    idx = _index(images)
    rows = _read_csv(gt_csv, idx)
    wrong = []
    for r in rows:
        pred = model.check_quality(idx[r["image"]]).passed
        if pred != bool(int(r["good"])):
            wrong.append(r["image"])
    n = max(len(rows), 1)
    acc = 1 - len(wrong) / n
    return {"task": "quality", "n": n, "errors": len(wrong), "accuracy": acc, "pass": acc >= 0.95, "wrong": wrong}


def eval_counts(model: LeafModel, gt_csv, images: list[Path], results: dict[str, ImageResult] | None = None) -> dict:
    idx = _index(images)
    rows = _read_csv(gt_csv, idx)
    details, correct, abs_err = [], 0, []
    for r in rows:
        res = results[r["image"]] if results and r["image"] in results else model.analyze_image(idx[r["image"]])
        gt = int(r["leaf_count"])
        pred = res.leaf_count
        ok = pred == gt
        correct += ok
        if pred is not None:
            abs_err.append(abs(pred - gt))
        details.append({"image": r["image"], "gt": gt, "pred": pred, "ok": ok})
    n = max(len(rows), 1)
    acc = correct / n
    return {"task": "count", "n": n, "exact_accuracy": acc, "mae": float(np.mean(abs_err)) if abs_err else None,
            "pass": acc >= 0.95, "details": details}


def eval_measures(model: LeafModel, gt_csv, images: list[Path], tol: float = 0.05, pass_rate: float = 0.95,
                  results: dict[str, ImageResult] | None = None) -> dict:
    idx = _index(images)
    rows = _read_csv(gt_csv, idx)
    cache: dict[str, ImageResult] = dict(results or {})
    details, ok_all = [], []
    for r in rows:
        name = r["image"]
        if name not in cache:
            cache[name] = model.analyze_image(idx[name])
        res = cache[name]
        x, y = float(r["x"]) * res.image_scale, float(r["y"]) * res.image_scale
        leaf = _leaf_at(res, int(round(x)), int(round(y)))
        d = {"image": name, "gt_length": float(r["length_mm"]), "gt_width": float(r["width_mm"]),
             "pred_length": None, "pred_width": None, "ok": False}
        if leaf is not None and leaf.measurement and leaf.measurement.length_mm:
            m = leaf.measurement
            el = abs(m.length_mm - d["gt_length"]) / d["gt_length"]
            ew = abs(m.width_mm - d["gt_width"]) / d["gt_width"]
            d.update(pred_length=m.length_mm, pred_width=m.width_mm, err_length=el, err_width=ew,
                     ok=el <= tol and ew <= tol)
        ok_all.append(d["ok"])
        details.append(d)
    rate = float(np.mean(ok_all)) if ok_all else 0.0
    return {"task": "measure", "n": len(rows), "within_tol_rate": rate, "tol": tol,
            "pass": rate >= pass_rate, "details": details}


def _leaf_at(res: ImageResult, x: int, y: int):
    for leaf, ins in zip(res.leaves, res._instances):
        h, w = ins.mask.shape
        if 0 <= y < h and 0 <= x < w and ins.mask[y, x]:
            return leaf
    return None


def tune_quality_thresholds(gt_csv, images: list[Path], max_side: int = 2048) -> QualityThresholds:
    """Chọn ngưỡng độ nét tối ưu (ít lỗi nhất) từ ảnh đã gán nhãn tốt/xấu."""
    idx = _index(images)
    rows = _read_csv(gt_csv, idx)
    feats, labels = [], []
    for r in rows:
        img = load_image(idx[r["image"]], max_side)
        feats.append(image_metrics(img.bgr)["sharpness"])
        labels.append(bool(int(r["good"])))
    feats, labels = np.array(feats), np.array(labels)
    best_t, best_err = QualityThresholds().min_sharpness, len(labels) + 1
    for t in np.unique(feats):
        err = int(((feats >= t) != labels).sum())
        if err < best_err:
            best_t, best_err = float(t), err
    return QualityThresholds(min_sharpness=best_t)
