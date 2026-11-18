"""CLI cho leafmodel.

Ví dụ:
    python main.py convert 11月 data/jpg                          # HEIC -> JPG để gán nhãn
    python main.py quality 11月                                   # phán đoán chất lượng ảnh
    python main.py analyze 11月 --backend fastsam --out out/      # đếm + đo (zero-shot)
    python main.py analyze 11月 --backend yolo --weights best.pt --aruco-mm 50 --out out/
    python main.py train dataset/ --epochs 150
    python main.py evaluate 11月 --backend yolo --weights best.pt --counts gt/counts.csv \
        --quality gt/quality.csv --measures gt/measures.csv --aruco-mm 50
    python main.py aruco marker.png                               # tạo marker để in
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leafmodel import LeafModel, build_segmenter
from leafmodel.io import convert_to_jpg, group_by_plant, list_images


def make_model(a) -> LeafModel:
    seg = build_segmenter(a.backend, a.weights)
    lpl = a.leaflets_per_leaf if a.leaflets_per_leaf else (1 if a.backend == "yolo" else 3)
    return LeafModel(seg, leaflets_per_leaf=lpl, max_side=a.max_side, aruco_mm=a.aruco_mm,
                     mm_per_px=a.mm_per_px, skip_bad_quality=not a.ignore_quality)


def add_model_args(p):
    p.add_argument("--backend", default="fastsam", choices=["yolo", "fastsam", "classical"])
    p.add_argument("--weights", help="file .pt (bắt buộc với yolo)")
    p.add_argument("--leaflets-per-leaf", type=int, default=0,
                   help="quy đổi lá chét -> lá kép (mặc định: 3 cho fastsam/classical, 1 cho yolo)")
    p.add_argument("--max-side", type=int, default=2048)
    p.add_argument("--aruco-mm", type=float, help="cạnh ArUco marker (mm) để quy đổi mm")
    p.add_argument("--mm-per-px", type=float, help="hệ số mm/px cố định (trên ảnh gốc)")
    p.add_argument("--ignore-quality", action="store_true", help="vẫn đếm/đo ảnh chất lượng kém")


def cmd_analyze(a):
    model = make_model(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plants = group_by_plant(list_images(a.input))
    report, rows = [], []
    for plant, paths in plants.items():
        pr = model.analyze_plant(paths, name=plant, aggregation=a.aggregation)
        print(f"[{Path(plant).name}] leaf_count={pr.leaf_count}  per-image={pr.to_dict()['per_image_counts']}")
        for ir in pr.images:
            if a.vis:
                model.visualize(ir, out / "vis" / Path(plant).name / (Path(ir.path).stem + ".jpg"))
            for l in ir.leaves:
                m = l.measurement
                rows.append({"plant": Path(plant).name, "image": Path(ir.path).name, "leaf_id": l.id,
                             "label": l.label, "counted": l.counted,
                             "length_mm": m and m.length_mm and round(m.length_mm, 1),
                             "width_mm": m and m.width_mm and round(m.width_mm, 1),
                             "length_px": m and round(m.length_px, 1), "width_px": m and round(m.width_px, 1),
                             "reliable": m and m.reliable})
        report.append(pr.to_dict())
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if rows:
        with open(out / "leaves.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"-> {out/'results.json'}, {out/'leaves.csv'}" + (f", {out/'vis'}/" if a.vis else ""))


def cmd_quality(a):
    from leafmodel.io import load_image
    from leafmodel.quality import assess_quality

    for p in list_images(a.input):
        img = load_image(p, a.max_side)
        q = assess_quality(img.bgr, img.orig_size)
        m = q.metrics
        print(f"{'OK ' if q.passed else 'BAD'} {p}  sharp={m['sharpness']:.0f} bright={m['brightness']:.0f} "
              f"veg={m['vegetation']:.2f} {','.join(q.reasons)}")


def cmd_evaluate(a):
    from leafmodel import evaluate as ev

    model = make_model(a)
    imgs = list_images(a.input)
    out = []
    if a.quality:
        out.append(ev.eval_quality(model, a.quality, imgs))
    if a.counts:
        out.append(ev.eval_counts(model, a.counts, imgs))
    if a.measures:
        out.append(ev.eval_measures(model, a.measures, imgs, tol=a.tol))
    for r in out:
        keys = {k: v for k, v in r.items() if k not in ("details", "wrong")}
        print(json.dumps(keys, ensure_ascii=False))
    if a.report:
        Path(a.report).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description="Đếm lá & đo kích thước lá")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="HEIC -> JPG")
    p.add_argument("input"); p.add_argument("output"); p.add_argument("--max-side", type=int)

    p = sub.add_parser("quality", help="phán đoán chất lượng ảnh")
    p.add_argument("input"); p.add_argument("--max-side", type=int, default=2048)

    p = sub.add_parser("analyze", help="đếm + đo lá (mỗi thư mục con = 1 cây)")
    p.add_argument("input"); p.add_argument("--out", default="out")
    p.add_argument("--aggregation", default="max", choices=["max", "median"])
    p.add_argument("--no-vis", dest="vis", action="store_false")
    add_model_args(p)

    p = sub.add_parser("train", help="fine-tune YOLO-seg")
    p.add_argument("dataset"); p.add_argument("--base", default="yolo11s-seg.pt")
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--device")

    p = sub.add_parser("evaluate", help="đánh giá theo tiêu chí faq.MD")
    p.add_argument("input"); add_model_args(p)
    p.add_argument("--quality"); p.add_argument("--counts"); p.add_argument("--measures")
    p.add_argument("--tol", type=float, default=0.05); p.add_argument("--report")

    p = sub.add_parser("aruco", help="tạo ArUco marker để in làm vật chuẩn kích thước")
    p.add_argument("output"); p.add_argument("--id", type=int, default=0)

    a = ap.parse_args()
    if a.cmd == "convert":
        print(f"converted {len(convert_to_jpg(a.input, a.output, a.max_side))} images")
    elif a.cmd == "quality":
        cmd_quality(a)
    elif a.cmd == "analyze":
        cmd_analyze(a)
    elif a.cmd == "train":
        from leafmodel.train import train, write_data_yaml

        print("best weights:", train(str(write_data_yaml(a.dataset)), a.base, a.epochs, a.imgsz, a.batch, a.device))
    elif a.cmd == "evaluate":
        cmd_evaluate(a)
    elif a.cmd == "aruco":
        from leafmodel.calibration import generate_aruco

        generate_aruco(a.output, a.id)
        print("saved", a.output)


if __name__ == "__main__":
    main()
