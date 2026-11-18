"""Fine-tune YOLO-seg cho bài toán tách lá.

Chuẩn bị dữ liệu (định dạng YOLO segmentation, export được từ CVAT / Label Studio / Roboflow):

    dataset/
      images/{train,val}/*.jpg     (dùng `python main.py convert` để đổi HEIC -> JPG)
      labels/{train,val}/*.txt     (mỗi dòng: class x1 y1 x2 y2 ... toạ độ chuẩn hoá 0-1)

Quy tắc gán nhãn (theo faq.MD):
    class 0 `leaf`       : lá đã xoè — vẽ polygon cả phần bị che đoán được (vẫn đếm)
    class 1 `young_leaf` : lá non chưa xoè — không đếm
    (tuỳ chọn) class 2 `leaflet` : lá chét dùng để đo, nếu chọn đo lá chét trung tâm
"""
from __future__ import annotations

from pathlib import Path

DATA_YAML = """\
path: {root}
train: images/train
val: images/val
names:
  0: leaf
  1: young_leaf
"""


def write_data_yaml(root: str | Path) -> Path:
    root = Path(root).resolve()
    p = root / "data.yaml"
    p.write_text(DATA_YAML.format(root=root))
    return p


def train(data_yaml: str, base: str = "yolo11s-seg.pt", epochs: int = 150, imgsz: int = 1024,
          batch: int = 8, device: str | None = None, project: str = "runs/leaf") -> str:
    from ultralytics import YOLO

    model = YOLO(base)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=project,
                name="seg", patience=40, degrees=15, fliplr=0.5, flipud=0.2, hsv_h=0.02, hsv_s=0.5,
                hsv_v=0.4, scale=0.4, mosaic=1.0, close_mosaic=20)
    return str(Path(model.trainer.save_dir) / "weights" / "best.pt")
