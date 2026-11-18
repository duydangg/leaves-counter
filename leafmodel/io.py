"""Đọc ảnh (bao gồm HEIC của iPhone) và chuẩn hoá về BGR numpy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC sẽ không đọc được, các định dạng khác vẫn OK
    pillow_heif = None

IMAGE_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class LoadedImage:
    path: str
    bgr: np.ndarray          # ảnh làm việc (đã resize)
    scale: float             # kích thước làm việc / kích thước gốc (<= 1)
    orig_size: tuple[int, int]  # (width, height) gốc


def load_image(path: str | Path, max_side: int | None = 2048) -> LoadedImage:
    """Đọc ảnh, xoay theo EXIF, resize để cạnh dài <= max_side."""
    path = Path(path)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        orig_w, orig_h = im.size
        scale = 1.0
        if max_side and max(orig_w, orig_h) > max_side:
            scale = max_side / max(orig_w, orig_h)
            im = im.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
        bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return LoadedImage(str(path), bgr, scale, (orig_w, orig_h))


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def group_by_plant(paths: list[Path]) -> dict[str, list[Path]]:
    """Mỗi thư mục cha = 1 cây (theo cách chụp: mỗi cây nhiều ảnh)."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(str(p.parent), []).append(p)
    return groups


def convert_to_jpg(src: str | Path, dst_dir: str | Path, max_side: int | None = None, quality: int = 95) -> list[Path]:
    """Chuyển HEIC -> JPG (giữ cấu trúc thư mục) để đưa vào công cụ gán nhãn."""
    src, dst_dir = Path(src), Path(dst_dir)
    out = []
    for p in list_images(src):
        rel = p.relative_to(src) if src.is_dir() else Path(p.name)
        target = (dst_dir / rel).with_suffix(".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        img = load_image(p, max_side=max_side)
        cv2.imwrite(str(target), img.bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(target)
    return out
