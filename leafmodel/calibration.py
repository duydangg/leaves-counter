"""Quy đổi pixel -> mm.

Ảnh không có thông tin khoảng cách, nên bắt buộc phải có vật chuẩn trong ảnh:
* ArUco marker (khuyến nghị: in marker cạnh đã biết, đặt sát lá) -> tự động.
* 2 điểm trên thước dây đã biết khoảng cách (vd vạch 10cm và 20cm) -> bán tự động.
* Hệ số mm/px cố định (khi camera cố định khoảng cách tới cây).

Mọi giá trị mm/px trả về đều tính trên *ảnh làm việc* (đã resize).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Scale:
    mm_per_px: float
    source: str


def from_manual(mm_per_px_original: float, image_scale: float) -> Scale:
    """mm/px đo trên ảnh gốc -> quy về ảnh làm việc."""
    return Scale(mm_per_px_original / image_scale, "manual")


def from_two_points(p1, p2, distance_mm: float, image_scale: float = 1.0) -> Scale:
    """p1, p2: toạ độ pixel trên ẢNH GỐC của 2 vạch thước cách nhau distance_mm."""
    d_px = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1])) * image_scale
    return Scale(distance_mm / d_px, "two_points")


_ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


def from_aruco(bgr: np.ndarray, marker_mm: float, dictionary: str = "4x4_50") -> Scale | None:
    """Tìm ArUco marker, trả về mm/px dựa trên độ dài cạnh trung bình. None nếu không thấy."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(_ARUCO_DICTS[dictionary]),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None or len(corners) == 0:
        return None
    sides = []
    for c in corners:
        q = c.reshape(4, 2)
        sides += [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return Scale(marker_mm / float(np.mean(sides)), f"aruco({len(corners)})")


def generate_aruco(path: str, marker_id: int = 0, px: int = 800, dictionary: str = "4x4_50") -> None:
    """Tạo file ảnh marker để in (in xong đo lại cạnh thực tế bằng thước)."""
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(_ARUCO_DICTS[dictionary]), marker_id, px)
    img = cv2.copyMakeBorder(img, px // 8, px // 8, px // 8, px // 8, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)
