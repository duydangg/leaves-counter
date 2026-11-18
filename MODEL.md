# leafmodel — đếm lá & đo lá dâu tây

## Cài đặt
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Các chức năng
| Task (README/faq) | Module | Tiêu chí pass |
|---|---|---|
| Phán đoán chất lượng ảnh | `leafmodel/quality.py` (độ nét, phơi sáng, độ phủ lá, độ phân giải) | sai < 5/100 ảnh |
| Đếm lá (lá non chưa xoè không đếm) | `segmentation.py` + `pipeline.py` | đúng ≥ 95/100 ảnh |
| Đo dài (gân chính) / rộng (vuông góc gân chính, chỗ rộng nhất) | `measure.py` + `calibration.py` | sai số ±5% |

Backend tách lá: `yolo` (cần train, dùng cho production), `fastsam` (zero-shot, baseline), `classical` (OpenCV thuần).

## Dùng
```bash
python main.py quality 11月
python main.py analyze 11月 --backend fastsam --out out/        # mỗi thư mục con = 1 cây
python main.py analyze 11月 --backend yolo --weights best.pt --aruco-mm 50 --out out/
```
Kết quả: `out/results.json` (số lá theo cây/ảnh), `out/leaves.csv` (kích thước từng lá), `out/vis/` (ảnh minh hoạ).
Một cây chụp nhiều góc → số lá của cây = MAX số lá trong các ảnh đạt chất lượng (`--aggregation median` để đổi).

```python
from leafmodel import LeafModel, build_segmenter
model = LeafModel(build_segmenter("yolo", "best.pt"), aruco_mm=50)
r = model.analyze_image("IMG_1423.HEIC")
print(r.quality.passed, r.leaf_count, [l.to_dict() for l in r.leaves])
```

## Đưa lên production (bắt buộc để đạt 95%)
1. `python main.py convert 11月 data/jpg`, rồi gán nhãn polygon (CVAT/Label Studio/Roboflow): class `leaf`, `young_leaf`.
   Nên có ≥ 150–300 ảnh, chia train/val theo **cây** (không theo ảnh) để tránh rò dữ liệu.
2. `python main.py train dataset/` → `runs/leaf/seg*/weights/best.pt`.
3. Tạo ground truth `counts.csv`, `quality.csv`, `measures.csv` (định dạng trong `leafmodel/evaluate.py`), rồi:
   `python main.py evaluate test_imgs --backend yolo --weights best.pt --counts ... --quality ... --measures ... --aruco-mm 50`

## Để đo được ±5%
- Ảnh không có thông tin khoảng cách, nên **phải có vật chuẩn**: in ArUco marker (`python main.py aruco marker.png`),
  đo lại cạnh sau khi in và đặt sát lá cần đo. Thước dây đang có trong ảnh mẫu dùng được ở chế độ bán tự động
  (`calibration.from_two_points`).
- Chụp vuông góc với phiến lá. Lá bị nghiêng hoặc che khuất sẽ bị đo ngắn hơn thực tế; những lá chạm mép ảnh
  hoặc bị che nhiều được đánh dấu `reliable=false`.

## File PT
Liên hệ dangnguyenduyvn@gmail.com