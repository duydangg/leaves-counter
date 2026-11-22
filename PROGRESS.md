# Tiến trình dự án đếm lá & đo lá dâu tây

*Cập nhật: 2026-11-22*

## 1. Mục tiêu (theo README.MD và faq.MD)

| Task | Định nghĩa | Tiêu chí nghiệm thu |
|---|---|---|
| Phán đoán chất lượng ảnh | Ảnh có đủ tốt để đếm/đo không | Trên 100 ảnh, sai dưới 5 ảnh |
| Đếm lá | Chụp từng cây, mỗi cây nhiều ảnh. Lá non chưa xoè không đếm. Lá bị che vẫn phải đếm | Đếm đúng số lá ở ≥ 95/100 ảnh |
| Đo lá | Dài = gân chính. Rộng = chỗ rộng nhất vuông góc với gân chính. Lá bị che không cần đo | Sai số trong ±5% so với số đo thực tế |

## 2. Đã hoàn thành

### Dữ liệu
- Đã xem 26 ảnh mẫu HEIC (5712×4284) trong `11月/`, gồm 5 cây (#1–#5) × 5 ảnh và 1 ảnh tổng quan (A株).
- Ảnh **chưa có nhãn**. Một số ảnh có thước dây, chưa ảnh nào có vật chuẩn để tự động quy đổi ra mm.

### Code (package `leafmodel/` + CLI `main.py`)
| Thành phần | File | Trạng thái |
|---|---|---|
| Đọc ảnh HEIC/JPG, xoay theo EXIF, resize, chuyển HEIC → JPG | `leafmodel/io.py` | ✅ Hoạt động |
| Kiểm tra chất lượng ảnh: độ nét, phơi sáng, tỉ lệ lá, độ phân giải | `leafmodel/quality.py` | ✅ Hoạt động. Đã bắt được ảnh mờ/tối/cháy sáng tạo giả |
| Tách lá, 3 backend: `yolo` (cần train), `fastsam` (zero-shot), `classical` (OpenCV) | `leafmodel/segmentation.py` | ✅ Code xong. ⚠️ Chưa có model YOLO đã train |
| Gắn nhãn lá non chưa xoè (`young_leaf`) và không đếm | `leafmodel/segmentation.py` | ⚠️ Zero-shot mới dùng luật hình dạng (lá dài-hẹp), chưa chính xác |
| Đo dài/rộng theo định nghĩa faq | `leafmodel/measure.py` | ✅ Hoạt động. Đánh dấu `reliable=false` cho lá chạm mép/bị che |
| Quy đổi px → mm: ArUco, 2 điểm trên thước, hệ số cố định | `leafmodel/calibration.py` | ✅ Hoạt động (ArUco đã thử trên marker tự tạo) |
| `LeafModel`: chất lượng → tách lá → đếm → đo; gộp nhiều ảnh của 1 cây (max/median) | `leafmodel/pipeline.py` | ✅ Hoạt động |
| Ảnh output: mask, bounding box, đường gân chính/chiều rộng, nhãn `#id L.. W..mm`, tiêu đề số lá + tỉ lệ | `leafmodel/pipeline.py` (`visualize`) | ✅ Hoạt động |
| Đánh giá theo 3 tiêu chí faq; dò ngưỡng độ nét từ ảnh đã gán nhãn | `leafmodel/evaluate.py` | ✅ Code xong, mới thử với dữ liệu giả |
| Script fine-tune YOLO-seg | `leafmodel/train.py` | ✅ Code xong, chưa chạy (chưa có nhãn) |
| CLI: `convert`, `quality`, `analyze`, `train`, `evaluate`, `aruco` | `main.py` | ✅ Hoạt động |
| Hướng dẫn sử dụng | `MODEL.md` | ✅ |

### Kết quả chạy thử trên 26 ảnh mẫu
- **Chất lượng ảnh:** 26/26 ảnh đạt. Chưa có ảnh xấu thật để kiểm chứng, ngưỡng hiện tại là giá trị khởi đầu.
- **Đếm (zero-shot FastSAM):** chạy hết 6 nhóm ảnh, nhưng **chưa đạt yêu cầu**:
  - sót các lá chét chồng lên nhau ở tâm cây;
  - đếm lẫn lá của cây bên cạnh;
  - số lá kép đang ước tính bằng số lá chét ÷ 3, chưa kiểm chứng.
- **Đo:** chưa đánh giá được độ chính xác vì chưa có số đo thực tế và chưa có vật chuẩn trong ảnh.

## 3. Các bước phát triển tiếp theo

### Bước 0 — Chốt định nghĩa với khách hàng (chặn các bước sau)
- [ ] "1 lá" là **lá kép** (3 lá chét) hay **từng lá chét**?
- [ ] Đo lá nào: **lá chét giữa** hay cả lá kép? Đo tất cả lá hay chỉ một số lá chỉ định?
- [ ] Chỉ đếm lá của **cây mục tiêu** (cây ở giữa ảnh / gần thẻ tên) đúng không?
- [ ] Thế nào là ảnh "chất lượng kém": mờ, ngược sáng, thiếu cây, che khuất nhiều…?

### Bước 1 — Quy trình chụp ảnh
- [ ] Đặt **vật chuẩn kích thước** trong mỗi ảnh cần đo. Khuyến nghị dùng ArUco marker (`python main.py aruco marker.png`), in ra, đo lại cạnh thật và đặt sát lá cần đo.
- [ ] Chụp **vuông góc** với phiến lá cần đo. Giữ khoảng cách và góc chụp ổn định giữa các cây.
- [ ] Quy định số ảnh/góc chụp cho mỗi cây (ví dụ: 1 ảnh từ trên xuống + 4 góc xung quanh).

### Bước 2 — Thu thập & gán nhãn dữ liệu
- [ ] Thu thêm ảnh: tối thiểu **150–300 ảnh**, nhiều cây, nhiều ngày, nhiều điều kiện ánh sáng.
- [ ] `python main.py convert <thư mục> data/jpg` rồi gán nhãn polygon (CVAT / Label Studio / Roboflow):
  - `leaf`: lá đã xoè, kể cả lá bị che (đếm);
  - `young_leaf`: lá non chưa xoè (không đếm);
  - `leaflet` (tuỳ kết quả Bước 0): lá chét dùng để đo.
- [ ] Ghi **ground truth**:
  - `quality.csv` (`image,good`), có cả ảnh xấu thật;
  - `counts.csv` (`image,leaf_count`), số lá đếm tay;
  - `measures.csv` (`image,x,y,length_mm,width_mm`), đo bằng thước thật trên cây.
- [ ] Chia train/val/test **theo cây**, không theo ảnh, để tránh rò dữ liệu. Giữ riêng ≥ 100 ảnh test cho nghiệm thu.

### Bước 3 — Train & đánh giá model
- [ ] `python main.py train dataset/` (YOLO11-seg), thử các cỡ model s/m/l và `imgsz` 1024–1536.
- [ ] `python main.py evaluate ...` trên tập test theo 3 tiêu chí faq.
- [ ] Dò ngưỡng chất lượng ảnh bằng `evaluate.tune_quality_thresholds` khi đã có ảnh xấu thật. Có thể chuyển sang bộ phân loại học máy nếu luật không đủ.
- [ ] Phân tích lỗi (đếm thiếu/thừa ở đâu), bổ sung dữ liệu cho các trường hợp khó rồi train lại.

### Bước 4 — Cải thiện thuật toán
- [ ] **Chỉ đếm cây mục tiêu:** nhận diện thẻ tên/thân cây chính, hoặc thêm nhãn phân biệt lá của cây khác.
- [ ] **Gộp nhiều ảnh của 1 cây:** thay `max` bằng cách ghép/khớp lá giữa các góc chụp để không đếm trùng hay bỏ sót.
- [ ] **Đo chính xác hơn:** xác định gân chính bằng skeleton/keypoint thay cho trục PCA; bù phối cảnh khi lá nghiêng (dùng marker để tính homography).
- [ ] **Tự động đọc thước dây** trong ảnh để quy đổi mm khi không có ArUco.
- [ ] Tự động bỏ qua lá `reliable=false` khi tổng hợp số đo.

### Bước 5 — Đóng gói & triển khai
- [ ] Export model sang ONNX/CoreML nếu chạy trên điện thoại hoặc máy yếu.
- [ ] API hoặc giao diện đơn giản: upload ảnh → nhận số lá, kích thước, ảnh minh hoạ.
- [ ] Xuất báo cáo theo cây/ngày (CSV/Excel) để theo dõi sinh trưởng.
- [ ] Viết unit test cho `measure`, `quality`, `calibration`.

## 4. Rủi ro chính
- **Tiêu chí đếm đúng 95% là rất cao** với lá dâu tây chồng chéo. Cần đủ dữ liệu gán nhãn và quy trình chụp chuẩn.
- **Đo ±5% không thể đạt nếu thiếu vật chuẩn** hoặc lá bị nghiêng nhiều so với camera.
- **Định nghĩa "lá"** (lá kép hay lá chét) thay đổi toàn bộ cách gán nhãn, nên phải chốt trước khi gán nhãn.
