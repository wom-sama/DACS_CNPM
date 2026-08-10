# So Sánh Nhánh Classification-Only Với Run `mango_hybrid_224`

Ngày cập nhật: 2026-06-02

## Run Cũ: `runs/mango_hybrid_224`

Thông tin chính từ `resolved_config.json`, `history.csv` và `best_metrics.json`:

| Mục | Giá trị |
|---|---|
| Run | `mango_hybrid_224` |
| Model | `vit_registers_hybrid` |
| Input size | `224` |
| Số class | `4` |
| Best epoch | `109` |
| Best val accuracy | `0.971034` |
| Best val macro F1 | `0.970986` |
| Best val loss | `0.944368` |
| Train valid objects | `9392` |
| Train selected samples | `7618` |
| Train ignored objects | `1774` |
| Train multi-object images | `844` |

Run này đạt kết quả phân loại tốt, nhưng logic vẫn là hybrid/detection-style. Với bài báo phân loại đơn, điểm yếu lớn nhất là ảnh nhiều object chỉ dùng object chính, làm mất 1774 object hợp lệ trong train.

## Logic Cũ

1. Dataset đọc nhãn YOLO.
2. Nếu crop classification, mỗi ảnh chỉ tạo 1 sample.
3. Object chính được chọn theo bbox lớn nhất/gần trung tâm.
4. Các object còn lại trong ảnh nhiều object bị bỏ qua với bài toán phân loại.
5. Model dùng `vit_registers_hybrid`, tức vẫn mang nhiều cấu hình detection như bbox head, objectness/detection-related path.
6. Augmentation cũ có batch mix, mosaic, cutmix, mixup ở mức cao. Các phép này có thể hợp với detection hoặc regularization tổng quát, nhưng không sạch cho phân loại độ chín vì dễ trộn màu và trộn nhãn.

## Logic Mới Trên Nhánh `classification-only-research`

1. Model chính là `vit_registers`, không dùng DETR decoder, objectness, bbox regression, count head hay quality head.
2. Khi `classification_target=True`, dataset mặc định bật `classification_object_crops=True`.
3. Mỗi bbox hợp lệ trong ảnh nhiều object trở thành một sample crop riêng.
4. Crop được căn theo bbox của chính sample đó, không luôn theo object lớn nhất.
5. Label phân loại là label của bbox đang crop.
6. `class_counts`, `labels()` và rare-class repeat đều tính theo label crop chính, tránh lặp sai theo toàn bộ object trong ảnh.
7. Khi không phải detection mode, train tự tắt `batch_mix_probability`, `mosaic_probability`, `mixup_probability`, `cutmix_probability`, `copy_paste_probability`.

## Điều Chỉnh Đã Thực Hiện

- Thêm object-level classification crops trong `MangoYOLOCropDataset`.
- Thêm `primary_object_index` cho `MangoSample` để crop đúng bbox đang được dùng làm label.
- Sửa `class_counts()` để classification đếm theo sample crop, không đếm toàn bộ object lặp lại.
- Sửa `RareClassRepeatDataset` để repeat theo label crop chính trong classification mode.
- Thêm flag `--disable-classification-object-crops` cho train/evaluate để tái lập kiểu primary-object-only khi cần ablation.
- Buộc tắt batch composition trong classification-only.
- Cập nhật tài liệu augmentation/preprocessing theo hướng phân loại đơn.
- Loại bỏ ảnh minh họa mosaic/cutmix/targeted copy-paste khỏi bộ ví dụ classification-only.
- Thêm ảnh minh họa `10_object_level_crops_from_multi_object_images`.

## Khuyến Nghị Cho Run Tiếp Theo

Nên train từ đầu bằng cấu hình classification-only:

- `--model-type vit_registers`
- `--best-metric macro_f1`
- object-level crops mặc định bật
- tắt brightness/contrast/saturation/hue/lighting trong run chính
- tắt random erasing trong run chính
- chỉ dùng affine nhẹ, horizontal flip, vertical flip/rotate90 rất thấp
- giữ `resize_mode=pad`
- dùng `class_weight_mode=sqrt_inverse`, LDAM/focal nhẹ nếu mất cân bằng còn rõ

Không nên dùng các cấu hình detection sau trên nhánh này:

- `--full-image-detection`
- `--model-type vit_registers_hybrid`
- `--quality-head`
- `--count-head`
- `--auxiliary-decoder-loss`
- `--best-metric macro_detection_hmean`
- `--mosaic-probability`
- `--cutmix-probability`
- `--copy-paste-probability`

