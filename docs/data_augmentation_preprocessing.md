# Data Augmentation and Preprocessing Cho Phân Loại Xoài

Tài liệu này mô tả pipeline tiền xử lý và tăng cường dữ liệu cho nhánh `classification-only-research`. Mục tiêu của nhánh này là phân loại độ chín/chất lượng xoài từ ảnh crop object, không tối ưu detection, bbox, objectness, count head hay quality head.

Luật ưu tiên vẫn giữ nguyên: không dùng pretrain, không dùng backbone pretrained, không dùng checkpoint ngoài repo.

Thư mục ảnh minh họa: [augmentation_examples](augmentation_examples/)

## Bảng Tóm Tắt

| Transformation type | Details | Vai trò trong phân loại |
|---|---|---|
| Kiểm tra ảnh gốc + YOLO bbox | Đọc ảnh gốc và vẽ bbox từ nhãn YOLO. Ảnh mẫu: [00_original_with_yolo_boxes](augmentation_examples/00_original_with_yolo_boxes/). | Dùng để kiểm tra chất lượng nhãn trước khi crop object. |
| Resize + padding | Giữ tỉ lệ ảnh, resize vào khung cố định 416x416 hoặc 640x640, sau đó pad phần thừa. Ảnh mẫu: [01_resize_pad_416](augmentation_examples/01_resize_pad_416/). | Tránh méo hình quả xoài, giữ hình dạng tự nhiên cho classifier. |
| Crop object chính | Crop quanh bbox object chính với margin nhỏ rồi resize/pad. Ảnh mẫu: [02_crop_primary_object](augmentation_examples/02_crop_primary_object/). | Giảm nhiễu nền, giúp mô hình tập trung vào quả xoài. |
| Object-level crops từ ảnh nhiều object | Mỗi bbox hợp lệ trong ảnh nhiều object được tách thành một sample phân loại riêng. Ảnh mẫu: [10_object_level_crops_from_multi_object_images](augmentation_examples/10_object_level_crops_from_multi_object_images/). | Tận dụng thêm object vốn từng bị bỏ qua khi chỉ crop object chính. Đây là thay đổi quan trọng cho classification-only. |
| Random affine nhẹ | Xoay, dịch chuyển và scale nhẹ quanh crop object. Ảnh mẫu: [03_random_affine](augmentation_examples/03_random_affine/). | Tăng khả năng chịu đựng với góc chụp và vị trí quả xoài, nhưng không làm biến dạng quá mạnh. |
| Horizontal flip | Lật trái/phải với xác suất vừa phải. Ảnh mẫu: [04_horizontal_flip](augmentation_examples/04_horizontal_flip/). | Phù hợp với phân loại vì hướng trái/phải không làm đổi class. |
| Vertical flip thấp | Lật trên/dưới với xác suất rất thấp. Ảnh mẫu: [05_vertical_flip](augmentation_examples/05_vertical_flip/). | Chỉ dùng rất nhẹ để tăng robustness; ảnh thật ít khi bị đảo ngược hoàn toàn. |
| Rotate90 thấp | Xoay theo bội số 90 độ với xác suất thấp. Ảnh mẫu: [06_rotate90](augmentation_examples/06_rotate90/). | Chỉ phù hợp khi dữ liệu thực tế có ảnh xoay do camera hoặc upload. |
| Color jitter rất nhẹ hoặc tắt | Hỗ trợ brightness, contrast, saturation, hue. Ảnh mẫu: [07_color_jitter](augmentation_examples/07_color_jitter/). | Màu sắc là tín hiệu chính của độ chín, nên chỉ dùng rất nhẹ trong ablation; run chính nên tắt hoặc gần như tắt. |
| Lighting/autocontrast rất nhẹ hoặc tắt | Mô phỏng thay đổi ánh sáng/camera. Ảnh mẫu: [08_lighting_autocontrast](augmentation_examples/08_lighting_autocontrast/). | Có thể giúp chống lệch ánh sáng, nhưng dễ làm sai màu chín/chưa chín. |
| Random erasing thận trọng | Che một vùng nhỏ trong crop. Ảnh mẫu: [09_random_erasing](augmentation_examples/09_random_erasing/). | Chỉ nên dùng trong thực nghiệm phụ; run chính nên tắt nếu vết/đốm trên vỏ là tín hiệu class. |
| Normalization | Pixel được scale về tensor và chuẩn hóa bằng mean/std. Ảnh mẫu: [11_normalization_visualized](augmentation_examples/11_normalization_visualized/). | Ổn định phân phối đầu vào, giúp gradient và huấn luyện ổn định hơn. |
| Val/test preprocessing | Val/test chỉ dùng crop object, resize/pad và normalization, không dùng augmentation ngẫu nhiên. | Đảm bảo metric phân loại công bằng và tái lập được. |

## Các Phép Đã Loại Bỏ Khỏi Nhánh Phân Loại

Các phép dưới đây phù hợp hơn với detection hoặc multi-object training, nhưng không còn là khuyến nghị cho classification-only:

- Mosaic: ghép nhiều ảnh/object vào một ảnh. Với phân loại crop đơn, cách này tạo nhãn hỗn hợp không còn rõ ràng.
- CutMix: chèn vùng ảnh từ mẫu khác. Với xoài, vùng màu/vết bệnh bị trộn có thể làm sai tín hiệu độ chín.
- Targeted copy-paste: dán object vào ảnh khác. Hữu ích cho detection/count, nhưng classifier chỉ cần crop object thật từ bbox.
- Copy-paste nhiều object: không cần thiết khi mỗi bbox đã được chuyển thành một crop sample riêng.
- Batch-mix/mixup mặc định: hiện bị tắt trong classification-only để giữ nhãn class rõ ràng. Nếu dùng, chỉ nên dùng như ablation riêng.

## Logic Dữ Liệu Mới

Trước đây, nếu một ảnh có nhiều bbox, dataset chỉ chọn object chính để crop. Điều này làm mất các object còn lại trong bài toán phân loại. Nhánh này thay đổi mặc định:

1. Đọc toàn bộ bbox hợp lệ trong label YOLO.
2. Với classification-only, mỗi bbox trở thành một `MangoSample` riêng.
3. Crop theo bbox của chính sample đó, không phải luôn theo bbox lớn nhất.
4. Label phân loại là class của bbox đang được crop.
5. `class_counts`, `labels()` và rare-class repeat đều tính theo crop sample chính, không tính lặp toàn bộ object trong ảnh.

Điều này giúp tận dụng ảnh nhiều object mà không cần tạo ảnh synthetic bằng mosaic/cutmix/copy-paste.

## Cấu Hình Khuyến Nghị

Với dữ liệu xoài, nên ưu tiên các biến đổi giữ nguyên tín hiệu màu sắc:

- `resize_mode=pad`
- `crop_margin_ratio` khoảng `0.05` đến `0.10`
- `brightness=0.0`
- `contrast=0.0`
- `saturation=0.0`
- `hue=0.0`
- `lighting_probability=0.0`
- `random_erasing_probability=0.0`
- `random_affine_degrees` khoảng `3` đến `6`
- `random_affine_translate` khoảng `0.02` đến `0.04`
- `random_affine_scale_min` khoảng `0.94` đến `0.97`
- `horizontal_flip_probability=0.5`
- `vertical_flip_probability` rất thấp, khoảng `0.0` đến `0.02`
- `rotate90_probability` thấp, khoảng `0.0` đến `0.06`

Trong code hiện tại, khi `model_type=vit_registers`, train sẽ tự tắt `batch_mix_probability`, `mosaic_probability`, `mixup_probability`, `cutmix_probability` và `copy_paste_probability` để tránh nhầm sang pipeline detection.

