# Data Augmentation and Preprocessing

Tài liệu này tóm tắt các bước tiền xử lý ảnh và tăng cường dữ liệu đang được hỗ trợ trong dự án TRKH. Tất cả ví dụ ảnh bên dưới được tạo từ ảnh thật trong tập train, không dùng dữ liệu ngoài và không dùng bất kỳ pretrain nào.

Thư mục ảnh minh họa: [augmentation_examples](augmentation_examples/)

Ghi chú: một số ảnh minh họa được đặt xác suất biến đổi = 1 hoặc dùng biến đổi mạnh hơn để dễ nhìn thấy tác động. Khi train chính thức, các xác suất và cường độ có thể được đặt nhẹ hơn để tránh làm hỏng tín hiệu màu sắc, hình dạng và bbox.

## Bảng Tóm Tắt

| Nhóm / phương pháp | Chi tiết | Mục đích | Ảnh minh họa |
|---|---|---|---|
| Ảnh gốc kèm YOLO bbox | Đọc ảnh gốc và vẽ bbox từ label YOLO để kiểm tra nhanh vị trí nhãn. | Xác nhận label đúng trước khi train; phát hiện lỗi bbox, class sai, ảnh thiếu nhãn. | [00_original_with_yolo_boxes](augmentation_examples/00_original_with_yolo_boxes/) |
| Resize + padding | Giữ tỉ lệ ảnh, scale vào khung 416x416 và pad phần thừa. Với lệnh 640 thì khung đích là 640x640. | Tạo đầu vào có kích thước cố định mà không làm méo trái xoài hay bbox. | [01_resize_pad_416](augmentation_examples/01_resize_pad_416/) |
| Crop primary object | Crop quanh object chính theo bbox và margin, sau đó resize về kích thước đầu vào. Thường dùng cho bài toán classification-only. | Giảm nhiễu nền, giúp mô hình phân loại tập trung vào quả xoài chính. | [02_crop_primary_object](augmentation_examples/02_crop_primary_object/) |
| Random affine | Xoay nhẹ, dịch chuyển, scale và shear trong giới hạn cấu hình. | Tăng khả năng chịu đựng với góc chụp, vị trí và kích thước object khác nhau. | [03_random_affine](augmentation_examples/03_random_affine/) |
| Horizontal flipping | Lật trái/phải theo xác suất cấu hình. | Tăng độ đa dạng hướng đặt quả xoài mà vẫn giữ logic nhãn. | [04_horizontal_flip](augmentation_examples/04_horizontal_flip/) |
| Vertical flipping | Lật trên/dưới với xác suất thấp. | Chỉ dùng nhẹ vì ảnh thật ít khi bị đảo ngược; giúp robustness nhưng không nên làm quá mạnh. | [05_vertical_flip](augmentation_examples/05_vertical_flip/) |
| Rotate90 | Xoay ảnh theo bội số 90 độ với xác suất thấp. | Mô phỏng hướng camera/ảnh bị xoay, hữu ích khi dữ liệu thu thập không đồng nhất. | [06_rotate90](augmentation_examples/06_rotate90/) |
| Color jitter | Hỗ trợ brightness, contrast, saturation và hue. | Tăng chịu đựng với điều kiện ánh sáng; với dữ liệu xoài, nên dùng rất nhẹ hoặc tắt nếu màu sắc là tín hiệu phân loại quan trọng. | [07_color_jitter](augmentation_examples/07_color_jitter/) |
| Lighting / autocontrast / sharpness | Biến đổi ánh sáng, tương phản tự động và độ nét. | Mô phỏng thay đổi độ sáng và camera; cần cảnh giác vì có thể làm lệch màu chín/chưa, hư/hỏng. | [08_lighting_autocontrast](augmentation_examples/08_lighting_autocontrast/) |
| Random erasing | Che một vùng nhỏ trong ảnh bằng màu trung bình/xám. | Tăng robustness khi object bị che khuất; với detection cần dùng thận trọng để không làm bbox mất ý nghĩa. | [09_random_erasing](augmentation_examples/09_random_erasing/) |
| Mosaic | Ghép 4 ảnh vào một canvas, cập nhật bbox theo vị trí mới. | Tạo ảnh nhiều object, tăng mật độ object và cải thiện detection khi tập gốc thiếu ảnh nhiều đối tượng. | [10_mosaic](augmentation_examples/10_mosaic/) |
| CutMix | Cắt một vùng từ ảnh khác và chèn vào ảnh hiện tại, đồng thời ghép label/bbox phù hợp. | Tăng đa dạng bối cảnh và số object trên ảnh; hữu ích cho detection nếu giữ bbox chính xác. | [11_cutmix](augmentation_examples/11_cutmix/) |
| Targeted copy-paste | Cắt object từ ảnh khác và dán vào ảnh hiện tại. Có thể tự động ưu tiên class có tỉ lệ tăng cường cao, ví dụ class có scale > 1.5. | Bù đắp class thiếu recall, đặc biệt khi một class ít object hoặc detection F1 thấp. | [12_targeted_copy_paste](augmentation_examples/12_targeted_copy_paste/) |
| Normalization | Pixel được đưa về dạng tensor, scale về [0,1], sau đó chuẩn hóa bằng mean/std. Ảnh minh họa là bản visualize lại tensor đã normalize. | Ổn định phân phối đầu vào, giúp tối ưu gradient và hội tụ tốt hơn. | [13_normalization_visualized](augmentation_examples/13_normalization_visualized/) |
| Class-aware repeat / scale | Tăng tần suất lấy mẫu hoặc cường độ augmentation theo class dựa trên độ mất cân bằng trong `canbang.yaml`. | Không phải một biến đổi hình ảnh riêng lẻ, nhưng ảnh hưởng trực tiếp đến số lần mô hình thấy class khó. | Không có thư mục riêng |
| Chia train/val/test | Train có thể dùng augmentation; val/test chỉ dùng resize/pad và normalization, không dùng random augmentation. | Đảm bảo metric val/test công bằng, không bị ảnh hưởng bởi biến đổi ngẫu nhiên. | Áp dụng trong pipeline |

## Cấu Hình Đang Được Ưu Tiên

Với dữ liệu xoài bị mất cân bằng và phân loại phụ thuộc mạnh vào màu sắc, các biến đổi màu như brightness, contrast, saturation, hue và lighting nên để rất nhẹ hoặc tắt trong run chính. Các biến đổi hình học và ghép ảnh nhiều object thường hữu ích hơn cho mục tiêu detection F1.

Các hướng đang ưu tiên:

1. Dùng resize/pad để giữ tỉ lệ thật của ảnh.
2. Dùng mosaic, cutmix và targeted copy-paste ở xác suất vừa phải để tạo ảnh nhiều object.
3. Tự động ưu tiên class khó bằng class-aware scale, rare-class repeat và targeted copy-paste.
4. Hạn chế biến đổi màu khi màu sắc là dấu hiệu phân biệt class.
5. Val/test chỉ dùng preprocessing tất định, không dùng augmentation ngẫu nhiên.

## Liên Hệ Với Mục Tiêu Mô Hình

- Macro F1 phân loại cần màu sắc, texture và độ chín được giữ ổn định.
- Detection F1 cần bbox chính xác, object đa dạng về vị trí/kích thước và có đủ ảnh nhiều object.
- Count head cần được thấy nhiều mẫu ảnh có 0, 1 và nhiều object; mosaic/cutmix/copy-paste giúp bổ sung trường hợp nhiều object.
- Các biến đổi làm mất thông tin bbox hoặc làm sai màu quả xoài cần được dùng nhẹ hơn so với biến đổi hình học/ghép ảnh.

## Danh Sách Thư Mục Ảnh Mẫu

Mỗi thư mục con trong [augmentation_examples](augmentation_examples/) hiện có 3 ảnh minh họa:

- [00_original_with_yolo_boxes](augmentation_examples/00_original_with_yolo_boxes/)
- [01_resize_pad_416](augmentation_examples/01_resize_pad_416/)
- [02_crop_primary_object](augmentation_examples/02_crop_primary_object/)
- [03_random_affine](augmentation_examples/03_random_affine/)
- [04_horizontal_flip](augmentation_examples/04_horizontal_flip/)
- [05_vertical_flip](augmentation_examples/05_vertical_flip/)
- [06_rotate90](augmentation_examples/06_rotate90/)
- [07_color_jitter](augmentation_examples/07_color_jitter/)
- [08_lighting_autocontrast](augmentation_examples/08_lighting_autocontrast/)
- [09_random_erasing](augmentation_examples/09_random_erasing/)
- [10_mosaic](augmentation_examples/10_mosaic/)
- [11_cutmix](augmentation_examples/11_cutmix/)
- [12_targeted_copy_paste](augmentation_examples/12_targeted_copy_paste/)
- [13_normalization_visualized](augmentation_examples/13_normalization_visualized/)

