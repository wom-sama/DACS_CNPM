# Data Augmentation and Preprocessing Cho Phân Loại Xoài

Tài liệu này mô tả pipeline tiền xử lý và tăng cường dữ liệu cho nhánh `classification-only-research`. Mục tiêu của nhánh này là phân loại độ chín/chất lượng xoài từ ảnh crop object, không tối ưu detection, bbox, objectness, count head hay quality head.

Luật ưu tiên vẫn giữ nguyên: không dùng pretrain, không dùng backbone pretrained, không dùng checkpoint ngoài repo.

Thư mục ảnh minh họa: [augmentation_examples](augmentation_examples/)

## Bảng Tóm Tắt

| Transformation type | Details | Vai trò trong phân loại |
|---|---|---|
| Kiểm tra ảnh gốc + YOLO bbox | Đọc ảnh gốc và vẽ bbox từ nhãn YOLO. Ảnh mẫu: [00_original_with_yolo_boxes](augmentation_examples/00_original_with_yolo_boxes/). | Dùng để kiểm tra chất lượng nhãn trước khi crop object. |
| Resize + padding | Giữ tỉ lệ ảnh, resize vào khung cố định của run, hiện khuyến nghị là 224x224; ảnh minh họa được render ở 416x416 để dễ quan sát. Ảnh mẫu: [01_resize_pad_416](augmentation_examples/01_resize_pad_416/). | Tránh méo hình quả xoài, giữ hình dạng tự nhiên cho classifier. |
| Crop object chính | Crop quanh bbox object chính với margin nhỏ rồi resize/pad. Ảnh mẫu: [02_crop_primary_object](augmentation_examples/02_crop_primary_object/). | Giảm nhiễu nền, giúp mô hình tập trung vào quả xoài. |
| Object-level crops từ ảnh nhiều object | Mỗi bbox hợp lệ trong ảnh nhiều object được tách thành một sample phân loại riêng. Ảnh mẫu: [10_object_level_crops_from_multi_object_images](augmentation_examples/10_object_level_crops_from_multi_object_images/). | Tận dụng thêm object vốn từng bị bỏ qua khi chỉ crop object chính. Đây là thay đổi quan trọng cho classification-only. |
| Auto rare-class crop margin | Class nào có `class_target_scale >= 1.5` sẽ được tự động crop rộng hơn, ví dụ margin gốc `0.08` có thể tăng tới trần `0.16`. | Tăng ngữ cảnh quanh object cho các class thiếu dữ liệu mà không hard-code class 1 và không tạo ảnh synthetic. |
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
- `class_crop_margin_scale_threshold=1.5`
- `class_crop_margin_max_ratio=0.16`
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

Với `canbang.yaml`, hệ thống đọc `auto_repeat_factors` và dùng cùng scale đó để xác định class cần được ưu tiên. Class có scale dưới `--class-crop-margin-scale-threshold` giữ margin gốc; class có scale bằng hoặc vượt ngưỡng được crop rộng hơn nhưng bị chặn bởi `--class-crop-margin-max-ratio`. Cách này tổng quát hơn việc nhắm riêng class 1.

## Hybrid CNN Stem Trong Nhánh Phân Loại

Nhánh classification-only vẫn giữ phần hybrid CNN stem. Đây là thành phần quan trọng với tập dữ liệu nhỏ vì nó giúp mô hình học đặc trưng cục bộ trước khi đưa vào Transformer.

Pipeline hiện tại:

```text
Object crop image
-> CNN stem
   Conv 3x3 + BatchNorm + GELU + MaxPool
   Conv 3x3 + BatchNorm + GELU + MaxPool
   Conv 3x3 + BatchNorm + GELU + MaxPool
-> Patch embedding
-> CLS token + register tokens + patch tokens
-> Transformer encoder
-> Mean(CLS + register tokens)
-> Linear classifier
-> Class logits
```

Điểm khác với mô hình ensemble CNN + ViT song song: mô hình hiện tại không chạy một nhánh ResNet riêng rồi concat với nhánh ViT. CNN stem nằm nối tiếp trước patch embedding. Vì vậy, ViT nhận feature map đã được lọc bởi CNN, không nhận trực tiếp pixel thô.

## Nhược Điểm Và Cách Giảm

| Nhược điểm | Cách giảm trong nhánh hiện tại |
|---|---|
| CNN stem downsample 8 lần, có thể mất chi tiết nhỏ trên vỏ xoài. | Run chính dùng 224 để so sánh công bằng với baseline cũ; giảm rủi ro mất chi tiết bằng object-level crop, crop margin hợp lý và không dùng augmentation làm mờ/méo màu. Nếu cần, chạy thêm ablation 416 sau. |
| BatchNorm trong CNN stem có thể kém ổn nếu batch quá nhỏ. | Input 224 cho phép batch lớn hơn, ví dụ batch 64, giúp thống kê BatchNorm ổn hơn so với input lớn. |
| Mô hình không có nhánh CNN global pooling riêng như ensemble ResNet+ViT. | Với dataset nhỏ, kiến trúc nối tiếp an toàn hơn và ít tham số hơn; nếu cần cho bài báo, chạy ablation sau với nhánh CNN fusion riêng thay vì thay ngay run chính. |
| Input 224 có thể kém hơn 416/512 nếu class phụ thuộc vào vết rất nhỏ. | Xem 224 là run chính để so sánh; nếu per-class recall còn nghẽn ở class cần chi tiết vỏ, chạy thêm ablation 416 với cùng logic object crop. |
| Biến đổi màu có thể phá tín hiệu độ chín. | Giữ brightness/contrast/saturation/hue/lighting bằng 0 trong run chính; chỉ dùng cho ablation riêng. |

## Lệnh Train Khuyến Nghị Hiện Tại

Lệnh dưới đây là cấu hình classification-only từ đầu đến cuối, không dùng pretrain, giữ CNN stem, dùng object-level crops, input 224 để so sánh trực tiếp với baseline `mango_hybrid_224`, và tắt toàn bộ batch composition không phù hợp với phân loại.

```powershell
cd D:\DataAI\AIEx\TRKH
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
$env:TRKH_AMP_DTYPE='bf16'
$env:OMP_NUM_THREADS='6'
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING='1'
$env:TRKH_ALLOW_WINDOWS_PIN_MEMORY='1'
$env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS='1'

D:\DataAI\.venv\Scripts\python.exe -m trkh.training.train `
  --data D:\DataAI\AIEx\dataset\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --run-name mango_cls_224_cnnstem_vitreg_5cls_objectcrops_rarecrop_local_v3 `
  --output-dir runs --disable-resume --seed 42 `
  --model-type vit_registers --image-size 224 --patch-size 16 `
  --stem-channels 32 --head-pooling cls_register_mean `
  --embed-dim 256 --depth 8 --num-heads 8 --num-registers 4 `
  --dropout 0.10 --drop-path-rate 0.10 `
  --batch-size 64 --grad-accum-steps 1 --epochs 0 --scheduler-total-epochs 140 --patience 70 `
  --learning-rate 3e-4 --min-learning-rate 1e-6 --weight-decay 0.05 --warmup-epochs 8 `
  --grad-clip-norm 0.75 --max-nonfinite-grad-steps 4 `
  --num-workers 8 --eval-num-workers 4 --train-image-cache-mb 0 --eval-image-cache-mb 0 `
  --class-weight-mode sqrt_inverse --focal-loss-gamma 2.0 --focal-loss-mix 0.20 `
  --label-smoothing 0.015 --ldam-scale 18.0 --best-metric macro_f1 `
  --resize-mode pad --crop-margin-ratio 0.08 `
  --class-crop-margin-scale-threshold 1.5 --class-crop-margin-max-ratio 0.16 `
  --brightness 0.0 --contrast 0.0 --saturation 0.0 --hue 0.0 --lighting-probability 0.0 `
  --random-erasing-probability 0.0 `
  --random-affine-degrees 3 --random-affine-translate 0.02 --random-affine-scale-min 0.96 `
  --horizontal-flip-probability 0.5 --vertical-flip-probability 0.0 --rotate90-probability 0.02 `
  --batch-mix-probability 0.0 --mosaic-probability 0.0 --mixup-probability 0.0 `
  --cutmix-probability 0.0 --copy-paste-probability 0.0
```

Nếu VRAM không đủ ở input 224, giảm `--batch-size 64` xuống `48` hoặc `32`. Không nên bật `--disable-cnn-stem` cho run chính; chỉ dùng flag đó cho ablation để chứng minh vai trò của hybrid CNN stem.

## Benchmark Num Workers

Trên máy local hiện tại:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM
- CPU: Intel Core i7-12700H, 14 core / 20 thread
- RAM: 16GB
- Batch size benchmark: 64
- Input: 224
- Model: `vit_registers` + CNN stem
- Dataset benchmark: `D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops`

Kết quả đo end-to-end gồm DataLoader wait, GPU transfer, forward, backward và optimizer step. Với dataset `cls_crops`, từ 4 worker trở lên DataLoader gần như không còn là nút nghẽn; thời gian chủ yếu nằm ở phần compute của mô hình.

| num_workers | mean batch seconds | data wait fraction | samples/second | total seconds including startup | Nhận xét |
|---:|---:|---:|---:|---:|---|
| 4 | 0.1843 | 0.14% | 347.29 | 17.55 | Tốt nhất trong phép đo, startup thấp nhất. |
| 6 | 0.1843 | 0.14% | 347.23 | 21.76 | Gần bằng 4 nhưng startup cao hơn. |
| 8 | 0.1845 | 0.15% | 346.81 | 27.27 | Không nhanh hơn 4 trên `cls_crops`. |
| 10 | 0.1850 | 0.19% | 346.02 | 32.61 | Chậm hơn nhẹ, startup cao. |
| 12 | 0.1858 | 0.17% | 344.51 | 38.33 | Không có lợi. |
| 14 | 0.1851 | 0.19% | 345.75 | 43.79 | Không có lợi. |
| 16 | 0.1857 | 0.23% | 344.65 | 49.72 | Startup nặng nhất, không tăng throughput. |

Vì vậy lệnh train local hiện đặt `--num-workers 4 --eval-num-workers 4`. Khi dùng `cls_crops`, tăng lên 6/8/10/12/14/16 không cải thiện throughput và chỉ làm startup worker nặng hơn. Nếu đổi lại sang dataset YOLO crop online hoặc bật augmentation nặng hơn, cần benchmark lại.

Lệnh benchmark có thể chạy lại khi đổi máy hoặc đổi dataset:

```powershell
cd D:\DataAI\AIEx\TRKH
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
$env:TRKH_AMP_DTYPE='bf16'
$env:OMP_NUM_THREADS='6'
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING='1'
$env:TRKH_ALLOW_WINDOWS_PIN_MEMORY='1'
$env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS='1'

D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.benchmark_num_workers `
  --data D:\DataAI\AIEx\image_baseline_experiments\data\cls_crops\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --workers 4,6,8,10,12,14,16 `
  --batch-size 64 --image-size 224 `
  --warmup-batches 3 --measure-batches 20 `
  --torch-threads 6
```
