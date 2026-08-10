# TRKH acoustic-firmness pilot R0 — 2026-07-29

Status: `planning_only` — chưa thu dữ liệu, chưa cho phép train/GPU, chưa thay đổi mô hình RGB hiện tại.

## 1. Câu hỏi khoa học

Một phép đo âm học có kiểm soát có cung cấp tín hiệu cơ học độc lập để phân biệt class 1 với các mẫu RGB đang bị hút nhầm từ class 0/2/4 hay không?

Đây là phép kiểm tra **giá trị thông tin trước giá trị kiến trúc**. Không xây encoder âm thanh sâu trước khi tín hiệu vượt qua các cổng lặp lại và đối chứng dưới đây.

## 2. Giả thuyết khóa trước

- `H1`: tần số cộng hưởng và độ tắt dần của cú gõ có độ lặp lại đủ tốt giữa các lần đo cùng quả.
- `H2`: sau khi giữ cố định dự đoán RGB, tín hiệu âm học loại được một phần false-positive vào class 1 mà vẫn giữ ít nhất 98% true-positive class 1.
- `H0-control`: lợi ích không được giải thích chỉ bằng khối lượng/kích thước, nguồn quả, thiết bị hoặc thứ tự đo.

## 3. Thiết kế hai pha

### F0 — khả năng đo lặp lại

- 20–30 quả, phủ nhiều kích thước và độ chín; chưa dùng để tuyên bố hiệu năng phân loại.
- Mỗi quả: 3 cú gõ tại vùng xích đạo × 2 phiên đo độc lập.
- Giữ cố định: đầu gõ, năng lượng gõ, khoảng cách và góc microphone, vị trí quả, phòng đo.
- Thu WAV mono PCM 48 kHz, tắt AGC/noise suppression nếu thiết bị cho phép; có 0,5 giây nền trước cú gõ.
- Ghi khối lượng, hai đường kính, nhiệt độ, nguồn/lô, thiết bị, người đo và vị trí gõ.

**Cổng F0**

- ICC của chỉ số độ cứng âm học `FI = f0^2 × mass^(2/3)` ≥ 0,75.
- CV trong cùng quả của `f0` ≤ 5%.
- Tỷ lệ bản ghi clipping, không bắt được onset hoặc SNR thấp ≤ 5%.
- Nếu không đạt: dừng nhánh âm học và sửa quy trình đo; không bù bằng mô hình phức tạp.

### A0 — kiểm định giá trị bổ sung

Chỉ mở khi F0 đạt. Mục tiêu ban đầu: khoảng 120 quả mới, ưu tiên ít nhất 24 quả/lớp; đơn vị chia fold là **fruit_id**, đồng thời khóa theo `source_id/session_id` để không rò rỉ các cú gõ của cùng quả.

- Giữ checkpoint RGB cố định; không tinh chỉnh trên tập pilot ở A0.
- Baseline 1: logits RGB.
- Baseline 2: RGB + khối lượng/kích thước.
- Candidate: RGB + các đặc trưng âm học khóa trước.
- Negative control: hoán vị đặc trưng âm học trong cùng nguồn/lô.
- Báo cáo riêng các cặp `0→1`, `2→1`, `4→1`, không chỉ macro-F1 tổng.

Đặc trưng A0 được khóa trước: `f0`, `FI`, Q-factor, hằng số tắt dần, spectral centroid và năng lượng theo các dải cố định. Mô hình đầu tiên chỉ là logistic/mixed-effects có regularization; chưa dùng CNN/Transformer/SSM cho waveform.

**Cổng A0**

- Giữ ≥ 98% true-positive class 1 của RGB.
- Lower bound bootstrap 95% của tỷ lệ loại restricted false-positive phải > 0.
- Vượt baseline metadata-only và shuffled-audio control.
- Không có đảo chiều hiệu ứng lớn theo nguồn/lô, người đo hoặc thiết bị.
- Báo cáo calibration và selective risk; không chọn threshold từ test niêm phong.

## 4. Ground truth cần bổ sung

Nhãn 5 lớp hiện tại vẫn được giữ, nhưng pilot nên đo thêm ít nhất một đại lượng vật lý sau lần ghi cuối:

- lực xuyên/độ cứng bằng penetrometer;
- TSS/Brix nếu có;
- điểm dập hoặc khả năng vận chuyển sau 24–48 giờ.

Không có đại lượng tham chiếu này, pilot chỉ chứng minh tương quan với nhãn chủ quan chứ chưa chứng minh cơ chế vật lý.

## 5. Nhánh tích hợp chỉ được mở sau A0

Thiết kế ưu tiên là late fusion có thể bỏ modality:

```text
RGB image ── TRKH encoder ─────────────── logits_rgb ───────────────┐
                                                                    ├─ logits_final
tap WAV ── physics features/tiny 1D encoder ─ reliability gate ─────┘
```

Residual âm học được zero-init; khi thiếu hoặc tín hiệu kém, gate bằng 0 để đầu ra quay chính xác về RGB. Đây mới là chỗ có thể tạo đóng góp kiến trúc mới, nhưng chỉ sau bằng chứng A0.

## 6. Trật tự ưu tiên nếu âm học thất bại

1. Ảnh cặp parallel/cross-polarized để tách phản xạ gương và phản xạ khuếch tán.
2. Theo dõi dọc cùng quả với firmness/TSS và outcome vận chuyển để mô hình hóa hazard/transition.
3. Không tiếp tục chồng thêm module RGB nếu chưa có giả thuyết lỗi và đối chứng mới.

## 7. Đường dẫn dữ liệu đề xuất

```text
D:\DataAI\AIEx\newdataset\trkh_acoustic_pilot_r1\
  metadata.csv
  wav\<fruit_id>\<session_id>\<tap_id>.wav
```

Dùng mẫu metadata đi kèm: `docs/TRKH_ACOUSTIC_FIRMNESS_CAPTURE_TEMPLATE_20260729.csv`.

## 8. Cơ sở thực nghiệm

- Nghiên cứu trên xoài Harumanis cho thấy fusion electronic-nose và acoustic firmness có thể cải thiện phân loại maturity/ripeness; phép đo dùng microphone, cú gõ kiểm soát và khối lượng quả: https://www.mdpi.com/1424-8220/12/5/6023
- Acoustic firmness của xoài thường dùng dạng tỷ lệ với `f^2 × mass^(2/3)`: https://www.sciencedirect.com/science/article/pii/S1350449521001055
- Đo firmness lặp lại theo thời gian đã được dùng để mô hình hóa biến đổi chất lượng xoài trong chuỗi vận chuyển: https://pubmed.ncbi.nlm.nih.gov/30524453/
- Cross-polarization là phương án dự phòng có cơ sở vì giảm glare/specular variability trong ảnh nông sản: https://www.sciencedirect.com/science/article/pii/S1537511016303099

