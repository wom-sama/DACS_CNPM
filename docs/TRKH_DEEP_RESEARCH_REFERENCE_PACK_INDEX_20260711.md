# TRKH Deep-Research Reference Pack - 2026-07-11

## Mục đích

Gói này là snapshot tự chứa để dùng cùng
`TRKH_DEEP_RESEARCH_PROBLEM_BRIEF_20260711.md`. Nó cung cấp tài liệu phương pháp,
code kiến trúc hiện tại, metadata hai dataset view, số liệu baseline, audit XAI,
các hướng gần nhất đã bị loại và metadata đối thủ AIDT. Mục tiêu là để một model
nghiên cứu khác có thể đề xuất hướng mới mà không lặp lại thí nghiệm cũ hoặc suy
diễn từ các metric không cùng protocol.

## Thứ tự đọc khuyến nghị

1. `01_docs/TRKH_DEEP_RESEARCH_PROBLEM_BRIEF_20260711.md`
2. `01_docs/TRKH_5CLASS_RESEARCH_JOURNAL.md`
3. `01_docs/TRKH_5CLASS_TOKEN_PRUNING_ARCHITECTURE.md`
4. `04_baseline_and_gate_evidence/`
5. `07_xai_visual_examples/`
6. `05_negative_methods/` và `06_latest_full_continuation/`
7. `03_current_model_code/` để kiểm tra tính khả thi của đề xuất
8. `08_competitor_aidt/` với các cảnh báo so sánh ở dưới

## Snapshot định lượng không được trộn protocol

| Mốc | Split/view | Macro F1 | Class-1 F1 | Class-1 P/R | Vai trò |
|---|---|---:|---:|---:|---|
| Raw TRKH keeper | `yolo_f/val=2606` | 0.8847 | 0.6860 | 0.6114/0.7815 | Baseline trainable hiện tại |
| Runtime softboost | `yolo_f/val=2606` | 0.8887 | 0.7030 | 0.6480/0.7682 | Diagnostic guard tốt nhất hiện tại |
| Raw TRKH final | locked test | 0.8865 | 0.6667 | xem evidence | Final raw test reference |
| Full continuation 20260711 | `yolo_f/val=2606` | 0.8868 | 0.6817 | 0.5931/0.8013 | Bị loại, không mở test |
| Continuation + softboost | `yolo_f/val=2606` | 0.8770 | 0.6373 | 0.6528/0.6225 | Bị loại, verifier phá recall |

Class 1 là `Xoai_Song_ChuaNhe_CoNguyCo`. Mọi đề xuất phải đồng thời bảo vệ true
class-1 recall và kiểm soát false positive `0/2/4→1`; tăng một phía bằng cách phá
phía còn lại không qua gate.

## Cấu trúc gói

### `01_docs`

Brief vấn đề, journal đầy đủ, TODO/gate history, command packet, mô tả kiến trúc,
dataset audit, cartography, expert-diversity/TTA, internal SSL, pretrained
retrieval và quyết định no-pretrain. Journal/TODO lớn vì chứa cả negative-result
history, không phải tài liệu quảng bá.

### `02_dataset_metadata`

Chứa YAML/stats của `class_f` và `yolo_f` cùng summary sinh từ manifest. Không
chứa ảnh hay manifest từng dòng. Hai view dùng cùng năm class nhưng khác crop và
wide-context representation; raw dataset là bất biến.

### `03_current_model_code`

Snapshot code cần để đọc kiến trúc hybrid CNN-ViT no-pretrain, token pruning,
attention views, loss, loader, launcher và XAI. Đây không phải một source archive
đầy đủ để build độc lập; commit nguồn đối chiếu là
`7020f0393a8d4260fc7c37372268ae9f27086224`, cộng các thay đổi local có trong
snapshot này.

### `04_baseline_and_gate_evidence`

Config/history/metric của raw keeper, runtime softboost, final raw test,
class-1 correction budget, current smoke gate và external-support taxonomy.
Test chỉ là final reference đã khóa, không được dùng để chọn hướng mới.

### `05_negative_methods`

Evidence gọn cho SPT, GPSA và LSA. Các route này đã được precheck/smoke/audit và
bị loại trên keeper hiện tại. Không đề xuất lại chỉ bằng sweep LR, layer, gate,
temperature, shift hoặc số batch.

### `06_latest_full_continuation`

Evidence compact của full continuation launcher-fixed: train history/config,
raw metric, frozen-verifier metric, balanced boundary review, XAI transition và
một số visualization đại diện. Không có checkpoint và không có test result.

### `07_xai_visual_examples`

Audit JSON/Markdown cùng một số class-1 FN và class-1 FP đại diện từ current
softboost. XAI chỉ là bằng chứng chẩn đoán, không chứng minh causal mechanism.
Kết quả lặp lại: far background perturbation yếu, object color mạnh, còn
stem/endpoint/border/crop cue tồn tại cục bộ.

### `08_competitor_aidt`

README, source model/train/data và metadata run của AIDT. File metric gốc trong
AIDT có thể dùng class order, crop view hoặc mapping khác với strict TRKH
`yolo_f` evaluation. Vì vậy không so trực tiếp một số trong AIDT `metrics.json`
với bảng trên nếu chưa kiểm tra sample alignment, class order và split. Journal
ghi cả các kết quả remap/alignment nghiêm ngặt; đây là lý do brief đánh dấu phép
so sánh hiện tại chưa hoàn toàn apples-to-apples.

### `09_followup_20260711`

Refresh sau hai báo cáo deep-research 14/15. Thư mục này chứa chính hai báo cáo
đầu vào, method-closure summaries mới (validation ceiling, CoAtNet,
EfficientFormerV2, LeFF, concurrent coupling, LDR/ALDR và transition-noise),
cùng infrastructure wrappers đã preflight cho full train-to-audit, TensorRT
export và video inference classification-only. Đọc thư mục này trước khi đề
xuất thêm stock hybrid, confidence/noise head hoặc paired-view fusion.

Snapshot code trong `03_current_model_code` cũng được refresh với các wrapper
một dòng, inference TensorRT classification-only, và các audit readiness mới.
Final test vẫn bị khóa bởi independent validation promotion gate; không được
dùng các final-test artifact để retune model.

## Câu hỏi nên giao cho model deep research

1. Xác định trần hiệu năng thực tế khi class 1 chỉ có 151 validation và 73 test
   samples, đồng thời có overlap ngữ nghĩa với class 0/2/4.
2. Đề xuất representation tạo cue interior-surface/boundary mới, không chỉ tăng
   foreground focus hoặc locality chung.
3. Chứng minh proposal khác SPT/GPSA/LSA, objectness, paired-view fusion,
   patch-verifier, current-embedding kNN/contrastive và các KD route đã bị loại.
4. Cho pre-smoke diagnostic độc lập, failure gate, ablation tối thiểu, VRAM dự
   kiến và kế hoạch hội tụ trong tối đa 30 epochs.
5. Thiết kế evaluation có seed variance, bootstrap CI, paired significance và
   locked test; không tune test hoặc sửa raw data.

## Các loại tệp cố ý không đóng gói

- Checkpoint `.pt` và optimizer/EMA state.
- Raw image dataset và full row-level dataset manifests.
- Superseded smoke/probe plots không còn giá trị đối chiếu.
- Prediction CSV lớn khi metric/changed-case summary đã đủ cho câu hỏi nghiên cứu.
- Tài liệu PDF bên ngoài; URL paper gốc nằm trong journal và brief.

## Tính toàn vẹn và trạng thái GitHub

`_MANIFEST.json` trong ZIP liệt kê từng entry, kích thước và SHA-256. File
`.sha256.txt` bên cạnh ZIP xác minh toàn archive. Tại thời điểm đóng gói, local
branch và upstream cùng commit `7020f0393a8d4260fc7c37372268ae9f27086224`, nhưng
brief/journal/command packet và run evidence mới chưa được push đầy đủ. Gói ZIP
này mới là snapshot hoàn chỉnh để chuyển sang model nghiên cứu khác; không nên
giả định GitHub đã chứa cùng nội dung.

## Snapshot refresh cuối ngày 2026-07-11

- Gói refresh có `266` payload cộng `_MANIFEST.json`, khoảng `16.82 MiB`
  uncompressed trước khi nén; `09_followup_20260711` có `63` tệp chọn lọc.
- Follow-up bổ sung hai deep-research report, validation information ceiling,
  và bằng chứng đóng phương pháp CoAtNet-Nano, EfficientFormerV2-S0, LeFF,
  concurrent local-global, LDR/ALDR, cùng transition-noise readiness.
- Snapshot code chứa full train-to-audit wrapper đã preflight, wrapper export
  TensorRT, wrapper test video và hỗ trợ classification-only TensorRT runtime.
- Không có checkpoint, ONNX, TensorRT engine hay raw dataset trong ZIP. Validation
  chọn mô hình; final test vẫn chỉ chạy sau promotion gate độc lập.
- Recipe và file lệnh full-train chỉ được cập nhật khi checkpoint mới thực sự
  vượt locked-validation gate và được chọn. Smoke, oracle hoặc kết quả tune theo
  test không đủ điều kiện promotion.
- SHA-256 toàn archive nằm trong file companion `.sha256.txt`; không nhúng hash
  archive vào chính README trong ZIP để tránh quan hệ tự tham chiếu.
- Archive cuối có `267` entries tính cả manifest, kích thước `8,951,829` byte
  (`8.537 MiB`). SHA-256 companion là
  `82be9ffd37b4d6361e5cddf11313a5f065dc64d524228324ebe260c48cd49385`.
