# TRKH 5-Class: Hồ sơ vấn đề hiện tại cho nghiên cứu chuyên sâu

Ngày chốt ngữ cảnh: **2026-07-11**
Nhánh đang phát triển: **`classification-only-research`**
Repo chính: **`D:\DataAI\AIEx\TRKH`**

## 1. Mục tiêu và các ràng buộc không được bỏ qua

Mục tiêu cuối cùng là một mô hình TRKH lai CNN-Transformer, train từ đầu hoặc ít
nhất không phụ thuộc backbone pretrained ở thời điểm triển khai, hội tụ trong
không quá 30 epoch và đạt F1 trên `0.98` cho **từng lớp**. Mô hình phải được đánh
giá công bằng trên cùng split, cùng ánh xạ lớp và cùng pipeline dữ liệu với đối
thủ.

Các ràng buộc hiện tại:

- Không bổ sung dữ liệu thô mới.
- Không sửa ảnh, nhãn hoặc cấu trúc raw dataset.
- Chỉ được thay đổi model, loss, sampling, augmentation, cách dùng metadata và
  các artifact train sinh ra ngoài raw dataset.
- Không dùng test để chọn kiến trúc, hyperparameter, threshold hoặc router.
- Mọi smoke/probe phải dùng full validation `2606` object trước khi được xem là
  bằng chứng chọn model.
- Phần cứng chính: RTX 4060 Laptop 8 GiB, RAM 16 GiB, i7-12700H.
- Recipe phải khả thi với batch hiện tại khoảng `32`, gradient accumulation `2`,
  image size `256` và tối đa 30 epoch.

## 2. Snapshot định lượng hiện tại

### 2.1 Keeper TRKH no-pretrain

Checkpoint:
`runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`

Keeper kế thừa model `mango_cls_256_5class_hardneg_maskfix_v4_30e`, sau đó
fine-tune thêm hai epoch với teacher-focus-binary, pairwise routing, bbox spatial
fusion, metric learning và boundary-band foreground dropout. Model có khoảng
`7.25M` tham số.

| Split/path | Macro F1 | Class-1 F1 | Class-1 P/R |
|---|---:|---:|---:|
| Full val, raw keeper | 0.8847 | 0.6860 | 0.6114 / 0.7815 |
| Full val, runtime patch-linear softboost | **0.8887** | **0.7030** | 0.6480 / 0.7682 |
| Locked test, raw keeper | 0.8865 | 0.6667 | 0.5978 / 0.7534 |

F1 theo lớp của runtime softboost trên validation:

| Class | Tên lớp | Support | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | `Xoai_Song_Chua_KhoDap` | 549 | 0.9306 | 0.9035 | 0.9168 |
| 1 | `Xoai_Song_ChuaNhe_CoNguyCo` | 151 | 0.6480 | 0.7682 | **0.7030** |
| 2 | `Xoai_Chin_NgotThanh_DeDap` | 544 | 0.8832 | 0.9449 | 0.9130 |
| 3 | `Xoai_ChinGia_NgotGat_KhongVanChuyen` | 712 | 0.9849 | 0.9171 | 0.9498 |
| 4 | `Xoai_Hu_KhongAnDuoc` | 650 | 0.9615 | 0.9600 | 0.9607 |

Như vậy, mục tiêu `>0.98` chưa đạt ở bất kỳ lớp nào; lớp 1 là bottleneck lớn
nhất nhưng không phải vấn đề duy nhất.

### 2.2 Ngân sách sửa lỗi lớp 1

Runtime softboost hiện có:

- `TP=116`
- `FP=63`
- `FN=35`
- F1 lớp 1: `0.7030`

Ngân sách oracle tối thiểu, chỉ để hình dung quy mô vấn đề:

- F1 `0.75`: cần ít nhất 13 FN được cứu, hoặc tổ hợp tương đương.
- F1 `0.80`: cần ít nhất 27 FN được cứu.
- F1 `0.85`: phải cứu cả 35 FN và loại ít nhất 10 FP.
- F1 `0.90`: phải cứu cả 35 FN và loại ít nhất 30 FP.
- F1 `0.98`: phải cứu cả 35 FN và loại ít nhất **57/63 FP**.

Đây là lý do các thay đổi chỉ sửa 2-5 mẫu không thể giải quyết mục tiêu.

### 2.3 External/pretrained upper bounds

Trên full `yolo_f/val` đã align theo `sample_index`:

| Nguồn | Macro F1 | Class-1 F1 | Ghi chú |
|---|---:|---:|---|
| AIDT aligned prediction | 0.9088 | 0.7169 | Pretrained/external diagnostic |
| Top-5 TTA teacher | 0.9139 | 0.7383 | Pretrained/external diagnostic |
| TRKH + Top-5 validation ensemble | 0.9166 | 0.7500 | Chỉ là upper bound, không deployable |

Artifact AIDT gốc `resnet50_vit_b16_5class` lại dùng `class_f`, thứ tự lớp khác,
112.7M tham số và hai backbone pretrained. Stored test của nó chỉ đạt macro/class-1
`0.8841/0.6225`, thấp hơn TRKH test raw `0.8865/0.6667`. Vì vậy cụm từ
“đánh bại AIDT” hiện chưa có một protocol so sánh duy nhất, hoàn toàn đồng nhất.

## 3. Các vấn đề ở dữ liệu

### D1. Chưa có kết luận tuyệt đối về dataset canonical

Hai nguồn còn được xem là có khả năng đúng nhất:

- `D:\DataAI\AIEx\newdataset\class_f`
- `D:\DataAI\AIEx\newdataset\yolo_f`

`class_f` là classification-folder crop; `yolo_f` giữ ảnh nguồn, bbox và có thể
sinh nhiều object crop từ một ảnh. Hai nguồn không chỉ khác format mà còn khác
ngữ cảnh, coordinate frame, số object và cách lập split. Kết quả trên một nguồn
không được xem là trực tiếp tương đương với nguồn kia nếu chưa remap theo object
và `sample_index`.

**Cần nghiên cứu:** phương pháp xác định lineage/canonical split và kiểm chứng
mapping mà không sửa raw data.

### D2. Lớp 1 thiếu mẫu và nằm giữa các biên ngữ nghĩa

Validation chỉ có 151 mẫu lớp 1, so với 544-712 mẫu ở các lớp lớn. Trên train,
lớp 1 chỉ xuất hiện trong 517 ảnh nguồn. Về hình ảnh, lớp 1 chồng lấn mạnh với:

- lớp 0 qua màu xanh, độ chua và bề mặt chưa chín;
- lớp 2 qua vùng vàng/chín không đồng đều;
- lớp 4 qua đốm, hư hỏng, ánh sáng vàng và texture bất thường.

Nhiều class-1 FN mà cả AIDT, Top-5, EffV2 và DINO đều đoán `0/2/4`. Điều này
gợi ý label boundary hoặc thiếu cue quan sát được, không chỉ là model yếu.

### D3. Không có thông tin về độ tin cậy nhãn và trần Bayes/annotator

Hiện chưa có:

- annotation guideline đủ chi tiết cho các biên `0↔1`, `1↔2`, `1↔4`;
- số annotator và inter-rater agreement;
- confidence theo nhãn;
- nhãn ordinal/attribute phụ được xác nhận bởi chuyên gia;
- ước lượng irreducible label noise hoặc Bayes error.

Nếu cùng một ảnh không thể được con người gán ổn định, F1 `0.98` cho mọi lớp có
thể vượt trần thông tin của dataset. Cần định lượng điều này thay vì giả định.

### D4. Một ảnh nguồn có thể chứa nhiều object và nhiều nhãn

Audit `yolo_f/train`:

- 8064 ảnh nguồn;
- 9215 object;
- 691 ảnh multi-object;
- 144 ảnh mixed-label;
- lớp 1 có trong 115 ảnh multi-object, trong đó 102 ảnh mixed-label.

Validation chỉ có 29 ảnh multi-object và chúng đều same-label thuộc lớp 0/3;
test là `1267/1267` single-object. Do đó các signal như same-source consistency,
global context hoặc layout multi-object không có support validation/test tương
xứng, và còn có thể truyền nhãn sai giữa các object cùng ảnh.

### D5. Context rộng vừa hữu ích tiềm năng vừa là shortcut

`yolo_f` cho phép giữ nền rộng hơn và vị trí bbox. Tuy nhiên:

- source-context branch, paired full/crop view, crop-margin sweep và simple
  context fusion đã không cải thiện keeper;
- background blur/gray thường làm xác suất thay đổi rất ít;
- stem, cành, padding, crop border và endpoint vẫn xuất hiện trong Grad-CAM;
- cùng loại nền có thể tương quan với nguồn thu thập hoặc nhãn, không phải trạng
  thái quả.

Vấn đề chưa giải quyết là khai thác context như một **negative cue có điều kiện**
hoặc localization prior mà không cho model học source/crop shortcut.

### D6. Bbox có hai coordinate frame dễ dùng sai

Raw `bbox` nằm trong frame ảnh nguồn; `crop_bbox` nằm trong frame sau crop/resize.
PiB audit cho thấy `crop_bbox + cls_register_mean` phản ánh foreground đúng hơn,
nhưng đổi token prior của keeper từ `bbox` sang `crop_bbox` lại giảm macro/class-1
từ `0.8847/0.6860` xuống `0.8822/0.6763`.

Điều này cho thấy:

- coordinate correctness không đồng nghĩa với downstream usefulness;
- model/checkpoint có thể đã thích nghi với semantics cũ của `bbox`;
- cần phân biệt bbox dùng để audit, bbox dùng để prune và bbox dùng để fuse.

### D7. Train/validation/test không đồng nhất về source structure

Train có nhiều mixed-label multi-object source, val gần như không có, test hoàn
toàn single-object. Một phương pháp học tốt source-level context trên train có
thể không generalize. Đây cũng là lý do OOF phải group theo source khi cần,
nhưng prediction vẫn phải object-level.

### D8. Raw data bất biến

Không thể:

- thu thêm class-1;
- relabel các case đáng ngờ;
- xóa duplicate hoặc sửa bbox tại raw root;
- tạo split mới bằng cách di chuyển raw files.

Mọi giải pháp phải làm qua index/manifest/cache sinh ngoài raw dataset và phải
giữ khả năng quay lại mapping gốc.

## 4. Các vấn đề ở model và representation

### M1. Precision-recall lớp 1 bị khóa trong trade-off

Keeper raw có recall lớp 1 khá cao `0.7815` nhưng precision thấp `0.6114`.
Runtime softboost tăng precision lên `0.6480` nhưng giảm recall xuống `0.7682`.
Hầu hết router, threshold, centroid, kNN, teacher consensus và margin loss đã
lặp lại cùng trade-off: chặn `0/2/4→1` đồng thời làm tăng `1→0/2/4`.

### M2. Thiếu cue surface/boundary có khả năng transfer

Các embedding hiện tại phân biệt tốt lớp 3/4 nhưng không tạo margin ổn định cho
lớp 1. Các probe prototype, Mahalanobis, kNN và neighbor consistency đều thất
bại hoặc đảo chiều khi chuyển từ train queue sang validation.

Vấn đề cần giải quyết không phải thêm một classifier nhỏ trên cùng embedding,
mà là học representation mới cho:

- độ chín cục bộ không đồng nhất;
- texture/đốm có ý nghĩa so với defect giả;
- chuyển tiếp xanh-vàng;
- cue interior so với stem/edge/crop border.

### M3. Attention nhìn đúng object nhưng chưa nhìn đúng thuộc tính

Full-val PiB cho thấy top patch đã ở foreground cao (`~0.93`), nên “model chỉ
nhìn nền” không còn là chẩn đoán đầy đủ. XAI tổng hợp cho thấy:

- foreground attention mass khoảng `0.9068`;
- background perturbation mean chỉ `0.0048`;
- object desaturation drop khoảng `0.1227`;
- Grad-CAM/rollout vẫn thường bám border, stem, endpoint và vùng màu rộng.

Model định vị được quả nhưng chưa tách được thuộc tính quyết định lớp 1.

### M4. Register/CLS chưa tạo các expert độc lập

Nhiều XAI audit cho heatmap similarity giữa CLS và register khoảng
`0.9998`. Các token được kỳ vọng học góc nhìn khác nhau nhưng đang gần như đồng
dạng. Tăng token hoặc thêm head không giúp nếu không có objective buộc chúng học
thuộc tính bổ sung một cách ổn định.

### M5. Local inductive bias đơn giản không giải quyết bottleneck

Đã thử và loại trên keeper hiện tại:

- `BlockLocalPatchMixer`;
- SoftPool và MaxSoft stem pooling;
- Shifted Patch Tokenization residual;
- GPSA-style gated relative-position attention;
- Locality Self-Attention: smoothing, sharpening và diagonal masking.

LSA full-val đặc biệt cho thấy self-attention diagonal mass chỉ
`0.0026-0.0044`; diagonal suppression 50% không đổi một prediction. Sharpening
làm nhiều harm hơn correction. Vì vậy locality/temperature không phải nút thắt
đơn giản.

### M6. Adapter smoke có thể đánh giá thấp kiến trúc cần train từ đầu

Các module mới thường được zero-init để resume checkpoint mà không đổi logits,
sau đó train 60-120 batch. Đây là gate tiết kiệm tài nguyên nhưng có hạn chế:

- backbone cũ có thể chống lại representation mới;
- optimizer học gate gần 0 và giữ nghiệm cũ;
- module cần co-adaptation từ đầu sẽ bị đánh giá thấp;
- full from-scratch ablation lại tốn nhiều thời gian và khó so sánh.

Cần tiêu chí khoa học để quyết định phương pháp nào xứng đáng full retrain, thay
vì vô hạn adapter smoke hoặc vô hạn full train.

### M7. Objective hiện tại phức tạp và có thể xung đột

Recipe tốt nhất đang kết hợp:

- LDAM-focal classification;
- teacher-focus binary;
- pairwise margin routing;
- head/patch metric learning;
- bbox spatial fusion;
- boundary-band foreground dropout;
- attention crop/drop views.

Chưa có phân tích gradient conflict hoặc loss contribution theo transition.
Một objective có thể tăng macro F1 nhưng làm mất class-1 recall; nhiều loss cùng
hoạt động có thể tối ưu các mục tiêu không tương thích.

### M8. Runtime verifier là cải thiện nhỏ, chưa phải representation mới

Patch-linear softboost tăng validation class-1 F1 từ `0.6860` lên `0.7030`,
nhưng:

- chỉ là post-hoc logit adjustment;
- được thiết kế trên validation evidence;
- chưa có final locked-test result tương đương cho full recipe mới;
- không giảm đủ ngân sách 35 FN/63 FP;
- có nguy cơ overfit threshold/pair behavior.

### M9. Token pruning có thể bỏ cue tinh tế, nhưng bỏ pruning cũng chưa chứng minh lợi ích

Keeper prune token ở các tầng giữa để giảm compute. Cue lớp 1 có thể là vùng nhỏ
và bị bỏ, nhưng các sensitivity test keep-rate/no-prune trước đây không tạo cải
thiện đủ an toàn. Cần một criterion giữ token theo **attribute evidence**, không
chỉ foreground score hay attention hiện tại.

### M10. Mô hình mạnh hơn bên ngoài có representation khác nhưng không cho teacher an toàn

AIDT/Top-5 sửa được nhiều class-1 FP hơn FN. Với current errors:

- consensus hỗ trợ `0→1`, `2→1`, `4→1` tương đối tốt;
- gần như không hỗ trợ `1→2` (`0/11`) và chỉ hỗ trợ `1→4` (`1/5`).

KD hoặc router trực tiếp từ chúng có xu hướng tăng precision nhưng phá recall.
Cần teacher uncertainty/OOF reliability theo transition, không phải confidence
toàn cục.

## 5. Vấn đề về train, optimization và hạ tầng

### T1. Full continuation hợp lệ không vượt keeper và làm rõ trade-off lớp 1

Run manual 30 epoch trước đó chết trước train vì PowerShell biến INFO trên stderr
thành `NativeCommandError`. Launcher đã sửa, smoke-test và continuation hợp lệ sau
đó đã hoàn tất với patience 3, không dùng test:

Original run name:
`full_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_30e_autorun_20260711`.
Reportable evidence is now compacted under
`runs\evidence_full_continuation_rejected_20260711`.

Run dừng sớm sau epoch 5, best tại epoch 2, thời gian `1078.84 s`, `7,245,590`
tham số. Best raw validation đạt macro/class-1 F1 `0.88681/0.68169`, class-1
P/R `0.59314/0.80132`; selection metric `0.76227`, thấp hơn keeper `0.76295`.
Như vậy continuation tăng recall nhưng mở rộng false positive và không vượt raw
keeper `0.88468/0.68605` theo class-1 F1.

Runtime softboost cố định cũng không cứu checkpoint mới. Full validation chỉ đạt
macro/class-1 F1 `0.87704/0.63729`, class-1 P/R `0.65278/0.62252`; confusion
chính là `1→0=41`, `1→2=10`, `0→1=26`, `4→1=12`. XAI trên 12 validation case
khóa trước cho thấy object-desaturation drop `0.15491`, cao hơn background
blur/gray `0.02342/0.02601`; `8/12` case nhạy object-color, Grad-CAM có
background/border flag `5/12` và `6/12`, rollout có `9/12` và `7/12`.
Continuation này do đó bị loại: nó vẫn dựa vào color cùng stem/endpoint/border
cục bộ, không tạo cue interior-surface mới và không được phép mở test.

### T2. Hạn mức phần cứng buộc phải chọn experiment rất kỹ

- GPU 8 GiB: model hiện dùng khoảng 5.3-6.4 GiB khi train.
- RAM 16 GiB: không phù hợp cache toàn bộ ảnh/feature lớn cùng lúc.
- Một full-val cho nhiều biến thể có thể mất vài phút; full train có thể mất
  hàng chục phút đến nhiều giờ.
- Không thể grid-search rộng một cách có trách nhiệm.

### T3. Hai lỗi hạ tầng từng làm sai cách đọc smoke

Đã sửa nhưng cần tính đến khi đọc lịch sử:

1. V8 từng ép `scheduler_total_epochs=Epochs`, khiến one-epoch smoke decay LR từ
   base xuống min trong một epoch. Nay có `-SchedulerTotalEpochs` độc lập.
2. Partial architecture resume từng kế thừa EMA update counter trưởng thành trong
   khi key mới bắt đầu từ zero, làm EMA gần như bỏ qua adapter. Nay EMA reset khi
   load architecture extension.

Một số smoke lịch sử trước hai fix này có thể bảo thủ hơn thực tế; không nên tự
động tái chạy tất cả, nhưng phải biết giới hạn bằng chứng.

### T4. Class-1 F1 biến động mạnh do support nhỏ

Validation lớp 1 chỉ có 151 mẫu. Một vài prediction thay đổi có thể dịch F1 rõ
rệt. Cần bootstrap confidence interval, seed variance và paired significance,
không chỉ so một số thập phân của một seed.

### T5. Gate hiện đóng vì không có signal độc lập đủ mạnh

Smoke-gate audit gần nhất:

- `smoke_gate_ready=false`;
- review readiness `0/6`;
- `0/858` manual fields được điền;
- external consensus class-1 upper bound `0.7237`, dưới milestone `0.75`;
- thiếu coverage cho `1→2` và `1→4`.

Full train hiện tại là explicit override của recipe đã chọn, không phải bằng
chứng rằng các manifest/router cũ đã được mở lại.

## 6. Vấn đề trong XAI và chẩn đoán

### X1. Các phương pháp attribution không hoàn toàn đồng thuận

Attention và gradient-rollout thường cho foreground mass cao, trong khi Grad-CAM
và rollout thường gắn cờ background/border nhiều hơn. Không thể kết luận nguyên
nhân chỉ từ một heatmap.

### X2. Color vừa là signal hợp lệ vừa là shortcut

Object desaturation làm prediction thay đổi lớn hơn background blur/gray, cho
thấy model dựa vào màu quả. Nhưng độ chín vốn liên quan màu; xóa color signal có
thể phá cue hợp lệ. Bài toán là tách color **ổn định theo illumination và surface**
khỏi yellow lighting, defect spot và camera/source style.

### X3. Border/stem/crop cue tồn tại cục bộ dù global background không phải bottleneck

Nhiều lỗi `0/4→1` bám stem, edge, điểm sáng hoặc crop-adjacent context. Simple
background suppression không sửa được vì shortcut nằm sát object boundary hoặc
trên chính bề mặt quả.

### X4. XAI hiện là retrospective, chưa tạo được objective tin cậy

Đã thử biến XAI/attention thành token selection, border suppression, local crop,
self-boosting hoặc verifier signal; hầu hết làm giảm class-1 recall. Cần phương
pháp biến explanation thành supervision mà không self-confirm bias của model.

## 7. Vấn đề evaluation, đối thủ và tính hợp lệ cho bài báo

### E1. So sánh AIDT/TRKH hiện chưa hoàn toàn apples-to-apples

Khác biệt gồm:

- `class_f` so với `yolo_f`;
- class order;
- object-level so với source/image-level;
- pretrained so với no-pretrain;
- model size 112.7M so với khoảng 7.25M;
- crop/context và normalization;
- stored test so với aligned validation prediction.

Paper cần một benchmark table cùng split, cùng sample mapping, cùng metric code,
cùng augmentation budget và ghi rõ pretrained compute.

### E2. Runtime softboost và nhiều diagnostics dùng validation

Các threshold/verifier/consensus trên validation chỉ là chẩn đoán. Không được
trình bày như test result hoặc SOTA nếu chưa khóa protocol và chạy test đúng một
lần sau model selection.

### E3. Chưa có uncertainty và statistical significance

Thiếu:

- nhiều seed cho keeper/final candidate;
- bootstrap CI cho macro/per-class F1;
- McNemar/paired bootstrap cho changed predictions;
- calibration/ECE theo lớp;
- sensitivity theo crop/source group.

### E4. Mục tiêu `0.98` có thể không phù hợp với mức ambiguous hiện tại

Không nên hạ mục tiêu tùy tiện, nhưng nghiên cứu cần trả lời:

- liệu `0.98` có nằm dưới annotator ceiling không;
- cần bao nhiêu label noise tối đa để vẫn đạt mục tiêu;
- có lớp nào về bản chất ordinal/multi-label thay vì categorical không;
- metric nào phản ánh rủi ro thực tế tốt hơn macro F1 đơn lẻ.

## 8. Vấn đề engineering và reproducibility

### R1. Worktree rất lớn và chưa sạch

Repo có nhiều file modified/untracked từ hàng trăm thử nghiệm. Điều này tạo rủi
ro:

- khó xác định code nào sinh checkpoint nào;
- dễ vô tình phụ thuộc module default-off hoặc config cũ;
- khó review/commit/bisect;
- khó tái hiện paper artifact trên máy khác.

Cần tạo checkpoint commit/tag rõ ràng sau khi full run và audit hoàn tất, nhưng
không được revert thay đổi của người dùng.

### R2. Test suite chưa hoàn toàn xanh

Broad relevant suite gần nhất có 138 test pass và 4 failure đã biết:

- ba test dataset cũ còn kỳ vọng tuple hai phần tử;
- một test dùng config field hybrid đã stale.

Các failure này được xem là không liên quan GPSA/LSA, nhưng một repo dùng cho
paper không nên giữ trạng thái “known red” lâu dài.

### R3. Dataset path mặc định có thể trỏ nhầm dataset cũ

`default_data_yaml()` từng có thể trỏ về `D:\DataAI\AIEx\dataset`. Mọi gate run
phải truyền explicit `class_f` hoặc `yolo_f` YAML và lưu resolved config.

### R4. Artifact churn và dung lượng

Checkpoint, XAI case images và plot lặp lại tạo hàng trăm MB mỗi vòng. Đã có
cleanup manifest và retention audit, hiện còn khoảng 80 GB trống. Mỗi phương
pháp vẫn phải compact evidence trước khi xóa source run để không mất khả năng
audit.

### R5. Windows/PowerShell-specific launcher

Đã gặp lỗi stderr pipeline, path quoting và scheduler forwarding. Cần một command
packet tối giản hoặc Python entry point trực tiếp để paper reproduction không phụ
thuộc hành vi PowerShell 5.

## 9. Các nhóm hướng đã thử và không nên đề xuất lại nguyên trạng

Danh sách này không cấm ý tưởng mới có cơ chế thực sự khác; nó cấm đổi vài
hyperparameter quanh cùng signal đã thất bại.

### Context/background/data-view

- source-context classification/fusion;
- full-context + crop paired view;
- crop-margin sweep;
- simple background suppression/blur/gray;
- hflip TTA/consistency;
- bbox prior switch `bbox→crop_bbox`;
- same-source consistency.

### Locality/token/architecture adapter

- BlockLocalPatchMixer;
- SoftPool/MaxSoft stem;
- Shifted Patch Token residual;
- GPSA relative-position mixing;
- LSA temperature/diagonal mask;
- late class-attention pooling;
- simple layer-token fusion;
- local zoom, part-token, micro-detail và high-frequency heads;
- simple patch objectness/token-label heads.

### Embedding/readout/reliability

- current-embedding kNN/neighbor smoothing;
- hard/soft centroids và prototype readout;
- Mahalanobis/spectral shallow readout;
- class-specific query readout;
- frozen DINOv2/EfficientNet feature readouts làm final teacher;
- current verifier threshold/router sweeps;
- in-sample AIDT/Top-5 consensus policies.

### Loss/policy đã không cho signal đủ mạnh

- cumulative/ordinal heads và simple ordinal losses;
- nhiều biến thể contrastive/center/pairwise quanh cùng embedding;
- auto sample weights/soft targets từ weak OOF signal;
- validation-derived targeted margin;
- auto-fill manual review từ visual heuristic.

Chi tiết từng run nằm trong `docs\TRKH_5CLASS_RESEARCH_JOURNAL.md` và
`docs\TODO_TRKH_5CLASS.md`.

## 10. Các câu hỏi ưu tiên cho nghiên cứu chuyên sâu

### Q1. Trần hiệu năng thực tế là bao nhiêu?

Tìm phương pháp ước lượng annotator/Bayes ceiling từ prediction disagreement,
near-duplicate, teacher ensemble và label-noise diagnostics khi không thể relabel
raw dataset. Cần biết điều kiện nào khiến per-class F1 `0.98` khả thi hoặc bất
khả thi.

### Q2. Kiến trúc hybrid nào học cue interior surface thay vì chỉ foreground?

Yêu cầu:

- train từ đầu trên khoảng 9.2k object;
- dưới khoảng 8-12M tham số nếu có thể;
- chạy được trên 8 GiB VRAM, image `256`;
- hội tụ trong 30 epoch;
- CNN xử lý texture/local frequency/illumination;
- Transformer xử lý quan hệ vùng và context;
- không chỉ thêm depthwise-conv/local-attention adapter đã thử.

Cần ưu tiên paper/code chính thức về fine-grained, small-data hybrid models và
phân tích vì sao cơ chế đó khác SPT/GPSA/LSA/LocalMixer.

### Q3. Làm thế nào mô hình hóa lớp 1 như một vùng biên bất định?

Tìm các phương pháp cho noisy ordinal boundary hoặc ambiguous class:

- evidential/Dirichlet classification;
- distributional/interval labels;
- class-conditional selective risk;
- partial-label hoặc complementary-label learning;
- heteroscedastic uncertainty;
- asymmetric precision-recall objectives;
- transition-aware experts với guarantee không phá recall.

Giải pháp phải dùng được khi nhãn raw không thể sửa.

### Q4. Có thể dùng bbox/context theo multi-task curriculum thế nào?

Nghiên cứu một curriculum hợp lệ như:

1. học object localization/objectness từ `yolo_f` bbox;
2. học interior-vs-boundary representation;
3. phân loại object bằng crop/context có kiểm soát;
4. bỏ detection head khi inference nếu cần.

Cần tránh lặp simple token-label/objectness head đã thất bại và giải thích cách
multi-task gradient thực sự cải thiện class attribute.

### Q5. Làm sao dùng external teacher mà không phá true class-1 recall?

Teacher hiện mạnh ở FP suppression nhưng yếu ở FN rescue. Cần phương pháp:

- OOF/fold-safe uncertainty;
- per-transition reliability;
- disagreement-aware distillation;
- teacher abstention;
- asymmetric positive/negative distillation;
- multi-teacher posterior với class-conditional calibration.

Không chấp nhận validation-tuned router hoặc dùng teacher confidence toàn cục.

### Q6. Làm sao học invariant color-surface cue?

Tìm cơ chế tách:

- reflectance/ripeness color;
- illumination/camera/source style;
- defect spot/texture;
- shadow/specular highlight;
- background-adjacent boundary.

Augmentation phải bảo toàn label và không biến lớp 1 thành lớp 0/2/4 về mặt
ngữ nghĩa.

### Q7. Làm sao ép token/register học các attribute bổ sung?

Cần cơ chế diversity có semantic grounding, không chỉ orthogonality. Ví dụ mỗi
token phụ trách maturity, defect, transportability, interior texture hoặc
boundary, nhưng supervision phải được suy ra hợp lệ từ dữ liệu hiện có và không
tự tạo pseudo-label sai.

### Q8. Protocol thí nghiệm tối thiểu nào đủ mạnh cho paper?

Đề xuất:

- baseline và ablation cùng code path;
- ít nhất 3 seed cho final candidates;
- bootstrap CI per-class F1;
- compute/parameter/FLOP/latency table;
- same-split comparison với AIDT;
- locked test một lần sau selection;
- XAI + robustness + failure taxonomy;
- negative-result table để chứng minh novelty không phải hyperparameter sweep.

## 11. Dạng câu trả lời mong muốn từ deep research

Mỗi đề xuất nên cung cấp:

1. Paper gốc và official implementation.
2. Cơ chế toán học cụ thể, không chỉ tên architecture.
3. Vì sao khác các tuyến đã bị loại ở M5 và mục 9.
4. Cách tích hợp vào TRKH hiện tại.
5. Số tham số/VRAM/compute dự kiến.
6. Khả năng resume keeper hay bắt buộc train từ đầu.
7. Curriculum/loss/augmentation cụ thể.
8. Pre-smoke diagnostic độc lập để tránh train mù.
9. Gate định lượng bảo vệ class-1 recall và kiểm soát `0/2/4→1` FP.
10. Ablation tối thiểu trong giới hạn 30 epoch.
11. Failure criterion rõ ràng để dừng sớm.
12. Rủi ro leakage, validation overfit và label-noise ceiling.

Ưu tiên 3-5 hướng có xác suất thành công cao nhất thay vì danh sách dài các
phương pháp chung chung.

## 12. Evidence quan trọng để đối chiếu

- Journal tổng: `docs\TRKH_5CLASS_RESEARCH_JOURNAL.md`
- TODO/gate history: `docs\TODO_TRKH_5CLASS.md`
- Current command packet:
  `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt`
- Keeper checkpoint:
  `runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`
- Current runtime softboost:
  `runs\eval_patch_linear_verifier_softboost001_full_val_20260705`
- Final raw test:
  `runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702`
- Correction budget:
  `runs\diagnostic_softboost_class1_correction_budget_20260706`
- Current smoke gate:
  `runs\audit_trkh_smoke_gate_current_rerun_20260706`
- External-support taxonomy:
  `runs\diagnostic_remaining_error_external_support_20260705`
- GPSA rejection evidence:
  `runs\evidence_gpsa_relative_position_rejected_20260711`
- LSA full-val precheck:
  `runs\diagnostic_lsa_patchonly_full_val_matrix_20260711`
- Full-continuation rejection evidence:
  `runs\evidence_full_continuation_rejected_20260711`
- Latest retention audit:
  `runs\artifact_retention_audit_after_full_continuation_cleanup_20260711`
- Self-contained deep-research reference pack:
  `docs\TRKH_DEEP_RESEARCH_REFERENCE_PACK_20260711.zip`

## Quantitative addendum 2026-07-11 (validation only)

- A strict 17-model audit is now available at
  `runs\diagnostic_validation_information_ceiling_multimodel_20260711`.
- The exact-label five-class oracle reaches macro F1 `0.98395`, but class-1 F1
  only `0.95765`; 29/2606 rows are wrong for every model.
- The class1-vs-rest oracle across all 17 models reaches only `0.97351`, leaving
  four unanimous false negatives and four unanimous false positives.
- Direct image and train/validation sequence review shows foreground-dominant
  maturity-boundary cases with nearby frames assigned conflicting labels. This
  does not prove a Bayes limit, but it makes an every-class F1 target above
  0.98 unsupported by the currently available visual/model information.
- A two-stage Deep Abstaining Classifier smoke was also rejected: closed-set
  macro/class1 F1 `0.87749/0.65487`, error-detection AUROC `0.42040`, and worse
  selective risk below full coverage. Evidence is at
  `runs\evidence_deep_abstention_w015_a130_reject_20260711`.
- Consequence for external research: do not assume that another confidence
  head, background filter, or validation-calibrated threshold can resolve the
  remaining class-1 rows. Any new claim needs a genuinely new representation
  signal plus explicit class1 FP/FN and risk-coverage evidence.

## Method-closure addendum 2026-07-11

Các kết luận dưới đây xuất hiện sau snapshot ban đầu và phải được xem như
no-repeat constraints mới:

- Stock/concurrent scratch hybrids đã được kiểm tra dưới cùng fast-convergence
  protocol. MobileViT-S đạt tốt nhất `0.8291/0.5459` macro/class1 ở epoch 8;
  EdgeNeXt X-Small/Small chỉ khoảng `0.7316/0.3878` và `0.7290/0.4020` ở epoch
  5; CoAtNet-Nano là mạnh nhất nhưng chỉ đạt `0.8616/0.6264` ở best epoch 13;
  EfficientFormerV2-S0 native-224 chỉ đạt highest-raw `0.8089/0.5136` ở epoch
  4 và `0.8077/0.4989` ở epoch 5. Không sweep tiếp stock backbone, LR, loss,
  sampler, teacher hay epochs quanh các route này.
- Partial transplant cũng không giải thích lợi thế CoAtNet: MBConv stem trong
  TRKH chỉ đạt `0.7973/0.5050`; true LeFF đạt `0.7917/0.4747`; persistent
  concurrent local-global coupling đạt `0.7845/0.4490`. XAI của cả ba vẫn cho
  thấy object-desaturation mạnh và far-background perturbation yếu.
- Fixed LDR-KL margin `2`, temperature `1` trên keeper chỉ đạt
  `0.8802/0.6725`. Adaptive LDR bị đóng trước GPU vì uncertainty lambda xếp
  class1 FN cao hơn FP, validation FP-vs-FN AUC chỉ `0.2464`, lambda spread
  `0.0162`, và gradient effect `0.000715`.
- Class-conditional transition correction cũng bị đóng trước GPU. EffV2 và
  DINO proxy experts không đồng thuận về class1 transition row (L1 disagreement
  `1.0272` train OOF và `0.5311` validation), nên invertible matrix không đồng
  nghĩa với identifiable label-noise process.
- Direct `yolo_f`/`class_f` paired-view attention không có gate: chỉ 28/2606
  validation predictions khác nhau; crop view rescue `0/33` context class1 FN
  và correction `2/76` class1 FP; label-assisted oracle chỉ `0.8883/0.6880`.
- Kết luận mới không phải là “mọi hybrid/uncertainty/noise method đều vô ích”.
  Nó là: các signal hiện tại không tách an toàn true-class1 recall khỏi
  `0/2/4->1` false positives. Một GPU route tiếp theo phải thay đổi supervision
  hoặc problem formulation và trước hết tạo được train-only/fold-safe/manual
  FN-vs-FP readiness signal; không chỉ đổi tên một confidence head, logit gate,
  prototype, background filter hay stock backbone.

Evidence mới cần đọc cùng journal:

- `runs\diagnostic_validation_information_ceiling_multimodel_20260711`
- `runs\evidence_compacthybrid_coatnet_nano_yolof_scratch_15e_reject_20260711`
- `runs\evidence_compacthybrid_efficientformerv2_s0_yolof_scratch_5e_reject_20260711`
- `runs\evidence_trkh_leff1234_scratch_5e_reject_20260711`
- `runs\evidence_trkh_concurrent_localglobal_scratch_5e_reject_20260711`
- `runs\evidence_ldrkl_m200_t100_stage2_smoke_reject_20260711`
- `runs\diagnostic_adaptive_ldr_readiness_keeper_20260711`
- `runs\diagnostic_class_noise_transition_readiness_20260711`

Hạ tầng final run hiện dùng một lệnh tại
`scripts\run_trkh_current_best_full_pipeline.ps1`. Train luôn giữ test đóng;
test chỉ được mở sau independent validation reload gate. Lệnh export engine và
video nằm riêng trong `scripts\run_trkh_export_engine.ps1` và
`scripts\run_trkh_test_video.ps1`.

## 13. Tiêu chuẩn để một hướng mới được triển khai

Một hướng mới chỉ nên đi từ research sang code khi:

- tạo signal khác bản chất với các tuyến đã loại;
- có precheck trên train/val hợp lệ, không dùng test;
- giải thích được nó nhắm vào cue nào của `1→0`, `1→2`, `1→4` và
  `0/2/4→1`;
- có cơ chế bảo vệ true class-1 recall;
- không yêu cầu sửa raw data;
- chạy được trên phần cứng hiện tại;
- có ablation và telemetry/XAI để biết module thực sự hoạt động;
- có failure gate để dừng trước full 30 epoch nếu signal sai.

Tài liệu này là snapshot vấn đề, không phải tuyên bố rằng mọi hướng còn lại đều
bất khả thi. Nó nhằm ngăn deep research đề xuất lại các tuyến đã thất bại và tập
trung vào khoảng trống representation/reliability thực sự còn mở.
