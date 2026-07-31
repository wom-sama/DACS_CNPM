# TRKH pretrained DINOv3 A0 protocol — 2026-07-29

> **Superseded on 2026-07-31.** This document is retained only to reproduce
> the historical `yolo_f` experiment. Do not use its keeper, teacher, data
> contract, scores or launcher for canonical `class_f`. The active protocol is
> `TRKH_PRETRAINED_CLASSF_B0_PROTOCOL_20260731.md`.

## Mục tiêu và phạm vi tuyên bố

A0 kiểm tra một giả thuyết duy nhất: đặc trưng DINOv3-S/16 có giúp nhánh TRKH
đang giữ lại phân tách lớp 1 khỏi các lớp 0/2/4 tốt hơn một bộ phân loại
DINOv3-S/16 thông thường hay không. Đây là nhánh nghiên cứu
`research/pretrained-hybrid-v1`; không thay đổi hay nhập trọng số vào nhánh
no-pretrain.

A0 là thí nghiệm label-only. Không dùng teacher CSV, distillation, threshold,
router hoặc ensemble hậu nghiệm. Mọi lệnh train đều có `--skip-final-test`.
Test lịch sử đã từng được mở nên chỉ có giá trị hồi cứu; tuyên bố xác nhận mới
cần một source/time/site holdout được niêm phong từ trước.

## Đầu vào bị khóa

- Dataset: `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml`, SHA-256
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- Thứ tự lớp cố định:
  `Xoai_Song_Chua_KhoDap`, `Xoai_Song_ChuaNhe_CoNguyCo`,
  `Xoai_Chin_NgotThanh_DeDap`,
  `Xoai_ChinGia_NgotGat_KhongVanChuyen`, `Xoai_Hu_KhongAnDuoc`.
- Validation đầy đủ có 2,606 ảnh; mọi quyết định probe/full phải dùng toàn bộ
  validation (`max_val_batches=0`) và bootstrap theo source group, không theo
  từng crop độc lập.
- Keeper: `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`,
  SHA-256 `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
  Mốc validation độc lập: macro-F1/class-1-F1 `0.882925/0.678261`.
- Backbone: `vit_small_patch16_dinov3.lvd1689m`, 256 px, patch 16,
  384 chiều, 5 prefix token (CLS + 4 register), 256 patch token, khoảng
  21.6 M tham số.
- Trọng số cục bộ: snapshot
  `3bf4720a82ec2066db88137180ff1f83a675cef0/model.safetensors`, SHA-256
  `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`,
  nguồn `https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m`,
  giấy phép/model-card identifier `dinov3-license`.
- Loader phải tạo kiến trúc với `pretrained=False`, kiểm tra SHA trước và sau
  khi load, rồi load strict từ file cục bộ. Không được tải mạng hoặc ngầm dùng
  cache registry trong một run A0.

## Hai arm A0

1. `direct`: `timm_classifier` với cùng checkpoint DINOv3-S/16, classifier 5
   lớp mới và recipe label-supervised. Đây là đối chứng cùng pretrained
   backbone.
2. `hybrid`: `vit_registers_pretrained_hybrid`, nạp keeper bằng `--resume`
   nhưng tuyệt đối không dùng `--resume-use-cli-config`. Kiến trúc, augmentation
   và các thành phần label-only của keeper được kế thừa; optimizer/scheduler/
   scaler/epoch được reset. DINOv3 đi qua nhánh residual có gate bị chặn trong
   `[-0.25, 0.25]`; A0 khóa `initial_scale=0.0` để logits keeper giữ nguyên tại
   bước 0. Teacher losses được đặt rõ về 0 và không truyền teacher CSV.

Hai arm dùng cùng split, class order, ImageNet/timm normalization, strict
balanced exposure, seed, precision, effective batch và lịch LR theo nhóm:
classifier/keeper/adapter/gate dùng `1.5e-4`, DINO backbone dùng `1.5e-5`
(`backbone_lr_scale=0.1`). Preflight phải thấy head và backbone optimizer group
ở cả direct lẫn hybrid; cờ scale tồn tại nhưng không thực sự tách direct là lỗi
fail-closed. Khác biệt được phép là treatment đã khai báo: direct dùng
classifier thường; hybrid giữ TRKH keeper và residual semantic specialist. Vì
vậy kết luận A0 là hiệu quả của toàn hệ hybrid so với cùng backbone, không phải
ước lượng riêng từng auxiliary loss.

## Tài nguyên và lịch chạy

Máy đích: RTX 4060 Laptop 8,188 MiB, BF16, workers train/eval `4/2`.
Stress smoke thực tế của hybrid ở batch 24, gradient checkpointing và gate 0.1
đã hoàn tất forward/backward/AdamW; toàn bộ 162 tensor DINO có gradient, peak
allocated/reserved là `2209.7/2976 MiB`, khi desktop đang dùng khoảng 2.9 GiB.
Gate 0.1 chỉ dùng để kiểm chứng đường gradient/tài nguyên, không phải tham số A0.
Số đo gốc và khóa tài nguyên nằm trong
`docs/TRKH_PRETRAINED_DINOV3_A0_RESOURCE_LOCK_20260729.json`.

Launcher mặc định batch 24, accumulation 2, effective batch 48. Khi chạy cùng
ứng dụng nặng, dùng một trong các cấu hình vẫn giữ effective batch 48: `16x3`,
`12x4`, hoặc `8x6`. Nếu OOM, dừng arm hiện tại và chạy lại **cả hai arm** bằng
cùng geometry thấp hơn; không so sánh hai arm có batch geometry khác nhau.

Các stage:

- `Preflight`: hash/provenance, branch/Git, parser thật, compile, focused tests,
  class order, CUDA/BF16, tạo thật hai model trên CPU và kiểm tra partial resume
  chỉ thiếu các key của nhánh pretrained. Không train.
- `Smoke`: 1 epoch, 4 train batch, 2 validation batch; chỉ là gate kỹ thuật.
- `Probe`: 5 epoch, 120 train batch/epoch, full validation 2,606 ảnh.
- `Full`: 30 epoch, toàn bộ train và validation; chỉ mở khi manifest probe đạt
  gate bên dưới. Full khởi tạo lại từ đúng DINO/keeper đã khóa, không resume từ
  checkpoint probe đã được chọn bằng validation.

Marker stage chỉ có hiệu lực khi cùng Git commit, micro-batch và accumulation
với stage kế tiếp. Probe/full yêu cầu worktree sạch; smoke chạy trên code bẩn
chỉ là chẩn đoán và phải chạy lại sau khi commit.

## Gate từ probe sang full

So sánh dự đoán paired trên cùng sample và bootstrap 10,000 lần theo
`source_image_group`. Manifest cấp quyền full phải ghi đúng run, Git commit,
support 2,606, test locked và thỏa đồng thời:

- `hybrid - direct` macro-F1 >= 0 và cận dưới CI 95% >= -0.003;
- `hybrid - direct` class-1-F1 >= +0.010 và cận dưới CI 95% > 0;
- class-1 precision delta và recall delta đều >= -0.020;
- không có tuning bằng test, threshold hoặc ensemble.

Schema tối thiểu của manifest:

```json
{
  "protocol": "TRKH_PRETRAINED_DINOV3_A0_20260729",
  "decision": "promotable_to_full",
  "git_commit": "<40 hex>",
  "direct_probe_run": "pretrained_dinov3_a0_direct_probe_<tag>",
  "hybrid_probe_run": "pretrained_dinov3_a0_hybrid_probe_<tag>",
  "full_validation_support": 2606,
  "bootstrap_unit": "source_image_group",
  "bootstrap_resamples": 10000,
  "test_locked": true,
  "batch_size": 24,
  "grad_accum_steps": 2,
  "effective_batch_size": 48,
  "hybrid_minus_direct": {
    "macro_f1": 0.005,
    "class1_f1": 0.015,
    "class1_precision": -0.005,
    "class1_recall": 0.020
  },
  "paired_source_group_bootstrap_95ci": {
    "macro_f1": [-0.003, 0.010],
    "class1_f1": [0.001, 0.025]
  }
}
```

Các số trong ví dụ chỉ minh họa schema, không phải kết quả. Launcher kiểm tra
lại toàn bộ bất đẳng thức và commit trước khi cho phép `Full`.

## Lệnh thực thi

```powershell
Set-Location D:\DataAI\AIEx\TRKH_pretrained

# Không train; phải qua trước.
powershell -ExecutionPolicy Bypass -File .\scripts\run_trkh_pretrained_dinov3_a0.ps1 -Mode Preflight

# Bắt buộc chạy hai smoke 1 epoch trước probe.
powershell -ExecutionPolicy Bypass -File .\scripts\run_trkh_pretrained_dinov3_a0.ps1 -Mode Smoke -Arm Both

# Probe dùng full validation nhưng vẫn khóa test.
powershell -ExecutionPolicy Bypass -File .\scripts\run_trkh_pretrained_dinov3_a0.ps1 -Mode Probe -Arm Both

# Chỉ sau khi tạo manifest gate hợp lệ từ paired source-group bootstrap.
powershell -ExecutionPolicy Bypass -File .\scripts\run_trkh_pretrained_dinov3_a0.ps1 `
  -Mode Full -Arm Both -GateManifest D:\path\to\dinov3_a0_probe_gate.json -ConfirmFull

# Fallback VRAM; launcher tự suy ra accumulation=4 để giữ effective batch=48.
powershell -ExecutionPolicy Bypass -File .\scripts\run_trkh_pretrained_dinov3_a0.ps1 `
  -Mode Smoke -Arm Both -BatchSize 12 -RunTag 20260729_b12
```

## Điều kiện báo cáo sau full

Giữ config/command/environment/Git/pretrain manifest, checkpoint hashes, full
confusion matrix và per-class P/R/F1, class-1 TP/FP/FN và các chuyển dịch
`0/2/4 <-> 1`, calibration/margin, dim/bright/low-contrast/occlusion, XAI có
provenance, independent reload, tham số, latency mean/p95, throughput, peak
VRAM, ONNX parity và TensorRT feasibility. Mốc đầu tiên là class-1-F1 >= 0.72
với macro-F1 cạnh tranh, nhưng chỉ số điểm không thay thế CI hoặc deployment
gate. Nếu không thắng direct theo gate đã khóa, kết luận là
`negative-but-reusable` hoặc `mechanism-only`, không đổi ngưỡng sau khi xem kết
quả.
