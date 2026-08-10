# Train Stop Diagnosis - 2026-06-10

## Scope

Project chinh van la `D:\DataAI\AIEx\TRKH`.

Run bi dung truoc do la AIDT phu:

`D:\DataAI\AIDT\runs\resnet50_vit_b16_class_f_5class_pretrained`

AIDT chi dung lam doi thu/nguon expert so sanh. Khong coi day la pipeline chinh cua TRKH.

## Ket qua kiem tra

- Sau reboot khong con process Python train nao dang chay.
- GPU dang ranh, chi dung khoang `1.2 GB / 8 GB`.
- Thu muc run AIDT partial co:
  - `config.json`
  - `best.pt`
  - `history.csv`
- Thu muc run partial khong co:
  - `metrics.json`
  - `predictions.csv`
  - `confusion_matrix.png`

`history.csv` co epoch 1 den 17. Best validation macro F1 la `0.908992` tai epoch 10. Dieu nay cho thay process khong treo o preprocessing/dataloader tu dau; no da train va validate duoc nhieu epoch roi bi dung truoc khi final test/metrics hoan tat.

## Nguyen nhan kha di

Khong co bang chung deadlock trong artifact sau reboot. Nguyen nhan kha di nhat:

1. Run full AIDT rat dai vi model `ResNet50 + ViT-B/16` co `112.7M` tham so va batch nho.
2. Moi epoch co khoang `9215 / 6 = 1536` train batches, chua tinh validation.
3. Log chi co tqdm/epoch JSON, khong co heartbeat batch/eval ro rang, nen VS Code co the nhin nhu bi treo.
4. Khi bi dung giua train, script AIDT cu khong co `finalize-only`, nen `best.pt` da co nhung chua xuat `metrics.json`.

## Huong xu ly trong TRKH

- Da xac nhan architecture trace v4 hien co day du 1 mau moi class va anh qua tung block.
- Da them `--trace-architecture` vao TRKH train de tu tao `run_dir/architecture_trace` sau khi run co `checkpoints/best.pt`.
- Da them block `Preflight Khong Train` vao `AIEx/image_baseline_experiments/Train.md`.
- Truoc bat ky smoke/full train nao can chay preflight: `py_compile`, `train --help`, `trace_architecture --help`, dataset count, GPU check.

## Artifact trace da xac nhan

Trace checkpoint hien co:

`runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\architecture_trace_v4_checkpoint`

Trace preflight moi:

`runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\architecture_trace_preflight_20260610`

Moi class co:

- `00_original.jpg`
- `01_model_input.png`
- `02_stem_activation.png`
- `03_patch_embedding_norm.png`
- `04_detail_map.png`
- `05_foreground_prior.png`
- `block_01_token_norm.png` den `block_08_token_norm.png`
- `prune_01_after_layer_2.png`
- `prune_02_after_layer_5.png`
- `shapes.json`
