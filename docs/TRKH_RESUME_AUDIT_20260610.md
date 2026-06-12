# TRKH Resume Audit - 2026-06-10

## Project boundary

Project chinh: `D:\DataAI\AIEx\TRKH`.

Dataset chinh: `D:\DataAI\AIEx\newdataset\class_f`.

AIDT chi la doi thu/nguon expert phu. Khong phat trien AIDT thay cho TRKH.

## Trang thai da xac nhan

- TRKH v4 run `mango_cls_256_5class_hardneg_maskfix_v4_30e` completed 30 epoch.
- Test macro F1: `0.882251`.
- Test accuracy: `0.923441`.
- Class 1 F1: `0.654545`.
- Nut that chinh van la class 1 va cac cap 0/1, 1/2, 2/3.
- XAI v4 cho thay nen da duoc loc tot hon; loi con lai nghieng ve boundary feature/color/defect, khong chi la background.

## Trace architecture

Da co artifact:

- `runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\architecture_trace_v4_checkpoint`
- `runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\architecture_trace_preflight_20260610`

Moi class co 1 mau train ngau nhien va cac file:

- original/input;
- stem activation;
- patch norm;
- detail map;
- foreground prior;
- token norm qua block 1-8;
- pruning overlay sau layer 2 va 5;
- `shapes.json`.

Da tich hop flag `--trace-architecture` vao `trkh.training.train`. Sau run, neu co `checkpoints/best.pt`, train se tao `run_dir/architecture_trace` va ghi ket qua vao `summary.json`.

## Pretrained toggle trong TRKH

Da them:

- `--pretrained`
- `--no-pretrained`

Mac dinh van la `--no-pretrained`.

`--pretrained` chi ho tro:

- `--model-type resnet50`
- `--model-type mobilenet_v3_large`
- `--model-type vit_b_16`

Custom TRKH `vit_registers` / `vit_registers_hybrid` se bao loi neu bat `--pretrained`, vi chua co adapter/distillation tu backbone pretrained. External weights path van bi chan de tranh leakage/nguon checkpoint khong ro.

## Verification

Da chay:

```text
py_compile: OK
train --help: nhan --pretrained/--no-pretrained va --trace-architecture
trace_architecture --help: nhan --class-name-mode va --expected-num-classes
direct pretrained toggle checks: passed
trace checkpoint preflight: passed, samples=5
```

`pytest` hien chua co trong venv, nen test file moi chua chay bang pytest. Da chay direct Python checks thay the.

## Smoke 2026-06-11

Run:

`runs\smoke_trkh_trace_preflight_20260611`

Ket qua:

- 1 train batch + 1 val batch completed;
- checkpoint `best.pt` duoc luu;
- `--skip-final-test` hoat dong;
- `--trace-architecture` tu tao 5 class samples;
- `summary.json` ghi trace status `completed`;
- tong thoi gian khoang `55` giay, trong do trace khoang `8.2` giay.

Smoke nay chi kiem tra wiring, metric `0.2` khong co y nghia khoa hoc.

## Batch/worker cho RTX 4060 Laptop 8 GB

Run v4:

- batch `64`;
- grad accumulation `1`;
- peak GPU allocated khoang `5.4 GB`;
- peak GPU reserved khoang `6.9 GB`;
- train workers `6`, eval workers `4`;
- 30 epoch khoang `4178` giay.

Khuyen nghi run tiep theo:

- batch `64`, grad accumulation `1` de giu chat luong strict-balanced + supervised contrastive;
- fallback batch `48` neu OOM/driver instability;
- train workers `4`, eval workers `2` tren Windows;
- chi set `TRKH_ALLOW_WINDOWS_MULTIPROCESSING=1`;
- khong bat persistent workers/pin memory env trong command mac dinh;
- image cache `0 MB` khi dung multiprocessing.

Ly do: workers `6/4` nhanh hon nhung de GPU util dao dong va kho dung process hon tren Windows. `4/2` la diem can bang tot hon giua throughput va on dinh.

## Class 1 audit 2026-06-11

Artifact:

`runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\class1_confusion_audit_20260611`

Class 1:

- TP `54`;
- FP `38`;
- FN `19`;
- precision `0.5870`;
- recall `0.7397`;
- F1 `0.6545`.

Nguon false positive class 1:

- class 0 -> 1: `23`;
- class 2 -> 1: `10`;
- class 4 -> 1: `3`;
- class 3 -> 1: `2`.

False negative class 1:

- class 1 -> 2: `9`;
- class 1 -> 0: `7`;
- class 1 -> 4: `3`.

Da copy anh review:

- 19 false negatives;
- 30 false positives;
- 30 close-margin;
- 30 high-confidence errors.

Ket luan: van de class 1 la ca precision va recall, nhung precision thap hon ro. Khong nen tiep tuc oversample class 1 vo dieu kien; can giam `0->1` va `2->1` bang pairwise boundary/label audit, neu khong recall tang se lam FP tang them.

## Tiep theo

1. Mo/canh bao thu cong cac anh trong `class1_confusion_audit_20260611/review_images`.
2. Neu nhan dung, huong train tiep theo la tang pairwise boundary 0/1 va 1/2, khong tang oversampling class 1.
3. Neu nhan sai/khong ro, relabel hoac dua vao ambiguous/exclude list tren train/val; khong sua test de tune.
4. Neu tiep tuc train TRKH custom, command chinh dung `--no-pretrained --trace-architecture`.
5. Neu muc tieu pretrained, custom TRKH can adapter/distillation; backbone torchvision pretrained chi la benchmark.
