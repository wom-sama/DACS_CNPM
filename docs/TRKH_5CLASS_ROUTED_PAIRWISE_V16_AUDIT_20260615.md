# TRKH 5-Class Routed Pairwise V16 Audit - 2026-06-15

## Muc tieu

V16 thu pairwise specialist/router cho cac bien de nham `0-1`, `1-2`, `2-3`
va `4-rest`. Diem khac V12 la pairwise margin head khong con adjustment toan
cuc tren moi sample; no chi mo khi top-2 base class nam trong boundary da khai
bao va probability margin du nho.

Huong nay van la no-pretrain TRKH. Test split chi dung audit sau khi chon
checkpoint bang validation.

## Kien truc

Input:

- base classification logits `[B,5]`;
- pooled transformer feature `[B,256]`;
- pair list `0-1,1-2,2-3,4-rest`.

Router:

1. Softmax base logits.
2. Lay top-2 class va `margin = p_top1 - p_top2`.
3. Voi pair thuong, route mo khi tap top-2 bang tap pair.
4. Voi `4-rest`, route mo khi class 4 nam trong top-2.
5. Route weight la:

   `clamp(1 - margin / route_max_probability_margin, 0, 1)`.

6. Pairwise score duoc nhan weight truoc khi tao logit residual.

Output:

- `pairwise_margin_route_weights [B,4]`;
- pairwise logit adjustment `[B,5]`;
- final logits `[B,5]`.

CLI/config moi:

- `pairwise_margin_routing`;
- `pairwise_margin_route_max_probability_margin`;
- `--pairwise-margin-routing`;
- `--pairwise-margin-route-max-probability-margin`.

Launcher:

`scripts/run_trkh_5class_routed_pairwise_v16.ps1`

Launcher V8/V12/V16 cung co them `-SkipFinalTest` de probe ket thuc sau train
validation, sau do chay test/trace rieng. Muc tieu la tranh timeout o buoc
audit cuoi khi dung tool automation.

## Kiem tra

- py_compile model/train/trace/config: pass.
- focused pytest pairwise/frequency: `6 passed`.
- full regression: `118 passed`.
- PowerShell parser cho V8/V12/V16: pass.
- preflight V16: pass tren RTX 4060 8 GB, batch `32`, grad accumulation `2`,
  workers `4/2`.
- smoke V16: exit `0`, loss finite, architecture trace completed.

Probe command da chay:

```powershell
& .\scripts\run_trkh_5class_routed_pairwise_v16.ps1 `
  -RunName 'probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615' `
  -Probe
```

Automation timeout sau khoang 904s, nhung day khong phai Python/train hang:
`history.csv` da co du 6 epoch va khong con python process. Timeout xay ra sau
train/validation, truoc khi launcher ghi `launcher_status.json`, final test va
architecture trace. Cac artifact con thieu da duoc xuat thu cong.

## Ket qua validation

Checkpoint duoc chon o epoch 3:

| Metric | V12 | V15 | V16 |
| --- | ---: | ---: | ---: |
| val macro F1 | 0.8866 | 0.8872 | 0.8874 |
| val class-1 precision | 0.6203 | 0.6257 | 0.6250 |
| val class-1 recall | 0.7682 | 0.7417 | 0.7616 |
| val class-1 F1 | 0.6864 | 0.6788 | 0.6866 |

V16 khong qua gate class-1 validation F1 `>=0.70`.

## Test audit sau validation selection

| Metric | V12 | V15 | V16 |
| --- | ---: | ---: | ---: |
| accuracy | 0.9321 | 0.9353 | 0.9321 |
| macro F1 | 0.8943 | 0.8975 | 0.8949 |
| class-1 precision | 0.6429 | 0.6506 | 0.6506 |
| class-1 recall | 0.7397 | 0.7397 | 0.7397 |
| class-1 F1 | 0.6879 | 0.6923 | 0.6923 |

Class-1 test audit:

- TP/FP/FN: `54/29/19`;
- `0->1=19`;
- `1->2=10`;
- `1->0=6`;
- `2->1=6`;
- `1->4=3`;
- `4->1=3`.

Artifacts:

- `runs/probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615`;
- `eval_test_detailed/predictions_detailed.csv`;
- `class1_confusion_audit_test`;
- `architecture_trace`.

## Trace audit

Route weights trong 5 mau trace, theo thu tu pair `0-1,1-2,2-3,4-rest`:

| Class | Route weights |
| --- | --- |
| 0 | `0.0000,0.0000,0.0000,0.3732` |
| 1 | `0.6195,0.0000,0.0000,0.0000` |
| 2 | `0.0000,0.0000,0.8376,0.0000` |
| 3 | `0.0000,0.0000,0.0000,0.5452` |
| 4 | `0.0000,0.0000,0.0000,0.4881` |

Router chi mo mot specialist tren moi mau trace, nen co han che adjustment
toan cuc. Tuy nhien class 0 va class 3 trace lai mo `4-rest`, cho thay base
logits van co ambiguity voi class 4 tren mot so anh. Viec nay khong tao buoc
nhay class 1.

Visual trace:

- class 1 attention tap trung vao vet lom/dom tren be mat qua, nhung heatmap
  van co diem nong o goc trai duoi;
- foreground pseudo-mask class 1 van lay nhieu nen co mau gan qua;
- class 0 va class 4 overlay cung cho thay co/la xanh hoac nen cung mau bi
  tinh la foreground.

Ket luan localization: pairwise/router dung vi tri logic hon V12, nhung dau vao
cho specialist van bi mask nen nhieu. Them head tren feature da nhiem nen chi
cai thien nho.

## Quyet dinh

Khong full-train V16:

- validation class-1 F1 `0.6866 < 0.70`;
- test class-1 F1 chi ngang V15 va chua du de thang cac baseline pretrained;
- trace xac nhan pseudo foreground mask chua object-tight.

Huong tiep theo nen uu tien truoc khi them specialist moi:

1. Tao object-tight crop/mask khong dung test split, uu tien classical/weak
   localization co cache va visual audit.
2. Tao group-clean split theo source sequence de metric validation khong bi
   lac quan do cung qua/phien chup xuat hien o nhieu split.
3. Tao train-only label-boundary review set cho `0/1/2`; cac mau ambiguous
   khong nen duoc fix bang oversampling class 1.
