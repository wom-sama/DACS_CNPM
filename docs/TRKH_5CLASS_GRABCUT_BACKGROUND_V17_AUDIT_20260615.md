# TRKH 5-Class GrabCut Background V17 Audit - 2026-06-15

## Muc tieu

Sau V16, trace cho thay pseudo foreground mask van lay nen xanh/co mau gan qua.
V17 thu foreground refinement classical bang GrabCut de giam shortcut nen truoc
khi them specialist moi. Huong nay no-pretrain va khong dung test de tune.

Tai lieu da doi chieu:

- GrabCut, Rother et al. 2004: https://dl.acm.org/doi/10.1145/1015706.1015720
- Cutout, DeVries and Taylor 2017: https://arxiv.org/abs/1708.04552
- Random Erasing, Zhong et al. 2017: https://arxiv.org/abs/1708.04896
- AugMix, Hendrycks et al. 2019: https://arxiv.org/abs/1912.02781

Codebase da co local exposure, obstacle augmentation va random erasing. Vi vay
V17 uu tien sua foreground mask thay vi tang augmentation.

## Kien truc / preprocessing

V17 giu nguyen model V16. Thay doi nam trong `trkh.data.dataset`:

- `_grabcut_foreground_mask_array`;
- background suppression modes:
  `grabcut`, `grabcut_gray`, `grabcut_blur`, `grabcut_mean`,
  `grabcut_desaturate_blur`, `grabcut_blur_gray`.

Input:

- anh RGB sau resize-pad;
- pseudo mask cu;
- ImageNet padding color;
- center prior va border prior.

Xu ly:

1. Tao pseudo foreground mask cu.
2. Mark pseudo foreground la probable foreground.
3. Erode core central foreground de tao sure foreground.
4. Mark padding/border ngoai center la sure background.
5. Chay OpenCV GrabCut.
6. Close/open mask, giu largest connected component.
7. Fallback ve pseudo mask neu mask qua nho, qua lon hoac OpenCV loi.
8. Composite background theo mode da chon, khong crop geometry.

Output:

- anh RGB cung kich thuoc, target khong bi doi;
- neu dung cache builder, output la dataset `classification_folder` moi voi
  cung split/class va `data.yaml` rieng.

## Tooling

Them:

- `trkh.tools.audit_grabcut_background`: xuat 1 mau moi class voi pseudo overlay,
  GrabCut overlay va anh sau suppression.
- `trkh.tools.build_preprocessed_classification_cache`: build cache dataset co
  `--dry-run`, `--max-samples-per-class`, `--workers`, `--overwrite`.
- `-DataYaml` cho launcher V8/V12/V16/V17 de co the tro toi cache dataset.
- `scripts/run_trkh_5class_grabcut_background_v17.ps1`.

Dry-run cache:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.build_preprocessed_classification_cache `
  --data D:\DataAI\AIEx\newdataset\class_f\data.yaml `
  --output-root D:\DataAI\AIEx\newdataset\class_f_grabcut_v17_smoke_cache `
  --mode grabcut_desaturate_blur `
  --workers 2 `
  --max-samples-per-class 2 `
  --dry-run
```

Small cache smoke built `30` images OK.

## Mask audit

Artifact:

`runs/grabcut_background_audit_v17_20260615`

Mask fractions:

| Class | Pseudo mask | GrabCut mask | Pseudo border | GrabCut border | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5871 | 0.4722 | 0.2930 | 0.0718 | 0.8043 |
| 1 | 0.7509 | 0.6239 | 0.3675 | 0.0729 | 0.8011 |
| 2 | 0.6002 | 0.5645 | 0.1819 | 0.0961 | 0.9076 |
| 3 | 0.8185 | 0.7054 | 0.4364 | 0.2004 | 0.8480 |
| 4 | 0.6550 | 0.5349 | 0.3212 | 0.0899 | 0.8166 |

Visual review:

- GrabCut giam ro foreground o border/padding;
- class 1 van giu mot phan nen xanh sat qua;
- class 3 van co truong hop leaf/hand/background dinh vao qua;
- vi vay `grabcut_desaturate_blur` duoc thu nhu suppression nhe, khong crop.

## Smoke

Command:

```powershell
& .\scripts\run_trkh_5class_grabcut_background_v17.ps1 `
  -RunName 'smoke_mango_cls_256_5class_grabcut_background_v17_20260615' `
  -Smoke
```

Ket qua:

- exit code `0`;
- loss finite;
- architecture trace completed;
- online GrabCut khong hang nhung cham hon validation/training thuong.

Smoke metric khong dung de so sanh vi chi `2` val batch.

## Probe

Command:

```powershell
& .\scripts\run_trkh_5class_grabcut_background_v17.ps1 `
  -RunName 'probe_mango_cls_256_5class_grabcut_background_v17_80b_4e_20260615' `
  -Probe `
  -Epochs 4 `
  -MaxTrainBatches 80 `
  -SkipFinalTest `
  -TraceArchitecture:$false
```

Tong thoi gian: `1226.8s`. `launcher_status.json` exit code `0`.

Per epoch:

| Epoch | train seconds | val seconds | val macro F1 | val class-1 F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 120.8 | 179.2 | 0.8763 | 0.6303 |
| 2 | 125.7 | 175.5 | 0.8789 | 0.6405 |
| 3 | 125.8 | 177.7 | 0.8739 | 0.6224 |
| 4 | 126.1 | 162.7 | 0.8721 | 0.6145 |

Best checkpoint: epoch 2.

| Metric | V16 | V17 |
| --- | ---: | ---: |
| val macro F1 | 0.8874 | 0.8789 |
| val class-1 precision | 0.6250 | 0.5889 |
| val class-1 recall | 0.7616 | 0.7020 |
| val class-1 F1 | 0.6866 | 0.6405 |

Trace artifact:

`runs/probe_mango_cls_256_5class_grabcut_background_v17_80b_4e_20260615/architecture_trace`

Class 1 model input cho thay background da bi xam/blur mot phan, nhung pseudo
mask trace van bao phu nen xanh sat qua. Attention van tap trung vao dom/lom,
nhung metric giam, kha nang do suppression lam thay doi phan bo mau va khong
loai sach vung nen dinh vao qua.

## Quyet dinh

Khong full-train V17:

- class-1 validation F1 `0.6405`, thap hon V16 `0.6866` va duoi gate `0.70`;
- online GrabCut qua cham cho full train neu khong cache;
- visual audit cho thay classical mask van giu nen xanh sat qua.

Giu lai:

- GrabCut modes o dang optional;
- audit tool;
- preprocessing cache builder.

Khong nen tiep tuc full/probe dai voi online GrabCut. Neu muon thu lai, chi nen
dung cache offline va gan `BackgroundSuppressionMode none` de tranh xu ly lap.
Huong tiep theo co gia tri hon:

1. Group-clean split theo source sequence de metric dang tin hon.
2. Train-only label-boundary review set cho `0/1/2`.
3. Learned object-tight localization co supervision hoac pseudo-label duoc audit
   thu cong, thay vi tiep tuc siet mask mau/GrabCut.
