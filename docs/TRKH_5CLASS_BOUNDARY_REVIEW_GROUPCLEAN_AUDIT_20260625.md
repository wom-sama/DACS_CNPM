# TRKH 5-Class Boundary Review + Group-Clean Audit - 2026-06-25

## Muc tieu

- Khong chay full train khi V16 best probe van chua qua gate class-1 validation F1 `0.70`.
- Tao review set leak-safe cho loi ranh gioi class `0/1`, `1/2`, `2/3`, `4/rest`.
- Tao split group-clean theo source sequence truoc khi dung metric de khang dinh generalization.
- Sua launcher de group-clean preflight khong doc nham split cu va co the tat hard-sample manifest cu.

## Code / Tool moi

- `trkh.tools.build_boundary_review_manifest`
  - Input: `predictions_detailed.csv`.
  - Output: `boundary_review_manifest.csv`, `summary.json`, `README.md`, va `review_images/...`.
  - Guard:
    - reject mixed split theo mac dinh;
    - reject test theo mac dinh;
    - train manifest moi duoc dung cho data-clean/loss decision;
    - val manifest chi dung de chan doan.
  - Toi uu toc do:
    - chon candidate bang prediction truoc;
    - chi doc anh/tinh foreground stats sau khi da gioi han review rows;
    - `--quality-scan-limit` mac dinh `0` de tranh quet foreground toan bo split.
- `scripts/run_trkh_5class_attention_views_v8.ps1`
  - Sua preflight `split_counts` khong con hard-code `D:\DataAI\AIEx\newdataset\class_f`.
  - Them `HardSampleManifest` va `HardSampleRepeatFactor`.
- `scripts/run_trkh_5class_boundary_contrastive_v12.ps1` va `scripts/run_trkh_5class_routed_pairwise_v16.ps1`
  - Truyen tiep `HardSampleManifest` / `HardSampleRepeatFactor`.

## Boundary Review Artifacts

Source checkpoint/predictions:

`runs/probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615`

Train review:

`runs/boundary_review_v16_train_20260625`

- input rows: `9215`
- selected before limits: `563`
- selected rows: `295`
- image stats rows: `295`
- by reason:
  - `focus_false_negative`: `6`
  - `focus_false_positive`: `120`
  - `low_margin_error`: `49`
  - `low_margin_correct_boundary`: `120`
- by pair:
  - `0-1`: `150`
  - `1-2`: `78`
  - `2-3`: `44`
  - `4-rest`: `23`
- notable buckets:
  - `0->1`: `99`
  - `2->1`: `19`
  - `1->0`: `6`
  - `over_bright_or_glare`: `125`
  - `center_border_lighting_gap`: `51`

Validation review:

`runs/boundary_review_v16_val_20260625`

- input rows: `2606`
- selected before limits: `341`
- selected rows: `246`
- image stats rows: `246`
- by reason:
  - `focus_false_negative`: `36`
  - `focus_false_positive`: `69`
  - `low_margin_error`: `51`
  - `low_margin_correct_boundary`: `90`
- by pair:
  - `0-1`: `110`
  - `1-2`: `58`
  - `2-3`: `52`
  - `4-rest`: `26`
- notable buckets:
  - `0->1`: `47`
  - `1->0`: `17`
  - `1->2`: `13`
  - `1->4`: `6`
  - `over_bright_or_glare`: `48`
  - `center_border_lighting_gap`: `23`
  - `background_heavy`: `6`

## Visual Review Nhanh

Da xem truc quan 4 case tu `runs/boundary_review_v16_val_20260625/review_images`:

- `t1->p0`: qua chiem gan toan khung, nen/padding khong ap dao; khac biet class `0/1` nam o sac xanh nhe va vet be mat.
- `t0->p1`: qua vang/sang hon ky vong cua class 0; co kha nang ranh nhan mem hoac nhan nhiem.
- `t1->p4`: glare manh, nen la sat qua, vet hu/do ro; nen danh dau lighting + dirty/obstacle, khong sua bang class weighting don thuan.
- `t3->p2`: nhieu dom den/defect lon; class `2/3` bi anh huong boi surface defect, khong chi boi do chin.

Ket luan: loc nen van can, nhung khong phai diem nghen duy nhat. Loi class 1 hien tai phan lon la boundary label + illumination + surface defect/dirty/partial. V30 foreground crop khong vuot V16 la hop ly.

## Group-Clean Split

Tao split moi bang hardlink:

`D:\DataAI\AIEx\newdataset\class_f_groupclean_v1`

Lenh da chay:

```powershell
D:\DataAI\.venv\Scripts\python.exe -m trkh.tools.build_classification_grouped_split `
  --data D:\DataAI\AIEx\newdataset\class_f\data.yaml `
  --output-dir D:\DataAI\AIEx\newdataset\class_f_groupclean_v1 `
  --class-name-mode raw --expected-num-classes 5 `
  --train-ratio 0.70 --val-ratio 0.20 `
  --near-id-window 3 --link-mode hardlink --overwrite
```

Split moi:

- total images: `13088`
- groups: `492`
- target split counts: train `9161`, val `2618`, test `1309`
- class 1 counts: train `535`, val `153`, test `77`
- largest groups: `1208`, `861`, `503`, `381`, `363`

Leak audit:

`runs/class_f_groupclean_v1_leak_audit_20260625`

- exact SHA1 cross-split: `0`
- same source stem cross-split: `0`
- same-class near numeric ID cross-split with window 3: `0`
- average-hash cross-split: `181` groups / `456` files

Average-hash overlap con lai la soft signal: nhieu crop xoai cung mau/bo cuc co the roi vao cung bucket hash. Hard leak chinh da duoc loai bo.

## Preflight / Smoke Group-Clean

Preflight sau khi sua launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_trkh_5class_routed_pairwise_v16.ps1 `
  -DataYaml D:\DataAI\AIEx\newdataset\class_f_groupclean_v1\data.yaml `
  -RunName preflight_groupclean_v16_20260625 `
  -SampleWeightManifest " " -HardSampleManifest " " `
  -PreflightOnly -SkipFinalTest
```

Ket qua: `status=ok`, split counts khop `class_f_groupclean_v1`.

Smoke da chay:

`runs/smoke_groupclean_v16_20260625`

- train audit: `9161`
- val audit: `2618`
- test audit: `1309`
- architecture trace: completed, 5 samples
- smoke chi 2 train / 2 val batch nen `val_macro_f1=0.1884` khong dung de danh gia chat luong.

## Lenh probe tiep theo de user chay neu can

Day la probe, khong phai full train. Khong dung sample/hard manifest cua split cu.

```powershell
cd D:\DataAI\AIEx\TRKH
powershell -ExecutionPolicy Bypass -File scripts\run_trkh_5class_routed_pairwise_v16.ps1 `
  -DataYaml D:\DataAI\AIEx\newdataset\class_f_groupclean_v1\data.yaml `
  -RunName probe_groupclean_v16_120b_6e_20260625 `
  -SampleWeightManifest " " `
  -HardSampleManifest " " `
  -Probe `
  -SkipFinalTest `
  -BatchSize 32 `
  -GradAccumSteps 2 `
  -NumWorkers 4 `
  -EvalNumWorkers 2
```

Neu probe group-clean class-1 val F1 van duoi `0.70`, khong nen full train. Buoc dung hon la review train manifest va tao manifest label-clean/ambiguous train-only.

## Ket luan hanh dong

1. Diem nghen hien tai khong con la thieu them mot loss/head nho; V9-V30 da cho thay nhieu ablation chi dao dong quanh V16.
2. Can label/data decision truoc:
   - dien cot `manual_label_status`, `quality_lighting`, `quality_dirty_obstacle`, `quality_partial_fruit`, `quality_background_mask` trong `runs/boundary_review_v16_train_20260625/boundary_review_manifest.csv`;
   - khong dung val/test de tao sample weights;
   - neu nhieu case train la `ambiguous/wrong`, tao train-only clean/soft-target/ignore manifest.
3. Neu van muon toi uu model sau label audit:
   - uu tien quality-aware auxiliary branch / defect pseudo-label tu train-only review;
   - object-tight localization co supervision that, khong tiep tuc siet pseudo-mask mau;
   - group-clean probe moi la protocol de so sanh nghiem tuc voi pretrained/AIDT.
