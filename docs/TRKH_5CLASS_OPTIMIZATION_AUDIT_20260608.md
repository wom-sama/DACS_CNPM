# TRKH 5-Class Optimization Audit - 2026-06-08

## Ket luan ngan

Run TRKH hien tai chua thua vi thieu recall class 1. Van de chinh la precision class 1 thap: model keo nhieu mau class 0 sang class 1. Vi vay khong nen tiep tuc tang class 1 bang oversampling chung; can hard-negative training cho cac bien 0/1, 1/2, 2/3 va 4/rest.

TRKH run hien tai cung khong so sanh ngang bang voi cac baseline pretrained. Cac output trong `D:\DataAI\AIEx\image_baseline_experiments\outputs` deu `pretrained=true`, trong khi TRKH reject pretrained/external weights. Neu muc tieu la danh bai ViT/AIDT pretrained, can mot ablation rieng cho phep pretrained hoac phai chap nhan rang bai toan hien tai la no-pretrain.

## So sanh voi baseline

| Model | Pretrained | Accuracy | Macro F1 |
|---|---:|---:|---:|
| MobileNetV3 | true | 0.9408 | 0.9050 |
| EfficientNet-B3 | true | 0.9432 | 0.9023 |
| EfficientNetV2-S | true | 0.9392 | 0.8976 |
| TRKH v3 maskfix | false | 0.9179 | 0.8751 |

AIDT hien co code ensemble `ResNet50 + ViT-B/16` pretrained qua `timm`, nhung chua thay thu muc run local. Day la doi thu manh hon ve prior feature, khong phai chi khac head.

## Metric hien tai

Checkpoint: `runs\mango_cls_256_5class_defectstat_v3_30e\checkpoints\best.pt`

Evaluate sau mask/prior fix:

- Accuracy: `0.9179`
- Macro F1: `0.8751`
- Weighted F1: `0.9201`
- Class 1 F1: `0.6380`

Confusion chinh:

- class 0 -> class 1: 27 mau test.
- class 1 -> class 0: 8 mau test.
- class 2 -> class 3: 5 mau test.
- class 3 -> class 2: 25 mau test.

Class 1 recall da kha cao so voi precision. Tang class 1 nua se lam precision xau them neu khong day du hard-negative class 0.

## Audit kien truc

### 1. Tien xu ly va loc nen

Da dat dung truoc model input:

- illumination normalization xu ly anh qua sang/qua toi;
- background suppression `desaturate_blur` lam nen kem hap dan hon;
- local exposure/obstacle augmentation mo phong anh qua toi, qua sang cuc bo, vat can, vet do, anh chi thay mot phan qua.

Loi da sua:

- Pseudo foreground mask cu OR central ellipse vao foreground, nen padding/nen o giua anh co the bi giu lai.
- Mask moi dung non-padding prior, mau xoai vang/xanh/nau, dark-defect prior va detail prior. Center chi la support nhe, khong con la foreground mac dinh.

Vi tri dung:

- preprocessing mask trong `trkh/data/dataset.py`;
- token foreground prior trong `trkh/models/model.py`;
- foreground consistency loss trong `trkh/training/train.py`;
- XAI/trace/evaluate doc checkpoint augmentation config.

### 2. Token pruning

Da dat dung sau Transformer block 2 va 5. Prefix tokens khong bi prune, patch tokens duoc rank bang attention score + foreground prior.

Ket qua XAI sau fix:

- class 3 co mau truoc do giu nhieu padding/ban tay/nen; sau fix mask giu it padding hon.
- mean GradCAM foreground mass khoang `0.7906`, background mass `0.2094`.
- attention audit van bao nen cao hon thuc te vi `return_attention=True` tam tat pruning de giu grid day du.

Diem can doi:

- keep rates `0.75,0.50` hoi manh voi dau hieu nho.
- De xuat run tiep theo dung `0.85,0.65`, chap nhan cham hon nhe de giu vet dap/texture nho.

### 3. So sanh dac trung

Da co cac co che:

- patch detail enhancer khuyech dai high-frequency RGB va edge;
- color-stat token dua mau/chroma/histogram vao attention;
- edge-stat token dua texture/bien vao attention;
- supervised contrastive loss tren `head,patch`;
- pairwise margin head cho bien class de nham.

Van de cua run cu:

- `classification_loss=ldam_focal` nhung `use_ldam=false`, `focal_loss_gamma=0`, `focal_loss_mix=0`, nen thuc chat gan voi CE + label smoothing.
- Aux loss hoi nang: metric `0.08`, foreground `0.05`, pairwise `0.08`, register diversity `0.01`.
- Pairwise pairs cu thieu bien `1-2`, trong khi class 1 bi nham sang class 2.

De xuat run tiep theo:

- bat LDAM/focal nhe: `ldam_max_margin=0.30`, `ldam_scale=18`, `focal_gamma=1.0`, `focal_mix=0.10`;
- giam aux: metric `0.04`, foreground `0.025`, pairwise `0.04`, register diversity `0`;
- pairwise pairs: `0-1,1-2,2-3,4-rest`.

### 4. Augmentation va can bang

Balanced epoch sampler dang dung train split only:

- train goc: `[1941, 541, 1920, 2520, 2293]`;
- exposure run cu: `[1843, 1843, 1843, 1844, 1843]`;
- gap < 10%.

Hard-mining manifest moi:

- file: `runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv`;
- total: `501` mau train-only;
- by target: class 0 `236`, class 1 `67`, class 2 `101`, class 3 `77`, class 4 `20`;
- reason: `353` focus misclassification, `148` near-boundary focus pair.

Loi wiring da sua:

- Truoc do hard-repeat bi skip khi strict balanced sampler bat.
- Nay hard-repeat boc dataset truoc sampler. Sampler van chia deu class, hard samples chi duoc lap lai ben trong class.

Smoke run xac nhan:

- repeat factor: `1.6`;
- effective samples: `9524`;
- balanced exposure: `[1906, 1906, 1906, 1906, 1906]`;
- relative gap: `0.0`.

## Command train tiep theo

Day la run nen chay tiep theo de do tac dong cua hard-negative + maskfix + aux nhe hon. Khong dung val/test de tao train sample.

```powershell
cd D:\DataAI\AIEx\TRKH
$env:PYTHONPATH='D:\DataAI\AIEx\TRKH'
$env:TRKH_AMP_DTYPE='bf16'
$env:OMP_NUM_THREADS='4'
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING='1'
$env:TRKH_ALLOW_WINDOWS_PIN_MEMORY='1'
$env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS='1'

D:\DataAI\.venv\Scripts\python.exe -m trkh.training.train `
  --data D:\DataAI\AIEx\newdataset\class_f\data.yaml `
  --class-name-mode raw --expected-num-classes 5 `
  --run-name mango_cls_256_5class_hardneg_maskfix_v4_30e `
  --output-dir runs --disable-resume --seed 42 `
  --model-type vit_registers --image-size 256 --patch-size 16 --stem-channels 32 `
  --cnn-feature-fusion --cnn-fusion-dropout 0.10 `
  --fine-grained-pooling --fine-grained-pooling-dropout 0.08 `
  --multi-branch-fusion --branch-color-tokens 1 --branch-edge-tokens 1 `
  --branch-cnn-tokens 0 --branch-token-dropout 0.08 `
  --detail-patch-enhancement --detail-patch-dropout 0.05 `
  --token-pruning --token-prune-layers 2,5 --token-keep-rates 0.85,0.65 `
  --token-prune-foreground-weight 0.45 `
  --pairwise-margin-head --pairwise-margin-pairs 0-1,1-2,2-3,4-rest `
  --pairwise-margin-logit-scale 0.25 --pairwise-margin-dropout 0.05 `
  --embed-dim 256 --depth 8 --num-heads 8 --num-registers 4 `
  --register-positional-embedding --head-pooling cls_branch_register_mean `
  --dropout 0.12 --attention-dropout 0.03 --drop-path-rate 0.10 `
  --batch-size 64 --grad-accum-steps 1 --epochs 30 --scheduler-total-epochs 30 --patience 8 `
  --learning-rate 2.5e-4 --min-learning-rate 1e-6 --warmup-epochs 4 `
  --weight-decay 0.05 --grad-clip-norm 0.7 --max-nonfinite-grad-steps 4 `
  --model-ema --model-ema-decay 0.995 `
  --num-workers 6 --eval-num-workers 4 --train-image-cache-mb 0 --eval-image-cache-mb 0 `
  --balanced-epoch-multiplier 1.0 --balanced-epoch-tolerance 0.10 `
  --disable-imbalance-auto-tune --disable-class-weights `
  --disable-class-aware-augmentation --disable-rare-class-repeat --disable-rare-class-recall-guard `
  --best-metric fair_macro_f1 --fair-f1-gap-target 0.08 `
  --classification-loss ldam_focal --ldam-max-margin 0.30 --ldam-scale 18 `
  --focal-loss-gamma 1.0 --focal-loss-mix 0.10 --label-smoothing 0.02 `
  --metric-learning-loss-weight 0.04 --metric-learning-temperature 0.16 `
  --metric-learning-sources head,patch `
  --foreground-consistency-loss-weight 0.025 --foreground-consistency-margin 0.07 `
  --register-diversity-loss-weight 0.0 --pairwise-margin-loss-weight 0.04 `
  --hard-sample-manifest runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv `
  --hard-sample-repeat-factor 1.6 `
  --resize-mode pad --brightness 0.04 --contrast 0.04 --saturation 0.02 --hue 0.01 `
  --illumination-normalization --illumination-normalization-strength 0.35 `
  --background-suppression-mode desaturate_blur --background-suppression-probability 0.8 `
  --background-suppression-margin 0.08 --background-suppression-blur-radius 7 `
  --local-exposure-probability 0.15 --local-exposure-strength 0.25 `
  --obstacle-probability 0.04 --obstacle-max-area 0.08 `
  --random-erasing-probability 0 --random-affine-degrees 3 --random-affine-translate 0.02 `
  --random-affine-scale-min 0.96 --horizontal-flip-probability 0.5 `
  --vertical-flip-probability 0 --rotate90-probability 0.03 --lighting-probability 0 `
  --batch-mix-probability 0 --mosaic-probability 0 --mixup-probability 0 `
  --cutmix-probability 0 --copy-paste-probability 0 --targeted-copy-paste-probability 0
```

## Debug nhanh truoc full train

Smoke command da chay thanh cong voi:

- `--epochs 1`
- `--max-train-batches 1`
- `--max-val-batches 1`
- `--skip-final-test`

Muon lap lai smoke, dung cung command full train nhung thay run name va them 4 flag tren. Hien chua co `--dry-run` rieng, nhung `max_train_batches/max_val_batches/skip_final_test` da du cho smoke run ngan.

## Rui ro con lai

- Neu ViT screenshot `1.0000` den tu split co near-duplicate theo sequence, ket qua do co the bi inflate. Audit hien co khong thay exact duplicate/source-stem leak, nhung average-hash va numeric-ID gan nhau xuyen split van cao.
- Mot so mau class 1 va 2 co ranh gioi label rat mong. XAI case `class 1 -> class 2` cho thay anh rat vang/ripe, co the la label boundary/noise.
- Neu muc tieu that su la macro F1 `0.95` va class 1 F1 `>0.85` tren split hien tai, kha nang can them relabel/bbox crop source-safe hoac cho phep pretrained. No-pretrain TRKH hien chua co bang chung du manh de dat nguong do trong 30 epoch.
