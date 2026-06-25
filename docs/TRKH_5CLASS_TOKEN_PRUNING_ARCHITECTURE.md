# TRKH 5-Class Token-Pruning Architecture

## Muc tieu

Kien truc nay danh cho dataset:

`D:\DataAI\AIEx\newdataset\class_f\data.yaml`

Rang buoc thiet ke:

- 5 class, train toi da 30 epoch.
- Giam anh huong cua padding va background.
- Tang do nhay voi khac biet nho giua class 0/1, class 2/3 va class 4 voi cac class chat luong thap.
- Khong pretrained.
- Khong dung val/test de can bang train.
- Tong sample input moi epoch gan bang kich thuoc train goc de khong lam cham train.

## Luong du lieu

| Khoi | Input | Output chinh | Cong dung |
|---|---|---|---|
| Optional foreground object crop | Anh RGB truoc resize | Anh RGB crop quanh pseudo foreground | Tang ty le qua/anh va giam padding/nen nhung giu mau goc; mac dinh tat. |
| Resize-pad + normalize | Anh RGB bat ky | `B x 3 x 256 x 256` | Giu ty le anh, khong crop mat dau hieu nho. |
| CNN stem | `B x 3 x 256 x 256` | `B x 256 x 32 x 32` | Lay bien, texture va mau cuc bo voi inductive bias cua CNN. |
| Patch embedding | Stem feature | `B x 256 x 256` | Tao grid `16 x 16`, moi patch la vector 256 chieu. |
| Patch detail enhancer | Anh normalize + grid | `B x 256 x 256` residual | Khuyech dai high-frequency RGB, local contrast va edge magnitude. |
| Prefix tokens | CLS + 4 register + 2 branch | `B x 7 x 256` | Luu global context va thong tin chuyen biet. |
| Transformer block 1 | 263 tokens | 263 tokens | Hoc quan he global truoc khi prune. |
| Transformer block 2 + prune | 263 tokens | 199 tokens | Giu 192/256 patch theo attention + foreground prior. |
| Transformer block 3-4 | 199 tokens | 199 tokens | Xu ly tap patch co kha nang huu ich cao. |
| Transformer block 5 + prune | 199 tokens | 135 tokens | Giu 128/256 patch goc. |
| Transformer block 6-8 | 135 tokens | 135 tokens | Tap trung compute vao patch con lai. |
| Fine-grained pooling | Global token + 128 patch | `B x 256` | Chon patch phan biet nho va fusion residual vao global feature. |
| Classification fusion | ViT + CNN-logit | `B x 5` | Cong logits tu global transformer va pooled CNN stem. |

So token trong bang khong tinh sai lech batch; `263 = 1 CLS + 4 register + 2 branch + 256 patch`.

## Cac nhanh

### 1. Main ViT branch

CNN stem tao feature map, patch embedding tao 256 spatial token. CLS va register token di qua 8 transformer block.

Register tokens khong bi prune. Chung dong vai tro bo nho global va nhan thong tin tu cac patch khac nhau. Regularizer register-diversity co the duoc bat de han che register collapse.

### 2. Color-statistic token

Input la anh RGB da normalize. Nhanh nay tinh:

- RGB mean/std;
- center-border difference;
- HSV summary;
- Lab mean/std/chroma;
- soft histogram mau.

Output la mot token 256 chieu. Token nay khong truc tiep phan loai; no tham gia self-attention cung CLS/register de mau sac va muc chin co the tuong tac voi texture.

### 3. Edge-statistic token

Input la anh RGB. Nhanh nay tinh Sobel edge, center-border edge, grayscale statistics va pooled edge grid.

Output la mot token 256 chieu. No ho tro tach:

- vo min va vo cang;
- vet dap nhe va be mat lanh;
- dom hu/benh va texture binh thuong.

### 4. Patch detail enhancer

Nhanh nay khac hai branch token vi output cua no co tinh khong gian:

1. Tinh local mean `5 x 5`.
2. Lay RGB high-frequency `image - local_mean`.
3. Tinh gradient magnitude tren grayscale.
4. Pool ve grid `16 x 16`.
5. Project tung vi tri thanh residual 256 chieu.
6. Cong residual vao patch embedding.

Nho vay dau hieu nho khong bi nen boi global color statistic.

### 5. CNN logit branch

Stem feature duoc global average pool, LayerNorm va linear thanh 5 logits. Head duoc zero-init, vi vay luc bat dau train model co hanh vi nhu ViT branch; sau do CNN branch chi hoc residual can thiet.

### Ket hop

- Color token va edge token duoc noi vao sequence sau CLS/register.
- Detail residual duoc cong vao dung spatial patch.
- Fine-grained pooling dung global feature de cham diem cac patch con lai.
- ViT logits va CNN logits duoc cong, khong concatenate, de head nhe va hoi tu nhanh.

## Token pruning va chong nhieu nen

TRKH cu khong co token pruning: tat ca 256 patch di qua day du 8 block.

Ban moi prune sau block 2 va block 5:

1. Lay attention trung binh tu CLS/register/branch token den moi patch.
2. Tao foreground prior tu non-padding mask, mau xoai xanh/vang/nau, dark-defect prior va local detail.
3. Chuan hoa hai score.
4. Xep hang bang:

`score = attention_score + foreground_weight * foreground_prior`

5. Top-k patch duoc giu; CLS/register/branch token luon duoc giu.

Pruning la hard top-k nen cac block sau thuc su xu ly sequence ngan hon. Khi chay XAI voi `return_attention=True`, pruning tam tat de attention rollout van co grid `16 x 16` day du.

Ban audit 2026-06-08 khong con coi center ellipse la foreground mac dinh, vi cach cu co the giu padding/nen o giua anh. Center prior chi con la support nhe khi patch co non-padding/detail phu hop.

Theo micro-benchmark tren RTX 4060 Laptop, batch 16, BF16:

- baseline: khoang 235.9 image/s, peak 1548.7 MB;
- cau hinh de xuat: khoang 253.8 image/s, peak 1261.1 MB.

Day la micro-benchmark compute, khong bao gom toc do doc dia.

## Optional foreground object crop

Tu V30, preprocessing co them crop object-tight co the cau hinh bang:

- `--foreground-crop-mode none|pseudo|grabcut`;
- `--foreground-crop-probability`;
- `--foreground-crop-margin-ratio`;
- `--foreground-crop-min-mask-area-ratio`;
- `--foreground-crop-max-mask-area-ratio`;
- `--foreground-crop-max-crop-area-ratio`.

Input la anh RGB truoc resize. Neu co target mask/bbox thi crop theo union target mask; neu khong co thi dung pseudo foreground mask hoac GrabCut. Output van la anh RGB, sau do moi resize-pad ve kich thuoc model.

Muc tieu cua block nay khac V17 background suppression:

- khong doi mau nen bang gray/blur;
- khong xoa texture goc;
- chi tang ty le vung qua va giam padding/nen ngoai object.

V30 probe cho thay block nay chay on dinh nhung chua vuot V16: val macro/class-1 F1 `0.8851/0.6786`, nen hien tai giu default `none` va khong dung lam cau hinh full train.

## Can bang 5 class khong leakage

Can bang chi dung label cua `train`:

- train goc: `[1941, 541, 1920, 2520, 2293]`;
- balanced exposure voi batch 64: `[1843, 1843, 1843, 1844, 1843]`;
- relative gap: khoang `0.054%`, nho hon gioi han `10%`;
- tong exposure: `9216`, gan bang `9215` sample train goc.

Sampler under-sample luan phien class lon va over-sample co augmentation class nho. No khong copy file, khong thay doi val/test va khong doc label val/test de tinh quota.

Neu `canbang.yaml` co `splits.train.classes`, loader chi doc phan nay va doi chieu voi thu muc train. Neu count khong khop, train dung ngay.

## Hard-negative mining

Sau run v3, class 1 khong thieu recall ma thieu precision: nhieu class 0 bi keo sang class 1. Vi vay hard mining chi dung train split de lap lai cac mau:

- 0 -> 1 va 1 -> 0;
- 1 -> 2 va 2 -> 1;
- 2 -> 3 va 3 -> 2;
- 4 -> rest.

Manifest hien tai:

`runs\mango_cls_256_5class_defectstat_v3_30e\hard_mining_train_only\hard_samples_train_only.csv`

Co `501` mau train-only. Khi lap lai voi factor `1.6`, smoke run tao `9524` effective samples va strict balanced sampler van giu exposure `[1906, 1906, 1906, 1906, 1906]`, gap `0%`.

## Data-centric weighting audit

Tu V27 co tool `trkh.tools.build_data_centric_sample_weights` de tao sample-weight manifest train-only tu prediction CSV:

- guard mac dinh bo qua path khong thuoc train split;
- ho tro `--dry-run`, `--max-issues`, `--max-per-reason`, `--max-per-pair`;
- copy anh review theo reason khi can audit;
- merge voi sample-weight manifest cu de tranh duplicate row vo hieu hoa downweight.

Output chinh:

- `sample_weights_train_only.csv`;
- `label_issue_review_train_only.csv`;
- `summary.json`;
- optional `review_images/`.

V27 dung manifest mined tu V3 stale nen probe thap hon V16. Tool van giu lai de phuc vu label-boundary audit, nhung khong dung output V27 lam full-train candidate.

## Loss va hoi tu 30 epoch

Cau hinh de xuat:

- LDAM/focal nhe hoac cross-entropy + label smoothing tuy run.
- Balanced epoch sampler, khong class weight de tranh double compensation.
- Supervised contrastive loss tren `head,patch`, weight nhe.
- Foreground consistency loss nhe.
- Pairwise margin head cho cac bien class gan nhau.
- AdamW, cosine decay, 2 epoch warmup, EMA `0.995`.
- Khong mixup/cutmix/mosaic/random erasing vi cac class khac nhau o chi tiet nho.
- Augmentation hinh hoc va mau chi o muc nhe.

SupCon duoc dung vi moi balanced batch co nhieu positive sample cho tung class. No keo embedding cung class lai gan va day cac class de nham ra xa.

Train loss co the cao hon val loss trong cau hinh nay vi train loss gom augmentation kho hon, LDAM/focal, foreground consistency, metric learning va pairwise margin loss; val loss thuong chi do classification loss tren anh eval it augmentation hon. Chi xem la bat thuong neu dong thoi co train accuracy thap bat thuong, val metric dao dong vo ly hoac data pipeline dung nham transform.

## Trace

Thu muc structural trace:

`docs/architecture_trace_5class`

Moi class co mot anh train ngau nhien va:

- anh goc, model input;
- stem activation;
- patch embedding norm;
- detail map;
- foreground prior;
- token norm sau tung block;
- patch giu/loai sau moi pruning stage;
- `shapes.json`.

Trace bang checkpoint EMA da train:

`docs/architecture_trace_5class_trained`

Moi trace co cung seed va cung mot anh ngau nhien moi class de co the doi chieu truc tiep. Ket qua run chinh duoc ghi tai `docs/TRKH_5CLASS_RUN_20260608.md`.

## Tai lieu nghien cuu

- DynamicViT, NeurIPS 2021: https://papers.nips.cc/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html
- EViT, ICLR 2022: https://openreview.net/forum?id=BjyvwnXXVn_
- TokenLearner, NeurIPS 2021: https://arxiv.org/abs/2106.11297
- TransFG, AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/19967
## V5 Ordinal Maturity Extension (2026-06-11)

V5 factorize bai toan thanh hai truc:

- `ordinal maturity`: mot scalar duoc hoc tu pooled transformer feature cho thu tu `class 0 < 1 < 2 < 3`;
- `defect`: pairwise head `class 4 vs rest`, vi class 4 la chat luong hu hong va khong nam tren cung truc do chin.

Ordinal head:

- input: pooled feature `[B, 256]`;
- `LayerNorm -> Dropout -> Linear(256, 1)`;
- output: maturity score `[B, 1]`;
- auxiliary loss: SmoothL1 den rank centered `[-1.5, -0.5, 0.5, 1.5]`, chi tren class 0-3;
- logit fusion: score nhan centered rank va residual scale, class 4 khong bi dieu chinh;
- final linear duoc zero-init, nen luc khoi tao khong lam thay doi logits cua classifier goc.

Ly do thay pairwise `0-1/1-2/2-3`: pairwise cu chi hoc tren tung cap va co the dua ra ba boundary mau thuan. Mot scalar chung dung tat ca sample class 0-3 va ap dat tinh nhat quan cua do chin.

Data augmentation v5 them `random_resized_crop_probability`. V4 scale-crop `0.8-1.0` tren moi sample; v5 dung `scale_min=0.88`, probability `0.35` de:

- giu anh day du cho phan lon sample, bao toan khac biet mau/vo nho;
- van mo phong qua bi cat mot phan tren 35% sample;
- khong tang rieng exposure class 1.

Architecture trace sau train co them:

- `01a_resized_before_preprocess.png`;
- `01b_illumination_normalized.png`;
- `01c_foreground_mask_overlay.png`;
- `foreground_mask_fraction` trong `shapes.json`.

## V8 Attention-Guided Views (2026-06-12)

V8 tai su dung attention cua fine-grained patch pooling sau token pruning. Patch
score duoc scatter ve grid goc bang `patch_indices`, tron voi foreground prior,
roi noi suy thanh score map cung kich thuoc anh.

Nhanh crop:

- input: anh `[B, 3, H, W]` va score map `[B, 1, H, W]`;
- lay bounding box cua vung score vuot threshold;
- them padding va dam bao dien tich crop toi thieu;
- resize ve `[B_selected, 3, H, W]`.

Nhanh drop:

- lay vung attention cao, dilation va gioi han dien tich trong khoang cau hinh;
- mac dinh blur `6%-16%` dien tich thay vi xoa bang mau hang;
- buoc model giu dung class khi cue chinh hoac cue nen dang duoc su dung bi lam mo.

Moi sample chi co toi da mot view phu. Raw batch van di qua nhanh chinh; cac view
duoc chon gom lai thanh mot batch nho va forward them mot lan. Loss ket hop:

`total_loss += attention_view_loss_weight * CE(attention_view, label)`

Co warm-up theo epoch; full v8 bat tu epoch 2. Architecture trace tu dong them:

- `06_attention_view_score.png`;
- `07_attention_crop.png`;
- `08_attention_drop.png`;
- `attention_drop_area_fraction` trong `shapes.json`.

History ghi view/crop/drop sample fraction va dien tich drop thuc te. Chi phi
forward ty le voi so sample duoc chon, khong co ba full forward nhu WS-DAN goc.

## V13 Foreground Surface Statistic Fusion (2026-06-12)

V13 them mot nhanh residual co the bat/tat de do mau, anh sang va sai khac be
mat chi trong pseudo foreground. Nhanh nay khong thay token-pruning path.

Input:

- model input `[B, 3, H, W]`;
- anh duoc denormalize va downsample toi da `64x64`.

Xu ly:

- tao soft foreground tu green/yellow/brown support, non-padding, local detail
  va center prior;
- dark/detail chi mo rong mask khi nam gan fruit-color support;
- tinh RGB/Lab/HSV truoc va sau gray-world normalization;
- tinh exposure, color ratio, histogram, dark/brown/bright spot,
  local-contrast/edge va center-border difference.

Output:

- vector thong ke `[B, 129]`;
- residual class logits `[B, 5]`;
- final logits la tong classifier logits va surface residual logits.

Head:

`LayerNorm(129) -> Linear(129, 64) -> GELU -> Dropout -> Linear(64, 5)`

Linear cuoi zero-init de checkpoint cu co the resume ma prediction ban dau
khong thay doi. Config/CLI:

- `foreground_surface_fusion`;
- `foreground_surface_fusion_dropout`;
- `--foreground-surface-fusion`;
- `--foreground-surface-fusion-dropout`.

Architecture trace them `09a` den `09f` cho weight, mask, edge, dark, brown va
bright map. Probe V13 cho thay mask tap trung phan lon tren qua nhung van nhan
mot phan nen co mau gan xoai; class-1 validation F1 `0.6822`, khong qua gate
`0.70`. Chi tiet tai
`docs/TRKH_5CLASS_FOREGROUND_SURFACE_V13_AUDIT_20260612.md`.

## V14 Compact Bilinear Patch Fusion (2026-06-12)

V14 them second-order local feature pooling sau token pruning. Nhanh nay hoc
tuong tac giua cac kenh patch feature de bat texture/dom/chuyen mau cuc bo.

Input:

- patch token `[B,N,256]` sau prune;
- fine-grained patch attention `[B,N]`;
- valid patch mask neu co.

Voi rank `R`:

- hai projection `256 -> R`;
- weighted outer product `[B,R,R]`;
- signed square-root va L2 normalize;
- noi left/right mean thanh descriptor `[B,R*R+2R]`.

Voi `R=32`, descriptor la `[B,1088]`. Head residual:

`LayerNorm(1088) -> Linear(1088,128) -> GELU -> Dropout -> Linear(128,5)`

Linear cuoi zero-init. Trace them
`09g_bilinear_patch_attention.png` va shape descriptor/attention.

V14 cung mask invalid/padding patch trong fine-grained attention. Probe dat
validation macro/class-1 F1 `0.8859/0.6802`, khong qua gate `0.70`. Heatmap van
co diem nong o bien qua, nen khong tang rank hoac full-train truoc khi sua
localization. Chi tiet:
`docs/TRKH_5CLASS_BILINEAR_PATCH_V14_AUDIT_20260612.md`.

## V15 Frequency-Selective Patch Pooling (2026-06-12)

V15 them mot pooling tuy chon theo LaSt-ViT sau token pruning. Muc tieu la
giam lazy aggregation: moi feature dimension chon patch co bieu dien on dinh
sau low-pass filtering tren chieu embedding.

Input:

- patch token `[B,N,256]` sau prune;
- valid-patch mask `[B,N]`;
- foreground prior `[B,N]`.

Xu ly:

1. FFT patch token tren chieu `256`.
2. Nhan Gaussian frequency kernel voi `sigma=sqrt(256)`.
3. IFFT va tinh `x / abs(filtered-x)`.
4. Mask padding va patch co foreground prior thap.
5. Top-k patch theo tung feature dimension.
6. Gather/average thanh selected feature `[B,256]`.
7. Residual blend voi pooled feature cu.

Output:

- `frequency_selective_feature [B,256]`;
- vote fraction `[B,N]`, tong bang 1;
- classifier input `[B,256]` sau blend.

Config/CLI:

- `frequency_selective_pooling`;
- `frequency_selective_top_k`;
- `frequency_selective_blend`;
- `frequency_selective_foreground_threshold`;
- cac flag CLI cung ten voi prefix `--`.

Trace them `09h_frequency_selective_votes.png`. Probe V15 dung
`top_k=1`, `blend=0.25`, threshold `0.35`; validation macro/class-1 F1
`0.8872/0.6788`, khong qua gate `0.70`. Vote van bam co/la xanh khi nen co mau
gan qua, nen khong full-train. Chi tiet:
`docs/TRKH_5CLASS_FREQUENCY_SELECTIVE_V15_AUDIT_20260612.md`.

## V16 Routed Pairwise Specialist (2026-06-15)

V16 doi pairwise margin head tu logit adjustment toan cuc sang adjustment co
router. Muc tieu la chi kich hoat specialist khi classifier goc that su dang
phan van giua cac bien class gan nhau.

Input:

- base logits `[B,5]`;
- pooled feature `[B,256]`;
- danh sach pair `0-1,1-2,2-3,4-rest`.

Xu ly:

1. Tinh probability tu base logits.
2. Lay top-2 class va khoang cach probability giua top-1/top-2.
3. Mot pair duoc mo neu top-2 khop pair khong phan biet thu tu. Rieng
   `4-rest` mo khi class 4 nam trong top-2.
4. Route weight:

   `weight = clamp(1 - margin / route_max_probability_margin, 0, 1)`

5. Pairwise score duoc nhan route weight truoc khi cong residual vao hai logit
   cua pair.

Output:

- `pairwise_margin_route_weights [B,num_pairs]`;
- pairwise logit adjustment `[B,5]`;
- final logits la `base_logits + routed_pairwise_adjustment`.

Config/CLI:

- `pairwise_margin_routing`;
- `pairwise_margin_route_max_probability_margin`;
- `--pairwise-margin-routing`;
- `--pairwise-margin-route-max-probability-margin`.

Trace them `pairwise_margin_route_weights` trong `shapes.json`. Probe V16 dung
`route_max_probability_margin=0.20`; validation macro/class-1 F1
`0.8874/0.6866`, chua qua gate `0.70`. Test audit sau khi chon checkpoint bang
validation dat macro/class-1 F1 `0.8949/0.6923`, nhung khong du de dao nguoc
quyet dinh. Chi tiet:
`docs/TRKH_5CLASS_ROUTED_PAIRWISE_V16_AUDIT_20260615.md`.

## V17 GrabCut Background Filtering (2026-06-15)

V17 them foreground refinement classical bang GrabCut vao preprocessing, khong
dung pretrained. No chi thay doi anh dau vao; kien truc model V16 giu nguyen.

Input:

- anh RGB sau resize-pad `[H,W,3]`;
- pseudo foreground mask tu mau/detail/padding;
- center prior va border/padding background prior.

Xu ly:

1. Khoi tao GrabCut mask tu pseudo foreground.
2. Mark core central foreground va border/padding background.
3. Chay OpenCV GrabCut bang `GC_INIT_WITH_MASK`.
4. Lay foreground/probable foreground, morphology close/open va largest
   connected component.
5. Neu mask qua nho/qua lon/khong on dinh thi fallback ve pseudo mask.
6. Dung mask de tao cac mode background suppression:
   `grabcut_gray`, `grabcut_blur`, `grabcut_mean`,
   `grabcut_desaturate_blur`, `grabcut_blur_gray`.

Output:

- anh RGB cung kich thuoc dau vao;
- target/bbox khong bi doi vi khong crop geometry;
- optional preprocessing cache dataset co cung split/class structure.

Tools/CLI:

- `trkh.tools.audit_grabcut_background`;
- `trkh.tools.build_preprocessed_classification_cache`;
- `--background-suppression-mode grabcut_desaturate_blur`;
- launcher V17: `scripts/run_trkh_5class_grabcut_background_v17.ps1`;
- launchers co `-DataYaml` de tro toi cache dataset neu build offline.

Audit mask cho thay GrabCut giam foreground o border manh hon pseudo mask, nhung
van co the giu nen xanh sat qua. Probe V17 80 batch x 4 epoch cho validation
macro/class-1 F1 `0.8789/0.6405`, thap hon V16; online GrabCut cung cham
~300s/epoch. Vi vay khong full-train V17. Chi tiet:
`docs/TRKH_5CLASS_GRABCUT_BACKGROUND_V17_AUDIT_20260615.md`.

## V25 Background Counterfactual + Ambiguous Soft Target (2026-06-25)

V25 them hai khoi train-only, mac dinh tat:

1. `AmbiguousSoftTargetDataset` cho mau boundary train-only.
2. Background-counterfactual consistency tren tensor da normalize.

Input ambiguous soft target:

- sample train `(image, hard_label, metadata)`;
- manifest train-only co `image_path`, `target_index`, `soft_target_index`
  hoac `soft_0..soft_4`.

Xu ly:

1. Dataset wrapper gan `metadata["soft_target"] [5]` cho sample co trong
   manifest.
2. Sample khong co manifest duoc gan hard one-hot target.
3. Collator stack thanh `soft_target [B,5]`.
4. Train loop dung soft target cho classification loss, nhung van giu hard
   label rieng de audit/metadata khac.

Output:

- `soft_target [B,5]` trong metadata;
- loss classification nhan target dang probability distribution.

Input background counterfactual:

- image tensor normalized `[B,3,H,W]`;
- pseudo foreground mask tu mau/detail/padding;
- logits goc `[B,5]`.

Xu ly:

1. Tao anh counterfactual bang cach thay doi nen: `gray`, `blur`, `mean`,
   hoac `desaturate_blur`.
2. Forward lai model tren anh counterfactual.
3. Tinh KL consistency giua probability goc va counterfactual co temperature.

Output:

- `train_background_counterfactual_consistency_loss`;
- `train_background_counterfactual_fraction`.

CLI:

- `--ambiguous-soft-target-manifest`;
- `--ambiguous-soft-target-alpha`;
- `--background-counterfactual-consistency-weight`;
- `--background-counterfactual-probability`;
- `--background-counterfactual-mode`;
- `--background-counterfactual-margin`;
- `--background-counterfactual-blur-kernel`;
- `--background-counterfactual-temperature`.

Probe V25 dat validation macro/class-1 F1 `0.8858/0.6787`, thap hon V16 va
khong qua gate `0.70`; khong full-train. Chi tiet:
`docs/TRKH_5CLASS_V25_V26_AUDIT_20260625.md`.

## V26 Targeted Directional Margin (2026-06-25)

V26 them loss train-only cho hard false-positive/false-negative quanh class 1,
khong tang sample class 1 toan cuc.

Input:

- logits `[B,5]`;
- hard labels `[B]`;
- manifest train-only co `image_path`, `target_index`, `negative_index`;
- metadata collated:
  - `targeted_margin_negative [B]`;
  - `targeted_margin_weight [B]`;
  - `targeted_margin_margin [B]`.

Xu ly:

1. Dataset wrapper `TargetedMarginDataset` gan negative class cho sample co
   trong manifest; sample khong co manifest dung `negative=-1`, `weight=0`.
2. Train loop tinh directional hinge:

   `max(0, margin + logit_negative - logit_target)`

3. Loss chi active voi sample co `negative>=0` va `weight>0`.

Output:

- `train_targeted_margin_loss`;
- `train_targeted_margin_fraction`;
- `train_targeted_margin_weight_mean`.

CLI/tool:

- `trkh.tools.build_targeted_margin_manifest`;
- `--targeted-margin-manifest`;
- `--targeted-margin-loss-weight`;
- `--targeted-margin-default-margin`;
- `--targeted-margin-default-weight`;
- `--targeted-margin-max-weight`.

Probe V26 dat validation macro/class-1 F1 `0.8864/0.6805`, van thap hon V16.
Forensic cho thay `0->1` tang len `51`, nen khong tang weight va khong
full-train. Chi tiet: `docs/TRKH_5CLASS_V25_V26_AUDIT_20260625.md`.
