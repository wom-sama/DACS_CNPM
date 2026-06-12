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
