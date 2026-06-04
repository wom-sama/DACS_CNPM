# De xuat tich hop XAI vao mo hinh TRKH

Ngay lap: 2026-06-03

## Pham vi

Trong bao cao nay, `XAI` duoc hieu la Explainable AI cho Computer Vision, khong phai cong ty xAI. Noi dung duoc tong hop tu:

- `D:/deep-research-report (8).md`
- `D:/DataAI/AIEx/TRKH/README.md`
- `D:/DataAI/AIEx/TRKH/trkh/models/model.py`
- `D:/DataAI/AIEx/TRKH/trkh/models/feature_hooks.py`
- `D:/DataAI/AIEx/TRKH/trkh/evaluation/attention_viz.py`
- `D:/DataAI/AIEx/TRKH/trkh/evaluation/robustness_eval.py`
- `D:/DataAI/AIEx/TRKH/trkh/evaluation/evaluate.py`
- `D:/DataAI/AIEx/TRKH/docs/data_augmentation_preprocessing.md`
- `D:/DataAI/AIEx/TRKH/docs/CLASSIFICATION_ONLY_COMPARISON_20260602.md`

Uu tien de xuat: tiet kiem compute/token truoc, tan dung code co san truoc, tranh thay doi lon vao training khi chua co bang chung.

## Kien truc TRKH hien tai lien quan den XAI

Nhanh hien tai dang tap trung vao classification-only cho xoai, khong dung pretrain, khong dung checkpoint ngoai repo. Pipeline chinh:

```text
Anh crop object tu bbox YOLO
-> CNN stem
-> patch embedding
-> CLS token + register tokens + patch tokens
-> Transformer encoder
-> mean(CLS + register tokens)
-> Linear classifier
-> logits phan lop
```

Dieu nay co hai he qua quan trong cho XAI:

1. Khong nen giai thich mo hinh nhu CNN thuan. CNN stem giu tin hieu cuc bo, nhung quyet dinh cuoi cung di qua token, register token, attention, LayerNorm va MLP.
2. Khong nen chi doc attention tu CLS token. Head hien tai mac dinh `head_pooling=cls_register_mean`, nen register tokens cung tham gia truc tiep vao classifier. Neu heatmap chi lay hang attention cua CLS thi co the bo sot mot phan ly do du doan.

Repo da co nen tang XAI ban dau:

- `attention_viz.py` co the sinh raw attention heatmap va Grad-CAM cho mot anh.
- `feature_hooks.py` co `HookRecorder`, reconstruct attention tu qkv, Grad-CAM tren feature map, va token norm summary.
- `robustness_eval.py` da co logic chon fail cases va goi `analyze_tensor`.
- `evaluate.py` co prediction records, confidence curves, confusion matrix va artifact stats.

Vi vay, huong tot nhat khong phai viet moi tu dau, ma la mo rong nhe cac module nay thanh mot pipeline audit XAI.

## Huong 1: XAI audit cho fail cases va mau do tin cay thap

Muc tieu: tao bao cao giai thich cho cac mau sai, mau confidence thap, va cac cap lop hay bi nham.

Ly do nen lam dau tien:

- Rat re vi chi chay tren mot tap nho, vi du 20-100 anh sau evaluation.
- Da gan voi logic hien co trong `evaluate.py` va `robustness_eval.py`.
- Huu ich ngay cho debug du lieu: crop sai, label sai, nen qua nhieu, anh thieu sang, class gan nhau.

Cach tich hop de xuat:

- Tao module moi `trkh/evaluation/xai_audit.py`.
- Doc checkpoint, data.yaml va prediction records tu evaluation.
- Chon mau theo cac nhom:
  - du doan sai;
  - confidence gan nguong;
  - top-2 probability sat nhau;
  - cap nham lan lon trong confusion matrix;
  - moi class lay it nhat N mau dung va N mau sai.
- Goi lai `analyze_tensor()` trong `attention_viz.py` de tao overlay attention/Grad-CAM.
- Xuat `xai_audit_summary.json`, `xai_audit.md`, va cac anh overlay vao `runs/<run>/xai_audit/`.

Chi phi: thap. Moi mau can 1 forward cho attention va 1 backward neu co Grad-CAM.

Gia tri: cao cho nghien cuu va bao cao, vi bien XAI thanh cong cu chan doan sau evaluation thay vi thanh phan phuc tap trong training.

## Huong 2: Sua raw attention cho dung voi register tokens

Muc tieu: lam heatmap attention hien tai phan anh dung head pooling `cls_register_mean`.

Van de hien tai:

- `build_attention_heatmap()` dang lay `attention[:, 0, prefix_tokens:]`, tuc chi lay attention tu CLS den patch tokens.
- Trong mo hinh chinh, classifier dung mean cua CLS va register tokens. Neu register tokens dang mang tin hieu manh, heatmap CLS-only co the misleading.

Cach tich hop de xuat:

- Them tuy chon `--query-tokens cls|registers|cls_register_mean`.
- Voi `cls_register_mean`, lay trung binh cac hang attention tu token 0 den token `num_registers`, roi moi chieu xuong patch grid.
- Ghi ro trong JSON output query token nao duoc dung.

Chi phi: rat thap. Chi sua cach tong hop attention, khong can backward, khong can train lai.

Gia tri: cao vi sua mot diem lech giai thich truc tiep voi kien truc hien tai.

## Huong 3: Attention Rollout cho ViT-Registers

Muc tieu: thay raw attention mot lop bang attention tich luy qua nhieu block co residual connection.

Co so tu bai nghien cuu:

- Raw attention thuong kem faithfulness vi khong theo doi residual connection va khong tong hop qua layer.
- Attention Rollout re hon Attention Flow nhieu va phu hop lam baseline ViT.

Cach tich hop de xuat:

- Them ham `build_attention_rollout_heatmap()` vao `feature_hooks.py`.
- Lay attention cua nhieu block bang `forward_features(return_attention=True)`.
- Voi moi block:
  - trung binh head;
  - cong identity de mo phong residual;
  - normalize theo hang;
  - nhan tich luy qua block.
- Sau rollout, lay dong query theo `cls`, `registers`, hoac `cls_register_mean`, roi bo prefix tokens de lay patch heatmap.
- Nen co tuy chon `--rollout-start-layer`, mac dinh bo qua 1-2 block dau neu heatmap nhieu noise.

Chi phi: thap. Mot forward co return attention la du.

Gia tri: cao. Day la baseline XAI ViT re, de giai thich trong paper, va tot hon raw attention mot lop.

Rui ro: van chua class-specific. Heatmap noi "model truyen thong tin tu dau", chua noi ro class nao gay ra logit nao.

## Huong 4: Gradient-weighted Attention Rollout, phien ban Chefer-lite

Muc tieu: them tin hieu class-specific vao rollout ma khong can trien khai LRP day du.

Co so tu bai nghien cuu:

- Voi ViT, explanation nen co class-specific signal, gradient va aggregation qua layer.
- Chefer-style relevance propagation manh hon raw attention/rollout, nhung trien khai full conservation qua LayerNorm, MLP, attention heads se ton cong hon.

Cach tich hop de xuat:

- Mo rong attention capture de giu attention tensor co gradient trong forward.
- Chon `target_class`: predicted class hoac ground-truth class.
- Backward `logits[:, target_class].sum()`.
- Voi tung block, dung:
  - `attention * positive_gradient`, hoac
  - `attention * relu(gradient)`, roi trung binh head.
- Cong identity, normalize va rollout nhu Huong 3.
- Them method moi trong CLI: `--method grad_rollout` hoac `--method all`.

Chi phi: trung binh thap. Khoang 1 forward + 1 backward cho moi mau, gan voi Grad-CAM.

Gia tri: rat cao cho fail-case audit vi heatmap gan voi class cu the.

Rui ro: day la Chefer-lite, chua phai AttnLRP/LRP bao toan relevance day du. Can ghi ro trong paper/report la "gradient-weighted attention rollout" de tranh claim qua muc.

## Huong 5: Grad-CAM co chon feature source cho CNN stem va patch embedding

Muc tieu: tan dung CNN stem cua TRKH de xem model nhin vao vung cuc bo nao truoc khi token hoa.

Van de hien tai:

- `resolve_feature_hook()` uu tien `patch_embed.proj` khi model co patch embedding.
- Voi TRKH, CNN stem nam truoc patch embedding. Neu chi Grad-CAM o `patch_embed.proj`, heatmap da sau downsample/tokenization va co the thieu chi tiet cuc bo.

Cach tich hop de xuat:

- Them tham so `--feature-source patch_embed|stem_last|last_conv|auto`.
- Voi `stem_last`, hook vao conv cuoi trong `HybridConvStem`.
- Xuat hai overlay rieng:
  - `gradcam_patch_embed_overlay.png`
  - `gradcam_stem_overlay.png`
- Trong audit, uu tien so sanh Grad-CAM stem voi attention rollout/grad-rollout.

Chi phi: thap. Moi feature source them mot backward hoac co the chay rieng tung lan.

Gia tri: cao cho bai toan xoai vi mau sac, vet vo, dom benh va bien dang nho deu la tin hieu cuc bo.

## Huong 6: Metric XAI nhe tren validation subset

Muc tieu: khong chi nhin heatmap bang mat, ma co so do dinh luong toi thieu.

Metric de xuat:

- Heatmap mass inside bbox/crop object: ti le tong heatmap nam trong bbox object sau khi map ve crop.
- Pointing hit: diem heatmap lon nhat co nam trong bbox object khong.
- Deletion AUC theo patch: xoa dan top-k patch tren token grid, xem confidence class muc tieu giam nhanh khong.
- Insertion AUC theo patch: bat dau tu anh nen/blur, them dan patch quan trong, xem confidence tang nhanh khong.
- Heatmap entropy: heatmap qua phan tan thi can flag, dac biet voi mau confidence cao.

Cach tich hop de xuat:

- Them module `trkh/evaluation/xai_metrics.py`.
- Chi chay tren subset nho: mac dinh 100-300 mau val, co seed.
- Dung patch-level perturbation thay vi pixel-level de re va hop voi ViT.
- Luu ket qua vao `runs/<run>/xai_metrics.json`.

Chi phi: trung binh. Deletion/insertion can nhieu forward, nen phai gioi han subset va so buoc.

Gia tri: trung binh-cao. Huu ich de so sanh raw attention, rollout, grad-rollout, Grad-CAM stem/patch_embed.

Rui ro: bbox khong phai ground-truth explanation hoan hao. Voi classification do chin, toan bo trai xoai co the la vung dung, khong nhat thiet chi mot diem.

## Huong 7: XAI-guided data cleaning

Muc tieu: dung heatmap de tim loi du lieu va shortcut thay vi ep model hoc heatmap ngay.

Cach tich hop de xuat:

- Trong `xai_audit.py`, them flag cac mau:
  - sai nhan nhung confidence cao;
  - heatmap nam ngoai object/crop qua nhieu;
  - heatmap tap trung vao padding/nen;
  - top-2 class sat nhau nhung heatmap khac biet qua it;
  - heatmap cua predicted class va ground-truth class gan nhu giong nhau.
- Xuat danh sach anh can review thu cong.

Chi phi: thap den trung binh, tuy so mau audit.

Gia tri: rat cao trong bo du lieu nho/vua. Neu nhan YOLO/crop/label co van de, cai thien du lieu thuong loi hon them loss phuc tap.

## Huong 8: Register-token diagnostics

Muc tieu: hieu register tokens dang giup hay dang hut thong tin khoi patch tokens.

Repo da co:

- `summarize_token_norms()` tinh `register_to_patch_ratio`, `high_norm_patch_fraction`.
- Training/evaluation da log artifact stats.

Mo rong de xuat:

- Them attention/relevance tu register tokens den patch tokens:
  - `register_to_patch_attention_mean`
  - `cls_to_patch_attention_mean`
  - `register_attention_entropy`
  - `cls_register_heatmap_similarity`
- Ve series theo epoch trong history artifacts.
- Trong XAI audit, ghi kem token norm va register attention stats cho moi mau.

Chi phi: rat thap neu dung attention co san.

Gia tri: cao vi kien truc TRKH dung register tokens, nen day la phan khac biet so voi ViT thong thuong.

## Huong 9: Robustness + XAI cho shortcut mau sac/anh sang/che khuat

Muc tieu: kiem tra model co dua vao vung/phong nen hoac dieu kien anh sang thay vi trai xoai khong.

Repo da co:

- `robustness_eval.py` voi clean, center occlusion, dim/bright lighting, low contrast.
- Fail cases da goi `analyze_tensor()`.

Cach tich hop de xuat:

- Them corruption nhe:
  - blur nen ngoai bbox;
  - gray-out nen ngoai bbox;
  - mask vung padding;
  - color desaturation nhe trong object vs ngoai object.
- So sanh heatmap clean vs corrupted bang overlap/entropy/confidence drop.

Chi phi: trung binh thap neu chi chay fail cases.

Gia tri: cao cho bai toan xoai vi mau va anh sang la tin hieu nhay cam.

## Huong 10: XAI regularization trong training, chi nen de sau

Muc tieu: dung heatmap/relevance de khuyen khich model tap trung vao object thay vi nen.

Cach co the lam:

- Dung YOLO bbox/crop mask lam weak supervision.
- Them loss nhe:
  - phat heatmap mass ngoai bbox;
  - khuyen khich heatmap entropy vua phai;
  - consistency heatmap giua anh goc va horizontal flip.

Ly do chua nen lam ngay:

- Tang chi phi training vi can attention/gradient trong train.
- Co nguy co lam model hoc heatmap "dep" nhung accuracy/faithfulness khong tot.
- Xoai classification co the can nhin ca vung object, mau sac tong the, vet nho va texture; ep heatmap qua chat co the hai recall.

Khuyen nghi: chi thu sau khi da co audit va metric subset cho thay model that su dua vao nen/padding qua nhieu.

## Nhung huong khong nen uu tien ngay

### LIME

Khong nen lam mac dinh vi can nhieu perturbation/superpixel, rat cham. Chi nen dung cho 5-10 mau minh hoa neu can model-agnostic comparison.

### Attention Flow

Khong nen lam mac dinh vi chi phi cao hon rollout nhieu. Co the dung lam benchmark nho neu can doi chieu nghien cuu.

### LRP/AttnLRP day du

Gia tri nghien cuu cao, nhung rui ro trien khai lon vi phai xu ly LayerNorm, attention heads, softmax, matrix multiplication va MLP dung quy tac conservation. Nen de thanh de tai sau khi pipeline rollout/grad-rollout on dinh.

### Heatmap regularization ngay trong run chinh

Khong nen dua vao run chinh khi chua co bang chung. Truoc mat nen dung XAI de audit, tim loi du lieu, va viet phan phan tich.

## Lo trinh tiet kiem de trien khai

### Buoc 1: 1-2 ngay, khong dung training

- Sua raw attention de ho tro `cls_register_mean`.
- Them `xai_audit.py` chon fail cases/low-confidence cases.
- Xuat Markdown/JSON/PNG overlay cho mot subset nho.

Ket qua mong doi: co ngay bao cao qualitative cho repo va paper.

### Buoc 2: 2-4 ngay, them baseline ViT tot hon

- Them attention rollout.
- Them tuy chon bo qua cac layer dau.
- So sanh raw attention vs rollout vs Grad-CAM tren cung fail cases.

Ket qua mong doi: heatmap ViT hop ly hon raw attention mot layer, chi phi van thap.

### Buoc 3: 3-7 ngay, them class-specific ViT explanation

- Them gradient-weighted rollout/Chefer-lite.
- Chay cho predicted class va ground-truth class tren fail cases.
- Them summary: heatmap cua class sai khac class dung the nao.

Ket qua mong doi: giai thich gan voi logit class cu the, phu hop hon voi bai nghien cuu XAI.

### Buoc 4: 1 tuan tro len, metric va ablation

- Them xai metrics tren subset.
- Chay deletion/insertion patch-level voi gioi han mau.
- Doi chieu voi bbox/pointing va confusion pairs.

Ket qua mong doi: co so do dinh luong de noi method nao dang faithful hon trong bo du lieu TRKH.

## De xuat uu tien cuoi cung

Nen chon thu tu sau:

1. `XAI audit` tren fail cases va low-confidence cases.
2. Sua attention heatmap de dung `cls_register_mean`.
3. Them `Attention Rollout` cho ViT-registers.
4. Them `Grad-CAM feature source` cho CNN stem vs patch embedding.
5. Them `Gradient-weighted Rollout` khi can class-specific explanation.
6. Them metric subset sau khi da co overlay on dinh.

Day la lo trinh tiet kiem nhat vi phan lon la evaluation-only, khong dung pretrain, khong thay doi training chinh, va tan dung duoc cac file co san trong repo. Neu token/compute han che, chi can lam buoc 1-3 la da co gia tri thuc dung cao.
