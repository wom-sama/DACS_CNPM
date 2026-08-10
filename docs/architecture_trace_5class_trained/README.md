# TRKH 5-Class Architecture Trace

Moi thu muc class chua mot anh train ngau nhien co seed co dinh va cac anh trung gian.
Trace dung checkpoint da train: `D:\DataAI\AIEx\TRKH\runs\mango_cls_256_5class_tokenprune_fg_v1_30e\checkpoints\best.pt`.

- `00_original.jpg`: anh crop goc.
- `01_model_input.png`: anh sau resize-pad va denormalize de xem.
- `02_stem_activation.png`: mean absolute activation cua CNN stem.
- `03_patch_embedding_norm.png`: norm patch token truoc transformer.
- `04_detail_map.png`: local color/high-frequency/edge map.
- `05_foreground_prior.png`: prior dung cung attention khi xep hang token.
- `block_XX_token_norm.png`: norm token sau tung transformer block; o da prune de trong.
- `prune_XX_after_layer_Y.png`: patch xanh duoc giu, patch toi bi loai.
- `shapes.json`: shape va patch index chi tiet.

Dataset: `D:\DataAI\AIEx\newdataset\class_f\data.yaml`
Seed: `42`
