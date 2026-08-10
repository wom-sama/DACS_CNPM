# TRKH Context Checkpoint 2026-06-12

## Du an va muc tieu bat buoc

- Repo: `D:\DataAI\AIEx\TRKH`.
- Dataset: `D:\DataAI\AIEx\newdataset\class_f`, 5 class.
- Uu tien no-pretrain, kien truc cau hinh duoc, audit day du.
- Train toi da 30 epoch; early stop sau 3 epoch khong cai thien.
- Muc tieu tham chieu: macro F1 `0.99`, class 1 F1 `0.96`.
- Khong tang rieng class 1 vo dieu kien; phai tranh data leakage.
- Truoc smoke/full train phai preflight va test loi.
- Luu architecture trace mot mau moi class qua cac khoi.

## Baseline va audit gan nhat

- TRKH v4 scratch: test macro F1 `0.8823`, class 1 F1 `0.6545`.
- V7 offline distillation: test macro F1 `0.8801`, class 1 F1 `0.6467`; reject.
- Top-5 pretrained TTA ensemble: test macro F1 `0.9248`, class 1 F1 `0.7786`.
- Class 1: TP `51`, FP `7`, FN `22`; confusion chinh `1->0` va `1->2`.
- XAI: foreground mass mean `0.7748`; background perturbation anh huong rat nho,
  object desaturation anh huong lon. Nut that chinh la mau/texture cuc bo va boundary
  nhan, khong chi la background.
- Oracle 10 expert: class 1 F1 `0.8722`, macro F1 `0.9567`; muc tieu cu khong kha thi
  voi tap expert hien tai neu khong co du lieu/nhan/feature tot hon.

## Nghien cuu ma nguon chinh thuc

Da clone read-only vao `D:\DataAI\_research_refs`:

- `WS-DAN.PyTorch`
- `pairwise-confusion`
- `ELR`
- `PMG`

Nguon:

- WS-DAN: https://arxiv.org/abs/1901.09891
- Pairwise Confusion: https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.pdf
- ELR: https://proceedings.neurips.cc/paper_files/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html
- PMG: https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training
- DHVT: https://proceedings.neurips.cc/paper_files/paper/2022/file/5e0b46975d1bfe6030b1687b0ada1b85-Paper-Conference.pdf

## Ket luan ky thuat tam thoi

1. Uu tien attention-guided crop/drop theo y tuong WS-DAN.
   - Dung foreground prior/attention cua TRKH de crop sat vung phan biet.
   - Drop vung noi bat co xac suat de ep mo hinh tim them dau hieu phu.
   - Shared backbone; loss tren anh goc va view tang cuong.
   - Can schedule/probability de khong tang chi phi len 3 lan nhu WS-DAN goc.
2. ELR la ung vien thu hai cho nhan ambiguous/noisy.
   - Can sample index on dinh qua sampler/repeat wrapper.
   - Can warmup va class-aware guard vi prediction som co the thien ve class 0/2,
     lam class 1 te hon.
3. Pairwise Confusion goc chi toi thieu hoa khoang cach feature giua hai nua batch.
   - TRKH da co pairwise margin head; khong them ngay khi chua co sampler cap dung.
4. PMG goc chay 4 optimizer steps moi batch va 200 epoch.
   - Khong phu hop muc tieu 30 epoch/train nhanh; chi co the lay y tuong local-view.
5. TRKH da co conv stem, detail enhancer, branch token va token pruning.
   - Loi ich lon hon den tu supervision view/vung phan biet, khong phai them head tuy y.

## Buoc tiep theo

1. Attention-guided view training da trien khai bang flag, mac dinh tat.
2. Research decision da ghi tai
   `docs/TRKH_NO_PRETRAIN_RESEARCH_DECISION_20260612.md`.
3. Full test suite da pass `100/100` bang pytest 8.4.2.
4. Buoc tiep theo la preflight dataset/config/leak.
5. Chay smoke gioi han 1-2 train/val batch va audit throughput/GPU.
6. Xuat original/score-map/crop/drop cho 1 mau moi class.
7. Chi full train neu smoke on dinh; early stop patience `3`.

## Nguyen tac phuc hoi

Neu context bi reset, doc file nay, `docs/TODO_TRKH_5CLASS.md`, va
`docs/TRKH_5CLASS_V7_TTA_FUSION_AUDIT_20260611.md` truoc khi tiep tuc.
