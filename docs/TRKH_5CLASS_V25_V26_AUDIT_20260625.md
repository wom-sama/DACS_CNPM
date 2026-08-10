# TRKH 5-Class V25/V26 Audit - 2026-06-25

## Muc tieu

- Uu tien no-pretrain TRKH tren `D:\DataAI\AIEx\newdataset\class_f`.
- Cai thien class 1 (`Xoai_Song_ChuaNhe_CoNguyCo`) ma khong oversample toan cuc class 1.
- Khong dung test split de tuning. Tat ca metric trong tai lieu nay la validation/probe.
- Full train 30 epoch chi duoc de xuat neu smoke + probe dat class-1 validation F1 >= `0.70`.

## Tham chieu hien tai

Best no-pretrain probe truoc do van la V16 routed pairwise:

- Run: `runs/probe_mango_cls_256_5class_routed_pairwise_v16_120b_6e_20260615`
- Best validation: macro F1 `0.8874`, class-1 F1 `0.6866`.
- Class 1: precision `0.6250`, recall `0.7616`.
- Forensic validation: `TP=115`, `FP=69`, `FN=36`.
- Loi chinh: `3->2=54`, `0->1=47`, `1->0=17`, `1->2=13`, `2->1=9`, `4->1=10`.

## V25 - Counterfactual + Ambiguous Soft Target

Thay doi:

- Them `AmbiguousSoftTargetDataset` train-only.
- Them `--ambiguous-soft-target-manifest` va `--ambiguous-soft-target-alpha`.
- Them background-counterfactual consistency tensor-only: `gray|blur|mean|desaturate_blur`.
- Them launcher `scripts/run_trkh_5class_counterfactual_ambiguity_v25.ps1`.

Manifest V25:

- Source train-only: `runs/mango_cls_256_5class_defectstat_v3_30e/train_maskfix_for_hard_mining/predictions_detailed.csv`.
- Output: `runs/ambiguous_soft_targets_v25_train_only_20260625/ambiguous_soft_targets_train_only.csv`.
- Rows: `364`; non-train skipped `0`.
- Pairs: `0->1=176`, `1->0=34`, `1->2=3`, `2->1=57`, `2->3=27`, `3->2=67`.

Probe:

- Run: `runs/probe_mango_cls_256_5class_counterfactual_ambiguity_v25_80b_4e_20260625`
- Best epoch: `3`.
- Best validation macro F1: `0.8858`.
- Best validation class-1 F1: `0.6787`, precision `0.6209`, recall `0.7483`.
- Forensic export class-1 F1: `0.6747`, `TP=112`, `FP=69`, `FN=39`.
- Loi chinh: `3->2=50`, `0->1=45`, `1->0=18`, `1->2=14`, `2->1=11`, `4->1=10`.

Ket luan V25:

- Khong qua gate `0.70`.
- Thap hon V16 o class-1 F1.
- Soft target tu teacher V3 cu lam mem nhan theo du doan sai; co the giam do sac boundary thay vi tang kha nang phan biet.
- Background-counterfactual consistency chay on-GPU, khong treo, nhung khong tao dot pha.

## V26 - Targeted Directional Margin

Thay doi:

- Them `TargetedMarginDataset`.
- Them `--targeted-margin-manifest`, `--targeted-margin-loss-weight`, `--targeted-margin-default-margin`, `--targeted-margin-default-weight`, `--targeted-margin-max-weight`.
- Them loss directional hinge tren logit:
  `max(0, margin + logit_negative - logit_target)`.
- Them tool `trkh.tools.build_targeted_margin_manifest`.
- Them launcher `scripts/run_trkh_5class_targeted_margin_v26.ps1`.

Manifest V26:

- Source train-only: `runs/mango_cls_256_5class_defectstat_v3_30e/train_maskfix_for_hard_mining/predictions_detailed.csv`.
- Output: `runs/targeted_margin_v26_train_only_20260625/targeted_margin_train_only.csv`.
- Rows: `260`; non-train skipped `0`.
- Reason: `focus_false_positive=227`, `focus_false_negative=33`.
- Pairs: `0->1=161`, `2->1=50`, `4->1=16`, `1->0=30`, `1->2=3`.

Smoke:

- Run: `runs/smoke_mango_cls_256_5class_targeted_margin_v26_20260625`.
- Exit code `0`.
- Architecture trace completed.
- Targeted margin active: `train_targeted_margin_loss=0.1504`, fraction `0.03125`.

Probe:

- Run: `runs/probe_mango_cls_256_5class_targeted_margin_v26_120b_6e_20260625`.
- Best epoch: `4`.
- Best validation macro F1: `0.8864`.
- Best validation class-1 F1: `0.6805`, precision `0.6150`, recall `0.7616`.
- Forensic validation: `TP=115`, `FP=72`, `FN=36`.
- Loi chinh: `0->1=51`, `3->2=50`, `1->0=16`, `1->2=14`, `2->1=9`, `4->1=9`.

Ket luan V26:

- Khong qua gate `0.70`.
- Thap hon V16 o class-1 F1.
- Directional margin train-only active nhung khong tong quat hoa; `0->1` tren validation tang tu V16 `47` len `51`.
- Tiep tuc tang weight V26 khong co co so tot vi no da lam precision class 1 giam.

## Kiem tra cac huong lien quan

- V19 ordinal boundary probe da co san: best class-1 F1 `0.6687`; reject.
- V20 pairwise confusion probe da co san: best class-1 F1 `0.6606`; reject.
- V26 calibrated threshold khong cuu duoc class 1: best per-class threshold F1 class 1 `0.6628`, thap hon raw best epoch.

## Quyet dinh

- Khong gui lenh full train cho V25/V26.
- Gate hien tai van la class-1 validation F1 `>=0.70`.
- V16 van la ung vien no-pretrain tot nhat trong nhom probe hien tai.

## Huong tiep theo co kha nang cao hon

1. Lam label-boundary audit train-only/val-only cho `0->1`, `1->0`, `1->2`, `3->2` de tach nhan sai, nhan ambiguous, va dieu kien anh sang/che khuat.
2. Tao group-clean split theo qua goc/phien chup truoc khi ket luan generalization. Split hien tai co dau hieu source-sequence leakage nen metric co the khong phan anh kha nang tren qua moi.
3. Neu van uu tien no-pretrain, can thu object-tight localization/mask co supervision nhe hoac crop/mask bang model rieng; pseudo foreground mau sac hien tai khong du vi nen xanh/co la co mau gan qua.
4. Neu chap nhan pretrained, dung pretrained expert/TTA lam baseline thuc dung; no-pretrain TRKH hien dang bi gioi han boi label ambiguity va dataset condition hon la thieu mot loss phu.
