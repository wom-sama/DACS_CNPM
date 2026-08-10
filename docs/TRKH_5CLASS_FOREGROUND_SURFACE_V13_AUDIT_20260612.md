# TRKH 5-Class Foreground Surface V13 Audit - 2026-06-12

## Muc tieu

V13 kiem tra gia thuyet rang loi class 1 den tu hai nguon:

- nen, padding, bong do va vat can chen vao thong ke mau;
- cac sai khac nho tren vo qua bi mat khi global transformer pooling.

Run nay giu nguyen pipeline V12, them mot nhanh thong ke be mat foreground co
the bat/tat. Khong dung test split de chon tham so hoac huan luyen.

## Thay doi kien truc

`ForegroundSurfaceStatisticFusion` nhan tensor anh da normalize
`[B, 3, H, W]`, dua ve toi da `64x64`, roi tao:

- pseudo foreground weight tu dai mau xanh/vang/nau-cam, non-padding,
  local edge/detail va center prior;
- gray-world normalized image chi dua tren foreground;
- mean/std RGB va Lab truoc/sau normalize;
- HSV summary va histogram truoc/sau normalize;
- ti le green/yellow/brown, dark, bright, under/over-exposure;
- thong ke dark spot, brown spot, bright spot, local contrast va edge detail;
- chenh lech center-border cho RGB, Lab va damage score.

Tong vector la `[B, 129]`. Head
`LayerNorm -> Linear -> GELU -> Dropout -> Linear(5)` tao residual logits va
cong vao classifier logits. Linear cuoi duoc zero-init, nen bat nhanh tren
checkpoint V8/V12 khong lam doi prediction truoc khi fine-tune.

CLI:

- `--foreground-surface-fusion`;
- `--foreground-surface-fusion-dropout`;
- launcher `scripts/run_trkh_5class_foreground_surface_v13.ps1`.

## Resume va smoke

Smoke dau tien phat hien checkpoint EMA van strict-load va dung khi checkpoint
cu khong co `foreground_surface_fusion_head.*`. Model thuong da cho phep
extension, nhung EMA chua dung cung helper.

Da sua ca model va EMA dung `_load_model_state_allowing_extensions`, chi cho
phep cac prefix extension da liet ke. Missing key ngoai allowlist van fail.

Ket qua:

- full regression test: `102 passed`;
- final smoke exit code `0`;
- model va EMA partial-load dung allowlist;
- architecture trace completed cho 5 class.

Smoke:

`runs/smoke_mango_cls_256_5class_foreground_surface_v13_final_20260612`

## Probe V13

Run:

`runs/probe_mango_cls_256_5class_foreground_surface_v13_120b_6e_20260612`

Cau hinh:

- resume V8 best checkpoint;
- V12 boundary sample weights va in-batch boundary contrastive;
- `120` train batch/epoch, full validation;
- toi da `6` epoch, patience `3`;
- batch `32`, gradient accumulation `2`, workers `4/2`;
- no-pretrain, foreground surface fusion bat.

Ket qua:

| Metric | V12 | V13 |
| --- | ---: | ---: |
| best val macro F1 | 0.8866 | 0.8868 |
| best val class-1 F1 | 0.6864 | 0.6822 |
| test macro F1 | 0.8943 | 0.8938 |
| test class-1 F1 | 0.6879 | 0.6835 |

V13 best epoch `2`, early-stop tai epoch `5`, tong `648.96s`.

Class-1 test:

- precision `0.6353`;
- recall `0.7397`;
- F1 `0.6835`;
- TP/FP/FN `54/31/19`.

Confusion chinh:

- `0 -> 1`: `20`;
- `1 -> 2`: `10`;
- `2 -> 1`: `7`;
- `1 -> 0`: `6`;
- `1 -> 4`: `3`;
- `4 -> 1`: `3`.

Mau confusion gan nhu khong thay doi so voi V12. Nhanh thong ke be mat khong
giai quyet duoc boundary 0/1/2.

## Audit hinh anh

Trace:

`runs/probe_mango_cls_256_5class_foreground_surface_v13_120b_6e_20260612/architecture_trace`

Moi class co them:

- `09a_foreground_surface_weight.png`;
- `09b_foreground_surface_mask.png`;
- `09c_foreground_surface_edge_detail.png`;
- `09d_foreground_surface_dark_spot.png`;
- `09e_foreground_surface_brown_spot.png`;
- `09f_foreground_surface_bright_spot.png`.

`shapes.json` cua class 1 xac nhan:

- surface stats `[1, 129]`;
- weight map `[1, 1, 64, 64]`;
- mask `[1, 64, 64]`.

Review mau class 1 cho thay mask nam phan lon tren qua, nhung van thu mot phan
nen be/mau vang-nau va bien bong do co mau gan vo xoai. Dark/brown map sau mask
da bot bat vung bong toi thuan tuy, nhung edge cua qua va chi tiet nen van co
the co score cao.

Ba ca close-margin review:

- `008_t0_p1`: class 0 xanh nhat nhung anh sang vang, rat gan class 1;
- `009_t1_p0`: class 1 xanh, anh toi va nen xanh/nam, rat gan class 0;
- `016_t1_p2`: class 1 sang-vang, nhieu dom be mat, rat gan class 2.

Day la bang chung rang loi con lai khong chi la background. Label boundary,
illumination va maturity continuum dang chi phoi class 1.

Audit CSV/anh:

`runs/probe_mango_cls_256_5class_foreground_surface_v13_120b_6e_20260612/class1_confusion_audit_test`

## Quyet dinh

Khong full train V13:

- class-1 validation F1 `0.6822 < 0.70`;
- thap hon V12 `0.6864`;
- test chi dung audit cuoi, khong dung de dao tham so;
- heuristic mask van nhay voi background co mau gan qua.

Khong nen tiep tuc them dai HSV hoac tang residual weight cua V13. Huong do co
nguy co hoc shortcut theo anh sang/nen va tang do phuc tap ma khong tang
boundary separation.

## Huong tiep theo

Thu tu uu tien:

1. Tao train-only label-boundary audit cho 0/1/2, gom high-confidence error,
   close-margin va near-duplicate; khong dua test sample vao train.
2. Thu learned object-tight mask/crop nhe, duoc hoc tu consistency cua hai view
   hoac foreground prior, thay vi threshold HSV co dinh.
3. Thu local patch comparator nhe theo PMG/API-Net: chon patch foreground co
   detail cao, encode cung shared backbone va hoc residual boundary logits.
4. Chi probe gioi han; full train khi validation class-1 F1 dat `>=0.70`.

Tai lieu tham khao:

- Pairwise Confusion: https://arxiv.org/abs/1705.08016
- API-Net: https://arxiv.org/abs/2002.10191
- PMG: https://arxiv.org/abs/2003.03836
- WS-DAN: https://arxiv.org/abs/1901.09891
