# C4 Rotation-Consensus A0 Closure (2026-07-20)

## Decision

Reject fixed C4 probability consensus and close the proposed four-pass C4 stem
orbit on the current no-pretrain keeper. The sole train-only A0 fails 12 of 14
mechanism gates. It does not authorize model integration, validation, shifted-
condition replay, XAI, a smoke, a probe, test access, or full training.

## Locked Evidence

- Accepted source: Cohen and Welling, ICML-2016 G-CNN paper.
- Official implementation reference: QUVA-Lab/e2cnn at commit/tree
  `022d6ca4...40238`/`7673db4f...df6ff`; no source or dependency was copied.
- Prospective protocol SHA-256: `e4852c34...9bb3`.
- Protocol commit: `df4d265`; implementation commit: `c0941ad`.
- Keeper SHA-256: `1f49d577...2677`.
- Formal summary SHA-256: `7695af82...3dfb`.
- Artifact manifest SHA-256: `9807a7b4...16ac`.
- Prediction CSV SHA-256: `9154f82f...a1a6`.

The formal opens all `9,215` `yolo_f/train` object rows exactly once and runs
the frozen keeper on `0/90/180/270` degree tensor rotations. The valid-image
mask follows the tensor; source-coordinate bbox remains fixed. The candidate
is the arithmetic mean of four softmax vectors. No angle, weight, threshold,
temperature, router, loss, seed, or checkpoint is fitted.

All structural gates pass. Targets, paths, source folds, and baseline
predictions match the existing clean CIDT cache exactly; maximum probability
difference is `8.94e-8`, below the locked `2e-5` tolerance. Validation/test
filenames are scanned only for source overlap, which is zero. Their pixels,
labels, predictions, and metrics remain unopened.

## Result

| Metric | Baseline K=0 | C4 mean | Delta |
|---|---:|---:|---:|
| Accuracy | 0.961910 | 0.956158 | -0.005751 |
| Macro F1 | 0.939876 | 0.931097 | -0.008779 |
| Class-1 precision | 0.700265 | 0.689133 | -0.011132 |
| Class-1 recall | 0.975970 | 0.926063 | -0.049908 |
| Class-1 F1 | 0.815444 | 0.790221 | -0.025223 |
| NLL | 1.163127 | 1.178503 | +0.015376 |
| Brier | 0.590597 | 0.598733 | +0.008137 |

C4 changes 173 decisions. It produces `54` corrections and `107` harms,
rescues one class-1 FN, and breaks 28 class-1 TP. It removes 53 restricted
`{0,2,4}->1` FP but creates 52, for only one net removal. The dominant change
is `1->0=69`, followed by `0->1=35`, `2->1=14`, and `1->2=12`.

Macro F1 decreases in all five source folds. Class-1 precision decreases in
four folds and improves only `+0.00379` in fold 3. Restricted-FP net removals
are `-4/+3/0/+4/-2`; class-1 net TP losses are `2/9/4/4/8`. The result is not
driven by one source fold.

## Mechanism Diagnosis

The full train set has high overall four-angle top-1 agreement (`0.93077`), but
the ambiguous cohorts are less stable: class-1 TP agreement is `0.78977` and
restricted-FP agreement is `0.55405`. Instability alone is therefore real, but
it is not a reliable negative-evidence signal.

Class-1 TP probability falls by mean `-0.01835`; restricted-FP probability
falls only `-0.01154`. The required separation is reversed by `-0.00682`.
`AUROC(-delta_p1)=0.42794`, and rotation-JS AUROC is `0.48461`. Averaging
orientation evidence suppresses true class 1 more strongly than its false
positives. This explains the recall loss and rules out interpreting cleaner
individual transitions as precision selectivity.

The earlier PDisco experiment showed that an equivariance loss can improve a
rotation-consistency statistic while collapsing useful foreground behavior.
This A0 now shows that fixed orbit averaging also lacks class-conditional
direction. Together they close nearby C4/D4, angle-subset, rotation-direction,
logit/probability/vote aggregation, temperature, bbox-policy, threshold,
router, regularizer, and seed variants on the current keeper.

## Runtime And Replay

Formal inference takes `355.995 s`, processes `25.885` source images/s or
`103.541` forward views/s, and peaks at `1.1326 GiB` CUDA allocation. Requested
workers are `4`; the Windows-safe loader clamps this read-only audit to zero
effective workers. This is recorded performance evidence, not a reason to
rerun a failed mechanism. The measured full-train recommendation remains
workers `4/2` for train/eval.

Independent replay reconstructs all metrics and gates from the prediction CSV
with maximum numerical difference `0.0` and verifies all four payload hashes.
The complete evidence is only `7,878,340` bytes, so it is retained in place at
`runs/audit_c4_rotation_consensus_a0_20260720`; no compaction or deletion is
needed.

The post-closure read-only retention audit covers 779 run directories and all
50 valid compaction manifests. All 219 manifest-derived original directories
remain absent, `deleted_anything=false`, and `blockers=[]`. Its summary
SHA-256 is `93c13f336e0ff26cbbd0fa9edc2a7ac5a142938a598a9cc68a04cfb65918389d`.

Current-best checkpoint, full-pipeline command, and update-history hashes stay
`1f49d577...2677`, `36b9aa1a...0faf`, and `39bd2879...8f53`.
