# TRKH 5-Class Class-1 Boundary CAGrad Readiness Protocol

Date: 2026-07-15
Status: immutable before auditor implementation or candidate execution
Scope: source-disjoint `yolo_f/train` Stage A only; validation and test forbidden

## Question

Can one conflict-averse multi-condition gradient step preserve the average
boundary objective while improving the worst local clean/dim/bright/
low-contrast descent direction, thereby suppressing class-1 false positives
without losing true class-1 support?

This is a fail-closed information gate. It is not permission to add CAGrad to
the shared trainer, run validation, run test, launch a probe/full train, or
update the current-best command packet.

## Primary evidence and transfer risk

- Paper: Liu et al., *Conflict-Averse Gradient Descent for Multi-task
  Learning*, NeurIPS 2021. Local SHA-256:
  `ad50131ada07c8c1b68fd7e31c3226f0a8980b667fc388c71e5be660f917098f`.
- Official repository:
  `D:/DataAI/external_sources/CAGrad`, commit
  `dc3d48152b6196945cfd56144879b9d42353b095`.
- Locked supervised-vision implementation:
  `nyuv2/utils.py`, SHA-256
  `b4ab48f2409979bdd07c31ec7712421d9a6f19462fdfe0d4e82d28d9a477eef5`.
- Locked supervised-vision recipe:
  `nyuv2/run.sh`, SHA-256
  `571d1abf065bd23ff5bbd6b9e81f1e530ce7ac97dca8d2f4dac3b11f36f2a714`.
- Locked license SHA-256:
  `057e0b7c6233c0ad86a3b8998333664967225ee761836407385d042d2583716c`.

The paper is multi-task learning evidence, not a mango-domain guarantee. It
searches `c` and reports `c=0.4` for the three-task NYU-v2 vision experiment.
The local A0 transfers that single value without tuning. No other `c` is
permitted.

The Algorithm 1 box writes `g_w=(1/K) sum_i w_i g_i`, while the derivation and
official supervised source use `g_w=sum_i w_i g_i`. This positive `1/K` scale
does not change the simplex minimizer and cancels in
`c ||g_0|| g_w / ||g_w||`. The auditor must verify paper/code direction
invariance rather than silently assume it.

## No-repeat boundary

- Direct quality/cartography GroupDRO, V-REx beta `1`, generic consistency,
  and illumination canonicalization are closed.
- A-GEM and original GEM optimized hard-negative versus class-1 memory
  constraints. They did not optimize four same-row condition objectives.
- The prior joint spatial PCGrad run combined classification with localization
  and objectness, then failed its matched image smoke. This A0 has no auxiliary
  task and does not reopen PCGrad, GradNorm, detector curriculum, or spatial
  gradients.
- CAGrad is admissible once because it explicitly maximizes the worst local
  task improvement inside a radius around the average gradient and, for
  `0 <= c < 1`, retains the paper's average-objective convergence target.

If this A0 fails, close condition-level CAGrad/MGDA/PCGrad/Agr/GradNorm and all
nearby `c`, task-weight, solver, step, or macrostep variants on this keeper.

## Immutable local inputs

- Keeper checkpoint:
  `runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt`
  SHA-256
  `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`.
- Keeper launcher args SHA-256:
  `908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff`.
- `yolo_f/data.yaml` SHA-256:
  `716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef`.
- CIDT summary/prediction SHAs:
  `d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad`,
  `2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c`.
- Prior V-REx summary/prediction/manifest SHAs:
  `0b78da68af731456faaca8664a4b97dc111121a0a5c7630cfede1141e800babb`,
  `7ca25f7ffad843e7cc4da4cc70567c1372209f2b28e4710599536ea42edc8de9`,
  `1ed6bec4753f0f581f0bed0c558f4a65814ca6692524cdc56a935b8e8f1d3f1e`.
- Prior V-REx evidence must report no validation/test use, no binary model,
  exact `4 x 1843` holdout rows, and zero prior argmax/CIDT mismatch.

## Immutable cohort and environments

- Source-disjoint CIDT fold `0`: fit/holdout `7372/1843`, source groups
  `6452/1612`, overlap empty.
- Fit restricted hard negatives: all `186` keeper `0/2/4 -> 1` rows from
  folds `1..4`.
- Fit class-1 positives: all `432` class-1 rows from folds `1..4`.
- Holdout boundary-risk rows: the immutable `36` restricted keeper false
  positives and all `109` class-1 rows in fold `0`.
- Conditions, with identical sample indices and labels:
  clean `(1.00, 1.00)`, dim `(0.70, 0.90)`, bright `(1.25, 1.10)`, and low
  contrast `(1.00, 0.65)`.
- Eval transform, object crop, bbox semantics, image size, and class order come
  from the keeper checkpoint. Batch size is `32`, workers `4`, seed `42`,
  deterministic CUDA, FP32/no-TF32 gradient accumulation.

## Locked objectives and CAGrad equation

For environment `e`, compute row-mean CE gradients and risks:

`H_e = mean CE(hard_e)`

`P_e = mean CE(class1_e)`

`R_e = 0.5 H_e + 0.5 P_e`

`g_e = grad R_e`

For `K=4`, define `g_0 = (1/K) sum_e g_e`, `c=0.4`, and probability simplex
`W={w: w_e >= 0, sum_e w_e=1}`. Use the official-code form
`g_w=sum_e w_e g_e` and solve in FP64:

`w* = argmin_w g_w^T g_0 + c ||g_0|| ||g_w||`.

The candidate direction is:

`d = g_0 + c ||g_0|| g_w / ||g_w||`.

The primary solver is deterministic SciPy SLSQP from the uniform simplex,
with analytic Gram-matrix objective/Jacobian, `ftol=1e-12`, and at most `1000`
iterations. Independent checks must include all four simplex vertices plus
four deterministic interior starts, official-source epsilon/rescale replay,
paper `g_w/K` scale invariance, KKT residual, and objective agreement. Solver
failure or inconsistent solutions closes the route.

Apply one all-parameter candidate step with normalized parameter ratio
`1e-4`, no optimizer, scheduler, augmentation, teacher, or saved model. The
prior environment-mean ERM and beta-1 V-REx predictions remain immutable
comparators. Candidate and locked ERM actual step norms may differ by at most
`5e-7` relative after FP32 parameter quantization.

## Structural and first-order gates

Every item must pass:

- all source/protocol hashes and official commit/worktree are exact;
- only source-disjoint train rows are constructed; no val/test path is read;
- all eight cohort gradients and all four `g_e` are finite and nonzero;
- every trainable parameter appears in every gradient and candidate direction;
- simplex sum error `<=1e-10`, minimum weight `>=-1e-10`, solver success,
  multistart objective spread `<=1e-10`, and KKT residual `<=1e-7`;
- paper/code scaled-direction relative error `<=1e-10` and official-source
  replay relative error `<=1e-6`;
- `||d-g_0|| / ||g_0||` differs from `0.4` by at most `1e-7`;
- CAGrad relative L2 from ERM is at least `0.35`;
- minimum task dot `min_e <g_e,d>` is nonnegative and exceeds the ERM minimum
  by at least `1e-6`;
- average-objective first-order change is negative;
- candidate actual parameter ratio is within `1e-8` of `1e-4`, and its update
  norm matches locked ERM within `5e-7` relative;
- initial state/schema are exact; no binary artifact or raw-data write exists.

## Clean behavior gates

Against the raw keeper on the `1843` holdout rows, all must pass:

- macro F1 delta `>=0`;
- class-1 F1 delta `>=+0.005`;
- class-1 precision delta `>=+0.005`;
- class-1 recall delta `>=-0.005`;
- at least four restricted `0/2/4 -> 1` false positives removed;
- corrections exceed harms; class-1 rescues are at least TP breaks;
- predicted class-1 support is at least `95%` of raw;
- maximum non-class-1 F1 drop `<=0.010`;
- class-1 F1 exceeds both locked ERM and V-REx by at least `0.005`;
- class-1 F1 is within `0.002` of aggregate-margin A-GEM;
- TP breaks do not exceed either locked ERM or V-REx.

## Illumination and risk gates

Across dim, bright, and low contrast, all must pass:

- every class-1 F1 delta versus raw is `>=-0.010`;
- every class-1 recall delta versus raw is `>=-0.015`;
- class-1 precision is nondecreasing in at least two conditions;
- no condition increases restricted class-1 false positives;
- aggregate rescues are at least aggregate TP breaks;
- worst-condition class-1 F1 gains at least `0.010` versus raw and `0.005`
  versus both locked ERM and V-REx;
- candidate boundary-risk mean is at most ERM mean plus `0.005`;
- candidate worst boundary risk is no greater than locked ERM worst risk.

## Artifacts and decision

Write only nonbinary `summary.json`, `report.md`,
`predictions_all_conditions.csv`, `gradient_groups.csv`,
`solver_starts.csv`, and `artifact_manifest.json` under a unique run directory.
Hash every artifact and independently replay the CSV, solver objective,
metrics, transitions, risks, and argmax before closure.

Any failed item sets status `closed`, denies Stage B/validation/test/probe/full
train/current-best command promotion, and forbids a neighboring sweep. Only a
complete pass authorizes a separately precommitted no-test Stage B smoke.
