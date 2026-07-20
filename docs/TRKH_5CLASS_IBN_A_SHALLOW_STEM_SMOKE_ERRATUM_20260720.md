# Shallow IBN-a matched-smoke resume erratum (2026-07-20)

## Scope

This erratum is prospective: it is committed before either matched-smoke role
is trained and does not change any metric, robustness, XAI, runtime, or stop
gate in `TRKH_5CLASS_IBN_A_SHALLOW_STEM_PROTOCOL_20260720.md`.

## Reason

The locked keeper `launcher_args.json` resumes from:

`runs\mango_cls_256_5class_hardneg_maskfix_v4_30e\checkpoints\best.pt`

That run was intentionally deleted on 2026-07-02 and was not byte-preserved.
The deletion is recorded in
`runs\cleanup_manifest_20260702_legacy_mango_cls_large_runs.json`. A full-drive
search on 2026-07-20 found no recoverable copy. Therefore, replaying the old
v4 initialization would be falsely described as reproducible.

## Locked repair

Both smoke roles instead resume from the current no-pretrain keeper:

`runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt`

Its locked SHA-256 is:

`1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`

The launcher still derives every other train argument from the keeper's
`launcher_args.json`. Both roles use `--resume-use-cli-config` and reset epoch,
optimizer, scheduler, and scaler. The common resume-path repair and run names
are provenance-only differences from that historical argument list. The sole
control-versus-candidate model difference remains:

`stem_normalization=batch -> ibn_a_first`

No test split is allowed. Validation support remains 2,606. The smoke budget
remains two epochs and 120 train batches per epoch with workers `4/2`. A failed
original gate closes this exact candidate; the erratum does not authorize a
ratio, placement, seed, loss, augmentation, or threshold sweep.
