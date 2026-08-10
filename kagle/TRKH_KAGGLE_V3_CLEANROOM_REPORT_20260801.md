# TRKH Kaggle B2/T4 V3 clean-room report

Status: **passed locally; real Kaggle v170 T4 acceptance remains pending**.

V3 fixes two observed Kaggle integration faults without changing the locked B2
model, data recipe, optimizer, scheduler, threshold or audit protocol:

- asset input now resolves either an intact ZIP or a Kaggle-mounted expanded
  tree containing `TRKH_KAGGLE_B2_UPLOAD_MANIFEST.json`;
- compact progress sets `TQDM_DISABLE=1`, preventing terminal carriage-return
  bars from becoming repeated notebook lines while retaining epoch metrics.

## Canonical artifacts

- Notebook: `a03188db743ed67b275cf016592d565d0f517a9cfdede67ad8e8ac108534a278`,
  97,116 bytes, 12/12 code cells compiled.
- Asset bundle: `d502efdefa03e76bcae5caff0567f372671a172d6ce350289d993a5e8f8bdb10`,
  86,494,944 bytes, 686 ZIP members and 685 manifest-covered files.
- Asset manifest: `8a3947c70adfdfc90d0fd397d1c0e0ed19ae6b6927f266666d4f13800bf6690b`.
- Dataset bundle unchanged: `c3695cf9dd44b559ce158b1ffee96533d4549c5724188f9df2f864b3d37dfadb`,
  891,390,651 bytes.
- A second independent asset build was byte-identical to the canonical ZIP.

## Input-mode clean rooms

Archive mode passed with the two canonical ZIPs in separate simulated Kaggle
mounts. Safe extraction verified CRC/member/path constraints, then recovered:

- source commit `7f7f0883cbb71b6a5620fee86c15b996c400a813`;
- source tree `7c8752efe6acb728ed913abe5db165b218c94177e3ef13f5b37ce1c62b19d965`;
- DINOv3 weight `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`;
- asset mode `archive_file` and dataset mode `archive_file`.

Expanded mode passed after the asset ZIP was fully unpacked and the compatible
dataset YAML/train/val tree was mounted separately. The notebook found the
manifest recursively, verified all 685 declared asset files and recovered the
same source/DINO identities with modes `kaggle_mounted_expanded`.

Both modes reject ambiguous/multiple representations, symlink/special entries,
asset/data sharing one Kaggle Dataset mount, a `test` path component, a `test`
YAML key, and train/val paths escaping the selected dataset input.

## Progress rendering

The v170 runtime family publishes tqdm 4.67.x. A subprocess probe with
`TQDM_DISABLE=1` produced empty stderr/no dynamic bar while retaining an
explicit epoch metric log. The notebook independently repeats this assertion
before training and records it in the runtime contract.

The remaining acceptance boundary is a fresh top-to-bottom run of the
unmodified V3 notebook on a real Kaggle `NvidiaTeslaT4` session with Internet
OFF and the two canonical inputs attached separately.
