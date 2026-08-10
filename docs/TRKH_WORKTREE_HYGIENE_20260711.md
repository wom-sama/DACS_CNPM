# TRKH Worktree Hygiene Plan - 2026-07-11

## Baseline

The immutable pre-cleanup inventory is under
`runs\worktree_hygiene_baseline_20260711` and contains SHA-256 hashes for every
tracked modification and untracked file.

| Item | Baseline |
|---|---:|
| Branch | `classification-only-research` |
| HEAD/upstream | `7020f0393a8d4260fc7c37372268ae9f27086224` |
| Ahead / behind | `0 / 0` |
| Staged files | `0` |
| Tracked modifications | `33` (`+45,782/-6,020`) |
| Untracked files | `269` (`12.336 MiB`) |
| Untracked tests / TRKH files | `147 / 85` |
| `git diff --check` findings | `0` |

Git is configured with `core.autocrlf=true`, while `.gitattributes` currently
uses `* text=auto`. The 33 stderr warnings are conversion notices, not
whitespace failures. Do not use a bulk formatter or renormalization commit
while behavior changes remain uncommitted.

## Protected Scope

- Never revert or delete a path whose ownership or purpose is unclear.
- Keep raw `class_f`/`yolo_f`, selected checkpoints, compact evidence,
  cleanup manifests, and retention audits outside source-cleanup actions.
- Treat `BaoCao/*` and root `deep-research-report (9|10).md` as user-owned until
  their intended repository scope is independently established.
- Keep generated-run cleanup separate from Git cleanup. It requires a preserved
  summary/XAI conclusion, a cleanup manifest, and a passing retention rerun.
- Do not stage with `git add .`, `git add -A`, wildcard directories, or an
  interactive console. Every batch uses an explicit reviewed path list.

## Scientific Batch Protocol

1. Record the batch objective, paths, dependencies, and tests before staging.
2. Confirm each path still matches the baseline hash or explain the later edit.
3. Run syntax/parse checks and the narrow tests for that batch.
4. Run broader regression tests when shared config, model, dataset, evaluator,
   inference, or trainer contracts are involved.
5. Run `git diff --check`, then stage the explicit path list only.
6. Inspect `git diff --cached --stat`, `--name-status`, and the full cached diff.
7. Commit only when the staged snapshot is self-consistent and reproducible.
8. Recount the worktree and append the result to the research journal.

## Planned Batches

| Order | Batch | Scope | Gate |
|---:|---|---|---|
| 1 | Diagnostic closure | Multi-stage readiness tool/test and its journal/TODO records | Compile, focused tests, no-test artifact review |
| 2 | Command/deployment safety | Full pipeline/export/video wrappers, inference fallback, command tests/docs | AST parse, preflights, focused inference tests |
| 3 | Shared research runtime | Config, data, model, losses, trainer, evaluator/XAI changes | Full collection plus broad regression suite |
| 4 | Research tools and tests | Reproducible builders/probes/audits and paired tests | Per-family tests plus import/CLI checks |
| 5 | Research documentation | Audits, reports, reference pack and hashes | Link/path/hash verification |

Batch boundaries may move when dependency inspection shows a file belongs to
another batch. The priority is a buildable history, not minimizing commit count.
No current-best command changes belong in any batch unless a new checkpoint
passes the independent locked-validation promotion gate.

## Progress

### Batch 01 - Verified research runtime

- Scope: 275 explicit paths covering source, config, scripts, tests, and the
  current command TXT required by its regression test.
- Verification: compileall passed; PowerShell AST `33/33`; focused fixes
  `20/20` plus `1/1`; complete pytest `620/620`; cached diff check passed;
  prohibited model artifact and secret-signature scans were empty.
- Review artifact: `runs\worktree_hygiene_baseline_20260711`, staged patch
  SHA-256 `6100001308b16567b0ac745b24ab6beb9ec9314ee246dbf62830e42a64ae9a6c`.
- Commit: `4c2c7cb824b48a76c9d796acaae944bf41369694`, 275 files,
  `+105,611/-6,030`.
- Result: worktree entries `302 -> 28`, staged entries `0`. Research docs and
  unclear user-owned files remained outside the index.

### Batch 02 - Research evidence and documentation

- Scope: 21 explicit paths under `docs`; all `BaoCao/*` and root reports 9/10
  remained excluded.
- Verification: UTF-8/conflict/relative-link checks passed; ZIP companion hash
  matched; all 266 payload stream hashes and 267 entries passed; reports 14/15
  matched their embedded copies byte-for-byte. Fifteen intentional Markdown
  hard breaks in verbatim report 15 were the complete whitespace allowlist.
- Commit: `7548c1f72ad7837d0adbb10b99bc3be4f81cdef9`, 21 files and 18,990 added
  lines.
- Push: local and `origin/classification-only-research` both resolved to
  `7548c1f`; ahead/behind `0/0`.
- Result: seven deliberately protected untracked paths remain, staged entries
  `0`. They are not deleted, ignored, staged, or reclassified without a later
  ownership decision.

### Batch 03 - Incremental research closures

- Scope: subsequent model/readiness stages were kept in dependency-coherent
  commits rather than rebuilding a large dirty batch. Each stage used explicit
  path staging, cached-diff review, focused plus full tests, evidence hashes,
  generated-run cleanup manifests, retention reruns, and push verification.
- Latest closure: same-class Fourier amplitude mix protocol/tool/tests/audit and
  journal/TODO records, commit `a2e3da34b6d35c0fe79f40bbfffcf19113649944`
  (`6` explicit paths, `+2,138`), pushed to
  `origin/classification-only-research`.
- Verification: exact full no-test CSV/payload reconstruction; compileall;
  focused tests `6/6`; complete pytest `820/820`; cached diff check and staged
  secret/binary scans passed. Keeper SHA remains `1f49d577...482677` and the
  current command SHA remains `3c718130...57dd`.
- Generated-run cleanup stayed separate from Git. The Fourier cleanup manifest
  hashed and removed only two superseded preflights (`18` files,
  `6,337,363` bytes); the retained full evidence SHA is
  `7a82f434...5020e`. Retention passed over `605` run directories and all `28`
  object compaction manifests with `blockers=[]`.
- Result after the research closure commit: local and origin both pointed to
  `a2e3da3`, and staged entries were `0`. This ledger follow-up is docs-only.
  The only remaining non-ledger worktree entries are the three deliberately
  protected user paths
  `BaoCao/`, `deep-research-report (9).md`, and
  `deep-research-report (10).md`. They remain untracked and untouched.
