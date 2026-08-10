# TRKH 5-Class Pair-Surface DDF v2 Runtime-Guard Test-Attestation Invalidation - 2026-07-29

## Decision

The developmental `55 passed` runtime-guard result and its local manifest are
withdrawn as reproducibility, guard-readiness and pre-lock evidence. This is
not an expansion of the earlier live-data incident.

Classification:

> DEVELOPMENTAL TEST-ATTESTATION INVALIDATED; ZERO DEMONSTRATED LIVE ACCESS;
> ZERO SCIENTIFIC, AUTHORITY OR QUOTA EFFECT.

## Cause

The manifest pinned six implementation/test files but omitted the transitively
imported `pair_surface_ddf_v2_prelock_validator.py`. Runtime guard test 52
imported that live module and expected its then-current API-v1 behavior. The
pre-lock verifier later changed in the shared worktree while every hash listed
in the guard manifest remained unchanged. Consequently the claimed exact suite
was no longer reproducible from the manifest's declared import closure.

The affected local manifest has SHA-256
`f36f95e2654a8bf8708f418716a99b6156947864632427176a00baa2d9e4c6af`.
It remains a retained developmental artifact only and must not be referenced
as a PASS gate. Its zero-access counters do not repair the incomplete source
identity.

## Required replacement

The trusted-runner v3 test must:

1. replace the live API-v1 import with an immutable synthetic-v1 fixture;
2. prospectively pin the full local import closure and exact expanded node
   registry;
3. install the forbidden-root tripwire before any local or pytest import;
4. run in a fresh isolated no-bytecode process and account for every child
   process or forbid children;
5. derive collection, negative-test and access counts instead of hard-coding
   them; and
6. emit a new create-once manifest under a new schema only after the reviewed
   sources and authority are committed and pushed.

The stale manifest cannot authorize a machine lock, S1, S2, formal A0, replay,
XAI, integration, full training, validation/test access or cleanup.
