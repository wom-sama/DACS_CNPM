from __future__ import annotations

import argparse
from pathlib import Path

from trkh.tools.pair_surface_ddf_v2_execution_guard import build_pending_authorization


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue one reviewed trusted-runner authorization.")
    parser.add_argument("--machine-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Deliberately fail before opening the partial/unreviewed machine lock.
    build_pending_authorization(machine_lock_path=args.machine_lock, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
