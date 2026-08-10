from __future__ import annotations

import argparse
from pathlib import Path

from trkh.tools.pair_surface_ddf_v2_execution_guard import build_machine_lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reviewed v3 trusted-runner machine lock.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Deliberately call the independent-review gate before reading either path.
    build_machine_lock(candidate_path=args.candidate, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
