from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from utils import ensure_dir, json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chay ablation grid cho ViT-Registers.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--study-name", type=str, default="vit640_ablation")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--train-script", type=Path, default=Path(__file__).resolve().parent / "train.py")
    parser.add_argument("--eval-script", type=Path, default=Path(__file__).resolve().parent / "evaluate.py")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--register-grid", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--register-pos-grid", nargs="+", default=["false", "true"])
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true", default=False)
    parser.add_argument("--skip-eval", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Cac arg truyen them cho train.py sau dau --",
    )
    return parser.parse_args()


def parse_bool_grid(values: List[str]) -> List[bool]:
    mapping = {
        "0": False,
        "1": True,
        "false": False,
        "true": True,
        "off": False,
        "on": True,
        "no": False,
        "yes": True,
    }
    resolved = []
    for value in values:
        normalized = value.strip().lower()
        if normalized not in mapping:
            raise ValueError(f"Gia tri bool khong hop le trong grid: {value}")
        resolved.append(mapping[normalized])
    return resolved


def normalize_train_args(train_args: List[str]) -> List[str]:
    if train_args and train_args[0] == "--":
        return train_args[1:]
    return train_args


def run_command(command: List[str], cwd: Path) -> None:
    print({"command": command})
    subprocess.run(command, cwd=str(cwd), check=True)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_markdown_table(rows: List[Dict[str, object]]) -> str:
    lines = [
        "| run_name | registers | register_pos_emb | macro_f1 | weighted_f1 | accuracy | reg/patch | high_patch_frac | best_epoch |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {num_registers} | {register_positional_embedding} | "
            "{macro_f1:.4f} | {weighted_f1:.4f} | {accuracy:.4f} | "
            "{register_to_patch_ratio:.4f} | {high_norm_patch_fraction:.4f} | {best_epoch} |".format(
                **row
            )
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_row(
    run_name: str,
    run_dir: Path,
    num_registers: int,
    register_positional_embedding: bool,
    split: str,
) -> Dict[str, object]:
    metrics_path = run_dir / f"eval_{split}_ablation" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "best_metrics.json"
    metrics = load_json(metrics_path)
    summary = load_json(run_dir / "summary.json")
    resolved_config = load_json(run_dir / "resolved_config.json")
    artifact_stats = metrics.get("artifact_stats", {})

    return {
        "run_name": run_name,
        "num_registers": num_registers,
        "register_positional_embedding": "true" if register_positional_embedding else "false",
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "register_to_patch_ratio": float(artifact_stats.get("register_to_patch_ratio", 0.0)),
        "high_norm_patch_fraction": float(artifact_stats.get("high_norm_patch_fraction", 0.0)),
        "best_epoch": int(summary.get("best_epoch", 0)),
        "parameter_count": int(resolved_config.get("parameter_count", 0)),
        "run_dir": str(run_dir.resolve()),
    }


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    base_train_args = normalize_train_args(args.train_args)
    register_pos_grid = parse_bool_grid(args.register_pos_grid)

    summary_dir = ensure_dir(args.output_dir / f"{args.study_name}_summary")
    rows: List[Dict[str, object]] = []

    for num_registers in args.register_grid:
        for register_positional_embedding in register_pos_grid:
            run_name = (
                f"{args.study_name}_r{num_registers}_"
                f"regpos{1 if register_positional_embedding else 0}"
            )
            run_dir = args.output_dir / run_name
            checkpoint_path = run_dir / "checkpoints" / "best.pt"

            if not args.skip_train:
                should_skip_train = args.skip_existing and checkpoint_path.exists()
                if not should_skip_train:
                    train_command = [
                        str(args.python),
                        str(args.train_script),
                        "--run-name",
                        run_name,
                        "--num-registers",
                        str(num_registers),
                        *base_train_args,
                    ]
                    if register_positional_embedding:
                        train_command.append("--register-positional-embedding")
                    run_command(train_command, cwd=project_dir)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Khong tim thay checkpoint sau train: {checkpoint_path}")

            eval_dir = run_dir / f"eval_{args.split}_ablation"
            if not args.skip_eval:
                should_skip_eval = args.skip_existing and (eval_dir / "metrics.json").exists()
                if not should_skip_eval:
                    eval_command = [
                        str(args.python),
                        str(args.eval_script),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--split",
                        args.split,
                        "--batch-size",
                        str(args.eval_batch_size),
                        "--num-workers",
                        str(args.eval_num_workers),
                        "--output-dir",
                        str(eval_dir),
                    ]
                    if args.override_image_size is not None:
                        eval_command.extend(["--override-image-size", str(args.override_image_size)])
                    run_command(eval_command, cwd=project_dir)

            rows.append(
                collect_row(
                    run_name=run_name,
                    run_dir=run_dir,
                    num_registers=num_registers,
                    register_positional_embedding=register_positional_embedding,
                    split=args.split,
                )
            )

    rows = sorted(rows, key=lambda row: (row["num_registers"], row["register_positional_embedding"]))
    payload = {
        "study_name": args.study_name,
        "split": args.split,
        "rows": rows,
    }
    json_dump(summary_dir / "summary.json", payload)
    write_csv(summary_dir / "summary.csv", rows)
    (summary_dir / "summary.md").write_text(build_markdown_table(rows), encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
