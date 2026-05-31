from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

from trkh.core.utils import (
    plot_all_training_metrics,
    plot_detection_training_metrics,
    plot_per_class_training_metrics,
    plot_training_history,
    plot_validation_convergence,
)


MAXIMIZE_METRICS = (
    "selection_metric",
    "val_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "val_detection_f1_50",
    "val_best_detection_f1_50",
    "val_detection_precision_50",
    "val_detection_recall_50",
    "val_bbox_iou",
    "val_bbox_giou",
)

MINIMIZE_METRICS = (
    "train_loss",
    "val_loss",
    "val_count_mae",
    "val_count_rmse",
)


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _infer_class_names_from_history(rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    indices = []
    for field_name in rows[0].keys():
        prefix = "val_class_"
        suffix = "_f1"
        if field_name.startswith(prefix) and field_name.endswith(suffix):
            index_text = field_name[len(prefix) : -len(suffix)]
            if index_text.isdigit():
                indices.append(int(index_text))
    if not indices:
        return []
    return [f"class_{index}" for index in range(max(indices) + 1)]


def _load_class_names(run_dir: Path, rows: Sequence[Dict[str, str]]) -> List[str]:
    config = _read_json(run_dir / "resolved_config.json")
    data_config = config.get("data")
    if isinstance(data_config, dict):
        class_names = data_config.get("class_names")
        if isinstance(class_names, list) and all(isinstance(name, str) for name in class_names):
            return list(class_names)
    return _infer_class_names_from_history(rows)


def _read_history(history_csv: Path) -> List[Dict[str, str]]:
    if not history_csv.exists():
        raise FileNotFoundError(f"Missing history CSV: {history_csv}")
    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _best_metric(
    rows: Sequence[Dict[str, str]],
    metric_name: str,
    *,
    maximize: bool,
) -> Optional[Dict[str, float]]:
    candidates: List[Tuple[int, float]] = []
    for row in rows:
        value = _row_float(row, metric_name)
        epoch = _row_float(row, "epoch")
        if value is None or epoch is None:
            continue
        candidates.append((int(epoch), value))
    if not candidates:
        return None
    if maximize:
        epoch, value = max(candidates, key=lambda item: item[1])
    else:
        epoch, value = min(candidates, key=lambda item: item[1])
    return {"epoch": epoch, "value": value}


def _write_history_summary(
    *,
    run_dir: Path,
    rows: Sequence[Dict[str, str]],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    first_epoch = int(float(rows[0]["epoch"])) if rows else None
    last_epoch = int(float(rows[-1]["epoch"])) if rows else None
    summary: Dict[str, object] = {
        "run_dir": str(run_dir),
        "history_rows": len(rows),
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "class_names": list(class_names),
        "best": {},
        "lowest": {},
    }
    best = summary["best"]
    lowest = summary["lowest"]
    assert isinstance(best, dict)
    assert isinstance(lowest, dict)
    for metric_name in MAXIMIZE_METRICS:
        metric = _best_metric(rows, metric_name, maximize=True)
        if metric is not None:
            best[metric_name] = metric
    for metric_name in MINIMIZE_METRICS:
        metric = _best_metric(rows, metric_name, maximize=False)
        if metric is not None:
            lowest[metric_name] = metric

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def render_history_artifacts(
    run_dir: Path,
    *,
    class_names: Optional[Sequence[str]] = None,
    write_summary: bool = True,
) -> List[Path]:
    history_csv = run_dir / "history.csv"
    rows = _read_history(history_csv)
    resolved_class_names = list(class_names) if class_names is not None else _load_class_names(run_dir, rows)

    outputs = [
        run_dir / "training_curves.png",
        run_dir / "results.png",
        run_dir / "all_training_metrics.png",
        run_dir / "per_class_training_metrics.png",
        run_dir / "detection_training_metrics.png",
        run_dir / "validation_convergence.png",
    ]

    plot_training_history(history_csv, outputs[0])
    plot_training_history(history_csv, outputs[1])
    plot_all_training_metrics(history_csv, outputs[2])
    plot_per_class_training_metrics(history_csv, resolved_class_names, outputs[3])
    plot_detection_training_metrics(history_csv, outputs[4])
    plot_validation_convergence(history_csv, outputs[5])

    generated = [path for path in outputs if path.exists()]
    if write_summary:
        summary_path = run_dir / "history_summary.json"
        _write_history_summary(
            run_dir=run_dir,
            rows=rows,
            class_names=resolved_class_names,
            output_path=summary_path,
        )
        generated.append(summary_path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render run-level training plots from a completed or interrupted history.csv.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to runs/<run_name> containing history.csv.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names. Defaults to resolved_config.json data.class_names, then class_N from history columns.",
    )
    parser.add_argument("--no-summary", action="store_true", help="Do not write history_summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = render_history_artifacts(
        args.run_dir,
        class_names=args.class_names,
        write_summary=not args.no_summary,
    )
    print({"generated": [str(path) for path in generated]}, flush=True)


if __name__ == "__main__":
    main()
