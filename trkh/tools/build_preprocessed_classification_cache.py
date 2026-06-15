from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.data.dataset import IMAGE_EXTENSIONS, _suppress_background_image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic preprocessed cache for classification_folder datasets."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to cache.",
    )
    parser.add_argument("--mode", type=str, default="grabcut_desaturate_blur")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--blur-radius", type=float, default=7.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Limit per split/class for smoke cache builds; 0 means all.",
    )
    return parser.parse_args()


def _iter_split_class_images(
    split_root: Path,
    class_name: str,
    *,
    max_samples: int = 0,
) -> List[Path]:
    class_root = split_root / class_name
    if not class_root.exists():
        return []
    paths = [
        path
        for path in sorted(class_root.rglob("*"), key=lambda item: str(item).lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if int(max_samples) > 0:
        return paths[: int(max_samples)]
    return paths


def _build_tasks(
    data_root: Path,
    output_root: Path,
    splits: Iterable[str],
    class_names: List[str],
    *,
    max_samples_per_class: int,
) -> List[Tuple[Path, Path]]:
    tasks: List[Tuple[Path, Path]] = []
    for split in splits:
        split_root = data_root / split
        for class_name in class_names:
            for src in _iter_split_class_images(
                split_root,
                class_name,
                max_samples=max_samples_per_class,
            ):
                relative = src.relative_to(data_root)
                tasks.append((src, output_root / relative))
    return tasks


def _save_image(image: Image.Image, dst: Path, quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = dst.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".webp"}:
        image.save(dst, quality=max(1, min(100, int(quality))))
    else:
        image.save(dst)


def _process_one(
    src: Path,
    dst: Path,
    *,
    mode: str,
    margin: float,
    blur_radius: float,
    quality: int,
    overwrite: bool,
) -> Dict[str, object]:
    if dst.exists() and not overwrite:
        return {"status": "skipped_exists", "src": str(src), "dst": str(dst)}
    try:
        with Image.open(src) as img:
            image = img.convert("RGB").copy()
        processed = _suppress_background_image(
            image,
            mode=mode,
            margin=margin,
            blur_radius=blur_radius,
        )
        _save_image(processed, dst, quality=quality)
        return {"status": "ok", "src": str(src), "dst": str(dst)}
    except Exception as exc:
        return {"status": "error", "src": str(src), "dst": str(dst), "error": str(exc)}


def _write_data_yaml(
    output_root: Path,
    source_payload: Dict[str, object],
    class_names: List[str],
    *,
    source_data: Path,
    mode: str,
    margin: float,
    blur_radius: float,
) -> None:
    payload = dict(source_payload)
    payload["path"] = "."
    payload["format"] = "classification_folder"
    payload["train"] = "train"
    payload["val"] = "val"
    payload["test"] = "test"
    payload["nc"] = len(class_names)
    payload["class_name_mode"] = "raw"
    payload["names"] = {int(index): name for index, name in enumerate(class_names)}
    payload["preprocessing_cache"] = {
        "source_data": str(source_data),
        "mode": str(mode),
        "margin": float(margin),
        "blur_radius": float(blur_radius),
        "leakage_note": (
            "Deterministic per-image preprocessing; no cross-split statistics and no label-derived "
            "parameters beyond preserving the folder structure."
        ),
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
    )
    source_payload = yaml.safe_load(Path(args.data).read_text(encoding="utf-8")) or {}
    data_root = data_spec.root
    output_root = Path(args.output_root)
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    tasks = _build_tasks(
        data_root,
        output_root,
        splits,
        data_spec.class_names,
        max_samples_per_class=int(args.max_samples_per_class),
    )

    summary: Dict[str, object] = {
        "source_data": str(args.data),
        "source_root": str(data_root),
        "output_root": str(output_root),
        "splits": splits,
        "mode": str(args.mode),
        "margin": float(args.margin),
        "blur_radius": float(args.blur_radius),
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "max_samples_per_class": int(args.max_samples_per_class),
        "tasks": len(tasks),
    }
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    status_counts: Dict[str, int] = {}
    errors: List[Dict[str, object]] = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_one,
                src,
                dst,
                mode=str(args.mode),
                margin=float(args.margin),
                blur_radius=float(args.blur_radius),
                quality=int(args.quality),
                overwrite=bool(args.overwrite),
            )
            for src, dst in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="preprocess-cache"):
            result = future.result()
            status = str(result.get("status", "unknown"))
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            if status == "error" and len(errors) < 20:
                errors.append(result)

    _write_data_yaml(
        output_root,
        source_payload,
        data_spec.class_names,
        source_data=Path(args.data),
        mode=str(args.mode),
        margin=float(args.margin),
        blur_radius=float(args.blur_radius),
    )
    canbang_src = Path(args.data).with_name("canbang.yaml")
    if canbang_src.exists():
        shutil.copy2(canbang_src, output_root / "canbang.yaml")

    summary["status_counts"] = status_counts
    summary["errors"] = errors
    (output_root / "preprocessing_cache_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "status_counts": status_counts}, ensure_ascii=True))
    return 1 if status_counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
