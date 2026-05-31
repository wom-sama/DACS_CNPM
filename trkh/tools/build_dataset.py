from __future__ import annotations

import argparse
import hashlib
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
LOGGER = logging.getLogger("build_dataset")


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class DataPair:
    image_path: Path
    label_path: Optional[Path]
    labels: Tuple[YoloLabel, ...]
    key: str

    @property
    def primary_class(self) -> Optional[int]:
        return self.labels[0].class_id if self.labels else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO-style dataset with train/val/test splits, data.yaml, and canbang.yaml."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing source images.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory containing YOLO .txt labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--audit-dir", type=Path, default=None, help="Optional directory for annotated audit images.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--names", nargs="*", default=None, help="Class names in class-id order.")
    parser.add_argument("--classes-file", type=Path, default=None, help="Optional classes.txt file.")
    parser.add_argument("--image-ext", nargs="*", default=list(IMAGE_EXTENSIONS))
    parser.add_argument("--include-empty", action="store_true", help="Keep images with no valid labels.")
    parser.add_argument("--clamp-boxes", action="store_true", help="Clamp bbox values into [0, 1] instead of dropping.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output images/labels/data.yaml/canbang.yaml.")
    parser.add_argument("--max-audit-images", type=int, default=300, help="Maximum annotated audit images; <=0 disables.")
    parser.add_argument("--log-file", type=Path, default=Path("dataset_processing.log"))
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode="w"),
            logging.StreamHandler(),
        ],
    )


def validate_ratios(train_ratio: float, val_ratio: float) -> float:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1).")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0.0:
        raise ValueError("--train-ratio + --val-ratio must be <= 1.")
    return test_ratio


def collect_files(root: Path, extensions: Sequence[str]) -> Dict[str, List[Path]]:
    normalized_ext = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    files_by_stem: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in normalized_ext:
            files_by_stem[path.stem].append(path)
    return files_by_stem


def collect_label_files(root: Path) -> Dict[str, List[Path]]:
    files_by_stem: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.txt")):
        if path.name.lower() == "classes.txt":
            continue
        files_by_stem[path.stem].append(path)
    return files_by_stem


def _maybe_clamp(value: float, clamp: bool) -> Optional[float]:
    if not value == value:
        return None
    if clamp:
        return min(1.0, max(0.0, value))
    if 0.0 <= value <= 1.0:
        return value
    return None


def parse_label_file(path: Path, *, clamp_boxes: bool) -> Tuple[List[YoloLabel], int]:
    labels: List[YoloLabel] = []
    invalid = 0
    for line_index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        parts = line.lstrip("\ufeff").strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            invalid += 1
            LOGGER.warning("Invalid label line %s:%s: expected at least 5 columns.", path, line_index)
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = (float(parts[i]) for i in range(1, 5))
        except ValueError:
            invalid += 1
            LOGGER.warning("Invalid label line %s:%s: non-numeric value.", path, line_index)
            continue
        if class_id < 0:
            invalid += 1
            LOGGER.warning("Invalid label line %s:%s: negative class id.", path, line_index)
            continue
        x = _maybe_clamp(x, clamp_boxes)
        y = _maybe_clamp(y, clamp_boxes)
        w = _maybe_clamp(w, clamp_boxes)
        h = _maybe_clamp(h, clamp_boxes)
        if x is None or y is None or w is None or h is None or w <= 0.0 or h <= 0.0:
            invalid += 1
            LOGGER.warning("Invalid label line %s:%s: bbox outside YOLO range.", path, line_index)
            continue
        labels.append(YoloLabel(class_id=class_id, x=x, y=y, w=w, h=h))
    return labels, invalid


def choose_unique_name(source: Path, used_names: set[str]) -> str:
    name = source.name
    if name not in used_names:
        used_names.add(name)
        return name
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    name = f"{source.stem}_{digest}{source.suffix}"
    used_names.add(name)
    return name


def build_pairs(
    images_dir: Path,
    labels_dir: Path,
    image_extensions: Sequence[str],
    *,
    include_empty: bool,
    clamp_boxes: bool,
) -> Tuple[List[DataPair], Dict[str, int]]:
    images = collect_files(images_dir, image_extensions)
    labels = collect_label_files(labels_dir)
    stats = Counter()
    pairs: List[DataPair] = []

    for stem, image_paths in sorted(images.items()):
        if len(image_paths) > 1:
            stats["duplicate_image_stems"] += len(image_paths) - 1
            LOGGER.warning("Duplicate image stem '%s'; using %s.", stem, image_paths[0])
        image_path = image_paths[0]
        label_candidates = labels.get(stem, [])
        label_path = label_candidates[0] if label_candidates else None
        if len(label_candidates) > 1:
            stats["duplicate_label_stems"] += len(label_candidates) - 1
            LOGGER.warning("Duplicate label stem '%s'; using %s.", stem, label_path)
        if label_path is None:
            stats["missing_labels"] += 1
            if include_empty:
                pairs.append(DataPair(image_path=image_path, label_path=None, labels=(), key=stem))
            continue
        parsed, invalid = parse_label_file(label_path, clamp_boxes=clamp_boxes)
        stats["invalid_label_lines"] += invalid
        if not parsed and not include_empty:
            stats["empty_or_invalid_labels"] += 1
            continue
        pairs.append(DataPair(image_path=image_path, label_path=label_path, labels=tuple(parsed), key=stem))

    for stem in sorted(set(labels) - set(images)):
        stats["missing_images"] += len(labels[stem])
        LOGGER.warning("Label has no matching image stem '%s'.", stem)
    return pairs, dict(stats)


def split_pairs(
    pairs: Sequence[DataPair],
    train_ratio: float,
    val_ratio: float,
    *,
    seed: int,
) -> Dict[str, List[DataPair]]:
    rng = random.Random(seed)
    groups: Dict[str, List[DataPair]] = defaultdict(list)
    for pair in pairs:
        key = str(pair.primary_class) if pair.primary_class is not None else "empty"
        groups[key].append(pair)

    splits = {split: [] for split in SPLITS}
    for class_key, class_pairs in sorted(groups.items()):
        rng.shuffle(class_pairs)
        total = len(class_pairs)
        train_count = max(1, int(total * train_ratio)) if total > 0 else 0
        val_count = int(total * val_ratio)
        if total >= 3 and train_count == total:
            train_count = total - 1
        if total >= 5 and val_count == 0:
            val_count = 1
            train_count = max(1, train_count - 1)
        if train_count + val_count > total:
            val_count = max(0, total - train_count)
        splits["train"].extend(class_pairs[:train_count])
        splits["val"].extend(class_pairs[train_count:train_count + val_count])
        splits["test"].extend(class_pairs[train_count + val_count:])
        LOGGER.info(
            "Class %s: total=%s train=%s val=%s test=%s",
            class_key,
            total,
            len(class_pairs[:train_count]),
            len(class_pairs[train_count:train_count + val_count]),
            len(class_pairs[train_count + val_count:]),
        )
    for split_pairs_list in splits.values():
        rng.shuffle(split_pairs_list)
    return splits


def prepare_output(output_dir: Path, audit_dir: Optional[Path], overwrite: bool) -> None:
    if overwrite:
        for child in (output_dir / "images", output_dir / "labels"):
            if child.exists():
                shutil.rmtree(child)
        for child in (output_dir / "data.yaml", output_dir / "canbang.yaml", output_dir / "test_split_manifest.json"):
            if child.exists():
                child.unlink()
        if audit_dir is not None and audit_dir.exists():
            shutil.rmtree(audit_dir)
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, *, move: bool) -> None:
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def write_clean_label(path: Path, labels: Sequence[YoloLabel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(f"{label.class_id} {label.x:.8f} {label.y:.8f} {label.w:.8f} {label.h:.8f}\n")


def write_split_files(
    splits: Dict[str, List[DataPair]],
    output_dir: Path,
    *,
    move: bool,
) -> List[DataPair]:
    written: List[DataPair] = []
    used_image_names: set[str] = set()
    for split, pairs in splits.items():
        for pair in pairs:
            image_name = choose_unique_name(pair.image_path, used_image_names)
            label_name = f"{Path(image_name).stem}.txt"
            image_dst = output_dir / "images" / split / image_name
            label_dst = output_dir / "labels" / split / label_name
            copy_or_move(pair.image_path, image_dst, move=move)
            write_clean_label(label_dst, pair.labels)
            written.append(DataPair(image_path=image_dst, label_path=label_dst, labels=pair.labels, key=pair.key))
    return written


def class_names_from_args(args: argparse.Namespace, max_class_id: int) -> List[str]:
    names: List[str] = []
    if args.classes_file is not None:
        names = [
            line.strip()
            for line in args.classes_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    elif args.names:
        names = list(args.names)
    while len(names) <= max_class_id:
        names.append(f"class_{len(names)}")
    return names


def write_data_yaml(output_dir: Path, names: Sequence[str]) -> None:
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": list(names),
        "class_name_mode": "raw",
        "canbang_yaml": "canbang.yaml",
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_balance_yaml(
    output_dir: Path,
    pairs: Sequence[DataPair],
    splits: Dict[str, List[DataPair]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    object_counts = Counter(label.class_id for pair in pairs for label in pair.labels)
    total_objects = sum(object_counts.values())
    payload = {
        "dataset_balance": {
            "version_note": "Keep raw class IDs; object-level counts; generated by trkh.tools.build_dataset",
            "total_images": len(pairs),
            "total_objects": int(total_objects),
            "train_ratio_config": float(train_ratio),
            "val_ratio_config": float(val_ratio),
            "test_ratio_config": float(test_ratio),
            "splits": {split: len(split_pairs) for split, split_pairs in splits.items()},
            "classes": {},
        }
    }
    for class_id in sorted(object_counts):
        count = int(object_counts[class_id])
        ratio = float(count / total_objects) if total_objects else 0.0
        payload["dataset_balance"]["classes"][str(class_id)] = {
            "count": count,
            "ratio": round(ratio, 6),
            "percent": round(ratio * 100.0, 2),
        }
    (output_dir / "canbang.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def draw_audit_images(pairs: Sequence[DataPair], audit_dir: Path, max_images: int) -> None:
    if max_images <= 0:
        return
    for pair in pairs[:max_images]:
        try:
            with Image.open(pair.image_path) as image:
                image = image.convert("RGB")
                draw = ImageDraw.Draw(image)
                width, height = image.size
                for label in pair.labels:
                    x1 = (label.x - label.w / 2.0) * width
                    y1 = (label.y - label.h / 2.0) * height
                    x2 = (label.x + label.w / 2.0) * width
                    y2 = (label.y + label.h / 2.0) * height
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((max(0, x1), max(0, y1 - 12)), str(label.class_id), fill="red")
                image.save(audit_dir / pair.image_path.name)
        except Exception as exc:  # pragma: no cover - best-effort audit output
            LOGGER.warning("Could not write audit image %s: %s", pair.image_path, exc)


def summarize_pairs(pairs: Iterable[DataPair]) -> Dict[str, int]:
    pairs = list(pairs)
    return {
        "images": len(pairs),
        "objects": sum(len(pair.labels) for pair in pairs),
        "single_object_images": sum(1 for pair in pairs if len(pair.labels) == 1),
        "multi_object_images": sum(1 for pair in pairs if len(pair.labels) > 1),
        "empty_images": sum(1 for pair in pairs if not pair.labels),
        "max_objects_per_image": max((len(pair.labels) for pair in pairs), default=0),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    test_ratio = validate_ratios(args.train_ratio, args.val_ratio)
    prepare_output(args.output_dir, args.audit_dir, args.overwrite)

    pairs, pair_stats = build_pairs(
        args.images_dir,
        args.labels_dir,
        args.image_ext,
        include_empty=args.include_empty,
        clamp_boxes=args.clamp_boxes,
    )
    if not pairs:
        raise RuntimeError("No valid image/label pairs found.")
    splits = split_pairs(pairs, args.train_ratio, args.val_ratio, seed=args.seed)
    written_pairs = write_split_files(splits, args.output_dir, move=args.move)
    max_class_id = max((label.class_id for pair in written_pairs for label in pair.labels), default=0)
    names = class_names_from_args(args, max_class_id)
    write_data_yaml(args.output_dir, names)
    write_balance_yaml(
        args.output_dir,
        written_pairs,
        splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )
    if args.audit_dir is not None:
        draw_audit_images(written_pairs, args.audit_dir, args.max_audit_images)

    LOGGER.info("Input pairing stats: %s", dict(pair_stats))
    LOGGER.info("Output summary: %s", summarize_pairs(written_pairs))
    LOGGER.info("Wrote dataset to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
