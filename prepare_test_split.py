from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from config import default_data_yaml, load_data_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tach test hold-out tu split val hien tai.")
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument(
        "--class-name-mode",
        choices=("auto", "raw", "mango"),
        default=None,
        help="Cach xu ly names trong data.yaml; dung raw cho dataset tuy bien hoac >4 lop.",
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=0,
        help="Neu > 0, validate so class trong data.yaml truoc khi tach split.",
    )
    parser.add_argument("--ratio", type=float, default=0.4, help="Ti le file trong val chuyen sang test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def load_label_counts(label_path: Path) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        counts[int(parts[0])] += 1
    return dict(counts)


def find_image_for_stem(images_dir: Path, stem: str) -> Path:
    for extension in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Khong tim thay image cho {stem} trong {images_dir}")


def primary_class(label_counts: Dict[int, int]) -> int:
    if not label_counts:
        return -1
    return max(sorted(label_counts), key=lambda class_index: label_counts[class_index])


def main() -> None:
    args = parse_args()
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    if data_spec.has_test_split and data_spec.test_images is not None and data_spec.test_images.exists():
        existing_files = list(data_spec.test_labels.glob("*.txt"))
        if existing_files and not args.force:
            raise RuntimeError(
                "Split test da ton tai va khong rong. Dung --force neu ban muon chay lai."
            )

    val_images_dir = data_spec.val_images
    val_labels_dir = data_spec.val_labels
    test_images_dir = data_spec.root / "images" / "test"
    test_labels_dir = data_spec.root / "labels" / "test"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    label_records: List[Dict[str, object]] = []
    for label_path in sorted(val_labels_dir.glob("*.txt")):
        class_counts = load_label_counts(label_path)
        if not class_counts:
            continue
        image_path = find_image_for_stem(val_images_dir, label_path.stem)
        label_records.append(
            {
                "stem": label_path.stem,
                "label_path": label_path,
                "image_path": image_path,
                "class_counts": class_counts,
                "primary_class": primary_class(class_counts),
            }
        )

    rng = random.Random(args.seed)
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for record in label_records:
        grouped[int(record["primary_class"])].append(record)

    selected_records: List[Dict[str, object]] = []
    for group_records in grouped.values():
        rng.shuffle(group_records)
        target_count = max(1, int(round(len(group_records) * float(args.ratio))))
        selected_records.extend(group_records[:target_count])

    selected_stems = {str(record["stem"]) for record in selected_records}
    moved_files = []
    for record in label_records:
        if str(record["stem"]) not in selected_stems:
            continue
        source_image = Path(record["image_path"])
        source_label = Path(record["label_path"])
        target_image = test_images_dir / source_image.name
        target_label = test_labels_dir / source_label.name
        source_image.replace(target_image)
        source_label.replace(target_label)
        moved_files.append(
            {
                "stem": record["stem"],
                "image": target_image.name,
                "label": target_label.name,
                "class_counts": record["class_counts"],
            }
        )

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = data_spec.root / "test_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_yaml": str(data_spec.data_yaml),
                "seed": args.seed,
                "ratio": args.ratio,
                "moved_files": moved_files,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        {
            "moved_files": len(moved_files),
            "test_images_dir": str(test_images_dir),
            "test_labels_dir": str(test_labels_dir),
            "manifest": str(manifest_path),
        }
    )


if __name__ == "__main__":
    main()
