from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from PIL import Image, ImageOps

from trkh.core.config import load_data_spec


SCHEMA_VERSION = "TRKH_CLASSF_BLIND_REVIEW_PACKET_V1"
DEFAULT_DATA_YAML = Path(r"D:\DataAI\AIEx\newdataset\class_f\data.yaml")
DEFAULT_PRIORITY_DIR = Path(
    r"D:\DataAI\AIEx\TRKH_pretrained\runs\classf_train_boundary_review_priority_20260802"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\DataAI\AIEx\TRKH_pretrained\runs\classf_blind_review_packet_20260805_r1"
)
DEFAULT_INTEGRITY_MANIFEST = Path(
    r"D:\DataAI\AIEx\TRKH_pretrained\runs\classf_integrity_review_trainval_p3_20260802"
    r"\classf_train_val_integrity_manifest.csv"
)
DEFAULT_REVIEW_SUMMARY = DEFAULT_PRIORITY_DIR / "review_summary.json"
P1_NAME = "priority_p1_class1_intersection.csv"
P2_NAME = "priority_p2_rgb_near_identical_pairs.csv"
LOCKED_INPUT_SHA256 = {
    "data_yaml_sha256": "312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8",
    "integrity_manifest_sha256": "fb4dd603851fb5d1f84e9afa589534835bc9587ddfbf32034de3a1a7dce0118b",
    "p1_csv_sha256": "6103ee946fb48147499fb167e23a8407614c04cf85d49b57c4e1d9e3cad20b46",
    "p2_csv_sha256": "7987a7c639df84e494c4603eca10712c1d510775c88ee8078aff3fba91aec56d",
    "review_summary_sha256": "7cb6efe68c2677bcd3a37e69937d9d27331eec2ce56051b1162e58c61fe88609",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_train_image(dataset_root: Path, train_root: Path, relative_path: str) -> Path:
    normalized = str(relative_path).replace("\\", "/").strip()
    rel = Path(normalized)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0].lower() != "train":
        raise ValueError(f"Only normalized TRAIN-relative paths are allowed: {relative_path!r}")
    resolved = (dataset_root / rel).resolve()
    if not _inside(resolved, train_root.resolve()) or not resolved.is_file():
        raise ValueError(f"Path escapes TRAIN or is missing: {relative_path!r}")
    return resolved


def _image_record(
    *,
    dataset_root: Path,
    train_root: Path,
    relative_path: str,
    current_class_id: int,
    expected_sha256: Optional[str],
    role: str,
) -> Dict[str, Any]:
    class_id = int(current_class_id)
    if not 0 <= class_id < 5:
        raise ValueError(f"Class id outside locked five-class contract: {class_id}")
    full_path = _resolve_train_image(dataset_root, train_root, relative_path)
    actual_sha256 = _sha256(full_path)
    if expected_sha256 and actual_sha256.lower() != str(expected_sha256).strip().lower():
        raise ValueError(f"Image SHA256 mismatch for {relative_path}")
    return {
        "relative_path": str(relative_path).replace("\\", "/"),
        "full_path": full_path,
        "current_class_id": class_id,
        "sha256": actual_sha256,
        "roles": [role],
    }


def _merge_image(records: MutableMapping[str, Dict[str, Any]], record: Dict[str, Any]) -> None:
    key = str(record["relative_path"])
    existing = records.get(key)
    if existing is None:
        records[key] = record
        return
    if existing["sha256"] != record["sha256"] or existing["current_class_id"] != record["current_class_id"]:
        raise ValueError(f"Conflicting metadata for {key}")
    existing["roles"] = sorted(set(existing["roles"]) | set(record["roles"]))


def _parse_partner_json(value: str, field: str) -> List[Mapping[str, Any]]:
    parsed = json.loads(value or "[]")
    if not isinstance(parsed, list) or any(not isinstance(item, Mapping) for item in parsed):
        raise ValueError(f"{field} must be a JSON list of objects")
    return list(parsed)


def _load_groups(
    data_yaml: Path,
    priority_dir: Path,
    integrity_manifest: Optional[Path],
    review_summary: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, str]]:
    spec = load_data_spec(data_yaml, expected_num_classes=5)
    dataset_root = spec.root.resolve()
    train_root = spec.train_images.resolve()
    if not _inside(train_root, dataset_root):
        raise ValueError("Resolved TRAIN root must be inside the dataset root")

    p1_path = priority_dir / P1_NAME
    p2_path = priority_dir / P2_NAME
    p1_rows = _read_csv(p1_path)
    p2_rows = _read_csv(p2_path)
    if not p1_rows or not p2_rows:
        raise ValueError("Both P1 and P2 queues must be non-empty")

    groups: List[Dict[str, Any]] = []
    source_ids: Set[str] = set()
    for row in p1_rows:
        source_id = str(row["review_id"])
        if source_id in source_ids:
            raise ValueError(f"Duplicate review_id: {source_id}")
        source_ids.add(source_id)
        images: Dict[str, Dict[str, Any]] = {}
        _merge_image(
            images,
            _image_record(
                dataset_root=dataset_root,
                train_root=train_root,
                relative_path=row["relative_path"],
                current_class_id=int(row["class_id"]),
                expected_sha256=row.get("sha256"),
                role="target",
            ),
        )
        for field, role in (("near_partners_json", "near_partner"), ("adjacent_partners_json", "adjacent_partner")):
            for partner in _parse_partner_json(row.get(field, "[]"), field):
                _merge_image(
                    images,
                    _image_record(
                        dataset_root=dataset_root,
                        train_root=train_root,
                        relative_path=str(partner["relative_path"]),
                        current_class_id=int(partner["class_id"]),
                        expected_sha256=None,
                        role=role,
                    ),
                )
        if len(images) < 2:
            raise ValueError(f"P1 group {source_id} must contain at least two distinct images")
        if int(row["class_id"]) != 1 or len({int(image["current_class_id"]) for image in images.values()}) < 2:
            raise ValueError(f"P1 group {source_id} violates the locked class-1 cross-label contract")
        groups.append({"source_review_id": source_id, "cohort": "P1", "images": list(images.values())})

    for row in p2_rows:
        source_id = str(row["review_id"])
        if source_id in source_ids:
            raise ValueError(f"Duplicate review_id: {source_id}")
        source_ids.add(source_id)
        images = [
            _image_record(
                dataset_root=dataset_root,
                train_root=train_root,
                relative_path=row["path_a"],
                current_class_id=int(row["class_a"]),
                expected_sha256=row.get("sha256_a"),
                role="pair_member",
            ),
            _image_record(
                dataset_root=dataset_root,
                train_root=train_root,
                relative_path=row["path_b"],
                current_class_id=int(row["class_b"]),
                expected_sha256=row.get("sha256_b"),
                role="pair_member",
            ),
        ]
        if images[0]["relative_path"] == images[1]["relative_path"]:
            raise ValueError(f"P2 group {source_id} repeats one path")
        if images[0]["current_class_id"] == images[1]["current_class_id"]:
            raise ValueError(f"P2 group {source_id} is not cross-label")
        groups.append({"source_review_id": source_id, "cohort": "P2", "images": images})

    inputs: Dict[str, str] = {
        "data_yaml_sha256": _sha256(data_yaml),
        "p1_csv_sha256": _sha256(p1_path),
        "p2_csv_sha256": _sha256(p2_path),
    }
    if integrity_manifest is not None:
        manifest_rows = _read_csv(integrity_manifest)
        manifest_index: Dict[str, Dict[str, str]] = {}
        for row in manifest_rows:
            if str(row.get("split", "")).strip().lower() != "train":
                continue
            relative_path = str(row.get("relative_path", "")).replace("\\", "/")
            if relative_path in manifest_index:
                raise ValueError(f"Duplicate TRAIN path in integrity manifest: {relative_path}")
            manifest_index[relative_path] = row
        checked_paths: Set[str] = set()
        for group in groups:
            for image in group["images"]:
                relative_path = str(image["relative_path"])
                if relative_path in checked_paths:
                    continue
                checked_paths.add(relative_path)
                row = manifest_index.get(relative_path)
                if row is None:
                    raise ValueError(f"Review image missing from locked integrity manifest: {relative_path}")
                if int(row["class_id"]) != int(image["current_class_id"]):
                    raise ValueError(f"Class mismatch against integrity manifest: {relative_path}")
                if str(row["sha256"]).strip().lower() != str(image["sha256"]).lower():
                    raise ValueError(f"Image mismatch against integrity manifest: {relative_path}")
        inputs["integrity_manifest_sha256"] = _sha256(integrity_manifest)
    if review_summary is not None:
        source_summary = json.loads(review_summary.read_text(encoding="utf-8"))
        scope = source_summary.get("scope", {})
        if scope.get("dataset_split_materialized") != ["train"]:
            raise ValueError("Source review summary is not TRAIN-only")
        for flag in (
            "validation_images_opened",
            "test_images_opened",
            "model_inference_run",
            "raw_dataset_modified",
            "automatic_relabel",
        ):
            if bool(scope.get(flag)):
                raise ValueError(f"Source review summary violates sealed scope: {flag}")
        outputs = source_summary.get("outputs", {})
        expected_outputs = {
            P1_NAME: (inputs["p1_csv_sha256"], len(p1_rows)),
            P2_NAME: (inputs["p2_csv_sha256"], len(p2_rows)),
        }
        for name, (expected_sha, expected_rows) in expected_outputs.items():
            item = outputs.get(name, {})
            if str(item.get("sha256", "")).lower() != expected_sha.lower() or int(item.get("rows", -1)) != expected_rows:
                raise ValueError(f"Source review summary does not bind {name}")
        inputs["review_summary_sha256"] = _sha256(review_summary)
    return groups, list(spec.class_names), inputs


def _derived_rng(seed: int, reviewer: int) -> random.Random:
    material = f"{SCHEMA_VERSION}:{int(seed)}:reviewer-{reviewer}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _review_units(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    p1_images: Dict[str, Dict[str, Any]] = {}
    p2_units: List[Dict[str, Any]] = []
    for group in groups:
        if group["cohort"] == "P2":
            p2_units.append(
                {
                    "unit_key": f"P2:{group['source_review_id']}",
                    "source_review_ids": [str(group["source_review_id"])],
                    "cohort": "P2",
                    "images": list(group["images"]),
                }
            )
            continue
        for image in group["images"]:
            key = str(image["relative_path"])
            existing = p1_images.get(key)
            if existing is None:
                p1_images[key] = {
                    **image,
                    "source_review_ids": [str(group["source_review_id"])],
                }
                continue
            if existing["sha256"] != image["sha256"] or existing["current_class_id"] != image["current_class_id"]:
                raise ValueError(f"Conflicting P1 review-unit metadata for {key}")
            existing["roles"] = sorted(set(existing["roles"]) | set(image["roles"]))
            existing["source_review_ids"] = sorted(
                set(existing["source_review_ids"]) | {str(group["source_review_id"])}
            )
    p1_units = [
        {
            "unit_key": f"P1:{image['sha256']}",
            "source_review_ids": image.pop("source_review_ids"),
            "cohort": "P1",
            "images": [image],
        }
        for _, image in sorted(p1_images.items())
    ]
    return p1_units + sorted(p2_units, key=lambda item: str(item["unit_key"]))


def _reviewer_plans(groups: Sequence[Dict[str, Any]], seed: int) -> List[List[Dict[str, Any]]]:
    units = _review_units(groups)
    plans: List[List[Dict[str, Any]]] = []
    prior_group_signature: Optional[List[str]] = None
    prior_image_orders: Dict[str, List[str]] = {}
    for reviewer in (1, 2):
        rng = _derived_rng(seed, reviewer)
        ordered = list(units)
        rng.shuffle(ordered)
        signature = [str(group["unit_key"]) for group in ordered]
        if reviewer == 2 and len(ordered) > 1 and signature == prior_group_signature:
            ordered = ordered[1:] + ordered[:1]
        display_groups: List[Dict[str, Any]] = []
        for group_index, group in enumerate(ordered, start=1):
            images = list(group["images"])
            rng.shuffle(images)
            unit_key = str(group["unit_key"])
            current_order = [str(item["relative_path"]) for item in images]
            if reviewer == 2 and len(images) > 1 and current_order == prior_image_orders.get(unit_key):
                images = images[1:] + images[:1]
            blind_group_id = f"R{reviewer}-G{group_index:03d}"
            display_images = []
            for image_index, image in enumerate(images, start=1):
                display_slot = chr(64 + image_index) if group["cohort"] == "P2" else f"I{image_index:02d}"
                display_images.append(
                    {
                        **image,
                        "blind_group_id": blind_group_id,
                        "blind_image_id": f"{blind_group_id}-{display_slot}",
                        "display_slot": display_slot,
                    }
                )
            display_groups.append(
                {
                    "unit_key": unit_key,
                    "source_review_ids": list(group["source_review_ids"]),
                    "cohort": group["cohort"],
                    "blind_group_id": blind_group_id,
                    "images": display_images,
                }
            )
        plans.append(display_groups)
        prior_group_signature = [str(group["unit_key"]) for group in ordered]
        prior_image_orders = {
            str(group["unit_key"]): [str(item["relative_path"]) for item in group["images"]]
            for group in display_groups
        }
    return plans


def _image_data_uri(path: Path, size: int = 640) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    payload = io.BytesIO()
    canvas.save(payload, format="JPEG", quality=95, optimize=False, progressive=False, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(payload.getvalue()).decode("ascii")


def _render_html(reviewer: int, groups: Sequence[Dict[str, Any]], class_names: Sequence[str]) -> str:
    class_contract = "".join(
        f"<li><b>{index}</b>: {html.escape(name)}</li>" for index, name in enumerate(class_names)
    )
    sections: List[str] = []
    for group in groups:
        cards = []
        for item in group["images"]:
            cards.append(
                "<figure><img src=\"{}\" alt=\"{}\"><figcaption>{}</figcaption></figure>".format(
                    _image_data_uri(Path(item["full_path"])),
                    html.escape(str(item["blind_image_id"])),
                    html.escape(str(item["blind_image_id"])),
                )
            )
        if group["cohort"] == "P2":
            prompt = (
                "Assign each image independently. Then answer in pair_answers.csv whether the visible state is "
                "the same and whether RGB evidence alone justifies different classes."
            )
        else:
            prompt = "Assign this image from its RGB evidence only."
        sections.append(
            f"<section><h2>{html.escape(str(group['blind_group_id']))}</h2>"
            f"<p>{html.escape(prompt)}</p><div class=\"images\">{''.join(cards)}</div></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TRKH blinded RGB review R{reviewer}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1180px;margin:24px auto;color:#172033}}
.warning{{border:2px solid #a22;background:#fff2f2;padding:12px}}
section{{break-inside:avoid;border:1px solid #ccd7e5;border-radius:8px;padding:12px;margin:18px 0}}
.images{{display:flex;flex-wrap:wrap;gap:12px}} figure{{margin:0;text-align:center;font-weight:700}}
img{{width:320px;height:320px;object-fit:contain;border:1px solid #555;background:white;cursor:zoom-in}}
img.zoom{{width:640px;height:640px;cursor:zoom-out}}
li{{margin:3px 0}} @media print{{body{{max-width:none}} section{{page-break-inside:avoid}}}}
</style></head><body>
<h1>TRKH TRAIN-only blinded RGB review — Reviewer {reviewer}</h1>
<div class="warning"><b>Independent review.</b> Current labels, source paths, relabel history and every model output are hidden.
Do not consult the other reviewer or the ADMIN_DO_NOT_SHARE directory before both answer files are locked.</div>
<h2>Fixed five-class contract</h2><ul>{class_contract}</ul>
<p>For each blind image, fill <code>answers.csv</code>: class id 0–4 when image-visible, otherwise leave it blank and mark
<code>ambiguous</code> or <code>unreviewable</code>. Give one short visual reason. Pair questions live in
<code>pair_answers.csv</code>. Click an image to toggle its 640-pixel review view.</p>
{''.join(sections)}
<script>document.querySelectorAll('img').forEach(x=>x.addEventListener('click',()=>x.classList.toggle('zoom')));</script>
</body></html>
"""


def _visible_readme(reviewer: int) -> str:
    return f"""TRKH DATA-01 — independent blinded reviewer {reviewer}

Open review.html and fill answers.csv plus pair_answers.csv.
Do not open ADMIN_DO_NOT_SHARE. Do not consult reviewer {2 if reviewer == 1 else 1} before both files are locked.

answers.csv
- assigned_class_id: 0,1,2,3,4 only when RGB evidence is sufficient; otherwise blank.
- visibility: classifiable, ambiguous, or unreviewable.
- evidence_notes: one short image-visible reason; no model reasoning.

pair_answers.csv
- image_a_id and image_b_id bind the displayed A/B order; do not edit them.
- same_visible_state: yes, no, or uncertain.
- different_class_visually_justified: yes, no, or uncertain.

These answers are review evidence only. They do not authorize automatic relabeling, exclusion, weighting, model selection,
validation access or test access.
"""


def build_packet(
    data_yaml: Path,
    priority_dir: Path,
    output_dir: Path,
    seed: int,
    integrity_manifest: Optional[Path] = None,
    review_summary: Optional[Path] = None,
    expected_input_sha256: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    data_yaml = Path(data_yaml).resolve()
    priority_dir = Path(priority_dir).resolve()
    final_output_dir = Path(output_dir).resolve()
    integrity_manifest = Path(integrity_manifest).resolve() if integrity_manifest is not None else None
    review_summary = Path(review_summary).resolve() if review_summary is not None else None
    spec = load_data_spec(data_yaml, expected_num_classes=5)
    protected_roots = [spec.root.resolve(), priority_dir]
    if integrity_manifest is not None:
        protected_roots.append(integrity_manifest.parent)
    if review_summary is not None:
        protected_roots.append(review_summary.parent)
    for protected in protected_roots:
        if (
            final_output_dir == protected
            or _inside(final_output_dir, protected)
            or _inside(protected, final_output_dir)
        ):
            raise ValueError(f"Output directory overlaps protected input root: {protected}")
    if final_output_dir.exists() and any(final_output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {final_output_dir}")

    groups, class_names, inputs = _load_groups(data_yaml, priority_dir, integrity_manifest, review_summary)
    if expected_input_sha256 is not None:
        if set(expected_input_sha256) != set(inputs):
            raise ValueError(
                f"Locked input SHA256 contract keys differ: expected={sorted(expected_input_sha256)}, "
                f"actual={sorted(inputs)}"
            )
        for name, expected in expected_input_sha256.items():
            actual = inputs.get(name)
            if actual is None or actual.lower() != str(expected).strip().lower():
                raise ValueError(f"Locked input SHA256 mismatch for {name}: expected={expected}, actual={actual}")
    plans = _reviewer_plans(groups, seed)
    staging_suffix = hashlib.sha256(f"{SCHEMA_VERSION}:{seed}".encode("utf-8")).hexdigest()[:12]
    output_dir = final_output_dir.parent / f".{final_output_dir.name}.staging-{staging_suffix}"
    if output_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    sealed_rows: List[Dict[str, Any]] = []
    reviewer_files: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for reviewer, display_groups in enumerate(plans, start=1):
        reviewer_dir = output_dir / f"reviewer_{reviewer}"
        html_path = reviewer_dir / "review.html"
        answers_path = reviewer_dir / "answers.csv"
        pair_answers_path = reviewer_dir / "pair_answers.csv"
        readme_path = reviewer_dir / "README.txt"
        _write_text(html_path, _render_html(reviewer, display_groups, class_names))
        _write_text(readme_path, _visible_readme(reviewer))

        answer_rows = []
        pair_rows = []
        for group in display_groups:
            if group["cohort"] == "P2":
                image_by_slot = {str(item["display_slot"]): str(item["blind_image_id"]) for item in group["images"]}
                pair_rows.append(
                    {
                        "blind_group_id": group["blind_group_id"],
                        "image_a_id": image_by_slot["A"],
                        "image_b_id": image_by_slot["B"],
                        "same_visible_state": "",
                        "different_class_visually_justified": "",
                        "pair_notes": "",
                    }
                )
            for item in group["images"]:
                answer_rows.append(
                    {
                        "blind_group_id": group["blind_group_id"],
                        "blind_image_id": item["blind_image_id"],
                        "assigned_class_id": "",
                        "visibility": "",
                        "evidence_notes": "",
                    }
                )
                sealed_rows.append(
                    {
                        "reviewer": reviewer,
                        "blind_group_id": group["blind_group_id"],
                        "blind_image_id": item["blind_image_id"],
                        "display_slot": item["display_slot"],
                        "source_review_ids": group["source_review_ids"],
                        "cohort": group["cohort"],
                        "relative_path": item["relative_path"],
                        "current_class_id": item["current_class_id"],
                        "sha256": item["sha256"],
                        "roles": item["roles"],
                    }
                )
        _write_csv(
            answers_path,
            ("blind_group_id", "blind_image_id", "assigned_class_id", "visibility", "evidence_notes"),
            answer_rows,
        )
        _write_csv(
            pair_answers_path,
            (
                "blind_group_id",
                "image_a_id",
                "image_b_id",
                "same_visible_state",
                "different_class_visually_justified",
                "pair_notes",
            ),
            pair_rows,
        )
        reviewer_files[f"reviewer_{reviewer}"] = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (readme_path, html_path, answers_path, pair_answers_path)
        }

    sealed_path = output_dir / "ADMIN_DO_NOT_SHARE" / "sealed_mapping.json"
    sealed_payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "class_names": list(class_names),
        "mapping": sealed_rows,
    }
    _write_text(sealed_path, json.dumps(sealed_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    distinct_paths = {row["relative_path"] for row in sealed_rows}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "scope": {
            "split": "train",
            "validation_opened": False,
            "test_opened": False,
            "model_inference_run": False,
            "raw_dataset_modified": False,
            "automatic_relabel": False,
        },
        "counts": {
            "source_groups": len(groups),
            "source_p1_groups": sum(group["cohort"] == "P1" for group in groups),
            "source_p2_groups": sum(group["cohort"] == "P2" for group in groups),
            "reviewer_display_groups_each": len(plans[0]),
            "reviewer_p1_singletons_each": sum(group["cohort"] == "P1" for group in plans[0]),
            "reviewer_p2_pairs_each": sum(group["cohort"] == "P2" for group in plans[0]),
            "distinct_train_images": len(distinct_paths),
            "reviewer_visible_rows_each": sum(len(group["images"]) for group in plans[0]),
        },
        "inputs": inputs,
        "input_hash_contract_verified": expected_input_sha256 is not None,
        "reviewer_files": reviewer_files,
        "sealed_mapping": {"sha256": _sha256(sealed_path), "bytes": sealed_path.stat().st_size},
        "review_policy": {
            "send_only_one_reviewer_directory_to_each_reviewer": True,
            "current_label_path_history_and_model_output_hidden": True,
            "two_independent_reviews_then_adjudication_required": True,
            "packet_authorizes_training_or_relabel": False,
        },
    }
    _write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_text(
        output_dir / "PACKET_README.txt",
        "Send reviewer_1 and reviewer_2 separately. Keep ADMIN_DO_NOT_SHARE sealed until both reviews are locked.\n"
        "The packet is TRAIN-only evidence and authorizes no relabel, model training, validation or test access.\n",
    )
    if final_output_dir.exists():
        final_output_dir.rmdir()
    output_dir.replace(final_output_dir)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic label/model/history-hidden DATA-01 review packets.")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--priority-dir", type=Path, default=DEFAULT_PRIORITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--integrity-manifest", type=Path, default=DEFAULT_INTEGRITY_MANIFEST)
    parser.add_argument("--review-summary", type=Path, default=DEFAULT_REVIEW_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_packet(
        args.data_yaml,
        args.priority_dir,
        args.output_dir,
        args.seed,
        integrity_manifest=args.integrity_manifest,
        review_summary=args.review_summary,
        expected_input_sha256=LOCKED_INPUT_SHA256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
