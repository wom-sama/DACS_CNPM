from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import trkh.tools.cross_colour_ratio_surface_a0_materializer as materializer
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.tools.cross_colour_ratio_surface_a0_materializer import (
    CohortRecord,
    EXPECTED_TRANSFORM_SEMANTICS,
    FEATURE_SIZE,
    LOCK_PATH,
    MODEL_IMAGE_SIZE,
    DataAccessLedger,
    build_frozen_eval_transform,
    direct_tensor_sample,
    feature_geometry,
    load_lock,
    model_input_to_srgb_uint8,
    pack_valid_mask,
    parse_yolo_label_bytes,
    structural_preflight,
    unpack_valid_masks,
)


def _synthetic_image(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        (
            (3 * xx + yy) % 256,
            (xx + 5 * yy + 17) % 256,
            (7 * xx + 2 * yy + 31) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb)


def _write_synthetic_yolo(root: Path) -> tuple[Path, Path]:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    first = _synthetic_image(173, 109)
    second = _synthetic_image(91, 157)
    first.save(images / "image_a.png")
    second.save(images / "image_b.png")
    (labels / "image_a.txt").write_text(
        "1 0.31 0.44 0.36 0.52\n"
        "2 0.55 0.48 0.40 0.35\n"
        "bad ignored line\n",
        encoding="utf-8",
    )
    (labels / "image_b.txt").write_text(
        "0 0.08 0.13 0.16 0.22\n"
        "4 0.71 0.73 0.42 0.38\n",
        encoding="utf-8",
    )
    return images, labels


def test_direct_loader_matches_production_object_crop_and_eval_tensor(
    tmp_path: Path,
) -> None:
    images, labels = _write_synthetic_yolo(tmp_path)
    transform = build_frozen_eval_transform(
        EXPECTED_TRANSFORM_SEMANTICS
    )
    dataset = MangoYOLOCropDataset(
        images_dir=images,
        labels_dir=labels,
        transform=transform,
        crop_margin_ratio=0.05,
        crop_to_primary_object=True,
        num_classes=5,
        split="train",
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    assert len(dataset) == 4
    for index in range(len(dataset)):
        production_tensor, production_label, production_metadata = (
            dataset[index]
        )
        sample = dataset.samples[index]
        image_bytes = sample.image_path.read_bytes()
        label_bytes = sample.label_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            image = decoded.convert("RGB").copy()
        objects = parse_yolo_label_bytes(label_bytes)
        primary = next(
            obj
            for obj in sample.objects
            if obj.object_index == sample.primary_object_index
        )
        direct = direct_tensor_sample(
            image,
            objects,
            target=int(production_label),
            model_bbox=primary.bbox,
            transform=transform,
        )
        torch.testing.assert_close(
            direct.model_input,
            production_tensor,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            direct.model_bbox,
            production_metadata["bbox"],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            direct.crop_bbox,
            production_metadata["crop_bbox"],
            atol=0.0,
            rtol=0.0,
        )
        assert torch.equal(
            direct.image_valid_mask,
            production_metadata["image_mask"],
        )


def test_srgb_and_valid_mask_cache_round_trip_is_exact(
    tmp_path: Path,
) -> None:
    images, labels = _write_synthetic_yolo(tmp_path)
    transform = build_frozen_eval_transform(
        EXPECTED_TRANSFORM_SEMANTICS
    )
    dataset = MangoYOLOCropDataset(
        images_dir=images,
        labels_dir=labels,
        transform=transform,
        crop_margin_ratio=0.05,
        crop_to_primary_object=True,
        num_classes=5,
        split="train",
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    tensors = []
    masks = []
    for index in range(len(dataset)):
        tensor, _, metadata = dataset[index]
        srgb, error, exact = model_input_to_srgb_uint8(tensor)
        assert exact
        assert error == 0.0
        assert srgb.dtype == torch.uint8
        tensors.append(srgb)
        masks.append(pack_valid_mask(metadata["image_mask"]))
    packed = np.stack(masks, axis=0)
    unpacked = unpack_valid_masks(packed)
    assert unpacked.shape == (
        len(dataset),
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
    )
    for index in range(len(dataset)):
        assert np.array_equal(
            unpacked[index],
            dataset[index][2]["image_mask"].numpy(),
        )


def test_feature_geometry_matches_locked_definition() -> None:
    valid = torch.zeros(
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
        dtype=torch.bool,
    )
    valid[31:225, 7:249] = True
    crop_bbox = torch.tensor(
        [0.52, 0.47, 0.61, 0.39],
        dtype=torch.float32,
    )
    valid16, bbox16 = feature_geometry(valid, crop_bbox)
    expected_valid = (
        torch.nn.functional.interpolate(
            valid.float().view(
                1,
                1,
                MODEL_IMAGE_SIZE,
                MODEL_IMAGE_SIZE,
            ),
            size=(FEATURE_SIZE, FEATURE_SIZE),
            mode="area",
        )[0, 0]
        >= 0.5
    )
    yy = (
        torch.arange(FEATURE_SIZE, dtype=torch.float32) + 0.5
    ) / FEATURE_SIZE
    xx = (
        torch.arange(FEATURE_SIZE, dtype=torch.float32) + 0.5
    ) / FEATURE_SIZE
    expected_bbox = (
        (xx[None, :] - crop_bbox[0]).abs()
        <= 0.5 * crop_bbox[2]
    ) & (
        (yy[:, None] - crop_bbox[1]).abs()
        <= 0.5 * crop_bbox[3]
    )
    assert np.array_equal(valid16, expected_valid.numpy())
    assert np.array_equal(
        bbox16,
        (expected_bbox & expected_valid).numpy(),
    )


def test_materializer_source_does_not_construct_the_broad_dataset() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "trkh"
        / "tools"
        / "cross_colour_ratio_surface_a0_materializer.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MangoYOLOCropDataset"
    ]
    assert calls == []


def test_access_ledger_blocks_unlisted_and_forbidden_paths(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    train = dataset_root / "images" / "train"
    validation = dataset_root / "images" / "val"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    allowed = train / "allowed.bin"
    undeclared = train / "undeclared.bin"
    forbidden = validation / "forbidden.bin"
    allowed.write_bytes(b"a")
    undeclared.write_bytes(b"b")
    forbidden.write_bytes(b"c")
    lock = {
        "immutable_inputs": {
            "allowed": {
                "path": str(allowed),
                "bytes": 1,
                "sha256": "unused",
            }
        },
        "protocol": None,
    }
    with DataAccessLedger(
        lock,
        lock_path=LOCK_PATH,
        domain_roots={"dataset": dataset_root},
    ) as ledger:
        assert allowed.read_bytes() == b"a"
        with pytest.raises(PermissionError, match="undeclared_input"):
            undeclared.read_bytes()
        with pytest.raises(
            PermissionError,
            match="forbidden_split_component",
        ):
            forbidden.read_bytes()
    snapshot = ledger.snapshot()
    assert snapshot["blocked_attempt_count"] == 2
    assert snapshot["validation_open_count"] == 1


def test_structural_preflight_reads_no_locked_cohort_pixels() -> None:
    result = structural_preflight()
    assert result["passed"]
    assert result["state"] == "structural_preflight_no_image_read_no_fit"
    assert result["rows"] == 763
    assert result["unique_images"] == 735
    assert result["checks"]["no_image_or_label_open"]
    assert not result["candidate_descriptor_created"]
    assert not result["candidate_model_state_created"]
    assert not result["candidate_metric_observed"]


def test_lock_and_cache_budget_are_prospectively_bounded() -> None:
    lock, _ = load_lock()
    srgb_bytes = 763 * 3 * 256 * 256
    mask_bytes = 763 * 256 * 256 // 8
    assert srgb_bytes + mask_bytes + 4 * 1024 * 1024 <= int(
        lock["resource_limits"]["temporary_cache_bytes"]
    )


def test_authorization_pins_resources_constraints_and_precedes_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, lock_sha = load_lock()
    output = (tmp_path / "authorized" / "formal").resolve()
    authorization_path = (tmp_path / "authorization.json").resolve()
    authorization_path.with_suffix(".sha256").write_text(
        "placeholder\n",
        encoding="utf-8",
    )
    expected = {
        "lock_sha256": materializer.sha256_file(materializer.LOCK_PATH),
        "erratum_sha256": materializer.sha256_file(materializer.ERRATUM_PATH),
        "module_sha256": materializer.sha256_file(materializer.MODULE_PATH),
        "engine_sha256": materializer.sha256_file(
            materializer.REPO_ROOT
            / "trkh"
            / "tools"
            / "cross_colour_ratio_surface_a0_engine.py"
        ),
        "test_sha256": materializer.sha256_file(Path(__file__)),
    }
    authorization = {
        "protocol_id": materializer.PROTOCOL_ID,
        "state": "materializer_authorized_no_fit",
        "output_dir": str(output),
        "expected": expected,
        "resource_limits": dict(lock["resource_limits"]),
        "execution_constraints": dict(
            materializer.REQUIRED_EXECUTION_CONSTRAINTS
        ),
    }
    result = materializer._verify_authorization(
        authorization,
        authorization_path=authorization_path,
        lock=lock,
        lock_sha256=lock_sha,
        output=output,
    )
    assert result["passed"]
    unauthorized = dict(authorization)
    unauthorized["execution_constraints"] = {
        **materializer.REQUIRED_EXECUTION_CONSTRAINTS,
        "head_fit": True,
    }
    with pytest.raises(ValueError, match="authorization differs"):
        materializer._verify_authorization(
            unauthorized,
            authorization_path=authorization_path,
            lock=lock,
            lock_sha256=lock_sha,
            output=output,
        )

    rejected_output = (tmp_path / "must_not_exist" / "formal").resolve()
    monkeypatch.setattr(
        materializer,
        "load_lock",
        lambda path: (_ for _ in ()).throw(ValueError("rejected lock")),
    )
    with pytest.raises(ValueError, match="rejected lock"):
        materializer.materialize(
            output=rejected_output,
            authorization_path=authorization_path,
        )
    assert not rejected_output.parent.exists()


def test_synthetic_formal_materializer_is_atomic_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images, labels = _write_synthetic_yolo(tmp_path / "dataset")
    transform = build_frozen_eval_transform(
        EXPECTED_TRANSFORM_SEMANTICS
    )
    dataset = MangoYOLOCropDataset(
        images_dir=images,
        labels_dir=labels,
        transform=transform,
        crop_margin_ratio=0.05,
        crop_to_primary_object=True,
        num_classes=5,
        split="train",
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    row_count = len(dataset)
    records = []
    targets = np.empty(row_count, dtype=np.int64)
    sources = []
    model_boxes = np.empty((row_count, 4), dtype=np.float32)
    crop_boxes = np.empty_like(model_boxes)
    valid_masks = np.empty(
        (row_count, FEATURE_SIZE, FEATURE_SIZE),
        dtype=np.bool_,
    )
    bbox_masks = np.empty_like(valid_masks)
    for index in range(row_count):
        _, label, metadata = dataset[index]
        sample = dataset.samples[index]
        primary = next(
            obj
            for obj in sample.objects
            if obj.object_index == sample.primary_object_index
        )
        source = sample.image_path.stem.casefold()
        targets[index] = int(label)
        sources.append(source)
        model_boxes[index] = metadata["bbox"].numpy()
        crop_boxes[index] = metadata["crop_bbox"].numpy()
        valid16, bbox16 = feature_geometry(
            metadata["image_mask"],
            metadata["crop_bbox"],
        )
        valid_masks[index] = valid16
        bbox_masks[index] = bbox16
        records.append(
            CohortRecord(
                position=index,
                sample_index=index,
                target=int(label),
                fold=index % 5,
                source_stem=source,
                image_path=sample.image_path.resolve(),
                label_path=sample.label_path.resolve(),
                model_bbox=tuple(
                    float(value) for value in primary.bbox
                ),
            )
        )
    keeper_probabilities = np.full(
        (row_count, 5),
        0.01,
        dtype=np.float32,
    )
    keeper_probabilities[
        np.arange(row_count),
        targets,
    ] = 0.96
    geometry = {
        "sample_indices": np.arange(row_count, dtype=np.int64),
        "targets": targets,
        "folds": np.arange(row_count, dtype=np.int64) % 5,
        "source_stems": np.asarray(sources),
        "keeper_probabilities": keeper_probabilities,
        "valid_masks": valid_masks,
        "bbox_masks": bbox_masks,
        "model_boxes": model_boxes,
        "crop_boxes": crop_boxes,
    }

    def observed_file(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "bytes": len(payload),
            "sha256": materializer.sha256_bytes(payload),
        }

    image_observed = {
        path: observed_file(path)
        for path in {record.image_path for record in records}
    }
    label_observed = {
        path: observed_file(path)
        for path in {record.label_path for record in records}
    }
    lock = {
        "protocol_id": materializer.PROTOCOL_ID,
        "immutable_inputs": {},
        "protocol": None,
        "cohort": {
            "image_files": materializer._file_manifest_from_observed(
                [record.image_path for record in records],
                image_observed,
            ),
            "label_files": materializer._file_manifest_from_observed(
                [record.label_path for record in records],
                label_observed,
            ),
        },
        "resource_limits": {
            "temporary_cache_bytes": 500 * 1024 * 1024,
            "process_rss_bytes": 20 * 1024 * 1024 * 1024,
            "clean_wall_seconds": 60,
        },
    }
    output = (tmp_path / "formal_output").resolve()
    authorization_path = (tmp_path / "authorization.json").resolve()
    authorization = {
        "protocol_id": materializer.PROTOCOL_ID,
        "state": "materializer_authorized_no_fit",
        "output_dir": str(output),
        "required_ancestor_commit": "synthetic",
    }

    monkeypatch.setattr(materializer, "ROWS", row_count)
    monkeypatch.setattr(
        materializer,
        "UNIQUE_FILES",
        len({record.image_path for record in records}),
    )
    monkeypatch.setattr(
        materializer,
        "TRACKED_DOMAIN_ROOTS",
        {"dataset": (tmp_path / "dataset").resolve()},
    )
    monkeypatch.setattr(
        materializer,
        "load_lock",
        lambda path: (lock, "synthetic-lock"),
    )
    monkeypatch.setattr(
        materializer,
        "load_authorization",
        lambda path: (authorization, "synthetic-authorization"),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_authorization",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        materializer,
        "_repository_state",
        lambda **kwargs: {"passed": True},
    )

    def load_inputs(_lock, *, ledger):
        ledger.authorize_cohort_paths(records)
        return {
            "geometry": geometry,
            "records": records,
            "resolved_config": {},
            "transform": {
                "semantics": dict(EXPECTED_TRANSFORM_SEMANTICS),
                "passed": True,
            },
            "source_verification": {"passed": True},
            "immutable_verification": {},
        }

    monkeypatch.setattr(
        materializer,
        "load_scientific_inputs",
        load_inputs,
    )
    monkeypatch.setattr(
        materializer.torch.cuda,
        "is_initialized",
        lambda: False,
    )

    result = materializer.materialize(
        output=output,
        authorization_path=authorization_path,
    )
    assert result["summary"]["automatic_passed"]
    assert output.is_dir()
    summary = materializer._load_output_summary(output)
    assert summary["rows"] == row_count
    assert summary["unique_images"] == 2
    for name, record in summary["outputs"].items():
        assert Path(record["path"]) == output / name
        assert (output / name).is_file()
    srgb = np.load(
        output / materializer.SRGB_CACHE_NAME,
        mmap_mode="r",
        allow_pickle=False,
    )
    packed = np.load(
        output / materializer.VALID_MASK_CACHE_NAME,
        mmap_mode="r",
        allow_pickle=False,
    )
    assert srgb.shape == (row_count, 3, 256, 256)
    assert unpack_valid_masks(packed).shape == (
        row_count,
        256,
        256,
    )
    ledger = json.loads(
        (output / "access_ledger.json").read_text(encoding="utf-8")
    )
    for path in [*image_observed, *label_observed]:
        normalized = materializer._normalized_path(path)
        assert ledger["path_counts"][normalized] == 2
        assert ledger["logical_path_counts"][normalized] == 1

    failed_output = (tmp_path / "failed_output").resolve()

    def forced_parity_failure(*args, **kwargs):
        raise RuntimeError("forced parity failure")

    monkeypatch.setattr(
        materializer,
        "model_input_to_srgb_uint8",
        forced_parity_failure,
    )
    with pytest.raises(RuntimeError, match="forced parity failure"):
        materializer.materialize(
            output=failed_output,
            authorization_path=authorization_path,
        )
    assert not failed_output.exists()
    assert not list(
        failed_output.parent.glob(f".{failed_output.name}.tmp-*")
    )
