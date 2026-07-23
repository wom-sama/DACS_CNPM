from __future__ import annotations

import copy
import csv
import inspect
from pathlib import Path

from trkh.tools import audit_saspa_synthetic_a0_f0 as audit


def test_protocol_and_requirements_hashes_are_locked() -> None:
    assert audit._sha256(audit.PROTOCOL_PATH) == audit.LOCKED_PROTOCOL_SHA256
    assert (
        audit._sha256(audit.PROTOCOL_MD_PATH)
        == audit.LOCKED_PROTOCOL_MD_SHA256
    )
    assert (
        audit._sha256(audit.PROTOCOL_LOCK_PATH)
        == audit.LOCKED_PROTOCOL_LOCK_SHA256
    )
    assert (
        audit._sha256(audit.REQUIREMENTS_PATH)
        == audit.LOCKED_REQUIREMENTS_SHA256
    )
    assert (
        audit._sha256(audit.CURRENT_COMMAND_PATH)
        == audit.LOCKED_CURRENT_COMMAND_SHA256
    )
    assert (
        audit._sha256(audit.COMMAND_HISTORY_PATH)
        == audit.LOCKED_COMMAND_HISTORY_SHA256
    )
    protocol = audit._read_json(audit.PROTOCOL_PATH)
    assert protocol["protocol_revision"] == 2
    assert protocol["phase_f0_no_output"]["may_generate_pixels"] is False
    assert protocol["generator"]["runtime"]["torch"] == "2.6.0+cu124"
    assert protocol["generator"]["runtime"]["torchvision"] == "0.21.0+cu124"


def test_requirements_are_exact_and_include_locked_runtime() -> None:
    pins = audit.parse_requirement_pins(audit.REQUIREMENTS_PATH)
    assert pins["torch"] == "2.6.0+cu124"
    assert pins["torchvision"] == "0.21.0+cu124"
    assert pins["diffusers"] == "0.32.2"
    assert pins["transformers"] == "4.48.3"
    assert pins["accelerate"] == "1.3.0"
    assert pins["opencv-python-headless"] == "4.8.1.78"
    assert len(pins) == 36


def _fixture_protocol(tmp_path: Path) -> dict:
    protocol = copy.deepcopy(audit._read_json(audit.PROTOCOL_PATH))
    yolo_root = tmp_path / "yolo_f"
    image_root = yolo_root / "images" / "train"
    label_root = yolo_root / "labels" / "train"
    class_root = tmp_path / "class_f"
    class_train = class_root / "train"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    class_train.mkdir(parents=True)
    rows = []
    for class_item in protocol["classes"]:
        class_index = int(class_item["index"])
        folder = str(class_item["folder"])
        (class_train / folder).mkdir()
        for item_index in range(4):
            stem = "Image_{}_{}".format(class_index, item_index)
            image_path = image_root / "{}.jpg".format(stem)
            label_path = label_root / "{}.txt".format(stem)
            crop_path = class_train / folder / "{}_box000.jpg".format(stem)
            image_path.write_bytes(
                "edge-{}-{}".format(class_index, item_index).encode("ascii")
            )
            label_path.write_text(
                "{} 0.5 0.5 0.4 0.4\n".format(class_index),
                encoding="utf-8",
            )
            crop_path.write_bytes(
                "subject-{}-{}".format(class_index, item_index).encode("ascii")
            )
            rows.append(
                {
                    "split": "train",
                    "source_split": "train",
                    "leakage_group": "g{}_{}".format(class_index, item_index),
                    "source_image": str(image_path),
                    "output_image": str(image_path),
                    "output_label": str(label_path),
                }
            )

    validation_image = tmp_path / "never_open_val.jpg"
    validation_label = tmp_path / "never_open_val.txt"
    rows.append(
        {
            "split": "val",
            "source_split": "val",
            "leakage_group": "forbidden_val",
            "source_image": str(validation_image),
            "output_image": str(validation_image),
            "output_label": str(validation_label),
        }
    )
    bad_image = image_root / "Image_bad.jpg"
    bad_label = label_root / "Image_bad.txt"
    bad_image.write_bytes(b"bad")
    bad_label.write_text(
        "0 0.5 0.5 0.4 0.4\n0 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    rows.append(
        {
            "split": "train",
            "source_split": "train",
            "leakage_group": "bad_multi",
            "source_image": str(bad_image),
            "output_image": str(bad_image),
            "output_label": str(bad_label),
        }
    )
    manifest = yolo_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "source_split",
                "leakage_group",
                "source_image",
                "output_image",
                "output_label",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    immutable = protocol["immutable_inputs"]
    immutable["source_manifest"]["path"] = str(manifest)
    immutable["yolo_train_images"] = str(image_root)
    immutable["yolo_train_labels"] = str(label_root)
    immutable["classification_train_root"] = str(class_train)
    return protocol


def test_tiny_cohort_is_deterministic_group_disjoint_and_train_only(
    tmp_path: Path,
) -> None:
    protocol = _fixture_protocol(tmp_path)
    first = audit.select_tiny_cohort(protocol)
    second = audit.select_tiny_cohort(protocol)
    assert first == second
    assert first["row_count"] == 10
    assert first["population"]["eligible_counts"] == [4, 4, 4, 4, 4]
    assert first["population"]["rejected"]["non_train"] == 1
    assert first["population"]["rejected"]["not_single_object"] == 1
    groups = []
    for row in first["rows"]:
        groups.extend(
            [row["edge_leakage_group"], row["subject_leakage_group"]]
        )
        assert 0 <= int(row["seed"]) < 2 ** 63
        expected_seed_hash = audit._selection_hash(
            "seed",
            protocol["protocol_id"],
            row["edge_relative_path"],
            row["subject_relative_path"],
            row["prompt_template_index"],
        )
        assert row["seed"] == int(expected_seed_hash[:16], 16) >> 1
        assert row["edge_leakage_group"] != row["subject_leakage_group"]
        assert "val" not in str(row["edge_relative_path"]).lower()
    assert len(groups) == 20
    assert len(groups) == len(set(groups))
    assert first["validation_test_pixels_opened"] is False
    assert first["synthetic_pixels_generated"] is False


def test_f0_resource_gates_are_conjunctive() -> None:
    protocol = audit._read_json(audit.PROTOCOL_PATH)
    gib = 1024 ** 3
    healthy = {
        "d_drive_free_bytes": 30 * gib,
        "physical_free_bytes": 5 * gib,
        "virtual_free_bytes": 16 * gib,
        "virtual_used_fraction": 0.5,
        "external_python_or_trtexec": [],
    }
    passed = audit.evaluate_f0_resource_gates(
        protocol,
        healthy,
        model_snapshot_bytes=audit.LOCKED_MODEL_REMOTE_BYTES,
        pipeline_load_seconds=60.0,
    )
    assert passed["passed"] is True
    failed = dict(healthy)
    failed["physical_free_bytes"] = 3 * gib
    failed["external_python_or_trtexec"] = [{"pid": 123}]
    rejected = audit.evaluate_f0_resource_gates(protocol, failed)
    assert rejected["passed"] is False
    assert rejected["checks"]["physical_ram_free"] is False
    assert rejected["checks"]["no_unknown_python_or_trtexec"] is False


def test_only_exact_venv_redirector_parent_is_allowed(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    executable = runtime / "Scripts" / "python.exe"
    current_command = [
        str(tmp_path / "base-python.exe"),
        "-m",
        "trkh.tools.audit_saspa_synthetic_a0_f0",
        "--formal-f0",
    ]
    candidate = {
        "pid": 40,
        "name": "python.exe",
        "exe": str(executable),
        "create_time": 100.0,
        "cmdline": [str(executable)] + current_command[1:],
    }
    assert audit._is_known_venv_redirector(
        candidate,
        current_parent_pid=40,
        current_command_line=current_command,
        current_create_time=101.0,
        runtime_prefix=runtime,
    )
    wrong_pid = dict(candidate, pid=41)
    assert not audit._is_known_venv_redirector(
        wrong_pid,
        current_parent_pid=40,
        current_command_line=current_command,
        current_create_time=101.0,
        runtime_prefix=runtime,
    )
    wrong_command = dict(
        candidate,
        cmdline=[str(executable), "-m", "unrelated.training"],
    )
    assert not audit._is_known_venv_redirector(
        wrong_command,
        current_parent_pid=40,
        current_command_line=current_command,
        current_create_time=101.0,
        runtime_prefix=runtime,
    )


def test_f0_pipeline_loader_has_no_generation_call() -> None:
    source = inspect.getsource(audit._load_pipeline_no_output)
    assert "BlipDiffusionControlNetPipeline.from_pretrained" in source
    assert "enable_model_cpu_offload" in source
    assert "enable_attention_slicing" in source
    assert "enable_vae_slicing" in source
    assert "pipe(" not in source
    assert '"pipeline_invoked": False' in source
    assert '"synthetic_pixels_generated": False' in source


def test_formal_hashes_runtime_without_premature_torch_import() -> None:
    source = inspect.getsource(audit.run_formal_f0)
    assert "hash_distribution_files=True" in source
    assert "import_runtime=False" in source
    runtime_source = inspect.getsource(audit.main)
    assert "hash_distribution_files=False" in runtime_source
    assert "import_runtime=True" in runtime_source


def test_artifact_manifest_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "source.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = audit._artifact_manifest(tmp_path, "manifest.json")
    names = [row["path"] for row in manifest["files"]]
    assert names == ["source.csv", "summary.json"]
    assert "manifest.json" not in names
