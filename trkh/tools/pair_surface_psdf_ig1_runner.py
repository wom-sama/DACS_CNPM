import sys


_AUTHORIZED_BOOTSTRAP = __name__ in ("__main__", "__mp_main__")
_AUDIT_INSTALLED = False
_AUDIT_PROBED = False
_AUDIT_POLICY = None
_AUDIT_EVENTS = []
_SPAWN_TICKET = None


def _stdlib_path(value):
    text = str(value or "").replace("/", "\\").rstrip("\\").casefold()
    bases = {
        str(sys.base_prefix).replace("/", "\\").rstrip("\\").casefold(),
        str(sys.base_exec_prefix).replace("/", "\\").rstrip("\\").casefold(),
    }
    return bool(text) and "site-packages" not in text and any(
        text == base or text.startswith(base + "\\") for base in bases
    )


def _scrub_to_stdlib():
    sys.path[:] = [entry for entry in sys.path if _stdlib_path(entry)]
    if not sys.path or not all(_stdlib_path(entry) for entry in sys.path):
        raise RuntimeError("PSDF bootstrap could not isolate the standard library")


def _bootstrap_audit(event, args):
    global _AUDIT_PROBED, _SPAWN_TICKET
    if event == "trkh.psdf.bootstrap_probe":
        _AUDIT_PROBED = True
    policy = _AUDIT_POLICY
    if policy is None:
        return
    if event == "open":
        policy.audit_open(args)
    elif event in ("os.mkdir", "os.rmdir", "os.remove", "os.unlink"):
        policy.audit_path(args[0], event)
    elif event in ("os.rename", "os.replace"):
        policy.audit_path(args[0], event + ":source"); policy.audit_path(args[1], event + ":destination")
    elif event.startswith("socket.") or event in (
        "os.system", "os.startfile", "pty.spawn", "urllib.Request"
    ):
        raise PermissionError("PSDF process forbids network and shell events")
    elif event == "subprocess.Popen" or event.startswith("os.spawn"):
        ticket = _SPAWN_TICKET
        if ticket is None or not policy.consume_spawn(ticket):
            raise PermissionError("PSDF process has no authorized spawn ticket")
        _SPAWN_TICKET = None


def _install_audit_hook():
    global _AUDIT_INSTALLED
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_bootstrap_audit)
        _AUDIT_INSTALLED = True
    sys.audit("trkh.psdf.bootstrap_probe", "ig1")
    if not _AUDIT_PROBED:
        raise RuntimeError("PSDF audit-hook probe did not fire")


def _audit_probe_live():
    return bool(_AUDIT_INSTALLED and _AUDIT_PROBED)


if _AUTHORIZED_BOOTSTRAP:
    _scrub_to_stdlib()
    _install_audit_hook()


import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from enum import Enum


PROTOCOL_ID = "trkh_psdf_information_gate_a0_r1"
AUTHORIZED_STATE = "authorized_train_only_primary_plus_replay"
ROLES = (
    "ddf_full", "static_matched", "ddf_spatial_only",
    "ddf_channel_only", "ddf_full_repeat",
)
RAW_WORKER_REPLY_KEYS = frozenset(
    ("kind", "run_name", "outer_fold", "target_blind_root_sha256",
     "raw_fold_root_sha256", "manifest_path", "access_root_sha256",
     "wall_seconds", "peak_rss_bytes", "peak_cuda_bytes")
)
FINALIZER_REPLY_KEYS = frozenset(
    ("kind", "outcome", "result_root_sha256", "receipt_path",
     "primary_preheld_sha256", "replay_preheld_sha256",
     "access_root_sha256", "wall_seconds", "peak_rss_bytes")
)
RAW_FIELDS = (
    "sample_indices", "held_folds", "component_ids", "component_order",
    "bootstrap_draws", "keeper_probabilities", "sidecar_preimages",
    "causal_scores", "invalid_fill_scores", "invalid_fill_maps",
    "attention_maps", "evidence_maps", "valid64", "valid32", "valid16",
    "bbox_valid16", "nonwrap_offsets", "donor_artifact_sha256",
    "artifact_identities", "spatial_filters32", "spatial_filters16",
    "channel_filters16", "channel_filters32", "cidt_sample_indices", "trace",
)
HEX = frozenset("0123456789abcdef")
REQUIRED_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
}
UNSET_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE")
FORBIDDEN_SPLITS = frozenset(("val", "valid", "validation", "test"))


def _hex64(value, name="SHA-256"):
    text = str(value)
    if len(text) != 64 or any(char not in HEX for char in text):
        raise ValueError(f"{name} must be lowercase 64-hex")
    return text


def _canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_sha(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _read_exact(path, spec, label):
    expected = _hex64(spec["sha256"], f"{label} SHA-256")
    payload = open(path, "rb").read()
    if len(payload) != int(spec["bytes"]) or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{label} bytes differ from authorization")
    return payload


def _verify_authorization(path, expected_sha256, required=None):
    expected = _hex64(expected_sha256, "expected authorization SHA-256")
    payload = open(os.path.abspath(path), "rb").read()
    observed = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise ValueError("Authorization SHA-256 differs")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Authorization must be one JSON object")
    if required is not None and set(value) != set(required):
        raise ValueError("Authorization top-level schema differs")
    return value, observed


def _create_once_bytes(path, payload):
    target = os.path.abspath(path)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(bytes(payload))
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("create-once write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(bytes(payload)).hexdigest()
    if _AUDIT_POLICY is not None:
        _AUDIT_POLICY.record_identity(target, "write_commit", len(payload), digest)
    return digest


def _mkdir_once(path):
    os.mkdir(os.path.abspath(path))
    return os.path.abspath(path)


def _under(path, root):
    try:
        return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
    except ValueError:
        return False


class _PSDFPolicy:
    def __init__(self, role, auth, *, boot=None, authorization_path=None, staging=None, spawn=()):
        if role not in ("parent", "worker", "finalizer"): raise ValueError("Unknown PSDF process role")
        repository = auth["repository_root"]; sources = [(os.path.join(repository, item["path"]), item) for item in auth["source_identities"]]
        input_names = tuple(auth["inputs"]) if role == "parent" else ("cohort_rgb", "cohort_valid", "cohort_arrays", "cidt_clean") if role == "worker" else ("cidt_clean",)
        reads = sources + [(auth["inputs"][name]["path"], auth["inputs"][name]) for name in input_names]
        if role == "parent":
            if boot is not None or not authorization_path or not staging: raise ValueError("Parent policy construction differs")
            reads.append((authorization_path, {"path": authorization_path})); write_roots = (staging, auth["outputs"]["pair_root"], auth["outputs"]["claim_path"], auth["outputs"]["receipt_path"]); normalization, fold = staging, None
        else:
            if boot is None or authorization_path is not None or staging is not None: raise ValueError("Child policy construction differs")
            if role == "worker": reads.extend((path, {"path": path}) for path in boot["fit_paths"].values())
            else:
                reads.extend((item["path"], item) for value in boot["aggregate_paths"].values() for item in [value, *value["files"]]); reads.extend((item["path"], item) for item in boot["target_paths"].values())
            write_roots = (boot["output_root"],) if role == "worker" else (boot["output_root"], auth["outputs"]["receipt_path"]); normalization, fold = boot["output_root"], boot["outer_fold"] if role == "worker" else None
        self.role = role; self.read_files = frozenset(os.path.normcase(os.path.abspath(path)) for path, _ in reads)
        self.read_roots = tuple(os.path.abspath(path) for path in auth["environment"]["site_paths"] + auth["environment"]["runtime_read_roots"]); self.write_roots = tuple(os.path.abspath(path) for path in write_roots)
        self.read_identities = {os.path.normcase(os.path.abspath(path)): dict(spec) for path, spec in reads}; self.normalization_root = os.path.abspath(normalization); self.fold = fold; self.spawn = list(spawn)
        self.events = []

    def _path(self, path):
        resolved = os.path.abspath(path)
        if self.normalization_root is not None and _under(resolved, self.normalization_root):
            suffix = os.path.relpath(resolved, self.normalization_root).replace("\\", "/")
            return "$RUN" if suffix == "." else "$RUN/" + suffix
        return os.path.normcase(resolved)

    def _append(self, operation, path, authoritative=True, **fields):
        row = {"order": len([item for item in self.events if item["authoritative"]]), "operation": operation,
               "path": self._path(path), "fold": self.fold, "member": None, "byte_count": None,
               "sha256": None, "authoritative": bool(authoritative)}
        row.update(fields); self.events.append(row)

    def audit_open(self, args):
        path = args[0] if args else None
        if isinstance(path, int) or not isinstance(path, (str, bytes, os.PathLike)):
            return
        resolved = os.path.abspath(os.fsdecode(path))
        mode = args[1] if len(args) > 1 else "r"
        write = isinstance(mode, int) and bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        write = write or isinstance(mode, str) and any(flag in mode for flag in "wax+")
        allowed = any(_under(resolved, root) for root in (self.write_roots if write else self.read_roots))
        allowed = allowed or (not write and os.path.normcase(resolved) in self.read_files)
        allowed = allowed or (not write and _stdlib_path(os.path.dirname(resolved)))
        parts = {part.casefold() for part in os.path.normpath(resolved).split(os.sep)}
        if not allowed or (not _stdlib_path(os.path.dirname(resolved)) and parts & FORBIDDEN_SPLITS):
            raise PermissionError(f"Unauthorized PSDF file access: {resolved}")
        identity = self.read_identities.get(os.path.normcase(resolved), {}) if not write else {}
        authoritative = write or os.path.normcase(resolved) in self.read_files
        self._append("write" if write else "read", resolved, authoritative=authoritative, mode=str(mode),
                     byte_count=identity.get("bytes"), sha256=identity.get("sha256"))

    def audit_path(self, path, operation):
        resolved = os.path.abspath(os.fsdecode(path))
        if not any(_under(resolved, root) for root in self.write_roots):
            raise PermissionError(f"Unauthorized PSDF path mutation: {resolved}")
        self._append(operation, resolved)

    def record_identity(self, path, operation, byte_count, digest):
        self._append(operation, path, byte_count=int(byte_count), sha256=_hex64(digest))

    def record_member(self, path, member, semantic, payload, identity):
        self._append("archive_member", path, member=str(member), semantic=str(semantic),
                     byte_count=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                     dtype=identity["dtype"], shape=list(identity["shape"]),
                     array_sha256=_hex64(identity["array_sha256"]))

    def consume_spawn(self, ticket):
        if self.role != "parent" or not self.spawn or ticket != self.spawn[0]:
            return False
        self.spawn.pop(0)
        return True

    def authoritative_rows(self):
        return [{key: value for key, value in row.items() if key != "authoritative"} for row in self.events if row["authoritative"]]

    def root(self): return _json_sha(self.authoritative_rows())


def _set_policy(policy):
    global _AUDIT_POLICY
    if not _audit_probe_live():
        _install_audit_hook()
    _AUDIT_POLICY = policy


class Stage(Enum):
    NEW = "new"
    AUTHORIZED = "authorized"
    CLAIMED = "claimed"
    TARGET_BLIND = "target_blind"
    TARGETS_MATERIALIZED = "targets_materialized"
    PRIMARY = "primary_complete"
    REPLAY = "replay_complete"
    PREHELD_PRIMARY = "preheld_primary"
    PREHELD_REPLAY = "preheld_replay"
    HELD = "held_authorized"
    CIDT = "cidt_released"
    FINAL = "finalized"


@dataclass
class StageMachine:
    stage: Stage = Stage.NEW
    target_blind_root: str = ""
    released: set = field(default_factory=set)

    def advance(self, expected, target):
        if self.stage is not expected:
            raise RuntimeError(f"IG1 stage {self.stage.value} cannot enter {target.value}")
        self.stage = target

    def authorize(self): self.advance(Stage.NEW, Stage.AUTHORIZED)
    def claim(self): self.advance(Stage.AUTHORIZED, Stage.CLAIMED)
    def freeze_target_blind(self, root):
        self.target_blind_root = _hex64(root, "target-blind root")
        self.advance(Stage.CLAIMED, Stage.TARGET_BLIND)
    def materialize_targets(self): self.advance(Stage.TARGET_BLIND, Stage.TARGETS_MATERIALIZED)
    def release_fit(self, run_name, outer_fold, observed_root):
        if self.stage not in (Stage.TARGETS_MATERIALIZED, Stage.PRIMARY):
            raise RuntimeError("Fit targets are not releasable at this stage")
        key = (str(run_name), int(outer_fold))
        if key in self.released or run_name not in ("primary", "replay") or not 0 <= int(outer_fold) < 5:
            raise RuntimeError("Fit-target release is not exactly once")
        if not hmac.compare_digest(self.target_blind_root, _hex64(observed_root)):
            raise RuntimeError("Worker target-blind root differs")
        if run_name == "replay" and self.stage is not Stage.PRIMARY:
            raise RuntimeError("Replay cannot precede the primary aggregate")
        self.released.add(key)
    def finish_primary(self):
        if {("primary", fold) for fold in range(5)} != self.released:
            raise RuntimeError("Primary fold set is incomplete")
        self.advance(Stage.TARGETS_MATERIALIZED, Stage.PRIMARY)
    def finish_replay(self):
        if not {("replay", fold) for fold in range(5)}.issubset(self.released):
            raise RuntimeError("Replay fold set is incomplete")
        self.advance(Stage.PRIMARY, Stage.REPLAY)
    def preheld(self, run_name):
        self.advance(Stage.REPLAY if run_name == "primary" else Stage.PREHELD_PRIMARY,
                     Stage.PREHELD_PRIMARY if run_name == "primary" else Stage.PREHELD_REPLAY)
    def authorize_held(self): self.advance(Stage.PREHELD_REPLAY, Stage.HELD)
    def release_cidt(self): self.advance(Stage.HELD, Stage.CIDT)
    def finalize(self): self.advance(Stage.CIDT, Stage.FINAL)


def _bootstrap_source_order():
    return ("import sys", "scrub_stdlib", "install_audit_hook", "emit_probe", "stdlib_imports")


AUTH_KEYS = frozenset((
    "schema_version", "protocol_id", "state", "authorization_id",
    "source_commit", "upstream_commit", "repository_root", "protocol",
    "registry", "source_identities", "interpreter", "environment", "inputs",
    "outputs", "resources", "lineage", "command", "quota", "fold_projection",
))
INPUT_KEYS = frozenset((
    "cohort_rgb", "cohort_valid", "cohort_arrays", "fold_manifest", "cidt_clean",
))


def _strict_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} schema differs")
    return value


def _validate_authorization(value, authorization_sha256, require_absent=True):
    row = _strict_keys(value, AUTH_KEYS, "authorization")
    if row["schema_version"] != 1 or row["protocol_id"] != PROTOCOL_ID or row["state"] != AUTHORIZED_STATE:
        raise ValueError("Authorization identity/state differs")
    _hex64(authorization_sha256, "authorization SHA-256")
    for name in ("source_commit", "upstream_commit"):
        text = str(row[name])
        if len(text) != 40 or any(char not in HEX for char in text):
            raise ValueError(f"Authorization {name} must be lowercase 40-hex")
    repository = os.path.abspath(row["repository_root"])
    if repository != os.path.abspath(os.getcwd()):
        raise ValueError("Authorized repository must be the current working directory")
    if row["protocol"] != {
        "path": "docs/TRKH_5CLASS_PSDF_INFORMATION_GATE_A0_R1_PROTOCOL_20260729.md",
        "sha256": "1365a9aa022ce20d0c5d8429365e89b0c891c16fd61b65c4642ffbf650a4d78b",
    } or row["registry"] != {
        "path": "docs/TRKH_5CLASS_PSDF_INFORMATION_GATE_A0_R1_MECHANISM_REGISTRY_20260729.json",
        "sha256": "cd1d16f7b0c6a906de8b5f43e84b64b0a38a8f817892fcba957befa567c24e70",
    }:
        raise ValueError("Protocol/registry identity differs")
    _strict_keys(row["inputs"], INPUT_KEYS, "authorization inputs")
    for name, spec in row["inputs"].items():
        if not isinstance(spec, dict) or not {"path", "sha256", "bytes"}.issubset(spec):
            raise ValueError(f"Input specification is incomplete: {name}")
        _hex64(spec["sha256"], f"{name} SHA-256")
        if int(spec["bytes"]) <= 0 or not os.path.isabs(spec["path"]):
            raise ValueError(f"Input path/size is invalid: {name}")
    interpreter = row["interpreter"]
    if interpreter != {"path": "D:/DataAI/.venv/Scripts/python.exe", "version": "3.9.11", "flags": ["-S", "-s", "-B"]}:
        raise ValueError("Interpreter lock differs")
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(interpreter["path"])):
        raise ValueError("Running interpreter differs")
    if list(sys.version_info[:3]) != [3, 9, 11] or not (sys.flags.no_site and sys.flags.no_user_site and sys.dont_write_bytecode):
        raise ValueError("Python version/flags differ")
    if any(os.environ.get(name) is not None for name in UNSET_ENV) or any(os.environ.get(k) != v for k, v in REQUIRED_ENV.items()):
        raise ValueError("Process-start environment differs")
    environment = row["environment"]
    if not isinstance(environment, dict) or environment.get("variables") != REQUIRED_ENV:
        raise ValueError("Authorized environment variables differ")
    for name in ("site_paths", "runtime_read_roots"):
        paths = environment.get(name)
        if not isinstance(paths, list) or not paths or len(paths) != len({os.path.normcase(os.path.abspath(path)) for path in paths}) or any(not os.path.isabs(path) or not os.path.isdir(path) for path in paths):
            raise ValueError(f"Authorized {name} differ")
    quota = row["quota"]
    if quota != {"pair": 1, "primary_aggregates": 1, "replay_aggregates": 1,
                  "fold_workers": 10, "scientific_finalizers": 1, "concurrent_children": 1}:
        raise ValueError("Execution quota differs")
    limits = row["resources"]
    required_limits = {"aggregate_wall_seconds": 2700, "pair_wall_seconds": 5700,
                       "worker_cuda_bytes": 2 * 1024 ** 3, "process_rss_bytes": 8 * 1024 ** 3,
                       "aggregate_temporary_bytes": 1024 ** 3,
                       "aggregate_retained_bytes": 512 * 1024 ** 2,
                       "pair_retained_bytes": 1024 ** 3}
    if limits != required_limits:
        raise ValueError("Resource ceiling differs")
    outputs = row["outputs"]
    if set(outputs) != {"pair_root", "claim_path", "receipt_path"} or any(not os.path.isabs(path) for path in outputs.values()):
        raise ValueError("Output schema/path differs")
    if require_absent and (os.path.exists(outputs["pair_root"]) or os.path.exists(outputs["claim_path"]) or os.path.exists(outputs["receipt_path"])):
        raise FileExistsError("Create-once pair output/claim/receipt already exists")
    if not isinstance(row["source_identities"], list) or not row["source_identities"]:
        raise ValueError("Source identity closure is empty")
    seen_sources = set()
    for source in row["source_identities"]:
        _strict_keys(source, ("name", "path", "sha256", "bytes"), "source identity")
        _hex64(source["sha256"], "source identity SHA-256")
        path = source["path"]
        resolved = os.path.abspath(os.path.join(repository, path)) if isinstance(path, str) else ""
        if not path or os.path.isabs(path) or not _under(resolved, repository) or int(source["bytes"]) <= 0 or os.path.normcase(resolved) in seen_sources:
            raise ValueError("Source path/size/uniqueness differs")
        seen_sources.add(os.path.normcase(resolved))
    _fold_projection(row["fold_projection"])
    return row


def _activate_paths(repository, site_paths):
    if not all(_stdlib_path(item) for item in sys.path):
        raise RuntimeError("Authorized path activation requires stdlib-only sys.path")
    additions = [os.path.abspath(path) for path in site_paths] + [os.path.abspath(repository)]
    for path in reversed(additions):
        if path not in sys.path:
            sys.path.insert(0, path)


def _array_sha(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _array_identity(array):
    return {"dtype": str(array.dtype), "shape": [int(v) for v in array.shape],
            "array_sha256": _array_sha(array)}


def _deep_identity(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, dict):
        return {str(key): _deep_identity(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_deep_identity(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Raw identity has unsupported type: {type(value).__name__}")


def _raw_root(raw):
    return _json_sha(_deep_identity(raw))


def _npy_bytes(array):
    import numpy as np
    output = io.BytesIO()
    np.lib.format.write_array(output, np.ascontiguousarray(array), allow_pickle=False)
    return output.getvalue()


def _save_array(path, array):
    payload = _npy_bytes(array)
    return {"path": os.path.basename(path), "bytes": len(payload),
            "sha256": _create_once_bytes(path, payload), "array": _array_identity(array)}


def _load_array(path, identity=None):
    import numpy as np
    value = np.load(path, allow_pickle=False)
    if identity is not None and _array_identity(value) != identity:
        raise ValueError(f"NPY identity differs: {path}")
    return np.ascontiguousarray(value)


def _verify_file_spec(spec, label):
    observed, size = _file_sha(spec["path"])
    if size != int(spec["bytes"]) or observed != _hex64(spec["sha256"]):
        raise ValueError(f"Authorized file differs: {label}")


def _npz_member(spec, name):
    import numpy as np
    member_spec = spec.get("members", {}).get(name)
    if not isinstance(member_spec, dict):
        raise ValueError(f"NPZ member is not authorized: {name}")
    with zipfile.ZipFile(spec["path"], "r") as archive:
        names = archive.namelist()
        expected = name + ".npy"
        if expected not in names or any(item.endswith("/") or ".." in item.split("/") for item in names):
            raise ValueError("NPZ archive members are invalid")
        payload = archive.read(expected)
    if "bytes" in member_spec and len(payload) != int(member_spec["bytes"]):
        raise ValueError(f"NPZ member byte count differs: {name}")
    if "sha256" in member_spec and hashlib.sha256(payload).hexdigest() != _hex64(member_spec["sha256"]):
        raise ValueError(f"NPZ member payload differs: {name}")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    identity = {"dtype": member_spec["dtype"], "shape": member_spec["shape"],
                "array_sha256": member_spec["array_sha256"]}
    if _array_identity(value) != identity:
        raise ValueError(f"NPZ array identity differs: {name}")
    if _AUDIT_POLICY is not None:
        _AUDIT_POLICY.record_member(spec["path"], expected,
                                    member_spec.get("semantic", name), payload, identity)
    return np.ascontiguousarray(value)


def _fold_projection(payload):
    row = _strict_keys(payload, ("sample_indices", "held_folds", "component_ids",
                                 "component_order", "projection_sha256"), "fold projection")
    core = {name: row[name] for name in ("sample_indices", "held_folds", "component_ids", "component_order")}
    if not hmac.compare_digest(_json_sha(core), _hex64(row["projection_sha256"], "projection SHA-256")):
        raise ValueError("Target-free fold projection root differs")
    indices, folds, components, order = (core[name] for name in core)
    if len(indices) != 763 or indices != sorted(indices) or len(set(indices)) != 763:
        raise ValueError("Fold projection IDs differ")
    if len(folds) != 763 or any(type(value) is not int or not 0 <= value < 5 for value in folds):
        raise ValueError("Fold projection values differ")
    if len(components) != 763 or any(not isinstance(value, str) for value in components):
        raise ValueError("Fold projection components differ")
    for value in components: _hex64(value, "component SHA-256")
    if len(order) != 158 or order != sorted(set(components)):
        raise ValueError("Component order differs")
    if len(set(zip(components, folds))) != 158:
        raise ValueError("A component crosses held folds")
    if _json_sha(list(zip(indices, folds))) != "fd61cd1c20c9ca66c113db1ec65d25c8e0ea8c4ec851f80f3c15516f993921df":
        raise ValueError("Sample/fold projection differs from V2R2")
    if _json_sha(order) != "e10e1045e6ede63ade0cb436b684be7a6ab377d0d27f74480ac8721756045ccb":
        raise ValueError("Component registry differs from V2R2")
    if _json_sha([[index, component] for index, component in zip(indices, components)]) != "39634b6949d578ec67d7c1a5475aac8249bddd7cfed346fac4939c1e24c204a6":
        raise ValueError("Sample/component projection differs from V2R2")
    return core


def _ordinal(values, indices, np):
    order = np.lexsort((indices, values))
    ranks = np.empty(len(indices), dtype=np.int64)
    ranks[order] = np.arange(len(indices), dtype=np.int64)
    return ranks


def _hungarian(costs):
    rows, columns = len(costs), len(costs[0])
    u, v, p, way = [0] * (rows + 1), [0] * (columns + 1), [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0], column0, minimum, used = row, 0, [None] * (columns + 1), [False] * (columns + 1)
        while True:
            used[column0] = True
            row0, delta, column1 = p[column0], None, 0
            for column in range(1, columns + 1):
                if not used[column]:
                    reduced = costs[row0 - 1][column - 1] - u[row0] - v[column]
                    if minimum[column] is None or reduced < minimum[column]:
                        minimum[column], way[column] = reduced, column0
                    if delta is None or minimum[column] < delta or (minimum[column] == delta and column < column1):
                        delta, column1 = minimum[column], column
            if delta is None:
                raise RuntimeError("Donor assignment is infeasible")
            for column in range(columns + 1):
                if used[column]: u[p[column]] += delta; v[column] -= delta
                elif minimum[column] is not None: minimum[column] -= delta
            column0 = column1
            if p[column0] == 0: break
        while column0:
            column1 = way[column0]; p[column0] = p[column1]; column0 = column1
    result = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]: result[p[column] - 1] = column - 1
    if min(result) < 0: raise RuntimeError("Donor assignment is incomplete")
    return result


def _donor_positions(indices, folds, components, keeper, valid32, valid16, bbox, fold, np):
    victim = np.flatnonzero(folds == fold); donor = np.flatnonzero(folds == ((fold + 1) % 5))
    if set(components[i] for i in victim) & set(components[i] for i in donor):
        raise ValueError("Donor components overlap held components")
    def ranks(rows, values): return _ordinal(values[rows], indices[rows], np)
    vr_b, dr_b = ranks(victim, bbox.sum((1, 2))), ranks(donor, bbox.sum((1, 2)))
    vr_v, dr_v = ranks(victim, valid16.sum((1, 2))), ranks(donor, valid16.sum((1, 2)))
    rivals = np.asarray((0, 2, 4), dtype=np.int64)
    strongest = rivals[np.argmax(keeper[:, rivals], axis=1)]
    vectors = np.empty((len(victim), len(donor), 6), dtype=np.int64)
    for i, left in enumerate(victim):
        for j, right in enumerate(donor):
            vectors[i, j] = (np.count_nonzero(valid32[left] ^ valid32[right]),
                             np.count_nonzero(valid16[left] ^ valid16[right]),
                             np.count_nonzero(bbox[left] ^ bbox[right]),
                             int(strongest[left] != strongest[right]), abs(int(vr_b[i] - dr_b[j])),
                             abs(int(vr_v[i] - dr_v[j])))
    maxima = vectors.max((0, 1)); coefficients = [0] * 6; coefficients[-1] = 1
    for i in range(4, -1, -1): coefficients[i] = len(victim) * sum(int(maxima[j]) * coefficients[j] for j in range(i + 1, 6)) + 1
    scalar = [[sum(int(vectors[i, j, k]) * coefficients[k] for k in range(6)) for j in range(len(donor))] for i in range(len(victim))]
    lower, extra = divmod(len(victim), len(donor)); base = len(donor) + 1; factor = base ** len(victim)
    edge = [[scalar[i][j] * factor + j * base ** (len(victim) - 1 - i) for j in range(len(donor))] for i in range(len(victim))]
    span = len(victim) * (max(map(max, edge)) - min(map(min, edge))) + 1
    slots = [(j, mandatory) for j in range(len(donor)) for mandatory in ([True] * lower + ([False] if extra else []))]
    assigned = _hungarian([[edge[i][j] - (span if mandatory else 0) for j, mandatory in slots] for i in range(len(victim))])
    mandatory_slots = {position for position, (_, mandatory) in enumerate(slots) if mandatory}
    if not mandatory_slots.issubset(set(assigned)):
        raise RuntimeError("Donor mandatory capacity slots were not selected")
    selected = np.asarray([slots[position][0] for position in assigned], dtype=np.int64)
    counts = np.bincount(selected, minlength=len(donor))
    upper = lower + int(bool(extra))
    if counts.min() < lower or counts.max() > upper or np.count_nonzero(counts == upper) != (extra if extra else len(donor)):
        raise RuntimeError("Donor capacity differs")
    picked = vectors[np.arange(len(victim)), selected]
    return victim, donor, selected, counts, picked, picked.sum(0), coefficients


def _offsets(sample_indices, folds, np):
    allowed = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    output = np.empty((len(sample_indices), 2, 2), dtype=np.int64)
    for fold in range(5):
        rows = np.flatnonzero(folds == fold)
        for block in range(2):
            ordered = sorted(rows, key=lambda pos: (hashlib.sha256(f"20260729|{fold}|{block}|{int(sample_indices[pos])}".encode()).hexdigest(), int(sample_indices[pos])))
            for rank, pos in enumerate(ordered): output[pos, block] = allowed[(rank + fold + 2 * block) % 8]
    return output


def _rss_bytes():
    if os.name != "nt":
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t)] + [(f"x{i}", ctypes.c_size_t) for i in range(7)]
    value = Counters(); value.cb = ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(value.peak_working_set)


def _tensor_state_sha(values):
    return _json_sha({name: _array_identity(value.detach().cpu().numpy()) for name, value in sorted(values)})


def _optimizer_sha(optimizer, model):
    names = {parameter: name for name, parameter in model.named_parameters()}
    state = {}
    for parameter, values in optimizer.state.items():
        state[names[parameter]] = {key: (_array_identity(value.detach().cpu().numpy()) if hasattr(value, "detach") else value)
                                   for key, value in sorted(values.items())}
    groups = [{key: ([names[p] for p in value] if key == "params" else value)
               for key, value in sorted(group.items())} for group in optimizer.param_groups]
    return _json_sha({"state": state, "param_groups": groups})


def _prepared_hashes(prepared, np):
    output = np.empty(763, dtype="<U64")
    for index in range(763):
        output[index] = _json_sha({name: _array_identity(getattr(prepared, name)[index].cpu().numpy())
                                  for name in ("values", "valid64", "valid32", "valid16")})
    return output


def _structural_context(auth, projection):
    import numpy as np
    import torch
    from trkh.tools import pair_surface_ddf_v2_engine as engine
    for name in ("cohort_rgb", "cohort_valid", "cohort_arrays", "cidt_clean"):
        _verify_file_spec(auth["inputs"][name], name)
    arrays = auth["inputs"]["cohort_arrays"]
    indices = _npz_member(arrays, "sample_indices").astype("<i8", copy=False)
    keeper = _npz_member(arrays, "keeper_probabilities").astype("<f8")
    boxes = _npz_member(arrays, "model_boxes")
    cidt_indices = _npz_member(auth["inputs"]["cidt_clean"], "sample_indices").astype("<i8", copy=False)
    if cidt_indices.shape != (9215,) or not np.all(cidt_indices[1:] > cidt_indices[:-1]):
        raise ValueError("CIDT sample-index projection differs")
    if indices.tolist() != projection["sample_indices"]:
        raise ValueError("Cache IDs differ from V2R2 projection")
    folds = np.asarray(projection["held_folds"], dtype="<i8")
    components = tuple(projection["component_ids"]); component_order = tuple(projection["component_order"])
    rgb = np.load(auth["inputs"]["cohort_rgb"]["path"], mmap_mode="r", allow_pickle=False)
    packed = np.load(auth["inputs"]["cohort_valid"]["path"], mmap_mode="r", allow_pickle=False)
    if tuple(rgb.shape) != (763, 3, 256, 256) or rgb.dtype != np.uint8:
        raise ValueError("RGB cache shape/dtype differs")
    valid256 = engine.unpack_valid_masks_little(np.asarray(packed))
    zero_rgb = np.where(valid256[:, None], np.asarray(rgb), np.uint8(0))
    prepared = engine.prepare_cached_model_inputs(torch.from_numpy(zero_rgb), torch.from_numpy(valid256))
    bbox = engine.rasterize_model_boxes_16(torch.from_numpy(boxes)).numpy() & prepared.valid16[:, 0].numpy()
    noise = np.random.Generator(np.random.PCG64(20260729)).integers(0, 256, size=rgb.shape, dtype=np.uint8, endpoint=False)
    if _array_sha(noise) != "9c18e8ab55f6bbf15d1372876bf2578cd094f2a3f1a755e89992c086bf365596": raise RuntimeError("Invalid-fill PCG64 noise identity differs")
    noise_rgb = np.where(valid256[:, None], np.asarray(rgb), noise)
    alternate = engine.prepare_cached_model_inputs(torch.from_numpy(noise_rgb), torch.from_numpy(valid256))
    hashes = np.stack((_prepared_hashes(prepared, np), _prepared_hashes(alternate, np)))
    if not np.array_equal(hashes[0], hashes[1]):
        raise RuntimeError("Invalid-fill prepared inputs differ")
    offsets = _offsets(indices, folds, np)
    donors, donor_rows = {}, []
    for fold in range(5):
        victim, donor, selected, counts, picked, totals, coefficients = _donor_positions(
            indices, folds, components, keeper, prepared.valid32[:, 0].numpy(), prepared.valid16[:, 0].numpy(), bbox, fold, np)
        donors[fold] = (victim, donor, selected)
        donor_rows.append({"fold": fold, "victim_ids": indices[victim].tolist(),
                           "donor_pool_ids": indices[donor].tolist(), "selected_donor_positions": selected.tolist(),
                           "donor_ids": indices[donor[selected]].tolist(), "donor_use_counts": counts.tolist(),
                           "selected_priority_vectors": picked.tolist(), "lexicographic_vector": totals.tolist(),
                           "priority_coefficients": coefficients})
    init_orders = []
    for role in ROLES:
        for fold in range(5):
            model = engine.initialize_sidecar(role, fold=fold, device="cpu")
            fit_ids = indices[(folds != fold) & (folds != ((fold + 1) % 5))]
            orders = np.stack(engine.build_epoch_orders(fit_ids, seed=engine.role_order_seed(role, fold)))
            init_orders.append((role, fold, _tensor_state_sha(model.state_dict().items()), _array_identity(orders)))
    bootstrap = np.random.Generator(np.random.PCG64(20260729)).integers(0, 158, size=(2000, 158), dtype=np.int64, endpoint=False)
    target_blind = {"sample_indices": _array_identity(indices), "held_folds": _array_identity(folds),
                    "component_ids": list(components), "component_order": list(component_order),
                    "keeper_probabilities": _array_identity(keeper), "valid64": _array_identity(prepared.valid64[:, 0].numpy()),
                    "valid32": _array_identity(prepared.valid32[:, 0].numpy()), "valid16": _array_identity(prepared.valid16[:, 0].numpy()),
                    "bbox_valid16": _array_identity(bbox), "nonwrap_offsets": _array_identity(offsets),
                    "bootstrap_draws": _array_identity(bootstrap), "donors": donor_rows, "initialization_orders": init_orders,
                    "prepared_input_sha256": _array_identity(hashes), "invalid_fill_noise": _array_identity(noise),
                    "invalid_fill_seed": 20260729, "cidt_sample_indices": _array_identity(cidt_indices)}
    return {"np": np, "torch": torch, "engine": engine, "indices": indices, "folds": folds,
            "components": components, "component_order": component_order, "keeper": keeper,
            "prepared": prepared, "alternate": alternate, "bbox": bbox, "offsets": offsets,
            "bootstrap": bootstrap, "donors": donors, "donor_rows": donor_rows,
            "prepared_hashes": hashes, "noise_identity": _array_identity(noise), "initialization_orders": init_orders, "cidt_indices": cidt_indices,
            "target_blind_root": _json_sha(target_blind)}


def _evaluate(model, prepared, positions, engine, torch, trace=False, **kwargs):
    values = {"scores": [], "attention_maps": [], "evidence_maps": [], "spatial_filters": [[], []], "channel_filters": [[], []]}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), 64):
            batch_pos = positions[start:start + 64]; batch = prepared.take(batch_pos)
            row = model.forward_with_trace(batch.values.cuda(), batch.valid64.cuda(), batch.valid32.cuda(), batch.valid16.cuda(), **kwargs)
            values["scores"].append(row["scores"].cpu().numpy())
            if trace:
                for name in ("attention_maps", "evidence_maps"): values[name].append(row[name].cpu().numpy())
                for block in range(2):
                    values["spatial_filters"][block].append(row["spatial_filters"][block].cpu().numpy())
                    values["channel_filters"][block].append(row["channel_filters"][block].cpu().numpy())
    np = __import__("numpy")
    output = {"scores": np.concatenate(values["scores"]).astype("<f8")}
    if trace:
        output.update({name: np.concatenate(values[name]).astype("<f8") for name in ("attention_maps", "evidence_maps")})
        output["spatial_filters"] = [np.concatenate(v).astype("<f8") for v in values["spatial_filters"]]
        output["channel_filters"] = [np.concatenate(v).astype("<f8") for v in values["channel_filters"]]
    return output


def _train_role(ctx, role, fold, fit_ids, fit_targets, output, channel, observation):
    np, torch, engine = ctx["np"], ctx["torch"], ctx["engine"]
    model = engine.initialize_sidecar(role, fold=fold, device="cuda")
    initial = _tensor_state_sha(model.state_dict().items()); optimizer = engine.build_adamw(model, lr=engine.learning_rate(0))
    optimizer_hashes, gradient_hashes, losses = [_optimizer_sha(optimizer, model)], [], []
    fit_positions = np.searchsorted(ctx["indices"], fit_ids)
    supervision = engine.build_task_supervision(torch.from_numpy(fit_targets), torch.from_numpy(ctx["bbox"][fit_positions].any((1, 2))))
    local = {int(sample): pos for pos, sample in enumerate(fit_ids.tolist())}
    global_pos = {int(sample): pos for pos, sample in enumerate(ctx["indices"].tolist())}
    orders = engine.build_epoch_orders(fit_ids, seed=engine.role_order_seed(role, fold)); step = 0
    for epoch, order in enumerate(orders):
        for group in optimizer.param_groups: group["lr"] = engine.learning_rate(epoch)
        for start in range(0, len(order), 64):
            batch_ids = order[start:start + 64]; positions = np.asarray([global_pos[int(v)] for v in batch_ids]); local_pos = np.asarray([local[int(v)] for v in batch_ids])
            batch = ctx["prepared"].take(positions); model.train(); optimizer.zero_grad(set_to_none=True)
            row = model.forward_with_trace(batch.values.cuda(), batch.valid64.cuda(), batch.valid32.cuda(), batch.valid16.cuda())
            loss = engine.pair_surface_loss(row["scores"], row["attention_maps"], supervision.take(local_pos),
                                            torch.from_numpy(ctx["bbox"][positions]).cuda(), batch.valid16.cuda())["total"]
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite PSDF loss")
            loss.backward(); gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(value).all() for _, value in gradients): raise FloatingPointError("Non-finite/missing gradient")
            gradient_hashes.append(_tensor_state_sha(gradients)); optimizer.step(); optimizer_hashes.append(_optimizer_sha(optimizer, model)); losses.append(float(loss.detach().cpu())); step += 1
            if not observation[0]: channel.send({"kind": "observation_started"}); observation[0] = True
    if step != 160: raise RuntimeError("PSDF fit did not execute exactly 160 steps")
    fit_prepared = ctx["prepared"].take(fit_positions)
    cuda_prepared = engine.PreparedSurfaceInputs(*(getattr(fit_prepared, name).cuda()
                                                   for name in ("values", "valid64", "valid32", "valid16")))
    recalibration = engine.recalibrate_masked_batch_norms(model, cuda_prepared, sample_indices=fit_ids)
    final = _tensor_state_sha(model.state_dict().items())
    state_dir = _mkdir_once(os.path.join(output, "state_" + role)); state_records = {}
    for name, value in sorted(model.state_dict().items()):
        filename = name.replace(".", "__") + ".npy"; state_records[name] = _save_array(os.path.join(state_dir, filename), value.cpu().numpy()); state_records[name]["path"] = "state_" + role + "/" + filename
    calibration = np.flatnonzero(ctx["folds"] == ((fold + 1) % 5)); held = np.flatnonzero(ctx["folds"] == fold)
    return model, {"calibration_ids": ctx["indices"][calibration], "calibration_scores": _evaluate(model, ctx["prepared"], calibration, engine, torch)["scores"],
                   "held_ids": ctx["indices"][held], "held_scores": _evaluate(model, ctx["prepared"], held, engine, torch)["scores"],
                   "losses": np.asarray(losses, dtype="<f8"), "gradient_sha256": np.asarray(gradient_hashes, dtype="<U64"),
                   "optimizer_state_sha256": np.asarray(optimizer_hashes, dtype="<U64"), "initialized_model_state_sha256": initial,
                   "final_model_state_sha256": final, "recalibration": recalibration, "state_records": state_records}


def _causal_fold(ctx, model, fold):
    np, torch, engine = ctx["np"], ctx["torch"], ctx["engine"]
    victim, donor_pool, selected = ctx["donors"][fold]; donor = donor_pool[selected]
    collected = {name: [] for name in ("causal", "zero_scores", "noise_scores", "zero_maps", "noise_maps",
                                        "attention", "evidence", "spatial32", "spatial16", "channel16", "channel32")}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(victim), 64):
            positions = victim[start:start + 64]; sources = donor[start:start + 64]
            batch = ctx["prepared"].take(positions); source_batch = ctx["prepared"].take(sources)
            args = [getattr(batch, name).cuda() for name in ("values", "valid64", "valid32", "valid16")]
            source_args = [getattr(source_batch, name).cuda() for name in ("values", "valid64", "valid32", "valid16")]
            clean = model.forward_with_trace(*args); source = model.forward_with_trace(*source_args)
            self_row = model.forward_with_trace(*args, filter_overrides=list(zip(clean["spatial_filters"], clean["channel_filters"])),
                                                 override_valid_masks=(source_args[2], source_args[3]))
            donor_row = model.forward_with_trace(*args, filter_overrides=list(zip(source["spatial_filters"], source["channel_filters"])),
                                                  override_valid_masks=(source_args[2], source_args[3]))
            displaced = model.forward_with_trace(*args, spatial_offsets=torch.from_numpy(ctx["offsets"][positions]).cuda(),
                                                  spatial_offset_direction="source_to_destination")
            spatial = model.forward_with_trace(*args, causal_mode=engine.CAUSAL_SPATIAL_NEUTRAL)
            channel = model.forward_with_trace(*args, causal_mode=engine.CAUSAL_CHANNEL_NEUTRAL)
            noise_batch = ctx["alternate"].take(positions); noise_args = [getattr(noise_batch, name).cuda() for name in ("values", "valid64", "valid32", "valid16")]
            noise = model.forward_with_trace(*noise_args)
            collected["causal"].append(torch.stack(tuple(row["scores"][:, 0] for row in (clean, self_row, donor_row, displaced, spatial, channel))).cpu().numpy())
            for key, row in (("zero", clean), ("noise", noise)):
                collected[key + "_scores"].append(row["scores"][:, 0].cpu().numpy()); collected[key + "_maps"].append(row["attention_maps"].cpu().numpy())
            for key in ("attention", "evidence"): collected[key].append(clean[key + "_maps"].cpu().numpy())
            collected["spatial32"].append(clean["spatial_filters"][0].cpu().numpy()); collected["spatial16"].append(clean["spatial_filters"][1].cpu().numpy())
            collected["channel16"].append(clean["channel_filters"][0].cpu().numpy()); collected["channel32"].append(clean["channel_filters"][1].cpu().numpy())
    output = {name: np.concatenate(values, axis=1 if name == "causal" else 0).astype("<f8") for name, values in collected.items()}
    output["held_ids"] = ctx["indices"][victim].astype("<i8", copy=False)
    return output


def _persist_tree(root, value):
    import numpy as np
    array_root = _mkdir_once(os.path.join(root, "arrays")); counter = [0]
    def visit(item):
        if isinstance(item, np.ndarray):
            name = f"{counter[0]:04d}.npy"; counter[0] += 1; record = _save_array(os.path.join(array_root, name), item)
            record["path"] = "arrays/" + name
            return {"__npy__": record}
        if isinstance(item, dict): return {str(key): visit(item[key]) for key in sorted(item)}
        if isinstance(item, (tuple, list)): return [visit(part) for part in item]
        if isinstance(item, (str, int, float, bool)) or item is None: return item
        raise TypeError(f"Artifact tree cannot persist {type(item).__name__}")
    return visit(value)


def _restore_tree(root, wire):
    if isinstance(wire, dict) and set(wire) == {"__npy__"}:
        record = wire["__npy__"]; path = os.path.abspath(os.path.join(root, record["path"]))
        if not _under(path, root): raise PermissionError("NPY artifact escapes its root")
        observed, size = _file_sha(path)
        if observed != record["sha256"] or size != int(record["bytes"]): raise ValueError("NPY artifact bytes differ")
        return _load_array(path, record["array"])
    if isinstance(wire, dict): return {key: _restore_tree(root, value) for key, value in wire.items()}
    if isinstance(wire, list): return [_restore_tree(root, value) for value in wire]
    return wire


def _preimage_root(preimages):
    return _json_sha([{"role": row["role"], "outer_fold": row["outer_fold"],
                       "calibration_sample_indices": _array_identity(row["calibration_sample_indices"]),
                       "calibration_raw_scores": _array_identity(row["calibration_raw_scores"]),
                       "held_sample_indices": _array_identity(row["held_sample_indices"]),
                       "held_raw_scores": _array_identity(row["held_raw_scores"])} for row in preimages])


def _artifact_rows(raw):
    names = ("sidecar_preimages", "causal_scores", "invalid_fill_scores", "invalid_fill_maps", "attention_maps",
             "evidence_maps", "spatial_filters32", "spatial_filters16", "channel_filters16", "channel_filters32",
             "donor_assignment", "nonwrap_offsets")
    values = {name: _array_sha(raw[name]) for name in names[1:10]}
    values.update(sidecar_preimages=_preimage_root(raw["sidecar_preimages"]), donor_assignment=raw["donor_artifact_sha256"],
                  nonwrap_offsets=_array_sha(raw["nonwrap_offsets"]))
    return [{"name": name, "sha256": values[name]} for name in names]


def _assemble_raw(folds):
    import numpy as np
    structural = folds[0]["structural"]
    if any(_raw_root(item["structural"]) != _raw_root(structural) for item in folds[1:]): raise ValueError("Fold structural roots differ")
    indices, held_folds = structural["sample_indices"], structural["held_folds"]; positions = {int(v): i for i, v in enumerate(indices)}
    raw = {name: structural[name] for name in ("sample_indices", "held_folds", "component_ids", "component_order", "bootstrap_draws",
                                                "keeper_probabilities", "valid64", "valid32", "valid16", "bbox_valid16", "nonwrap_offsets", "cidt_sample_indices")}
    raw["donor_artifact_sha256"] = structural["donor_artifact_sha256"]
    raw["sidecar_preimages"] = [folds[fold]["preimages"][role] for role in range(5) for fold in range(5)]
    shapes = {"causal_scores": (6, 763), "invalid_fill_scores": (2, 763), "invalid_fill_maps": (2, 763, 4, 16, 16),
              "attention_maps": (763, 4, 16, 16), "evidence_maps": (763, 4, 16, 16), "spatial_filters32": (763, 9, 32, 32),
              "spatial_filters16": (763, 9, 16, 16), "channel_filters16": (763, 16, 9), "channel_filters32": (763, 32, 9)}
    for name, shape in shapes.items(): raw[name] = np.empty(shape, dtype="<f8")
    for item in folds:
        causal = item["causal"]; destination = np.asarray([positions[int(v)] for v in causal["held_ids"]])
        raw["causal_scores"][:, destination] = causal["causal"]
        raw["invalid_fill_scores"][:, destination] = np.stack((causal["zero_scores"], causal["noise_scores"]))
        raw["invalid_fill_maps"][:, destination] = np.stack((causal["zero_maps"], causal["noise_maps"]))
        for target, source in (("attention_maps", "attention"), ("evidence_maps", "evidence"), ("spatial_filters32", "spatial32"),
                               ("spatial_filters16", "spatial16"), ("channel_filters16", "channel16"), ("channel_filters32", "channel32")):
            raw[target][destination] = causal[source]
    trace = {}
    for name, shape, dtype in (("losses", (5, 5, 160), "<f8"), ("gradient_sha256", (5, 5, 160), "<U64"),
                               ("optimizer_state_sha256", (5, 5, 161), "<U64"), ("initialized_model_state_sha256", (5, 5), "<U64"),
                               ("final_model_state_sha256", (5, 5), "<U64")):
        trace[name] = np.empty(shape, dtype=dtype)
        for fold in range(5):
            for role in range(5): trace[name][role, fold] = folds[fold]["traces"][role][name]
    trace["prepared_input_sha256"] = structural["prepared_input_sha256"]; raw["trace"] = trace
    raw["artifact_identities"] = _artifact_rows(raw)
    return raw


def _child_handshake(channel, bootstrap, child_kind):
    keys = ("kind", "nonce", "parent_pid", "authorization_bytes", "authorization_sha256", "output_root",
            "site_paths", "runtime_read_roots", "run_name", "outer_fold", "fit_paths", "aggregate_paths", "target_paths")
    row = _strict_keys(bootstrap, keys, "child bootstrap")
    if row["kind"] != child_kind or multiprocessing.parent_process() is None or multiprocessing.parent_process().pid != int(row["parent_pid"]):
        raise PermissionError("Child lacks its live spawning parent")
    nonce = _hex64(row["nonce"], "channel nonce")
    if not all(hasattr(channel, name) for name in ("send", "recv", "poll")): raise PermissionError("Child channel differs")
    channel.send({"kind": "child_ready", "nonce": nonce, "pid": os.getpid()})
    if not channel.poll(30): raise PermissionError("Parent channel was not live")
    message = channel.recv()
    if message != {"kind": "parent_authorized", "nonce": nonce, "authorization_sha256": row["authorization_sha256"]}:
        raise PermissionError("Parent channel binding differs")
    payload = bytes(row["authorization_bytes"])
    if hashlib.sha256(payload).hexdigest() != _hex64(row["authorization_sha256"]): raise ValueError("Child authorization bytes differ")
    auth = _validate_authorization(json.loads(payload.decode("utf-8")), row["authorization_sha256"], require_absent=False)
    if row["site_paths"] != auth["environment"]["site_paths"] or row["runtime_read_roots"] != auth["environment"]["runtime_read_roots"]: raise PermissionError("Child runtime roots differ from authorization")
    return row, auth


def _child_policy(auth, boot, role):
    repository = auth["repository_root"]; sources = [os.path.join(repository, item["path"]) for item in auth["source_identities"]]
    inputs = [auth["inputs"][name]["path"] for name in ("cohort_rgb", "cohort_valid", "cohort_arrays", "cidt_clean")]
    fit_files = list(boot["fit_paths"].values()) if role == "worker" else []
    aggregate = [path for value in boot["aggregate_paths"].values() for path in ([value["path"]] + [item["path"] for item in value["files"]])] if role == "finalizer" else []
    targets = [value["path"] for value in boot["target_paths"].values()] if role == "finalizer" else []
    identities = {spec["path"]: spec for spec in auth["inputs"].values()}; identities.update({os.path.join(repository, item["path"]): item for item in auth["source_identities"]})
    if role == "finalizer": identities.update({item["path"]: item for value in boot["aggregate_paths"].values() for item in [value, *value["files"]]}); identities.update({item["path"]: item for item in boot["target_paths"].values()})
    write_roots = (boot["output_root"], auth["outputs"]["receipt_path"]) if role == "finalizer" else (boot["output_root"],)
    policy = AccessPolicy(role, read_files=sources + inputs + fit_files + aggregate + targets,
                          read_roots=boot["site_paths"] + boot["runtime_read_roots"], write_roots=write_roots,
                          read_identities=identities, normalization_root=boot["output_root"],
                          fold=boot["outer_fold"] if role == "worker" else None)
    _set_policy(policy)
    for item in auth["source_identities"]: _verify_file_spec({**item, "path": os.path.join(repository, item["path"])}, item["name"])
    return policy


def _fold_worker_entry(channel, bootstrap):
    observation, started = [False], time.monotonic()
    try:
        boot, auth = _child_handshake(channel, bootstrap, "fold_worker"); policy = _child_policy(auth, boot, "worker")
        local = json.loads(json.dumps(auth)); local["inputs"]["cohort_arrays"]["members"] = {name: auth["inputs"]["cohort_arrays"]["members"][name] for name in ("sample_indices", "keeper_probabilities", "model_boxes")}
        local["inputs"]["cidt_clean"]["members"] = {"sample_indices": auth["inputs"]["cidt_clean"]["members"]["sample_indices"]}
        _activate_paths(auth["repository_root"], boot["site_paths"]); projection = _fold_projection(auth["fold_projection"])
        output = _mkdir_once(boot["output_root"]); ctx = _structural_context(local, projection)
        target_record = {"schema": "psdf_target_blind/v1", "target_blind_root_sha256": ctx["target_blind_root"]}
        _create_once_bytes(os.path.join(output, "target_blind.json"), _canonical_bytes(target_record))
        channel.send({"kind": "target_blind", "run_name": boot["run_name"], "outer_fold": boot["outer_fold"],
                      "target_blind_root_sha256": ctx["target_blind_root"]})
        if not channel.poll(300): raise TimeoutError("Fit-target release did not arrive")
        release = _strict_keys(channel.recv(), ("kind", "run_name", "outer_fold", "target_blind_root_sha256", "fit_ids", "fit_targets"), "fit release")
        if release["kind"] != "fit_release" or release["run_name"] != boot["run_name"] or release["outer_fold"] != boot["outer_fold"] or release["target_blind_root_sha256"] != ctx["target_blind_root"]:
            raise PermissionError("Fit-target stage binding differs")
        records = []
        for name in ("fit_ids", "fit_targets"):
            record = release[name]; path = os.path.abspath(record["path"])
            if path != os.path.abspath(boot["fit_paths"][name]): raise PermissionError("Fit shard path differs")
            policy.read_identities[os.path.normcase(path)] = dict(record)
            observed, size = _file_sha(path)
            if observed != record["sha256"] or size != record["bytes"]: raise ValueError("Fit shard bytes differ")
            records.append(_load_array(path, record["array"]))
        fit_ids, fit_targets = records; fold = int(boot["outer_fold"])
        expected = ctx["indices"][(ctx["folds"] != fold) & (ctx["folds"] != ((fold + 1) % 5))]
        if not ctx["np"].array_equal(fit_ids, expected) or fit_targets.shape != fit_ids.shape or not ctx["np"].isin(fit_targets, ctx["np"].arange(5)).all(): raise ValueError("Fit shard role differs")
        ctx["torch"].cuda.reset_peak_memory_stats(); preimages, traces, training, causal = [], [], [], None
        for role in ROLES:
            model, record = _train_role(ctx, role, fold, fit_ids, fit_targets, output, channel, observation)
            preimages.append({"role": role, "outer_fold": fold, "calibration_sample_indices": record["calibration_ids"],
                              "calibration_raw_scores": record["calibration_scores"], "held_sample_indices": record["held_ids"], "held_raw_scores": record["held_scores"]})
            traces.append({name: record[name] for name in ("losses", "gradient_sha256", "optimizer_state_sha256", "initialized_model_state_sha256", "final_model_state_sha256")})
            training.append({name: record[name] for name in ("recalibration", "state_records")})
            if role == "ddf_full": causal = _causal_fold(ctx, model, fold)
            del model; ctx["torch"].cuda.empty_cache()
        structural = {"sample_indices": ctx["indices"], "held_folds": ctx["folds"], "component_ids": list(ctx["components"]), "component_order": list(ctx["component_order"]),
                      "bootstrap_draws": ctx["bootstrap"], "keeper_probabilities": ctx["keeper"], "valid64": ctx["prepared"].valid64[:, 0].numpy(),
                      "valid32": ctx["prepared"].valid32[:, 0].numpy(), "valid16": ctx["prepared"].valid16[:, 0].numpy(), "bbox_valid16": ctx["bbox"],
                      "nonwrap_offsets": ctx["offsets"], "cidt_sample_indices": ctx["cidt_indices"], "donor_artifact_sha256": _json_sha(ctx["donor_rows"]),
                      "prepared_input_sha256": ctx["prepared_hashes"], "invalid_fill_noise": ctx["noise_identity"], "invalid_fill_seed": 20260729,
                      "donor_preimages": ctx["donor_rows"], "initialization_orders": ctx["initialization_orders"]}
        fold_raw = {"structural": structural, "preimages": preimages, "causal": causal, "traces": traces, "training_records": training}
        raw_root = _raw_root(fold_raw); tree = _persist_tree(output, fold_raw)
        manifest_path = os.path.join(output, "fold_manifest.json"); _create_once_bytes(manifest_path, _canonical_bytes({"schema": "psdf_fold_raw/v1", "raw_fold_root_sha256": raw_root, "tree": tree}))
        reply = {"kind": "raw_fold_complete", "run_name": boot["run_name"], "outer_fold": fold, "target_blind_root_sha256": ctx["target_blind_root"],
                 "raw_fold_root_sha256": raw_root, "manifest_path": manifest_path, "access_root_sha256": policy.root(), "wall_seconds": time.monotonic() - started,
                 "peak_rss_bytes": _rss_bytes(), "peak_cuda_bytes": int(ctx["torch"].cuda.max_memory_allocated())}
        if set(reply) != set(RAW_WORKER_REPLY_KEYS): raise RuntimeError("Worker reply schema differs")
        channel.send(reply)
    except BaseException as error:
        try: channel.send({"kind": "worker_error", "observed": observation[0], "error": type(error).__name__, "traceback": traceback.format_exc()})
        except BaseException: pass
        raise


def _target_shards(auth, projection, root):
    import numpy as np
    targets = _npz_member(auth["inputs"]["cohort_arrays"], "targets").astype("<i8", copy=False)
    indices = np.asarray(projection["sample_indices"], dtype="<i8"); folds = np.asarray(projection["held_folds"], dtype="<i8")
    if targets.shape != (763,) or not np.isin(targets, np.arange(5)).all(): raise ValueError("Cohort target member differs")
    _mkdir_once(root); output = []
    for fold in range(5):
        roles = {}
        for role, mask in (("fit", (folds != fold) & (folds != ((fold + 1) % 5))), ("calibration", folds == ((fold + 1) % 5)), ("held", folds == fold)):
            roles[role] = {}
            for name, value in (("ids", indices[mask]), ("targets", targets[mask])):
                path = os.path.join(root, f"fold{fold}_{role}_{name}.npy"); record = _save_array(path, np.ascontiguousarray(value)); record["path"] = path; roles[role][name] = record
        output.append(roles)
    return output


def _close_aggregate(run_root, replies):
    folds = []
    for reply in replies:
        payload = open(reply["manifest_path"], "rb").read(); manifest = _strict_keys(json.loads(payload), ("schema", "raw_fold_root_sha256", "tree"), "fold manifest")
        if manifest["schema"] != "psdf_fold_raw/v1" or manifest["raw_fold_root_sha256"] != reply["raw_fold_root_sha256"]: raise ValueError("Fold manifest binding differs")
        value = _restore_tree(os.path.dirname(reply["manifest_path"]), manifest["tree"])
        if _raw_root(value) != reply["raw_fold_root_sha256"]: raise ValueError("Fold semantic root differs")
        for training in value["training_records"]:
            for state in training["state_records"].values():
                state_path = os.path.join(os.path.dirname(reply["manifest_path"]), state["path"]); _read_exact(state_path, state, "retained model state"); _load_array(state_path, state["array"])
        folds.append(value)
    raw = _assemble_raw(folds); raw_root = _raw_root(raw); aggregate = _mkdir_once(os.path.join(run_root, "aggregate")); tree = _persist_tree(aggregate, raw)
    path = os.path.join(aggregate, "raw_manifest.json"); body = _canonical_bytes({"schema": "psdf_raw_aggregate/v1", "raw_root_sha256": raw_root, "tree": tree})
    files = []
    def collect(wire):
        if isinstance(wire, dict) and set(wire) == {"__npy__"}: files.append({**wire["__npy__"], "path": os.path.join(aggregate, wire["__npy__"]["path"])}); return
        for value in (wire.values() if isinstance(wire, dict) else wire if isinstance(wire, list) else ()): collect(value)
    collect(tree); record = {"path": path, "bytes": len(body), "sha256": _create_once_bytes(path, body), "files": files, "raw_root_sha256": raw_root,
              "access_root_sha256": _json_sha([reply["access_root_sha256"] for reply in replies]),
              "target_blind_root_sha256": replies[0]["target_blind_root_sha256"],
              "fold_artifact_root_sha256": _json_sha([reply["raw_fold_root_sha256"] for reply in replies])}
    return raw, record


def _spawn_one(policy, target, boot, ticket, handler, deadline):
    global _SPAWN_TICKET
    _scrub_to_stdlib(); context = multiprocessing.get_context("spawn"); parent, child = context.Pipe(); process = context.Process(target=target, args=(child, boot))
    _SPAWN_TICKET = ticket; process.start(); child.close(); observed = False
    try:
        while time.monotonic() < deadline:
            if parent.poll(1):
                message = parent.recv(); kind = message.get("kind") if isinstance(message, dict) else None
                if kind == "child_ready":
                    if message != {"kind": "child_ready", "nonce": boot["nonce"], "pid": process.pid}: raise PermissionError("Child-ready binding differs")
                    parent.send({"kind": "parent_authorized", "nonce": boot["nonce"], "authorization_sha256": boot["authorization_sha256"]})
                elif kind == "target_blind": parent.send(handler(message))
                elif kind == "observation_started": observed = True
                elif kind in ("raw_fold_complete", "scientific_complete"):
                    process.join(10)
                    if process.is_alive() or process.exitcode != 0: raise RuntimeError("Child did not exit cleanly after its receipt")
                    return message
                elif kind in ("worker_error", "finalizer_error"): raise RuntimeError(f"child failure after_observation={message.get('observed', observed)}: {message.get('error')}")
                else: raise RuntimeError("Unauthorized child message")
            if not process.is_alive() and not parent.poll(): raise RuntimeError(f"Child exited without receipt: {process.exitcode}")
        raise TimeoutError("Authorized child exceeded its wall ceiling")
    finally:
        _SPAWN_TICKET = None
        if process.is_alive(): process.terminate(); process.join(10)
        parent.close()


def _scientific_finalizer_entry(channel, bootstrap):
    started = time.monotonic()
    try:
        boot, auth = _child_handshake(channel, bootstrap, "scientific_finalizer"); policy = _child_policy(auth, boot, "finalizer"); _activate_paths(auth["repository_root"], boot["site_paths"])
        import numpy as np
        from trkh.tools.pair_surface_psdf_ig1_scientific import evaluate_ig1_pair
        raw, records = {}, boot["aggregate_paths"]
        for run in ("primary", "replay"):
            record = records[run]; body = _read_exact(record["path"], record, run + " aggregate"); manifest = _strict_keys(json.loads(body), ("schema", "raw_root_sha256", "tree"), "aggregate manifest")
            if manifest["schema"] != "psdf_raw_aggregate/v1" or manifest["raw_root_sha256"] != record["raw_root_sha256"]: raise ValueError("Aggregate manifest binding differs")
            raw[run] = _restore_tree(os.path.dirname(record["path"]), manifest["tree"])
            if _raw_root(raw[run]) != record["raw_root_sha256"]: raise ValueError("Aggregate raw root differs")
        def read_record(record):
            body = _read_exact(record["path"], record, "target shard"); value = _load_array(record["path"], record["array"])
            if len(body) != record["bytes"]: raise ValueError("Target shard bytes differ")
            return value
        paths = boot["target_paths"]; shards, calibration, union_targets = [], [], np.full(763, -1, dtype="<i8")
        for fold in range(5):
            shard = {"outer_fold": fold}
            for role in ("fit", "calibration", "held"):
                shard[role] = {"sample_indices": paths[f"fold{fold}_{role}_ids"]["array"], "targets": paths[f"fold{fold}_{role}_targets"]["array"]}
            shards.append(shard); ids = read_record(paths[f"fold{fold}_calibration_ids"]); truth = read_record(paths[f"fold{fold}_calibration_targets"])
            calibration.append({"outer_fold": fold, "sample_indices": ids, "targets": truth}); union_targets[np.searchsorted(raw["primary"]["sample_indices"], ids)] = truth
        if not np.isin(union_targets, np.arange(5)).all(): raise ValueError("Calibration target union differs")
        unusable = ~raw["primary"]["bbox_valid16"].any((1, 2)); unusable_ids = raw["primary"]["sample_indices"][unusable]
        member = auth["inputs"]["cidt_clean"]["members"]; identity = lambda name: {key: member[name][key] for key in ("dtype", "shape", "array_sha256")}
        target_registry = _json_sha(shards); cidt_registry = _json_sha({"sample_indices": _array_identity(raw["primary"]["cidt_sample_indices"]), "targets": identity("targets"), "baseline_predictions": identity("baseline_predictions")})
        lock = {"schema": "trkh_psdf_ig1_lineage_lock/v1", "protocol_id": PROTOCOL_ID, "repository_root": os.path.abspath(auth["repository_root"]), "authorization_sha256": boot["authorization_sha256"],
                "machine_lock_sha256": _hex64(auth["lineage"]["machine_lock_sha256"]), "environment_lock_sha256": _json_sha(auth["environment"]), "access_policy_sha256": _json_sha({"protocol": PROTOCOL_ID, "authorization": boot["authorization_sha256"]}),
                "target_blind_root_sha256": records["primary"]["target_blind_root_sha256"], "primary_raw_root_sha256": records["primary"]["raw_root_sha256"], "replay_raw_root_sha256": records["replay"]["raw_root_sha256"],
                "primary_access_root_sha256": records["primary"]["access_root_sha256"], "replay_access_root_sha256": records["replay"]["access_root_sha256"], "component_order_sha256": _json_sha(raw["primary"]["component_order"]),
                "sample_component_mapping_sha256": _json_sha([[int(index), component] for index, component in zip(raw["primary"]["sample_indices"], raw["primary"]["component_ids"])]),
                "bootstrap_draws": _array_identity(raw["primary"]["bootstrap_draws"]), "keeper_probabilities": _array_identity(raw["primary"]["keeper_probabilities"]),
                **{name: _array_identity(raw["primary"][name]) for name in ("valid64", "valid32", "valid16", "bbox_valid16", "nonwrap_offsets")},
                "donor_artifact_sha256": raw["primary"]["donor_artifact_sha256"], "sidecar_preimage_registry_sha256": _preimage_root(raw["primary"]["sidecar_preimages"]),
                "artifact_registry_sha256": _json_sha(raw["primary"]["artifact_identities"]), "bbox_unusable_sample_indices": _array_identity(unusable_ids),
                "bbox_unusable_records_sha256": _json_sha([{"sample_index": int(index), "target": int(target)} for index, target in zip(unusable_ids, union_targets[unusable])]),
                "target_shards": shards, "target_shard_registry_sha256": target_registry, "cidt_sample_indices": _array_identity(raw["primary"]["cidt_sample_indices"]),
                "cidt_targets": identity("targets"), "cidt_baseline_predictions": identity("baseline_predictions"), "cidt_registry_sha256": cidt_registry,
                "source_identities": auth["lineage"]["scientific_sources"]}
        lock["frozen_record_sha256"] = _raw_root(lock); wrappers = {run: {"target_blind_root_sha256": records[run]["target_blind_root_sha256"], "raw": raw[run], "calibration_target_slices": calibration} for run in ("primary", "replay")}
        def release_held():
            held = [{"outer_fold": fold, "sample_indices": read_record(paths[f"fold{fold}_held_ids"]), "targets": read_record(paths[f"fold{fold}_held_targets"])} for fold in range(5)]
            return {run: {"target_blind_root_sha256": records[run]["target_blind_root_sha256"], "raw": raw[run], "held_target_slices": held} for run in ("primary", "replay")}
        def release_cidt(): return {"cidt_targets": _npz_member(auth["inputs"]["cidt_clean"], "targets").astype("<i8", copy=False), "cidt_baseline_predictions": _npz_member(auth["inputs"]["cidt_clean"], "baseline_predictions").astype("<i8", copy=False)}
        output = _mkdir_once(boot["output_root"]); preheld = {run: os.path.join(output, run + "_preheld.bin") for run in ("primary", "replay")}
        result = evaluate_ig1_pair(lock=lock, calibration_runs=wrappers, release_held_runs=release_held, release_cidt=release_cidt, preheld_paths=preheld)
        decision = _canonical_bytes(result.decision_summary); decision_path = os.path.join(output, "decision.json"); result_root = _create_once_bytes(decision_path, decision)
        receipts = {item.run_name: item.sha256 for item in result.preheld_receipts}; receipt_body = _canonical_bytes({"protocol_id": PROTOCOL_ID, "outcome": result.outcome, "result_root_sha256": result_root,
                                                                                                                 "primary_preheld_sha256": receipts["primary"], "replay_preheld_sha256": receipts["replay"],
                                                                                                                 "primary_fold_artifact_root_sha256": records["primary"]["fold_artifact_root_sha256"], "replay_fold_artifact_root_sha256": records["replay"]["fold_artifact_root_sha256"]})
        _create_once_bytes(auth["outputs"]["receipt_path"], receipt_body); reply = {"kind": "scientific_complete", "outcome": result.outcome, "result_root_sha256": result_root, "receipt_path": auth["outputs"]["receipt_path"],
                                                                                   "primary_preheld_sha256": receipts["primary"], "replay_preheld_sha256": receipts["replay"], "access_root_sha256": policy.root(),
                                                                                   "wall_seconds": time.monotonic() - started, "peak_rss_bytes": _rss_bytes()}
        if set(reply) != set(FINALIZER_REPLY_KEYS): raise RuntimeError("Finalizer reply schema differs")
        channel.send(reply)
    except BaseException as error:
        try: channel.send({"kind": "finalizer_error", "observed": True, "error": type(error).__name__, "traceback": traceback.format_exc()})
        except BaseException: pass
        raise
