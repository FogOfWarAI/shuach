# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Atomic JSON writes and integrity records for completed native fits."""

import hashlib
import json
import tempfile
from pathlib import Path

from .data.heroes import file_hash


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fit_files(model):
    if model in {"prior", "history_frequency", "current", "mean_delta"}:
        extension = "json"
    elif model in {"logistic", "lightgbm", "ridge"}:
        extension = "joblib"
    else:
        extension = "pt"
    return ("metrics.json", "predictions.npz", f"checkpoint.{extension}")


def _artifact_hashes(directory, names, root):
    result = {}
    for name in names:
        path = directory / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping completed artifact: {path}")
        result[name] = file_hash(path)
    return result


def _run_context_hash(manifest):
    context = {
        key: value
        for key, value in manifest.items()
        if key not in {"status", "error", "updated_utc"}
    }
    return hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seal_artifacts(directory, names, root, marker="completion.json"):
    directory, root = Path(directory), Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    write_json(
        directory / marker,
        {
            "schema_version": 1,
            "run_fingerprint": manifest["fingerprint"],
            "run_context_sha256": _run_context_hash(manifest),
            "fit_path": directory.resolve().relative_to(root.resolve()).as_posix(),
            "artifacts": _artifact_hashes(directory, names, root),
        },
    )


def completed_fit(directory, names, root, marker="completion.json"):
    directory, root = Path(directory), Path(root)
    receipt = directory / marker
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("run_fingerprint") != manifest.get("fingerprint")
        or record.get("run_context_sha256") != _run_context_hash(manifest)
        or record.get("fit_path") != directory.resolve().relative_to(root.resolve()).as_posix()
        or not isinstance(record.get("artifacts"), dict)
        or not record["artifacts"]
        or set(record["artifacts"]) != set(names)
    ):
        raise ValueError(f"Invalid completion identity: {receipt}")
    if _artifact_hashes(directory, names, root) != record["artifacts"]:
        raise ValueError(f"Completed artifacts changed: {receipt}")
    return True


def verify_completions(root):
    """Check existing completions before overwriting shared preparation artifacts."""
    root = Path(root)
    if not root.exists():
        return
    for pattern in ("completion.json", "fit-completion.json"):
        for receipt in sorted(root.rglob(pattern)):
            record = json.loads(receipt.read_text())
            if not isinstance(record, dict):
                raise ValueError(f"Invalid completion identity: {receipt}")
            completed_fit(receipt.parent, record.get("artifacts", {}), root, receipt.name)
