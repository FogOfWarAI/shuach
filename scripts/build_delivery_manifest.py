# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Inventory or verify a local research delivery without changing its evidence.

Raw data are covered by the acquisition/adapter provenance reports. This
inventory covers completed run artifacts, current source, documentation,
analysis, manuscripts, release archives and reviewed PDF outputs. It is an
integrity aid, not an authenticity signature or a public release.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--tuning", action="store_true", help="Require and include both completed tuning stages"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / (
        "reports/delivery-manifest-tuning.json" if args.tuning else "reports/delivery-manifest.json"
    )
    if args.verify:
        payload = json.loads(target.read_text())
        failures = []
        for record in payload["files"]:
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root):
                failures.append({"path": record["path"], "reason": "outside workspace"})
                continue
            if not path.is_file():
                failures.append({"path": record["path"], "reason": "missing"})
            elif path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                failures.append({"path": record["path"], "reason": "content changed"})
        if failures:
            raise SystemExit(json.dumps(failures, indent=2))
        print(f"Verified all {len(payload['files'])} inventoried files")
        return

    status = json.loads((root / "design/runtime-status.json").read_text())
    run_names = list(status["runs"]) + ["hero-calibration-v1"]
    for name in run_names:
        manifest = json.loads((root / "runs" / name / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"Run is not completed: {name}")
    tuning_runs = []
    if args.tuning:
        for path in [
            "reports/hyperparameter-search/results.json",
            "reports/specialist-search/results.json",
        ]:
            if not (root / path).exists() or not json.loads((root / path).read_text())["complete"]:
                raise ValueError(f"Tuning result is incomplete: {path}")
        for path in [
            "reports/tuning-quality/quality-verification.json",
            "reports/tuning-data-integrity.json",
            "reports/tuned-predictor-reload-audit.json",
        ]:
            if json.loads((root / path).read_text())["status"] != "passed":
                raise ValueError(f"Tuning audit is incomplete: {path}")
        from gamematch.runner import code_hash

        reload_audit = json.loads((root / "reports/tuned-predictor-reload-audit.json").read_text())
        if reload_audit["final_fits_checked"] != 211 or reload_audit["source_sha256"] != code_hash():
            raise ValueError("Refresh CPU reload audit for all 211 final fits and current source")
        data_audit = json.loads((root / "reports/tuning-data-integrity.json").read_text())
        if data_audit["manifests_checked"] != 34:
            raise ValueError("Refresh input audit for all 34 declared data manifests")
        tuning_runs = ["hpo-v1", "hpo-v1-baselines", "specialist-hpo-v2"]
    required = [
        "output/pdf/gamematch-software-paper.pdf",
        "output/pdf/gamematch-technical-report.pdf",
        "reports/pdf-quality-review.json",
        "reports/eight-hour-review.md",
        "reports/quality-verification.json",
        "reports/package-inspection.json",
        "reports/final-model-metric-audit.json",
    ]
    for path in required:
        if not (root / path).is_file():
            raise ValueError(f"Missing delivery artifact: {path}")
    folders = [
        "src",
        "tests",
        "docs",
        "benchmarks",
        "examples",
        "scripts",
        "design",
        "reports",
        "datasets/reports",
        "datasets/metadata",
        "reading/software",
        "dist",
        "output/pdf",
        "manuscript/tables",
        "manuscript/figures",
        *["runs/" + name for name in run_names],
        *["runs/" + name for name in tuning_runs],
    ]
    paths = {path for folder in folders for path in (root / folder).rglob("*") if path.is_file()}
    paths.update(root.glob("*.md"))
    for pattern in ("*.md", "*.py", "*.json"):
        paths.update((root / "datasets").glob(pattern))
    paths.update(
        root / path
        for path in [
            "pyproject.toml",
            "MANIFEST.in",
            "LICENSE",
            "NOTICE",
            "requirements-local-lock.txt",
            "papers.psv",
            "cache-status.csv",
            "reading/reading-notes.md",
            "reading/text-manifest.json",
            "reading/cache-audit-20261114.json",
            "reading/version-audit-20261114.json",
            "reading/late-source-audit-20261114.json",
            "reading/fifth-reading-audit-20261114.json",
            "reading/sixth-reading-audit-20261114.json",
            "reading/seventh-reading-audit-20261114.json",
            "reading/eighth-reading-audit-20261114.json",
            "reading/evidence-cards.json",
            "reading/build_evidence_cards.py",
            "reading/nrt-nct-implementation-notes.json",
            "reading/README.md",
            "manuscript/main.tex",
            "manuscript/technical-report.tex",
            "manuscript/tuning-followup.tex",
            "manuscript/references.bib",
            "manuscript/jmlr2e.sty",
            "manuscript/README.md",
            ".github/workflows/tests.yml",
        ]
    )
    paths = sorted(
        path
        for path in paths
        if path != target
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".tmp"}
        and path.name != ".DS_Store"
    )
    records = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": __doc__.strip(),
        "run_directories": run_names + tuning_runs,
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "verification": ".venv/bin/python scripts/build_delivery_manifest.py --verify"
        + (" --tuning" if args.tuning else ""),
        "files": records,
    }
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Inventoried {len(records)} files, {payload['total_bytes']} bytes")


if __name__ == "__main__":
    main()
