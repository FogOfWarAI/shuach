# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Inspect local release archives without publishing or extracting executable code."""

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from gamematch import __version__

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def inspect(path):
    """Check a local release archive against the current source tree.

    Args:
        path: Path to a GameMatch wheel or gzipped source distribution.

    Returns:
        Archive size, digest, license checks and counts of verified files and links.

    Raises:
        ValueError: A package contract differs from the source or release metadata.
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            files = {
                name: archive.read(name) for name in archive.namelist() if not name.endswith("/")
            }
    else:
        with tarfile.open(path) as archive:
            if any(member.issym() or member.islnk() for member in archive):
                raise ValueError("Unexpected link in source distribution")
            files = {
                member.name.split("/", 1)[1]: archive.extractfile(member).read()
                for member in archive
                if member.isfile()
            }
    forbidden = []
    for name in files:
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).is_absolute() or ".." in parts:
            raise ValueError(f"Escaping archive path: {name}")
        if any(
            part
            in {
                ".git",
                ".cache",
                "__pycache__",
                ".DS_Store",
                "datasets",
                "runs",
                "pdf",
            }
            or part.startswith("._")
            for part in parts
        ):
            forbidden.append(name)
        if name.endswith((".pyc", ".pt", ".npz", ".joblib")):
            forbidden.append(name)
    if forbidden:
        raise ValueError(f"Research/cache artifacts leaked into package: {forbidden}")
    for name, content in files.items():
        if name.endswith(".py") and b"SPDX-License-Identifier: Apache-2.0" not in content[:180]:
            raise ValueError(f"Missing source license header: {name}")
    metadata_name = (
        next(name for name in files if name.endswith(".dist-info/METADATA"))
        if path.suffix == ".whl"
        else "PKG-INFO"
    )
    metadata = BytesParser().parsebytes(files[metadata_name])
    if metadata["Name"] != "gamematch-research" or metadata["Version"] != __version__:
        raise ValueError("Package name/version mismatch")
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError("Unexpected package licensing metadata")
    required = {"LICENSE", "NOTICE", "THIRD_PARTY.md"}
    if set(metadata.get_all("License-File", [])) != required:
        raise ValueError("License file declarations differ from the project attributions")
    license_root = (
        PurePosixPath(metadata_name).parent / "licenses" if path.suffix == ".whl" else PurePosixPath()
    )
    for name in sorted(required):
        if files.get((license_root / name).as_posix()) != (SOURCE_ROOT.parent / name).read_bytes():
            raise ValueError(f"Missing or stale licensing/attribution file: {name}")
    source_files = []
    package = SOURCE_ROOT / "gamematch"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Missing source tree for archive comparison: {package}")
    for source in sorted(package.rglob("*.py")):
        relative = source.relative_to(SOURCE_ROOT).as_posix()
        packaged = relative if path.suffix == ".whl" else "src/" + relative
        if files.get(packaged) != source.read_bytes():
            raise ValueError(f"Distribution contains stale or missing source: {source}")
        source_files.append(packaged)
    links = 0
    if path.suffix != ".whl":
        for name, content in files.items():
            if not name.endswith(".md"):
                continue
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content.decode()):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#")[0]
                # Resolve archive-relative paths without touching the filesystem.
                parts = []
                for part in (PurePosixPath(name).parent / target).parts:
                    if part == "..":
                        if not parts:
                            raise ValueError(f"Escaping documentation link: {name}: {target}")
                        parts.pop()
                    elif part != ".":
                        parts.append(part)
                destination = "/".join(parts)
                if destination not in files:
                    raise ValueError(f"Broken packaged documentation link: {name}: {target}")
                links += 1
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "members": len(files),
        "forbidden_files": forbidden,
        "has_license": True,
        "has_notice": True,
        "has_third_party_notice": True,
        "source_files_matching_workspace": len(source_files),
        "checked_internal_documentation_links": links,
    }


def current_distributions(directory=Path("dist")):
    """Select the current release without deleting or inspecting older releases."""
    prefix = f"gamematch_research-{__version__}"
    wheels = sorted(Path(directory).glob(prefix + "-*.whl"))
    source = Path(directory) / (prefix + ".tar.gz")
    if len(wheels) != 1 or not source.is_file():
        raise ValueError(f"Expected one wheel and source archive for version {__version__}")
    return [wheels[0], source]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir", type=Path, default=Path("dist"),
        help="Directory containing the wheel and source archive",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/package-inspection.json"),
        help="Destination for the JSON inspection report",
    )
    args = parser.parse_args(argv)
    paths = current_distributions(args.dist_dir)
    result = {
        "status": "passed",
        "version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local archives only: expected source, source headers, licensing, metadata, packaged documentation links, and exclusion of research/cache/binary artifacts.",
        "artifacts": [inspect(path) for path in paths],
    }
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print("Inspected wheel and source archive successfully")


if __name__ == "__main__":
    main()
