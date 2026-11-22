# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Acquire the exact public files used by the reference adapters.

Only explicit registry entries are downloaded. Pinned hashes fail closed when a
publisher changes an asset. No third-party code, pickle or archive entry executes.
Dataset licenses are separate from the package's software license.
"""

import io
import json
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .heroes import file_hash

REGISTRY = {
    "uci-dota2": {
        "url": "https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip",
        "source": "https://archive.ics.uci.edu/dataset/367/dota2+games+results",
        "license": "CC-BY-4.0",
        "filename": "uci-dota2.zip",
        "bytes": 1821241,
        "sha256": "5bf3d2fe75c55ffba506f4aaee0d88684aa8e651f622ff32a0ec7928ae476717",
        "members": {
            "dota2Train.csv": "extracted/uci-dota2/dota2Train.csv",
            "dota2Test.csv": "extracted/uci-dota2/dota2Test.csv",
        },
    },
    "neuralac-dota2018": {
        "url": "https://codeload.github.com/bigdata-ustc/NAC/zip/refs/heads/main",
        "source": "https://github.com/bigdata-ustc/NAC",
        "license": "Repository GPL-3.0; independent data license unconfirmed",
        "filename": "neuralac-main.zip",
        "bytes": 15895241,
        "sha256": "b9ad0cf52bcd015bdb90b623dce02813d6731985e2febc784d84cd3785df3d7a",
        "members": {"NAC-main/data/dota2018.csv": "extracted/NAC-main/data/dota2018.csv"},
    },
    "overwatch": {
        "url": "https://zenodo.org/records/10359600/files/overwatch.jsonl?download=1",
        "source": "https://zenodo.org/records/10359600",
        "license": "CC-BY-4.0",
        "filename": "overwatch.jsonl",
        "bytes": 63587768,
        "sha256": "ffefc2e77e5769e79f25951af70c04bdda56fa2c7934067a823a76a1400b5d1a",
    },
    "dbd-2022": {
        "url": "https://raw.githubusercontent.com/quantitative-ecologist/predator-foraging-mode-videogames/main/data/FraserFrancoetal2022-data.csv",
        "source": "https://github.com/quantitative-ecologist/predator-foraging-mode-videogames",
        "license": "CC-BY-NC-4.0",
        "filename": "dbd-2022.csv",
        "bytes": 13699886,
        "sha256": "cbd0195a5ce337bbc1a5e1696a1a7321c2ef55427c286234b905cd1c3647fff0",
    },
    "dbd-2023": {
        "url": "https://raw.githubusercontent.com/quantitative-ecologist/experience-hunting-tactics/main/data/FraserFrancoetal2023-data.csv",
        "source": "https://github.com/quantitative-ecologist/experience-hunting-tactics",
        "license": "CC-BY-NC-4.0",
        "filename": "dbd-2023.csv",
        "bytes": 15242394,
        "sha256": "c6dcff1195e1966f9f22c49a31ce816ed8f0ee9d1cc0ae63e023644602563456",
    },
    "dota2-matches-v2": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/devinanzelmo/dota-2-matches?datasetVersionNumber=2",
        "source": "https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches",
        "license": "CC-BY-SA-4.0",
        "filename": "dota2-matches-v2.zip",
        "bytes": 454666789,
        "sha256": "dbebbc6d67a5af7bf4616de03467e56bf79a966a268301873e0472e4cff47013",
        "members": {
            name: f"extracted/dota2-matches-v2/{name}"
            for name in ["match.csv", "players.csv", "match_outcomes.csv", "player_time.csv"]
        },
    },
}
REGISTRY["neuralac-lol"] = {
    **REGISTRY["neuralac-dota2018"],
    "nested_archive": "NAC-main/data/lol_75W.zip",
    "members": {"lol_75W.csv": "extracted/neuralac-lol/lol_75W.csv"},
}


def _within(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Data destination escapes the selected root")
    return target


def _verify(path, asset):
    if path.stat().st_size != asset["bytes"] or file_hash(path) != asset["sha256"]:
        raise ValueError(f"Pinned size/SHA-256 mismatch: {path}")


def _download(asset, target):
    if not asset["url"].startswith("https://"):
        raise ValueError("Public acquisitions require HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(asset["url"], headers={"User-Agent": "GameMatch-Research/0.1"})
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with urlopen(request, timeout=60) as response, pending.open("wb") as stream:
            if response.status != 200 or not response.geturl().startswith("https://"):
                raise ValueError("Expected successful HTTPS data response")
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Received a webpage instead of the pinned data file")
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > asset["bytes"]:
                    raise ValueError("Response exceeds pinned asset size")
                stream.write(chunk)
        _verify(pending, asset)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def _extract_member(archive, member, destination):
    info = archive.getinfo(member)
    if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError("Selected archive entry is not a regular data file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with archive.open(info) as source, pending.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        # ZipFile validates the selected member's CRC while reading to EOF.
        digest = file_hash(pending)
        if destination.exists():
            if file_hash(destination) != digest:
                raise ValueError(f"Existing extracted file differs: {destination}")
        else:
            pending.replace(destination)
        return {"sha256": digest, "bytes": destination.stat().st_size}
    finally:
        pending.unlink(missing_ok=True)


def fetch_dataset(name, data_root):
    """Fetch or verify one pinned public dataset and write its provenance receipt.

    Args:
        name: A dataset key in REGISTRY.
        data_root: Base directory for raw files, extracted inputs and receipts.

    Returns:
        A dictionary with the source, license, hashes, cache status and
        extracted members, also saved as acquisition/<name>.json.

    Raises:
        ValueError: The key is unknown, a destination escapes data_root, or
            the size or hash differs from the registered identity.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    root = Path(data_root).resolve()
    asset = REGISTRY[name]
    target = _within(root, f"raw/{asset['filename']}")
    receipt = _within(root, f"acquisition/{name}.json")
    cached = target.exists()
    if cached:
        _verify(target, asset)
    else:
        _download(asset, target)
    extracted = {}
    if asset.get("members"):
        with ZipFile(target) as outer:
            nested = asset.get("nested_archive")
            archive = ZipFile(io.BytesIO(outer.read(nested))) if nested else outer
            try:
                for member, relative in asset["members"].items():
                    extracted[relative] = _extract_member(archive, member, _within(root, relative))
            finally:
                if nested:
                    archive.close()
    record = {
        "dataset": name,
        "status": "verified",
        "cached_download": cached,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "source": asset["source"],
        "url": asset["url"],
        "data_license": asset["license"],
        "raw_file": str(target.relative_to(root)),
        "raw_sha256": asset["sha256"],
        "raw_bytes": asset["bytes"],
        "extracted": extracted,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=receipt.parent, delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        pending.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        pending.replace(receipt)
    finally:
        pending.unlink(missing_ok=True)
    return record
