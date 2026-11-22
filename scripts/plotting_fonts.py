# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Choose a font covering every label, or request an English-only fallback."""

import os
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font


def _font_candidates():
    explicit = os.environ.get("GAMEMATCH_PLOT_FONT")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ValueError(f"GAMEMATCH_PLOT_FONT is not a font file: {path}")
        yield path
    local = Path("/Library/Fonts/Arial Unicode.ttf")
    if local.is_file():
        yield local
    preferred = ("noto sans cjk", "source han sans", "pingfang", "microsoft yahei", "simhei")
    fonts = sorted(
        font_manager.fontManager.ttflist,
        key=lambda entry: (
            not any(name in entry.name.lower() for name in preferred),
            entry.name,
            entry.fname,
        ),
    )
    yield from (Path(entry.fname) for entry in fonts)


def configure_plot_font(chinese_labels):
    """Return True for fully supported Chinese labels, False for English labels.

    Callers must supply every Chinese/dynamic label and provide ASCII English
    alternatives. Checking actual glyph coverage avoids apparently successful
    plots that contain missing-glyph boxes. No fonts are installed or downloaded.
    """
    required = {ord(char) for text in chinese_labels for char in text if not char.isspace()}
    seen = set()
    family = "DejaVu Sans"
    chinese = False
    for path in _font_candidates():
        if path in seen:
            continue
        seen.add(path)
        try:
            available = FT2Font(str(path)).get_charmap()
        except (OSError, RuntimeError):
            continue
        if required.issubset(available):
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            chinese = True
            break
    matplotlib.rcParams.update(
        {"font.family": [family], "axes.unicode_minus": False, "svg.fonttype": "path"}
    )
    return chinese
