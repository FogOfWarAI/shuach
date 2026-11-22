# SPDX-License-Identifier: Apache-2.0
"""Generate the API index and module pages from the source being deployed."""

import ast
import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "gamematch"
GROUPS = {
    "matching": ("Team assignment", "matching-api"),
    "models": ("Models and ratings", "model-cards"),
    "data": ("Data and features", "data-contracts"),
    "evaluation": ("Evaluation", "research-api"),
    "inference": ("Saved predictors", "inference-api"),
    "tuning": ("Hyperparameter search", "hyperparameter-search"),
    "research": ("Experiment runners and utilities", "research-api"),
}


def on_files(files, config):
    files.remove(files.get_file_from_path("reference/index.md"))
    sections = {key: [] for key in GROUPS}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.stem.startswith("_") or path.stem == "cli":
            continue
        parts = path.relative_to(PACKAGE).with_suffix("").parts
        module = ".".join(("gamematch", *parts))
        symbols = [
            node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_")
        ]
        if not symbols:
            continue
        group = parts[0] if parts[0] in GROUPS else "research"
        if path.stem == "indexed_inference":
            group = "inference"
        title, guide = GROUPS[group]
        slug = "-".join(parts)
        detail = [f"# `{module}`", f"[API index](index.md) · [{title} guide](../{guide}.md)"]
        entries = []
        for node in symbols:
            detail.append(f"::: {module}.{node.name}")
            names = [node.name]
            if isinstance(node, ast.ClassDef):
                names.extend(
                    f"{node.name}.{method.name}" for method in node.body
                    if isinstance(method, ast.FunctionDef) and not method.name.startswith("_")
                )
            for name in names:
                entries.append(f"- [`{name}`]({slug}.md#{module}.{name})")
        files.append(File.generated(config, f"reference/{slug}.md", content="\n\n".join(detail)))
        sections[group].append(
            f'<div class="api-module" data-name="{module}" markdown="1">\n\n'
            f'[`{module}`]({slug}.md)\n\n' + "\n".join(entries) + "\n\n</div>"
        )
    index = [
        "# Python API reference",
        "Search by symbol or module. Select an entry for its signature and implementation documentation.",
        '<label class="api-filter" for="api-query">Filter the API index'
        '<input id="api-query" type="search" placeholder="e.g. exact_balance or HeroPredictor"></label>'
        '<p id="api-count" aria-live="polite"></p>',
    ]
    for key, (title, guide) in GROUPS.items():
        index.extend([
            '<section class="api-group" markdown="1">', f"## {title}",
            f"[Usage and rules](../{guide}.md)", *sections[key], "</section>",
        ])
    files.append(File.generated(config, "reference/index.md", content="\n\n".join(index)))
    return files


def on_page_markdown(markdown, page, config, **kwargs):
    """Keep repository-only files reachable from guides deployed under /shuach/."""
    parent = posixpath.dirname(page.file.src_uri)

    def resolve(match):
        target = posixpath.normpath(posixpath.join(parent, match[1]))
        if target.startswith("../"):
            return f"]({config.repo_url}/blob/main/{target[3:]})"
        return match[0]

    return re.sub(r"\]\((\.\./[^)]+)\)", resolve, markdown)
