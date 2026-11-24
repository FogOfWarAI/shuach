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

# Keys refer to individual publications in the downloadable bibliography.
SYMBOL_PAPERS = {
    "models.heroes.SignedHero": "draftai",
    "models.heroes.FactorizationMachine": "fm",
    "models.heroes.DeepSets": "deepsets",
    "models.heroes.RosterSetTransformer": "settransformer",
    "models.heroes.MLP": "backprop",
    "models.heroes.make_model": "fm deepsets settransformer neuralac",
    "models.neuralac.NeuralAC": "neuralac",
    "models.history.HistoryDeepSets": "deepsets",
    "models.history.make_history_model": "deepsets cupid champ omninet",
    "models.outcomes.build_history_outcome_model": "deepsets cupid champ omninet",
    "models.omninet.OmniEncoder": "omninet transformer",
    "models.omninet.OwO": "cupid champ omninet",
    "models.omninet.MMRNet": "quickskill omninet",
    "models.ratings.TeamElo": "elo",
    "models.ratings.TrueSkillRating": "trueskill",
    "models.ratings.OpenSkillRating": "openskill weng2011",
    "models.skill.SnapshotBaseline": "gru transformer",
    "models.skill.make_skill_model": "gru transformer quickskill",
    "models.ordinal.OrdinalMLP": "ordinal",
    "models.regularization.ActivationDropoutSequential": "dropout",
    "models.tabular.train_tabular": "logistic lightgbm",
    "evaluation.calibration.fit_binary_calibration": "temperature",
    "evaluation.calibration.apply_binary_calibration": "temperature",
}


def read_papers():
    text = (ROOT / "docs" / "models.bib").read_text(encoding="utf-8")
    papers = {}
    for entry in re.split(r"(?m)(?=^@)", text.strip()):
        if entry.strip():
            header = re.match(r"@\w+\{([^,]+),", entry)
            if not header:
                raise ValueError(f"Invalid BibTeX header: {entry.splitlines()[0]}")
            key = header[1]
            if key in papers:
                raise ValueError(f"Duplicate BibTeX key: {key}")
            papers[key] = entry.strip()
    return papers


def bibtex_blocks(keys, papers):
    blocks = []
    for key in keys.split():
        entry = papers[key]
        doi = re.search(r"\bdoi\s*=\s*\{([^}]+)", entry)
        url = re.search(r"\burl\s*=\s*\{([^}]+)", entry)
        source = f"https://doi.org/{doi[1]}" if doi else (url[1] if url else "")
        link = f"[Source]({source})\n\n" if source else ""
        blocks.append(
            f'<details markdown="1"><summary>BibTeX · {key}</summary>\n\n'
            f"{link}```bibtex\n{entry}\n```\n\n</details>"
        )
    return "\n\n".join(blocks)


def on_files(files, config):
    papers = read_papers()
    pending_symbols = set(SYMBOL_PAPERS)
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
            symbol = ".".join((*parts, node.name))
            keys = SYMBOL_PAPERS.get(symbol)
            if keys:
                detail.append(bibtex_blocks(keys, papers))
                pending_symbols.remove(symbol)
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
    if pending_symbols:
        raise ValueError(f"BibTeX citations reference missing API symbols: {sorted(pending_symbols)}")
    index = [
        "# Python API reference",
        "Search by symbol or module. Select an entry for its signature and implementation documentation.",
        "[Model papers and BibTeX](../papers.md) · [Download the bibliography](../models.bib)",
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

    if "<!-- bib:" in markdown:
        papers = read_papers()
        markdown = re.sub(r"<!-- bib: ([\w ]+) -->", lambda m: bibtex_blocks(m[1], papers), markdown)
    return re.sub(r"\]\((\.\./[^)]+)\)", resolve, markdown)
