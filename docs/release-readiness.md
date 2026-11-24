# Building and verifying a distribution

This page is a packaging procedure for this checkout. The version comes from
`gamematch.__version__`. Run the commands from the project root or from an
unpacked source archive; the build frontend creates the source archive first
and builds the wheel from it.

## Build and inspect

```bash
python -m pip install -e '.[neural,dev]'
python -m build --outdir dist
python scripts/inspect_distribution.py --dist-dir dist --output reports/package-inspection.json
```

The inspector records SHA-256 digests for both archives, compares packaged
Python modules with the source tree, checks license metadata and resolves
relative Markdown links inside the source archive. The wheel installs the
library and the `gamematch` console entry point. The source archive also
carries the user guides, examples, benchmark configurations, contributor
policies and tests.

## Verify an installed wheel

Use a second virtual environment and run these commands outside the project
directory:

```bash
python -m pip install --force-reinstall --no-deps --no-index dist/*.whl
python -m pip check
gamematch --version
gamematch doctor
gamematch lobby-demo --seed 0
gamematch feature-schema --profile hero-roster
```

`gamematch doctor` reports which optional extras are importable. PyTorch is
optional: a base install runs the catalog, feature contracts and the CPU
assignment examples, while training runners need `.[neural]`.

## Tests and static checks

```bash
python -m pip install -e '.[neural,tabular,rating,dev]'
ruff check src tests scripts examples
python -m coverage run --source=gamematch -m unittest discover -s tests -v
python -m coverage report --fail-under=85
```

The CI workflow runs the same checks on Ubuntu and macOS with Python 3.10 and
3.12, first with the base install and then with the neural extra. See
`CONTRIBUTING.md` for the contributor workflow and `docs/reproduction.md` for
auditing saved runs.
