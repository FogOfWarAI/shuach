# Contributing

Open a GitHub issue for reproducible bugs or a pull request for code and documentation
changes. Include the package version so reviewers can distinguish your installation
from frozen experiment sources. For sensitive
reports, use the private route in the [security policy](SECURITY.md). Project
discussions follow the [code of conduct](CODE_OF_CONDUCT.md).

For documentation changes, start with the [writing guide](docs/writing-guide.md). It uses a real predictor to show how to describe arguments, array shapes, output units, failure cases and an example a reader can verify.

## Set up and test a change

Start feature branches from `develop` and open pull requests back to `develop`.
Promote a tested commit from `develop` to `release` by pull request. After
release validation, merge `release` into `main`. Use a merge commit when
promoting between long-lived branches to preserve their shared history.

To publish packages on GitHub Releases, tag the chosen commit on `release`
as `v<package version>` and push that tag. CI tests the tag on Linux and
macOS with Python 3.10 and 3.12. The publish job checks release ancestry,
the exact version and the artifact hashes before attaching the inspected
wheel, source archive and inspection report. Development and release-candidate
versions are marked as prereleases.

From the directory containing this version's `pyproject.toml`, activate a
Python 3.10+ virtual environment and run:

```bash
python -m pip install -e '.[tabular,rating,dev]'
gamematch --version
gamematch doctor
ruff check src tests scripts examples
python -m coverage run --source=gamematch -m unittest discover -s tests
python -m coverage report --fail-under=85
python examples/constrained_assignment.py
```

Tests use generated fixtures and CPU execution. For a focused test, use
`python -m unittest discover -s tests -p 'test_constraints.py'`.
Describe the user-visible change and its inputs, add a regression case where
the old behavior was wrong, and update the relevant API reference or example.
Record changes to saved artifacts or public behavior in the [change log](CHANGELOG.md).
The test suite also runs the Python examples in the introductory API, matching
and getting-started guides. Keep those examples self-contained and data-free.

## Write a reproducible bug report

Linux and macOS are the full-suite CI targets. On Windows, the synthetic smoke
benchmark works, but the full suite currently encounters Windows-specific
symbolic-link permissions and file-locking errors in indexed-data and atomic-write tests.

Include the output of `gamematch --version` and `gamematch doctor`, the exact
command or short Python example, the expected result and the observed result.
For research runs, include the protocol, model, seed, array shapes and the
first relevant traceback. Replace private data with generated values that
preserve the behavior. For a documentation issue, identify the page and the
first step whose result differs.

## Extend a research contract

For a data adapter, document entity identity, side/role meaning, class ordering, feature and target availability, split policy, source/license, exclusions and unsupported claims. Include a small generated fixture that exercises delayed or boundary feedback. Do not embed downloaded records, account identifiers or private paths in the software package.

For a model, specify tensor shapes and output semantics, cited equations, deviations from author code and the common input information set. Test masking, permutation behavior where intended, finite gradients and checkpoint reload. Public-data adaptation should remain labeled as such. Do not infer original-paper effectiveness from a different task or cohort.

For a solver, keep model scores separate from an independently available evaluator. State the feasible set, objective, tie behavior, side-exchangeability assumption, and whether the search is exact. A model's predicted improvement alone is not an intervention result.

Preserve source/config/data fingerprints for research runs. Changes that invalidate saved outputs should use a new output directory and an explicit changelog entry. Report failed or inconclusive experiments as well as favorable ones. Project-authored source follows Apache-2.0; cite third-party code and keep data licenses distinct.
