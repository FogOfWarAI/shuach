# Local release readiness

This directory contains the `0.1.0rc5.dev3` development version. The package
version comes from `gamematch.__version__`. Use the following commands from the
project directory or unpacked source archive to build and inspect a new release.
The build frontend creates the source archive first and builds the wheel from it.

```bash
python -m pip install -e '.[dev]'
python -m build --outdir dist
python scripts/inspect_distribution.py --dist-dir dist --output reports/package-inspection.json
```

The inspector records SHA-256 digests for both archives, compares packaged Python
modules with the source tree, checks license metadata, and resolves relative
Markdown file links inside the source archive. The wheel installs the library
and the `gamematch` console entry point. The source archive also carries the
user guides, examples, benchmark configurations, contributor policies and tests.

For a consumer check, create a second virtual environment and install the
generated wheel. Run `python -m pip check`, `gamematch doctor` and
`gamematch lobby-demo --seed 0` outside the project directory. The
`optional_dependencies` section of the doctor output reports whether the optional
benchmarks can import their dependencies. The synthetic lobby demo requires
only the base installation.

On November 21, **dev3** passed all 220 tests on macOS ARM64 using Python 3.12.7 and CPU PyTorch 2.12.0. Statement coverage was 3,778 of 4,098. Ruff passed for the library, tests, scripts and examples. The suite executes the Python examples in the English and Chinese API and quickstart guides and the team assignment guide.

A separate virtual environment used the installed wheel with CPU PyTorch 2.15.0. The tests from the source archive also passed there. The synthetic quickstart produced the two model results and 18 assignment results documented in the README.

The UCI pilot trained the two added roster models on 1,500 rows. Reloading either checkpoint reproduced all 250 held-out predictions exactly in this environment. Overwatch replay processed 58,211 eligible events. Reloading each of the four validation calibrators reproduced the three probabilities on each of 8,732 test events to within `2e-15`. The [worked examples](model-additions.md) give the inputs, commands and output order.

The November 20 **dev2** source check passed all 218 CPU tests on macOS ARM64 with
Python 3.12.7 and PyTorch 2.12.0. Statement coverage was 3,741 of 4,061
(92.12%). `ruff check src tests scripts examples` passed.

The local dev2 installation check used macOS ARM64, Python 3.12.7 and PyTorch
2.15.0. Two virtual environments with system site packages disabled resolved
dependencies from cached wheels. One installed the wheel and the other built
directly from the source archive. Both passed `pip check` and ran the installed
console script outside the checkout. A separate base environment exercised the
CPU quickstart without LightGBM, TrueSkill or OpenSkill. The synthetic example
completed two fitted models and produced all 18 expected decision rows.

## Earlier validation

The November 18 development version (`0.1.0rc5.dev1`) passed 208 local CPU tests
with 92.03% core statement coverage. Its macOS and local Ubuntu ARM64 guest
results are recorded under `reports/software-quality-20261118/` in the parent
research workspace. The following table describes the earlier rc4 baseline.

GameMatch is a local research artifact. Public publication and submission have not occurred. This page records completed engineering evidence and remaining release work.

The [JMLR MLOSS guidance](https://www.jmlr.org/mloss/mloss-info.html), checked on 14 November 2026, calls for a short software description, a public repository, clear licensing and community evidence in the cover letter. Its review guidance considers documentation, contribution pathways, comparative software measurements, testing, platform support and maintainability. These are review criteria, not a promise that completing a checklist produces acceptance.

Area | Local evidence | Remaining work
--- | --- | ---
Task scope | Seven reference public-data contracts, an optional LoL individual-performance history extension and a separate synthetic decision evaluator | Queue/party observations, verified native identity and longer temporal confirmation
Reproduction boundaries | NeuralAC function/gradient equivalence; explicit OwO/DAWN/MMR-Net adaptations | Original industrial datasets are unavailable; original result tables are not reproduced
Source and provenance | Frozen source, exact configuration, data hashes, split IDs, predictions and checkpoints | A versioned public repository and reviewed release archive
Acquisition | Seven fixed-hash acquisition entries; fresh UCI download and cached-input verification | Source maintainers can change URLs or licensing; review pinned revisions when updating
Installation | rc4 wheel/sdist; 176 tests pass in four macOS dependency environments and an Ubuntu 24.04 ARM64 guest on the same Mac; direct sdist installation also tested on macOS | Linux x86_64, CUDA execution and an external-user installation
Documentation | English API, data contracts, model cards, reproduction guide and executable examples | Public documentation hosting and usability feedback
Numerical correctness | Semantic/integration tests, independent metrics and checkpoint reloads | Maintain regression coverage as features and dependency versions change
Comparison methodology | Group/temporal protocols, fixed references and separately frozen validation searches, complete-seed gates, paired cluster sensitivities | Broader tuning budgets and new untouched confirmation datasets
Software performance | Matched local implementation and search measurements are reported separately when complete | Multiple hosts and production workloads; no universal speed ranking
Licensing | Apache-2.0 authored source, source headers, dependency attribution; raw data excluded | Confirm ownership and declarations before public release
Maintenance | Contribution instructions and a CPU CI matrix are prepared; local Linux guest additionally passes CLI, dependency checks and assignment examples | Execute hosted CI, establish issue triage and maintain releases
Community impact | No independent adoption evidence yet | Recruit external users, record reproducible use cases and address their issues
Authorship | Draft placeholders are explicit | Confirm authors, affiliations, funding and competing-interest declarations

The Linux check uses Ubuntu 24.04.5, Python 3.12.3 and CPU PyTorch 2.12.0 in a temporary ARM64 virtual machine. Only the supplied package input and a dedicated output directory are shared with the host. Each environment creates its own test fixtures; these runs establish installed-package behavior rather than portability of every historical checkpoint or repetition of the full empirical study. The explicit HPO `--device cpu` option is tested; omitting it retains the older mixed CPU/MPS research plan.

The practical submission sequence is: publish a reviewed version with accessible documentation; execute hosted Linux/macOS CI; obtain at least one independent reproduction; improve the package from actual user feedback; then finalize the four-page description and cover letter using evidence available at submission. No user count, adoption claim or hosted-CI result should be invented to close these gaps.
