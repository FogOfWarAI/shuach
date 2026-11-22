# Third-party boundaries

The package distributes project-authored Python modules. Dependencies are installed separately; their licenses are not replaced by the project license. Downloaded research/data caches are not bundled in the source or wheel distribution.

| Dependency | Local version | Declared license |
|---|---:|---|
| PyTorch | 2.12.0 | BSD-3-Clause; wheel also includes separately licensed native components |
| NumPy | 2.3.4 | BSD; bundled numerical libraries have additional notices |
| pandas | 2.3.3 | BSD-3-Clause |
| SciPy | 1.17.1 | BSD; bundled numerical libraries have additional notices |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| LightGBM | 4.7.0 | MIT |
| TrueSkill | 0.4.5 | BSD |
| OpenSkill | 6.2.0 | MIT |

These declarations were checked against installed package metadata. The local launcher can point LightGBM at PyTorch's existing LLVM OpenMP library; it does not copy that library into the distribution.

GameMatch implements its signed hero baseline and compact Set Transformer in project-authored PyTorch code. The rating adapters call the installed OpenSkill package. The model guide links the original paper, authors' code and the Dota draft project, and describes what GameMatch implements for each. [Sources and implementation scope](docs/model-additions.md#sources-and-scope-of-the-update)

Data remain governed by source-specific terms. UCI and Overwatch records list CC BY 4.0. The new Dota 2 Matches v2 release lists CC BY-SA 4.0 and requests attribution to YASP/OpenDota. DBD 2022/2023 sources use CC BY-NC 4.0. Some cached repositories or data have unresolved redistribution terms. The downloader is not a blanket authorization to redistribute data.

Cached NeuralAC source is GPL-3.0 and stays outside the package. The project implementation follows published mathematical operations using independently written vectorized code. OwO, DAWN and MMR-Net adaptations document departures from original industrial feature sets, dimensions and task protocols. Citations acknowledge methods; they do not imply endorsement or original-result reproduction.

Skillbench reference modules (MIT, pinned commit `4749a9a5b3d852a753ec82d320809bc2c82831f0`) are cached under `reading/software/skillbench-runtime` only for the explicit interface probe. Their original license is retained there. They are not copied into the GameMatch library or distributed in its wheel/source archive; `scripts/verify_skillbench_reference.py` is the separately authored probe harness.
