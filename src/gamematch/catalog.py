# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Model tasks, input contracts and executable adapters are separate metadata.

Compatibility describes the current implementation, not measured performance or
successful reproduction of a paper. No data are loaded by this module.
"""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ModelCard:
    """Describe a callable model ID, its targets, inputs and optional base model."""
    name: str
    label: str
    kind: str
    tasks: tuple[str, ...]
    input_schemas: tuple[str, ...]
    scope: str
    variant_of: str | None = None


@dataclass(frozen=True)
class InputProfile:
    """Describe a target, available input fields and the adapters that consume them."""
    name: str
    task: str
    input_schema: str
    target: str
    inputs: tuple[str, ...]
    datasets: tuple[str, ...]
    native_models: tuple[str, ...]
    external_models: tuple[str, ...] = ()
    external_entry: str | None = None
    limitation: str = ""
    feature_schema: str | None = None

    @property
    def executable_schema(self):
        """Return the runtime FeatureSchema behind the legacy string label.

        Unlike ``input_schema`` (a metadata string), this object drives tensor
        validation, canonical layout and encoding. Import is local so catalog
        stays usable when only the contract layer is installed.
        """
        if self.feature_schema is None:
            raise ValueError(f"Profile {self.name!r} has no executable feature schema")
        from .features import SCHEMAS

        try:
            return SCHEMAS[self.feature_schema]
        except KeyError as exc:
            raise ValueError(f"Unknown feature schema: {self.feature_schema}") from exc


OUTCOME = "outcome_prediction"
RATING = "future_observed_rating"
HERO_MODELS = (
    "prior",
    "linear",
    "signed_hero",
    "logistic",
    "lightgbm",
    "fm",
    "mlp",
    "deepsets",
    "set_transformer",
    "neuralac",
    "neuralac_no_cooperation",
    "neuralac_no_competition",
    "neuralac_no_attention",
    "neuralac_author_bias",
)
HISTORY_MODELS = ("static_deepsets", "history_deepsets", "owo_transformer", "owo", "dawn")
ASYMMETRIC_MODELS = ("prior", "history_frequency", "logistic", "lightgbm", "mlp", "ordinal")
RATING_MODELS = ("elo", "trueskill", "openskill", "openskill_bt_full", "openskill_tm_full")
SKILL_MODELS = ("current", "mean_delta", "ridge", "mlp", "gru", "transformer", "mmrnet")


def _cards():
    cards = {}

    def add(name, label, kind, tasks, schemas, scope, variant_of=None):
        cards[name] = ModelCard(name, label, kind, tasks, schemas, scope, variant_of)

    for name, label, kind in [
        ("prior", "Prior", "statistical_baseline"),
        ("linear", "Linear (PyTorch)", "linear_model"),
        ("signed_hero", "Shared signed hero effects", "linear_model"),
        ("logistic", "Logistic regression", "linear_model"),
        ("lightgbm", "LightGBM", "tree_model"),
        ("fm", "Factorization machine", "interaction_model"),
        ("mlp", "MLP", "neural_model"),
        ("deepsets", "Deep Sets", "neural_model"),
        ("set_transformer", "Roster Set Transformer", "neural_model"),
        ("neuralac", "NeuralAC", "neural_model"),
    ]:
        schemas = ["hero_roster"]
        if name in ASYMMETRIC_MODELS:
            schemas.append("asymmetric_context")
        if name in {"logistic", "lightgbm"}:
            schemas.append("player_history_aggregates")
        if name == "mlp":
            schemas.append("performance_sequence")
        add(
            name,
            label,
            kind,
            (OUTCOME, RATING) if name == "mlp" else (OUTCOME,),
            tuple(schemas),
            "Implemented estimator; input representation depends on the adapter.",
        )
    for name in HERO_MODELS:
        if not name.startswith("neuralac_"):
            continue
        add(
            name,
            name.replace("neuralac_", "NeuralAC / "),
            "neural_variant",
            (OUTCOME,),
            ("hero_roster",),
            "Component or author-code-bias variant; not a separate algorithm.",
            "neuralac",
        )
    for name, label in zip(
        HISTORY_MODELS,
        [
            "Static Deep Sets",
            "History Deep Sets",
            "OwO Transformer",
            "OwO",
            "DAWN",
        ],
    ):
        scope = "Equal-team player-history outcome model; configurable class count."
        if name in {"owo", "dawn"}:
            scope += " Public-data adaptation of " + ("CUPID" if name == "owo" else "CHAMP")
            scope += "; original industrial results are not reproduced."
        if name == "dawn":
            scope += " Requires a target-domain identifier; one domain cannot test transfer."
        add(name, label, "neural_model", (OUTCOME,), ("player_history",), scope)
    for name, label in zip(
        RATING_MODELS,
        [
            "Elo", "TrueSkill", "OpenSkill (Plackett–Luce)",
            "OpenSkill (Bradley–Terry Full)", "OpenSkill (Thurstone–Mosteller Full)",
        ],
    ):
        add(
            name,
            label,
            "online_rating",
            (OUTCOME,),
            ("event_replay",),
            "Online state updates plus declared validation-fitted outcome calibration.",
        )
    for name, label in [
        ("current", "Current rating persistence"),
        ("mean_delta", "Mean rating delta"),
        ("ridge", "Ridge"),
        ("gru", "GRU"),
        ("transformer", "Transformer"),
        ("mmrnet", "MMR-Net"),
    ]:
        kind = "neural_model" if name in {"gru", "transformer", "mmrnet"} else "baseline"
        scope = "Future observed-rating proxy; not true MMR or a verified novice cohort."
        if name == "mmrnet":
            scope += " Public-data QuickSkill module adaptation."
        add(name, label, kind, (RATING,), ("performance_sequence",), scope)
    add(
        "history_frequency",
        "History frequency",
        "statistical_baseline",
        (OUTCOME,),
        ("asymmetric_context",),
        "Previously completed predator outcome frequencies.",
    )
    add(
        "ordinal",
        "Ordinal MLP",
        "neural_model",
        (OUTCOME,),
        ("asymmetric_context",),
        "Ordered five-class elimination count, including all outcomes 0 through 4.",
    )
    return cards


MODELS = _cards()
PROFILES = {
    "hero-roster": InputProfile(
        "hero-roster",
        OUTCOME,
        "hero_roster",
        "Left-side loss/win (0/1)",
        ("two ordered hero rosters",),
        ("uci-dota2", "neuralac-dota2018", "neuralac-lol"),
        HERO_MODELS,
        limitation="Hero IDs are not persistent player IDs or player histories.",
    ),
    "overwatch-history": InputProfile(
        "overwatch-history",
        OUTCOME,
        "player_history",
        "Left-side loss/draw/win (0/1/2)",
        ("completed player histories", "static summaries", "history mask", "target mode"),
        ("overwatch",),
        HISTORY_MODELS,
        limitation="Current native adapter: unordered 6v6; two observed modes; 5/13 features.",
    ),
    "league-history": InputProfile(
        "league-history",
        OUTCOME,
        "player_history",
        "Team 100 loss/win (0/1)",
        ("completed player histories", "static summaries", "history mask"),
        ("leagueData",),
        (),
        ("history_deepsets", "owo", "logistic", "lightgbm"),
        "scripts/run_league_neural.py; scripts/run_league_baselines.py",
        "Separate frozen drivers: 5v5, one mode, compact store. DAWN is available through "
        "the model API but has no integrated leagueData training driver or result here.",
    ),
    "asymmetric-context": InputProfile(
        "asymmetric-context",
        OUTCOME,
        "asymmetric_context",
        "Elimination count 0/1/2/3/4",
        ("map/avatar", "completed predator summaries"),
        ("dbd-2022", "dbd-2023"),
        ASYMMETRIC_MODELS,
        limitation="Four prey identities and their histories are unavailable.",
    ),
    "overwatch-ratings": InputProfile(
        "overwatch-ratings",
        OUTCOME,
        "event_replay",
        "Left-side loss/draw/win (0/1/2)",
        ("persistent player IDs", "ordered teams", "delayed outcome feedback", "target mode"),
        ("overwatch",),
        RATING_MODELS,
        limitation="Online rating state is not the same input as the neural history tensors.",
    ),
    "future-rating": InputProfile(
        "future-rating",
        RATING,
        "performance_sequence",
        "Future observed OpenSkill mean",
        ("completed-game snapshots", "hero lineups", "current observed rating"),
        ("dota2-matches-v2",),
        SKILL_MODELS,
        limitation="Post-game regression, not pre-game outcome prediction; no ACC metric.",
    ),
}


# Executable counterparts of the legacy ``input_schema`` strings. Adding a
# feature now means editing one FeatureSpec in features/schema.py, not the
# adapter, the model signature and the runner.
PROFILE_SCHEMA_KEYS = {
    "hero-roster": "hero_binary",
    "overwatch-history": "overwatch_history",
    "league-history": "league_history",
    "asymmetric-context": "dbd_predator",
    "overwatch-ratings": "overwatch_rating",
    "future-rating": "dota_future_rating",
}
PROFILES = {
    name: replace(profile, feature_schema=PROFILE_SCHEMA_KEYS[name])
    for name, profile in PROFILES.items()
}


def feature_schema_for(name):
    """Return the executable FeatureSchema for a profile ID.

    Args:
        name: A registered input profile ID such as ``overwatch-history``.

    Returns:
        A FeatureSchema whose declared modalities, scopes, timings and output
        head can be inspected, validated and serialized without loading data.
    """
    return get_profile(name).executable_schema


def get_model(name):
    """Return the frozen ModelCard for a catalog ID, raising ValueError for unknown IDs."""
    try:
        return MODELS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown model: {name}") from exc


def get_profile(name):
    """Return the frozen InputProfile for a catalog ID, raising ValueError for unknown IDs."""
    try:
        return PROFILES[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown input profile: {name}") from exc


def model_catalog(task=None):
    """Return model records and separate counts of base IDs and variants.

    Args:
        task: Optional filter, either outcome_prediction or future_observed_rating.

    Returns:
        A dictionary with base_count, variant_count, models and counting_unit.
        Each entry in models is a copy of a catalog record.
    """
    if task not in {None, OUTCOME, RATING}:
        raise ValueError(f"Unknown task: {task}")
    cards = [c for c in MODELS.values() if task is None or task in c.tasks]
    return {
        "counting_unit": "callable model IDs, not distinct algorithms or completed experiments",
        "base_count": sum(c.variant_of is None for c in cards),
        "variant_count": sum(c.variant_of is not None for c in cards),
        "models": [asdict(c) for c in cards],
    }


def compatibility(model, profile):
    """Describe whether a model accepts a profile and how it can be executed.

    Args:
        model: Registered model ID.
        profile: Registered input profile ID.

    Returns:
        A dictionary with input_compatible, execution, reason and model scope.
        Execution is native_runner, external_driver, model_api_only or unavailable.
    """
    card, inputs = get_model(model), get_profile(profile)
    task_ok = inputs.task in card.tasks
    schema_ok = inputs.input_schema in card.input_schemas
    if inputs.name == "league-history" and "player_history_aggregates" in card.input_schemas:
        schema_ok = True
    compatible = task_ok and schema_ok
    if not task_ok:
        reason = "Different target task: " + inputs.task
    elif not schema_ok:
        reason = "No implemented input adapter from " + inputs.input_schema
        reason += " to " + ", ".join(card.input_schemas)
    else:
        reason = "Input contract is supported; this is not evidence of model quality."
    execution = "unavailable"
    if compatible:
        execution = (
            "native_runner"
            if model in inputs.native_models
            else "external_driver"
            if model in inputs.external_models
            else "model_api_only"
        )
    warnings = []
    if model == "dawn" and inputs.name == "league-history":
        warnings.append("A declared constant domain ID permits computation, not transfer evidence.")
    return {
        "model": model,
        "profile": profile,
        "task": inputs.task,
        "input_compatible": compatible,
        "execution": execution,
        "reason": reason,
        "warnings": warnings,
        "scope": card.scope,
    }


def check_models(profile, models):
    """Check a group of model IDs against one input profile without accessing data.

    Args:
        profile: Registered input profile ID such as hero-roster.
        models: Nonempty list or tuple of distinct model IDs.

    Returns:
        The profile, a record per model, and the aggregate flags input_compatible
        and native_runner_ready. Inspect execution to find the available entry point.

    Raises:
        ValueError: A name is unknown or the requested list is invalid.
    """
    inputs = get_profile(profile)
    if (
        not isinstance(models, (list, tuple))
        or not models
        or any(not isinstance(name, str) for name in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("Request distinct model IDs in a nonempty list")
    checks = [compatibility(name, profile) for name in models]
    return {
        "profile": asdict(inputs),
        "models": checks,
        "input_compatible": all(c["input_compatible"] for c in checks),
        "native_runner_ready": all(c["execution"] == "native_runner" for c in checks),
        "data_validation": "Not performed: inspect raw identities, splits and availability at run time.",
    }
