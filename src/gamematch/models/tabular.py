# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Converged logistic and LightGBM adapters with the same side-specific hero counts."""

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


def require_logistic_convergence(model):
    """Return iterations only when the fitted solver stayed below its declared cap."""
    iterations = int(np.max(model.n_iter_))
    if iterations >= model.max_iter:
        raise RuntimeError(f"Logistic solver hit its iteration cap ({model.max_iter})")
    return iterations


def hero_counts(encoded, vocab_size):
    columns = (encoded + np.arange(2)[None, :, None] * vocab_size).reshape(-1)
    rows = np.repeat(np.arange(len(encoded)), encoded.shape[1] * encoded.shape[2])
    return csr_matrix(
        (np.ones(len(columns), dtype=np.float32), (rows, columns)),
        shape=(len(encoded), 2 * vocab_size),
    )


def predict_kill_counts(model, features):
    """Return five probability columns in literal elimination-count order.

    Args:
        model: Fitted classifier whose classes_ lists observed counts in 0..4.
        features: Preprocessed feature matrix accepted by the classifier.

    Returns:
        An array [N, 5]. Counts absent during fitting have zero probability
        before the runner's numerical clipping and temperature calibration.
    """
    probabilities = model.predict_proba(features)
    counts = np.asarray(model.classes_)
    if not np.issubdtype(counts.dtype, np.integer) or not np.isin(counts, np.arange(5)).all():
        raise ValueError("Expected fitted elimination counts in 0..4")
    ordered = np.zeros((len(probabilities), 5), dtype=probabilities.dtype)
    ordered[:, counts] = probabilities
    return ordered


def train_tabular(name, splits, vocabulary, config, seed, directory):
    data = {
        key: hero_counts(vocabulary.transform(value.rosters), vocabulary.size)
        for key, value in splits.items()
    }
    if name == "logistic":
        model = LogisticRegression(
            C=config.logistic_c, max_iter=1000, solver="lbfgs", random_state=seed
        )
        model.fit(data["train"], splits["train"].labels)
        epoch = require_logistic_convergence(model)
    elif name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM could not be imported. Install 'gamematch-research[tabular]' "
                "and run 'gamematch doctor' to inspect the import error."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "LightGBM's native library is unavailable. Check your platform's OpenMP "
                "runtime (libomp on macOS) and run 'gamematch doctor'. In a source checkout "
                "with PyTorch's bundled libomp, use scripts/run_local.py."
            ) from exc
        model = lgb.LGBMClassifier(
            n_estimators=config.max_trees,
            num_leaves=31,
            learning_rate=config.tree_learning_rate,
            n_jobs=config.threads,
            random_state=seed,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(
            data["train"],
            splits["train"].labels,
            eval_set=[(data["valid"], splits["valid"].labels)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(config.tree_patience, verbose=False)],
        )
        epoch = model.best_iteration_
    else:
        raise ValueError(name)
    joblib.dump(model, directory / "checkpoint.joblib")
    return model, data, epoch
