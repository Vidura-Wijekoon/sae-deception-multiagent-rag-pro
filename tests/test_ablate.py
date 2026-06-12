"""Phase 6 gate: the causal harness behaves as designed on the CPU proxy.

All tests use cheap representers (tfidf_svd / random_init) - no model download.
"""

from __future__ import annotations

import numpy as np

from sae_deception.attacks.corpus import build_corpus
from sae_deception.interp.ablate import (
    context_swap_intervention,
    feature_ablation,
    mean_ablate,
    top_probe_dims,
)
from sae_deception.interp.represent import get_representer


def _views(ex):
    return [f"Question: {e.query}\n\nContext:\n{e.context_text}\n\nAnswer:" for e in ex]


def test_mean_ablate_zeroes_information_only_in_chosen_dims():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 10))
    Xa = mean_ablate(X, np.array([2, 5]))
    assert np.allclose(Xa[:, 2], X[:, 2].mean())
    assert np.allclose(Xa[:, [0, 1, 3, 4, 6, 7, 8, 9]], X[:, [0, 1, 3, 4, 6, 7, 8, 9]])


def test_ablating_top_dims_hurts_more_than_random():
    """On the naive corpus the signal is concentrated; killing the probe's own
    top dims must cost more AUROC than killing random dims."""
    ex = build_corpus(seed=0, n_base_facts=40, style="naive")
    y = np.array([e.label for e in ex])
    rep = get_representer("random_init", dim=128, seed=0).fit([])
    Xw = rep.encode(_views(ex))
    Xr = rep.encode([e.context_text for e in ex])
    fa = feature_ablation(Xw, Xr, y, k=16, n_random=5, n_folds=5, seed=0)
    assert fa["drop_ablate_writer_top_dims"] >= fa["drop_ablate_random_dims_mean"]
    assert fa["baseline_writer_auroc"] > 0.9  # naive labels are lexically trivial


def test_top_probe_dims_finds_a_planted_feature():
    rng = np.random.RandomState(0)
    y = np.array([0, 1] * 50)
    X = rng.normal(size=(100, 20))
    X[:, 7] += 3.0 * y  # plant a perfectly informative dim
    dims = top_probe_dims(X, y, k=3)
    assert 7 in dims.tolist()


def test_context_swap_necessity_is_near_total_for_worst_case_reader():
    """With the deterministic worst-case writer, removing the poisoned doc
    should almost always remove the attack behaviour, and inserting it should
    almost always create it. Not exactly 1.0: a false-answer token can occur
    *benignly* in another retrieved doc (e.g. "nitrogen" answers q7 falsely but
    appears truthfully in q13's clean passage), so the token-matching reader
    still surfaces it post-swap - a real confound worth measuring, not a bug.
    The probe score must mostly fall under the swap."""
    ex = build_corpus(seed=0, n_base_facts=30, style="naive")
    rep = get_representer("tfidf_svd", dim=96, seed=0)
    rep.fit([e.context_text for e in ex] + _views(ex))
    out = context_swap_intervention(ex, rep, top_k=4, seed=0)
    assert out["n_queries_poison_retrieved"] > 0
    assert out["necessity_flip_rate"] >= 0.85
    assert out["sufficiency_flip_rate"] >= 0.85
    assert out["frac_queries_probe_score_fell"] > 0.8
