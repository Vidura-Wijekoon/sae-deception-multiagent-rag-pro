"""Fast, model-free tests for the de-risk harness (no network, no GPU).

Covers corpus determinism/balance, the probe AUROC machinery, and the two
controls that carry the scientific weight (random-init lexical-triviality and
shuffled-retriever). Runs in a second on CPU with only numpy + scikit-learn.

    pytest -q
"""

from __future__ import annotations

import numpy as np

from sae_deception.attacks.corpus import build_corpus, corpus_summary
from sae_deception.interp.represent import get_representer
from sae_deception.probes.train import _auroc, run_probe, shuffle_rows


def test_corpus_is_deterministic_balanced_and_unique():
    a = build_corpus(seed=0, n_base_facts=60, style="naive")
    b = build_corpus(seed=0, n_base_facts=60, style="naive")
    assert [e.sha256 for e in a] == [e.sha256 for e in b]      # deterministic
    assert len({e.sha256 for e in a}) == 120                    # all unique
    s = corpus_summary(a)
    assert s["by_condition"] == {"clean": 60, "poisoned": 60}   # balanced
    assert s["by_attack_type"]["poisonedrag"] == 30
    assert s["by_attack_type"]["greshake"] == 30


def test_hardened_poisonedrag_differs_from_clean_by_answer_only():
    ex = build_corpus(seed=0, n_base_facts=60, style="hardened")
    clean = next(e for e in ex if e.example_id == "q001_clean")
    pois = next(e for e in ex if e.qid == 1 and e.label == 1)
    # France q1 is a poisonedrag slot; hardened swaps the answer token only
    # (templates apply .capitalize(), so compare case-insensitively).
    assert "lyon" in pois.context_text.lower() and "paris" not in pois.context_text.lower()
    assert "paris" in clean.context_text.lower()


def test_auroc_matches_sklearn_reference():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    # known AUROC for this toy case is 0.75
    assert abs(_auroc(y, s) - 0.75) < 1e-9


def test_random_init_control_separates_naive_attacks():
    """A1.1: an untrained representation should already ace the naive labels,
    proving they are lexically trivial."""
    ex = build_corpus(seed=0, n_base_facts=60, style="naive")
    y = np.array([e.label for e in ex])
    rep = get_representer("random_init", dim=384, seed=0).fit([e.context_text for e in ex])
    X = rep.encode([e.context_text for e in ex])
    p = run_probe("ri", X, y, n_folds=5, n_boot=200, seed=0)
    assert p.auroc > 0.95
    assert p.ci95[0] <= p.auroc <= p.ci95[1]


def test_shuffling_retriever_destroys_its_signal():
    """A3.6: row-shuffling a feature block must not preserve label alignment."""
    ex = build_corpus(seed=0, n_base_facts=60, style="naive")
    y = np.array([e.label for e in ex])
    rep = get_representer("tfidf_svd", dim=128, seed=0).fit([e.context_text for e in ex])
    X = rep.encode([e.context_text for e in ex])
    true = run_probe("t", X, y, n_folds=5, n_boot=200, seed=0).auroc
    shuf = run_probe("s", shuffle_rows(X, seed=0), y, n_folds=5, n_boot=200, seed=0).auroc
    assert shuf < true                      # shuffled block loses discriminative power
    assert abs(shuf - 0.5) < 0.15           # ...back toward chance
