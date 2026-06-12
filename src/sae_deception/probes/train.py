"""Linear probes, cross-validated AUROC, bootstrap CIs, and controls (Phase 5).

The probe is a plain L2-logistic regression on a frozen representation — the
same recipe as MacDiarmid et al. 2024 and Goldowsky-Dill et al. 2025. We report:

  * pooled out-of-fold (OOF) AUROC from stratified k-fold CV, so every example
    is scored by a probe that never saw it;
  * a bootstrap 95% CI on that AUROC over examples (ASSUMPTIONS A2.2: with ~120
    examples the SE is ~0.05, so a CI is mandatory before claiming any gap);
  * per-fold AUROC mean +/- std as a secondary stability check.

Direction 1 (cross-agent): we compare writer-only, retriever-only, and combined
(concatenated) probes, plus a shuffled-retriever negative control (A3.6).
Direction 2 (amplification): see `amplification_ratios`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    name: str
    auroc: float                       # pooled OOF AUROC
    ci95: tuple[float, float]
    fold_mean: float
    fold_std: float
    n: int
    n_features: int
    oof_scores: np.ndarray = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "auroc": round(self.auroc, 4),
            "ci95_low": round(self.ci95[0], 4),
            "ci95_high": round(self.ci95[1], 4),
            "fold_mean": round(self.fold_mean, 4),
            "fold_std": round(self.fold_std, 4),
            "n": self.n,
            "n_features": self.n_features,
        }


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U); ties handled via average ranks."""
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for k, c in enumerate(counts):
        avg[k] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cross_val_oof_scores(X: np.ndarray, y: np.ndarray, n_folds: int, C: float, seed: int):
    """Return (oof_scores, per_fold_aurocs). Standardises features inside each fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_aurocs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y[tr])
        s = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        oof[te] = s
        fold_aurocs.append(_auroc(y[te], s))
    return oof, np.array(fold_aurocs)


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap CI on AUROC, resampling examples with replacement."""
    rng = np.random.RandomState(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        # guard against a resample with only one class
        if len(np.unique(y[idx])) < 2:
            boots[b] = np.nan
            continue
        boots[b] = _auroc(y[idx], scores[idx])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_probe(name: str, X: np.ndarray, y: np.ndarray, *, n_folds=5, C=1.0, n_boot=2000, seed=0) -> ProbeResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    oof, fold_aurocs = cross_val_oof_scores(X, y, n_folds, C, seed)
    auroc = _auroc(y, oof)
    ci = bootstrap_ci(y, oof, n_boot, seed)
    return ProbeResult(
        name=name, auroc=auroc, ci95=ci,
        fold_mean=float(np.nanmean(fold_aurocs)), fold_std=float(np.nanstd(fold_aurocs)),
        n=len(y), n_features=X.shape[1], oof_scores=oof,
    )


def shuffle_rows(X: np.ndarray, seed: int) -> np.ndarray:
    """Permute rows of a feature block — destroys its alignment to labels.
    Used for the A3.6 shuffled-retriever negative control."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    return X[perm]


# ---------------------------------------------------------------------------
# Direction 2 — retrieval amplification
# ---------------------------------------------------------------------------

def amplification_ratios(no_ctx: np.ndarray, clean: np.ndarray, poisoned: np.ndarray) -> dict:
    """Per-feature amplification: how much each representation dimension's mean
    absolute activation grows from clean to poisoned context.

    README Direction 2 falsifier: ratios concentrated near 1.0 => retrieval is
    neutral (neither amplifier nor filter).
    """
    m_no = np.mean(np.abs(no_ctx), axis=0)
    m_clean = np.mean(np.abs(clean), axis=0)
    m_pois = np.mean(np.abs(poisoned), axis=0)
    eps = 1e-8
    amp_pois_vs_clean = (m_pois + eps) / (m_clean + eps)
    amp_clean_vs_no = (m_clean + eps) / (m_no + eps)
    return {
        "median_amp_poisoned_vs_clean": float(np.median(amp_pois_vs_clean)),
        "p90_amp_poisoned_vs_clean": float(np.percentile(amp_pois_vs_clean, 90)),
        "p10_amp_poisoned_vs_clean": float(np.percentile(amp_pois_vs_clean, 10)),
        "frac_features_amplified_gt_1_2": float(np.mean(amp_pois_vs_clean > 1.2)),
        "frac_features_suppressed_lt_0_8": float(np.mean(amp_pois_vs_clean < 0.8)),
        "median_amp_clean_vs_nocontext": float(np.median(amp_clean_vs_no)),
        "_amp_vector_poisoned_vs_clean": amp_pois_vs_clean.tolist(),
    }
