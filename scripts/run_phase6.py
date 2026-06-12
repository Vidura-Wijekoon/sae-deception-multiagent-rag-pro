"""Phase 6 — run the causal harness and write phase6_ablation.json.

    python scripts/run_phase6.py --config configs/default.yaml

Reuses the Phase 4 corpus and the cached activations from the Phase 5 run
(data/acts/<experiment_id>/) when present, so the marginal cost on CPU is the
ablation fits only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_deception.attacks.corpus import build_corpus  # noqa: E402
from sae_deception.config import set_global_seed  # noqa: E402
from sae_deception.interp.ablate import context_swap_intervention, feature_ablation  # noqa: E402
from sae_deception.interp.represent import get_representer  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_experiment import load_config  # noqa: E402


def _writer_view(e) -> str:
    return f"Question: {e.query}\n\nContext:\n{e.context_text}\n\nAnswer:"


def _encode_cached(rep, rep_name: str, view: str, texts: list[str], acts_dir: Path) -> np.ndarray:
    """Same cache layout as experiment.py, so Phase 5 activations are reused."""
    acts_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(("␟".join(texts)).encode()).hexdigest()[:10]
    path = acts_dir / f"{rep_name}__{view}__{h}.npy"
    if path.exists():
        return np.load(path)
    X = rep.encode(texts)
    np.save(path, X)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--ablate-k", type=int, default=32)
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    set_global_seed(cfg.seed)
    out_dir = cfg.out_dir(REPO_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_dir = REPO_ROOT / "data" / "acts" / cfg.experiment_id

    rep = get_representer(cfg.representation, dim=cfg.repr_dim, seed=cfg.seed,
                          neural_model=cfg.neural_model)
    results: dict = {
        "experiment_id": cfg.experiment_id,
        "representation": cfg.representation,
        "representation_revision": rep.revision,
        "config_hash": cfg.config_hash(),
        "feature_ablation": {},
    }

    # fit (label-free) the representation if it needs it (tfidf_svd)
    fit_texts: list[str] = []
    corpora = {}
    for style in ("naive", "hardened"):
        ex = build_corpus(seed=cfg.seed, n_base_facts=cfg.n_base_facts,
                          attack_mix=cfg.attack_mix, style=style)
        corpora[style] = ex
        fit_texts += [e.context_text for e in ex] + [_writer_view(e) for e in ex] + [e.query for e in ex]
    rep.fit(fit_texts)

    # --- Arm 2: feature ablation per corpus style -------------------------
    for style, ex in corpora.items():
        y = np.array([e.label for e in ex], dtype=int)
        Xw = _encode_cached(rep, cfg.representation, f"{style}_writer", [_writer_view(e) for e in ex], acts_dir)
        Xr = _encode_cached(rep, cfg.representation, f"{style}_retriever", [e.context_text for e in ex], acts_dir)
        fa = feature_ablation(Xw, Xr, y, k=args.ablate_k, n_random=args.n_random,
                              n_folds=cfg.n_folds, C=cfg.probe_C, seed=cfg.seed)
        results["feature_ablation"][style] = fa
        print(f"[phase6:ablate:{style}] base={fa['baseline_writer_auroc']} "
              f"self_drop={fa['drop_ablate_writer_top_dims']} "
              f"transfer_drop={fa['drop_ablate_retriever_top_dims_in_writer']} "
              f"random={fa['drop_ablate_random_dims_mean']}+/-{fa['drop_ablate_random_dims_std']} "
              f"(transfer = {fa['transfer_drop_sigma_vs_random']} sigma vs random) "
              f"overlap={fa['topk_overlap_writer_retriever']} vs {fa['topk_overlap_expected_by_chance']} expected")

    # --- Arm 1: context-swap intervention (naive corpus, like Phase 1/4) --
    cs = context_swap_intervention(corpora["naive"], rep, top_k=cfg.top_k,
                                   seed=cfg.seed, probe_C=cfg.probe_C)
    results["context_swap"] = cs
    print(f"[phase6:swap] poison_retrieved={cs['n_queries_poison_retrieved']} "
          f"necessity={cs['necessity_flip_rate']} sufficiency={cs['sufficiency_flip_rate']} "
          f"probe_score_drop_mean={cs['probe_score_drop_mean']} "
          f"frac_fell={cs['frac_queries_probe_score_fell']}")

    (out_dir / "phase6_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out_dir / 'phase6_ablation.json'}")


if __name__ == "__main__":
    main()
