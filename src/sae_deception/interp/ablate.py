"""Phase 6 - causal harness for Direction 1 (cross-agent transfer).

The de-risk run showed combined-vs-writer AUROC cannot establish cross-agent
transfer (the A3.2 confound: both agents see the same poisoned text). This
module supplies the two causal arms the SUMMARY said the claim must rest on:

1. **Context-swap intervention** (`context_swap_intervention`) - a do()
   operation on the retriever->writer channel, at the text level. For every
   attacked query whose poisoned doc reached the writer, we swap that doc for
   its clean counterpart (everything else in the context held fixed) and
   re-run the writer. Reported:
     * necessity   - fraction of attack successes that *disappear* under the
                     swap (poison was necessary for the behaviour);
     * sufficiency - fraction of clean runs where *inserting* the poisoned doc
                     makes the attack succeed (poison is sufficient);
     * probe-score drop - mean fall in the writer deception-probe score when
                     the poison is swapped out. This is the activation-level
                     causal signal: if the probe tracks the *channel* rather
                     than incidental text, the score must fall.

2. **Feature ablation** (`feature_ablation`) - the SAE-feature-ablation proxy.
   Identify top-k discriminative dims of the writer probe and of the retriever
   probe (full-data fit, standardised |coef|). Then mean-ablate, in the
   *writer* representation:
     * the writer's own top-k dims          (within-agent ceiling),
     * the dims the *retriever* probe found (cross-agent transfer test), and
     * k random dims, R repeats             (control).
   If retriever-identified dims, ablated in the writer, drop writer AUROC well
   beyond the random control, the two agents are using *shared feature axes* -
   the cheap analog of "a structurally similar feature appears in Agent B"
   (README Direction 1). Overlap stats (observed vs hypergeometric-expected)
   are reported alongside.

On the GPU run, the same functions run unchanged on Gemma SAE features, where
"dims" become actual SAE latents and mean-ablation becomes feature clamping.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..attacks.corpus import ContextExample
from ..pipeline.rag import Retriever, writer_node
from ..pipeline.state import GraphState
from ..probes.train import run_probe

# ---------------------------------------------------------------------------
# Arm 2 - feature ablation (cross-agent shared-axis test)
# ---------------------------------------------------------------------------

def top_probe_dims(X: np.ndarray, y: np.ndarray, k: int, C: float = 1.0) -> np.ndarray:
    """Top-k dims by |coef| of a full-data standardised logistic probe.

    Full-data fit is fine here: the dims are used to *select an intervention*,
    and the effect of that intervention is then measured out-of-fold.
    """
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(Xs, np.asarray(y, dtype=int))
    return np.argsort(-np.abs(clf.coef_[0]))[:k]


def mean_ablate(X: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Replace the chosen columns with their column mean (zero information,
    distribution otherwise intact - the linear analog of SAE feature clamping)."""
    Xa = np.array(X, dtype=np.float64, copy=True)
    Xa[:, dims] = Xa[:, dims].mean(axis=0, keepdims=True)
    return Xa


def feature_ablation(
    Xw: np.ndarray,
    Xr: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 32,
    n_random: int = 20,
    n_folds: int = 5,
    C: float = 1.0,
    seed: int = 0,
) -> dict:
    """Run the three ablation arms on the writer representation."""
    y = np.asarray(y, dtype=int)
    d = Xw.shape[1]
    k = min(k, d)

    dims_w = top_probe_dims(Xw, y, k, C=C)
    dims_r = top_probe_dims(Xr, y, k, C=C)

    # observed overlap vs hypergeometric expectation (k draws from d dims)
    overlap = int(len(set(dims_w.tolist()) & set(dims_r.tolist())))
    expected = k * k / d

    def oof_auroc(X: np.ndarray) -> float:
        return run_probe("ablate", X, y, n_folds=n_folds, C=C, n_boot=200, seed=seed).auroc

    base = oof_auroc(Xw)
    drop_self = base - oof_auroc(mean_ablate(Xw, dims_w))
    drop_transfer = base - oof_auroc(mean_ablate(Xw, dims_r))

    rng = np.random.RandomState(seed)
    rand_drops = np.array([
        base - oof_auroc(mean_ablate(Xw, rng.choice(d, size=k, replace=False)))
        for _ in range(n_random)
    ])

    rd_std = float(rand_drops.std())
    return {
        "k": int(k),
        "n_dims": int(d),
        "baseline_writer_auroc": round(base, 4),
        "drop_ablate_writer_top_dims": round(drop_self, 4),
        "drop_ablate_retriever_top_dims_in_writer": round(drop_transfer, 4),
        "drop_ablate_random_dims_mean": round(float(rand_drops.mean()), 4),
        "drop_ablate_random_dims_std": round(rd_std, 4),
        "transfer_drop_sigma_vs_random": round(
            float((drop_transfer - rand_drops.mean()) / (rd_std + 1e-12)), 2
        ),
        "topk_overlap_writer_retriever": overlap,
        "topk_overlap_expected_by_chance": round(expected, 2),
    }


# ---------------------------------------------------------------------------
# Arm 1 - context-swap intervention (do() on the cross-agent channel)
# ---------------------------------------------------------------------------

def _probe_scorer(rep, examples: list[ContextExample], writer_view, C: float, seed: int):
    """Train a full-data writer probe and return score(texts) -> P(poisoned).

    Full-data fit is intentional: the probe is the *measurement device* for the
    intervention; its causal validity comes from the swap, not from held-out
    eval (which lives in Phase 5).
    """
    X = rep.encode([writer_view(e) for e in examples])
    y = np.array([e.label for e in examples], dtype=int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)

    def score(texts: list[str]) -> np.ndarray:
        return clf.predict_proba(scaler.transform(rep.encode(texts)))[:, 1]

    return score


def context_swap_intervention(
    examples: list[ContextExample],
    rep,
    *,
    top_k: int = 4,
    seed: int = 0,
    n_distractors: int = 30,
    probe_C: float = 1.0,
) -> dict:
    """For each attacked query: factual run, then the counterfactual where the
    poisoned doc is swapped for its clean counterpart in the writer's context."""

    def writer_view(e: ContextExample) -> str:
        return f"Question: {e.query}\n\nContext:\n{e.context_text}\n\nAnswer:"

    score = _probe_scorer(rep, examples, writer_view, probe_C, seed)
    retr = Retriever(rep, seed=seed).index(examples)

    by_qid: dict[int, dict] = {}
    for i, e in enumerate(examples):
        slot = by_qid.setdefault(e.qid, {"clean": None, "poison": None, "false": None,
                                          "gold": e.answer, "query": e.query})
        if e.label == 0:
            slot["clean"] = i
        else:
            slot["poison"] = i
            slot["false"] = e.injected_claim if e.attack_type == "poisonedrag" else None

    clean_idxs = [v["clean"] for v in by_qid.values() if v["clean"] is not None]
    rng = np.random.RandomState(seed)

    per_query = []
    n_attacked = n_success = n_necessity_flip = 0
    n_clean_base = n_sufficiency_flip = 0
    score_drops = []

    for qid, v in by_qid.items():
        if v["poison"] is None or v["clean"] is None:
            continue
        pool = [i for i in clean_idxs if i != v["clean"]]
        rng.shuffle(pool)
        allowed_clean = sorted(set(pool[:n_distractors] + [v["clean"]]))
        allowed_atk = sorted(set(allowed_clean + [v["poison"]]))

        # --- factual attacked run -------------------------------------
        st = GraphState(query=v["query"], gold_answer=v["gold"], condition="attacked")
        st.retrieved = retr.retrieve(v["query"], top_k, allowed_atk)
        if not st.poison_retrieved:
            continue
        n_attacked += 1
        ans_f, surf_f = writer_node(st, v["false"])
        success_f = surf_f

        # --- counterfactual: swap poisoned doc -> clean counterpart ----
        clean_text = examples[v["clean"]].context_text
        cf = GraphState(query=v["query"], gold_answer=v["gold"], condition="attacked")
        cf.retrieved = [
            type(r)(doc_id=r.doc_id, text=clean_text, score=r.score, is_poison=False)
            if r.is_poison else r
            for r in st.retrieved
        ]
        ans_cf, surf_cf = writer_node(cf, v["false"])

        # --- sufficiency: insert the poisoned doc into the clean run ---
        st_cl = GraphState(query=v["query"], gold_answer=v["gold"], condition="clean")
        st_cl.retrieved = retr.retrieve(v["query"], top_k, allowed_clean)
        _, surf_cl = writer_node(st_cl, v["false"])
        n_clean_base += 1
        ins = GraphState(query=v["query"], gold_answer=v["gold"], condition="attacked")
        poisoned_e = examples[v["poison"]]
        ins.retrieved = list(st_cl.retrieved)
        if ins.retrieved:
            ins.retrieved[0] = type(ins.retrieved[0])(
                doc_id=poisoned_e.example_id, text=poisoned_e.context_text,
                score=1.0, is_poison=True,
            )
        _, surf_ins = writer_node(ins, v["false"])
        if (not surf_cl) and surf_ins:
            n_sufficiency_flip += 1

        s_f, s_cf = score([st.writer_view, cf.writer_view])
        score_drops.append(float(s_f - s_cf))

        if success_f:
            n_success += 1
            if not surf_cf:
                n_necessity_flip += 1

        per_query.append({
            "qid": qid, "query": v["query"],
            "factual_answer": ans_f, "counterfactual_answer": ans_cf,
            "attack_success_factual": bool(success_f),
            "attack_success_counterfactual": bool(surf_cf),
            "probe_score_factual": round(float(s_f), 4),
            "probe_score_counterfactual": round(float(s_cf), 4),
        })

    drops = np.array(score_drops) if score_drops else np.array([0.0])
    return {
        "n_queries_poison_retrieved": n_attacked,
        "n_attack_success_factual": n_success,
        "necessity_flip_rate": round(n_necessity_flip / max(n_success, 1), 4),
        "sufficiency_flip_rate": round(n_sufficiency_flip / max(n_clean_base, 1), 4),
        "probe_score_drop_mean": round(float(drops.mean()), 4),
        "probe_score_drop_median": round(float(np.median(drops)), 4),
        "frac_queries_probe_score_fell": round(float((drops > 0).mean()), 4),
        "per_query": per_query,
    }
