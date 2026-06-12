"""Render the de-risk figures from a completed run (reads metrics.json + the
cached activations). Saves PNGs into experiments/<id>/figures/.

    python scripts/plot_results.py --repo-root . --experiment 260605_phase5_probe_derisk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DEFAULT / "src"))

from sae_deception.attacks.corpus import build_corpus  # noqa: E402
from sae_deception.config import ExperimentConfig  # noqa: E402
from sae_deception.experiment import _encode_cached, _writer_view  # noqa: E402
from sae_deception.interp.represent import get_representer  # noqa: E402
from sae_deception.probes.train import _auroc, amplification_ratios, run_probe  # noqa: E402

C_NEU, C_TFIDF, C_RAND = "#2563eb", "#16a34a", "#dc2626"


def fig_auroc(metrics: dict, out: Path) -> None:
    reps = ["neural", "tfidf_svd", "random_init"]
    colors = {"neural": C_NEU, "tfidf_svd": C_TFIDF, "random_init": C_RAND}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, style in zip(axes, ["naive", "hardened"]):
        xs = np.arange(len(reps))
        for off, arm, hatch in [(-0.18, "writer_only", None), (0.18, "combined", "//")]:
            vals, los, his, cols = [], [], [], []
            for r in reps:
                d = metrics["direction1_probes"][style][r][arm]
                vals.append(d["auroc"]); los.append(d["auroc"] - d["ci95_low"]); his.append(d["ci95_high"] - d["auroc"])
                cols.append(colors[r])
            ax.bar(xs + off, vals, width=0.34, color=cols, alpha=0.85 if arm == "writer_only" else 0.5,
                   hatch=hatch, edgecolor="black", linewidth=0.6,
                   yerr=[los, his], capsize=3, label=arm.replace("_", "-"))
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.text(2.35, 0.51, "chance", color="gray", fontsize=8, va="bottom", ha="right")
        ax.set_xticks(xs); ax.set_xticklabels(["MiniLM\n(neural)", "TF-IDF\nSVD", "random\ninit"])
        ax.set_title(f"{style} attacks"); ax.set_ylim(0, 1.05)
        if style == "naive":
            ax.set_ylabel("probe AUROC (5-fold OOF, 95% CI)")
    axes[1].legend(["writer-only", "combined (W+R)"], loc="upper right", fontsize=8)
    fig.suptitle("Probe AUROC by representation and attack style - random-init is the lexical-triviality control (A1.1)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_per_attack(repo_root: Path, exp_id: str, out: Path) -> None:
    cfg = ExperimentConfig()
    acts = repo_root / "data" / "acts" / exp_id
    reps = ["neural", "random_init"]
    data = {}  # (rep, style) -> (PR, GR)
    for rep_name in reps:
        rep = get_representer(rep_name, dim=cfg.repr_dim, seed=cfg.seed, neural_model=cfg.neural_model)
        fit_texts = []
        for style in ["naive", "hardened"]:
            ex = build_corpus(seed=cfg.seed, n_base_facts=cfg.n_base_facts, style=style)
            fit_texts += [e.context_text for e in ex] + [_writer_view(e) for e in ex] + [e.query for e in ex]
        rep.fit(fit_texts)
        for style in ["naive", "hardened"]:
            ex = build_corpus(seed=cfg.seed, n_base_facts=cfg.n_base_facts, style=style)
            y = np.array([e.label for e in ex]); atk = np.array([e.attack_type for e in ex])
            Xw = _encode_cached(rep, rep_name, f"{style}_writer", [_writer_view(e) for e in ex], acts)
            p = run_probe("x", Xw, y, n_folds=cfg.n_folds, n_boot=400, seed=cfg.seed)
            prm = (atk == "none") | (atk == "poisonedrag"); grm = (atk == "none") | (atk == "greshake")
            data[(rep_name, style)] = (_auroc(y[prm], p.oof_scores[prm]), _auroc(y[grm], p.oof_scores[grm]))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    groups = ["PoisonedRAG\n(factual lie)", "Greshake\n(prompt injection)"]
    xs = np.arange(2)
    bars = [
        ("neural · naive", C_NEU, 0.9, -0.3, [data[("neural", "naive")][0], data[("neural", "naive")][1]]),
        ("neural · hardened", C_NEU, 0.5, -0.1, [data[("neural", "hardened")][0], data[("neural", "hardened")][1]]),
        ("rand-init · hardened", C_RAND, 0.5, 0.1, [data[("random_init", "hardened")][0], data[("random_init", "hardened")][1]]),
    ]
    w = 0.2
    for i, (lab, col, alpha, off, vals) in enumerate(bars):
        ax.bar(xs + off, vals, width=w, color=col, alpha=alpha, edgecolor="black", linewidth=0.6, label=lab)
    ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(1.45, 0.51, "chance", color="gray", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(groups); ax.set_ylim(0, 1.05)
    ax.set_ylabel("writer-probe AUROC"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Hardened factual poisoning is undetectable by a frozen probe (≈chance);\ninjection keeps a surface signature")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_amplification(repo_root: Path, exp_id: str, out: Path) -> None:
    cfg = ExperimentConfig()
    acts = repo_root / "data" / "acts" / exp_id
    rep = get_representer("neural", neural_model=cfg.neural_model)
    ex = build_corpus(seed=cfg.seed, n_base_facts=cfg.n_base_facts, style="naive")
    by = {}
    for e in ex:
        by.setdefault(e.qid, {})[e.condition] = e
    qids = sorted(by)
    Xnc = _encode_cached(rep, "neural", "naive_nocontext", [by[q]["clean"].query for q in qids], acts)
    Xcl = _encode_cached(rep, "neural", "naive_writer_clean", [_writer_view(by[q]["clean"]) for q in qids], acts)
    Xpo = _encode_cached(rep, "neural", "naive_writer_poison", [_writer_view(by[q]["poisoned"]) for q in qids], acts)
    amp = amplification_ratios(Xnc, Xcl, Xpo)["_amp_vector_poisoned_vs_clean"]
    amp = np.array(amp)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.hist(np.clip(amp, 0, 3), bins=50, color=C_NEU, alpha=0.8, edgecolor="white")
    ax.axvline(1.0, ls="--", c="gray", lw=1.2); ax.text(1.02, ax.get_ylim()[1] * 0.9, "neutral (×1.0)", fontsize=8)
    ax.axvline(np.median(amp), ls="-", c=C_RAND, lw=1.2, label=f"median ×{np.median(amp):.2f}")
    ax.set_xlabel("per-feature activation ratio  (poisoned ÷ clean context)")
    ax.set_ylabel("number of MiniLM features"); ax.legend(fontsize=9)
    ax.set_title("Direction 2: retrieval is ~neutral in embedding space (ratios cluster at 1.0, modest amplified tail)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_DEFAULT))
    ap.add_argument("--experiment", default="260605_phase5_probe_derisk")
    args = ap.parse_args()
    repo = Path(args.repo_root)
    exp_dir = repo / "experiments" / args.experiment
    figs = exp_dir / "figures"; figs.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    fig_auroc(metrics, figs / "fig1_auroc_by_representation.png")
    fig_per_attack(repo, args.experiment, figs / "fig2_per_attack_type.png")
    fig_amplification(repo, args.experiment, figs / "fig3_amplification.png")
    print("wrote figures to", figs)


if __name__ == "__main__":
    main()
