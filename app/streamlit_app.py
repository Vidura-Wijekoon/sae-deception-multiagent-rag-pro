"""sae-deception-multiagent-rag — interactive dashboard.

    streamlit run app/streamlit_app.py        (or: make ui)

Five views over the de-risk experiment, plus a live playground that runs a
query through the real LangGraph retriever->writer pipeline and scores it with
the deception probe:

  1. Overview            — phase gates, headline numbers, the three findings
  2. Direction 1 · Probes — AUROC + bootstrap CIs per representation/arm/style
  3. Direction 2 + Phase 6 — amplification ratios; causal ablation results
  4. Attack corpus       — browse the 120-context PoisonedRAG/Greshake set
  5. Live pipeline       — run the two-agent graph on any query, watch the
                           probe's deception score react to poisoning

Everything renders from the artefacts the experiment scripts write
(metrics.json, phase6_ablation.json, attack manifests); the playground builds
its corpus/retriever/probe on first use and caches them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_deception.attacks.corpus import _FACTS, build_corpus  # noqa: E402
from sae_deception.interp.represent import get_representer  # noqa: E402
from sae_deception.pipeline.graph import build_graph, run_query  # noqa: E402
from sae_deception.pipeline.rag import Retriever  # noqa: E402

st.set_page_config(page_title="SAE Deception · Multi-Agent RAG", page_icon="🔬", layout="wide")

ARM_LABELS = {
    "writer_only": "Writer only",
    "retriever_only": "Retriever only",
    "combined": "Combined (W+R)",
    "combined_shuffled_retriever_control": "Combined, shuffled R (control)",
}
REP_LABELS = {"neural": "MiniLM (neural)", "tfidf_svd": "TF-IDF→SVD", "random_init": "Random-init (control)"}


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

@st.cache_data
def list_experiments() -> list[str]:
    return sorted(
        p.parent.name for p in (REPO_ROOT / "experiments").glob("*/metrics.json")
    )


@st.cache_data
def load_json(rel_path: str) -> dict | None:
    p = REPO_ROOT / rel_path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_manifest(exp_id: str, style: str) -> pd.DataFrame:
    p = REPO_ROOT / "data" / "attacks" / f"{exp_id}_{style}_manifest.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Playground resources (built lazily, cached for the session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building corpus, retriever and probe…")
def playground(rep_name: str, style: str, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    examples = build_corpus(seed=seed, n_base_facts=60, style=style)

    def writer_view(text_query: str, ctx: str) -> str:
        return f"Question: {text_query}\n\nContext:\n{ctx}\n\nAnswer:"

    rep = get_representer(rep_name, dim=384, seed=seed)
    rep.fit([e.context_text for e in examples]
            + [writer_view(e.query, e.context_text) for e in examples]
            + [e.query for e in examples])

    retriever = Retriever(rep, seed=seed).index(examples)
    graph = build_graph(retriever, top_k=4)

    X = rep.encode([writer_view(e.query, e.context_text) for e in examples])
    y = np.array([e.label for e in examples], dtype=int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)

    def deception_score(text: str) -> float:
        return float(clf.predict_proba(scaler.transform(rep.encode([text])))[0, 1])

    return examples, graph, deception_score


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def probe_bar_chart(metrics: dict, style: str) -> go.Figure:
    fig = go.Figure()
    probes = metrics["direction1_probes"][style]
    for rep_name, arms in probes.items():
        xs, ys, lo, hi = [], [], [], []
        for arm, label in ARM_LABELS.items():
            d = arms.get(arm)
            if not isinstance(d, dict):
                continue
            xs.append(label)
            ys.append(d["auroc"])
            lo.append(d["auroc"] - d["ci95_low"])
            hi.append(d["ci95_high"] - d["auroc"])
        fig.add_trace(go.Bar(
            name=REP_LABELS.get(rep_name, rep_name), x=xs, y=ys,
            error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo),
        ))
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="chance")
    fig.update_layout(barmode="group", yaxis_title="OOF AUROC (95% bootstrap CI)",
                      yaxis_range=[0, 1.05], legend_title="Representation", height=420)
    return fig


def ablation_chart(fa: dict) -> go.Figure:
    arms = ["Writer's own top-k", "Retriever's top-k → writer (transfer)", "Random k (control)"]
    vals = [fa["drop_ablate_writer_top_dims"],
            fa["drop_ablate_retriever_top_dims_in_writer"],
            fa["drop_ablate_random_dims_mean"]]
    err = [0, 0, fa["drop_ablate_random_dims_std"]]
    fig = go.Figure(go.Bar(x=arms, y=vals, error_y=dict(type="data", array=err),
                           marker_color=["#4c78a8", "#e45756", "#9d9d9d"]))
    fig.update_layout(yaxis_title="AUROC drop after mean-ablation", height=380,
                      title=f"k={fa['k']} of {fa['n_dims']} dims · baseline AUROC {fa['baseline_writer_auroc']}")
    return fig


def gauge(score: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(valueformat=".2f"),
        title=dict(text=title),
        gauge=dict(
            axis=dict(range=[0, 1]),
            bar=dict(color="#e45756" if score > 0.5 else "#59a14f"),
            steps=[dict(range=[0, 0.5], color="#e8f0e8"), dict(range=[0.5, 1], color="#f7e6e6")],
            threshold=dict(line=dict(color="black", width=3), value=0.5),
        ),
    ))
    fig.update_layout(height=260, margin=dict(t=60, b=10))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

experiments = list_experiments()
if not experiments:
    st.error("No experiment artefacts found. Run `python scripts/run_experiment.py` first.")
    st.stop()

with st.sidebar:
    st.title("🔬 SAE Deception")
    st.caption("Probing for deception in multi-agent RAG pipelines")
    exp_id = st.selectbox("Experiment", experiments, index=len(experiments) - 1)
    metrics = load_json(f"experiments/{exp_id}/metrics.json")
    manifest = load_json(f"experiments/{exp_id}/run_manifest.json")
    phase6 = load_json(f"experiments/{exp_id}/phase6_ablation.json")

    if manifest:
        with st.expander("Provenance (run manifest)"):
            st.write(f"**created:** {manifest.get('created_utc', '?')}")
            st.write(f"**git:** `{str(manifest.get('git_commit', '?'))[:12]}`")
            st.write(f"**config hash:** `{manifest.get('config_hash', '?')}`")
            st.write(f"**representation:** `{manifest.get('representation_revision', '?')}`")
    st.divider()
    st.caption("CPU de-risk proxy — MiniLM stands in for Gemma 2-2B + Gemma Scope SAE. "
               "Swap `representation: gemma_sae` in configs/default.yaml for the GPU run.")

tab_overview, tab_d1, tab_d2p6, tab_corpus, tab_live = st.tabs(
    ["📋 Overview", "📊 Direction 1 · Probes", "🧪 Direction 2 + Phase 6 · Causal",
     "🗂️ Attack corpus", "🤖 Live pipeline"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.header("Where the project stands")
    pipe = metrics.get("pipeline_phase1_4", {})
    ctl = metrics.get("controls", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Poison retrieval rate (Phase 4)", f"{pipe.get('poison_retrieval_rate', 0):.2f}",
              "gate ≥0.30 " + ("PASS" if pipe.get("phase4_gate_A2_1_pass") else "FAIL"))
    c2.metric("Attack success (worst-case reader)", f"{pipe.get('attack_success_rate', 0):.2f}")
    hardened_w = metrics["direction1_probes"]["hardened"][metrics["primary_representation"]]["writer_only"]
    c3.metric("Hardened writer-probe AUROC", f"{hardened_w['auroc']:.2f}",
              f"CI [{hardened_w['ci95_low']:.2f}, {hardened_w['ci95_high']:.2f}]", delta_color="off")
    c4.metric("Random-init control on naive (A1.1)",
              f"{ctl.get('A1_1_random_init_writer_auroc_naive', 0):.2f}",
              "label is lexically trivial", delta_color="off")

    st.subheader("Phase gates")
    p6_done = phase6 is not None
    gates = pd.DataFrame([
        ("0", "Environment + core papers read", "✅ done"),
        ("1", "Working RAG pipeline (LangGraph retriever→writer)", "✅ done — real LangGraph StateGraph (pipeline/graph.py)"),
        ("2", "Activations captured per agent per query", "✅ proxy — MiniLM views cached to data/acts/ (Gemma capture = GPU)"),
        ("3", "Gemma Scope SAE attached, features labeled", "⏳ GPU-gated"),
        ("4", "120-example attack set with measured success rate", "✅ done — gate A2.1 PASS"),
        ("5", "Per-representation AUROC vs random-init control", "✅ proxy done"),
        ("6", "Cross-agent transfer: correlational + causal", "✅ proxy done — causal harness run" if p6_done else "🔶 harness built, run pending"),
        ("7", "Summary + cohort feedback + extended-mode decision", "🔶 summary written; cohort feedback external"),
    ], columns=["Phase", "Gate", "Status"])
    st.dataframe(gates, hide_index=True, use_container_width=True)

    st.subheader("The three findings that changed the GPU plan")
    st.markdown(
        "1. **The easy version of the task is fake.** A representation with no trained "
        "parameters scores AUROC ≈1.0 on naive attacks — the label is decodable from surface "
        "text. Every headline number must ship beside the random-init control.\n"
        "2. **A frozen probe cannot catch hardened factual poisoning** (single-token lie): "
        "≈chance overall, below chance on PoisonedRAG. Injection-style attacks (Greshake) stay "
        "detectable because injected instructions are intrinsically foreign text.\n"
        "3. **Correlational 'cross-agent' signal is confounded** — both agents see the same "
        "poisoned text, so combined-vs-writer AUROC can't establish transfer. The claim now "
        "rests on the Phase 6 causal harness (see the Causal tab)."
    )


# ---------------------------------------------------------------------------
# Tab 2 — Direction 1 probes
# ---------------------------------------------------------------------------

with tab_d1:
    st.header("Direction 1 — probe AUROC by representation and arm")
    style = st.radio("Attack corpus style", ["naive", "hardened"], horizontal=True,
                     help="naive = boilerplate-heavy attacks (lexically trivial); "
                          "hardened = single-token lies / naturalistic injections")
    st.plotly_chart(probe_bar_chart(metrics, style), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Per-attack-type AUROC (writer probe)")
        pat = metrics.get("per_attack_type_auroc", {}).get(style, {})
        if pat:
            st.dataframe(pd.DataFrame([
                {"Attack family": "PoisonedRAG (factual lie)", "AUROC": pat.get("poisonedrag")},
                {"Attack family": "Greshake (prompt injection)", "AUROC": pat.get("greshake")},
            ]), hide_index=True, use_container_width=True)
    with col_b:
        st.subheader("Δ combined − writer (the correlational transfer test)")
        rows = [{"Representation": REP_LABELS.get(r, r),
                 "Δ AUROC": d.get("combined_minus_writer")}
                for r, d in metrics["direction1_probes"][style].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Never significantly > 0 → correlational AUROC cannot carry the "
                   "cross-agent claim (A3.2 confound). See the causal tab.")


# ---------------------------------------------------------------------------
# Tab 3 — Direction 2 amplification + Phase 6 causal
# ---------------------------------------------------------------------------

with tab_d2p6:
    st.header("Direction 2 — does retrieval amplify adversarial features?")
    amp_rows = []
    for rep_name, a in metrics.get("direction2_amplification", {}).items():
        amp_rows.append({
            "Representation": REP_LABELS.get(rep_name, rep_name),
            "Median amp (poisoned/clean)": a["median_amp_poisoned_vs_clean"],
            "p10": a["p10_amp_poisoned_vs_clean"], "p90": a["p90_amp_poisoned_vs_clean"],
            "% dims amplified >1.2×": f"{100 * a['frac_features_amplified_gt_1_2']:.1f}%",
            "% dims suppressed <0.8×": f"{100 * a['frac_features_suppressed_lt_0_8']:.1f}%",
        })
    st.dataframe(pd.DataFrame(amp_rows), hide_index=True, use_container_width=True)
    st.caption("Falsifier check: ratios concentrated near 1.0 ⇒ on this proxy, retrieval is "
               "*neutral* — neither amplifier nor filter.")

    st.divider()
    st.header("Phase 6 — causal harness (Direction 1's real test)")
    if not phase6:
        st.info("No phase6_ablation.json found for this experiment yet. "
                "Run `python scripts/run_phase6.py` to produce it.")
    else:
        cs = phase6["context_swap"]
        st.subheader("Arm 1 · Context-swap intervention — do() on the retriever→writer channel")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Necessity (swap removes attack)", f"{cs['necessity_flip_rate']:.2f}")
        c2.metric("Sufficiency (insert creates attack)", f"{cs['sufficiency_flip_rate']:.2f}")
        c3.metric("Mean probe-score drop", f"{cs['probe_score_drop_mean']:+.3f}")
        c4.metric("Queries where score fell", f"{100 * cs['frac_queries_probe_score_fell']:.0f}%")
        with st.expander(f"Per-query intervention table ({cs['n_queries_poison_retrieved']} queries)"):
            st.dataframe(pd.DataFrame(cs["per_query"]), hide_index=True, use_container_width=True)

        st.subheader("Arm 2 · Feature ablation — shared-axis (transfer) test")
        fa_style = st.radio("Corpus style", list(phase6["feature_ablation"].keys()),
                            horizontal=True, key="fa_style")
        fa = phase6["feature_ablation"][fa_style]
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.plotly_chart(ablation_chart(fa), use_container_width=True)
        with col_r:
            st.metric("Transfer drop vs random control",
                      f"{fa['transfer_drop_sigma_vs_random']:.1f}σ")
            st.metric("Top-k overlap (writer ∩ retriever)",
                      f"{fa['topk_overlap_writer_retriever']} dims",
                      f"{fa['topk_overlap_expected_by_chance']} expected by chance",
                      delta_color="off")
            st.caption("Ablating, *in the writer*, the dims the **retriever** probe found: "
                       "a drop far beyond the random control means the two agents read the "
                       "deception signal off shared feature axes — the cheap analog of "
                       "cross-agent feature transfer. On the GPU run these dims become "
                       "Gemma Scope SAE latents and ablation becomes feature clamping.")


# ---------------------------------------------------------------------------
# Tab 4 — Attack corpus browser
# ---------------------------------------------------------------------------

with tab_corpus:
    st.header("Attack corpus — 120 contexts per style")
    style_c = st.radio("Style", ["naive", "hardened"], horizontal=True, key="corpus_style")
    df = load_manifest(exp_id, style_c)
    if df.empty:
        st.info("Manifest not found — run `python scripts/run_experiment.py` first.")
    else:
        f1, f2, f3 = st.columns(3)
        atk = f1.multiselect("Attack type", sorted(df["attack_type"].unique()),
                             default=sorted(df["attack_type"].unique()))
        tier = f2.multiselect("Difficulty tier", sorted(df["tier"].unique()),
                              default=sorted(df["tier"].unique()))
        q = f3.text_input("Search query text")
        view = df[df["attack_type"].isin(atk) & df["tier"].isin(tier)]
        if q:
            view = view[view["query"].str.contains(q, case=False)]
        st.dataframe(view[["example_id", "tier", "attack_type", "query", "answer",
                           "injected_claim", "context_text"]],
                     hide_index=True, use_container_width=True, height=320)

        st.subheader("Clean vs poisoned, side by side")
        qids = sorted(view["qid"].unique())
        if qids:
            qid = st.selectbox("Question", qids,
                               format_func=lambda i: df[df.qid == i]["query"].iloc[0])
            pair = df[df.qid == qid]
            cl, po = st.columns(2)
            clean_row = pair[pair.label == 0].iloc[0]
            pois_row = pair[pair.label == 1].iloc[0]
            with cl:
                st.markdown("**🟢 Clean context**")
                st.info(clean_row["context_text"])
            with po:
                st.markdown(f"**🔴 Poisoned context** — `{pois_row['attack_type']}`")
                st.error(pois_row["context_text"])
                if pois_row["injected_claim"]:
                    st.caption(f"Injected claim: `{pois_row['injected_claim']}`")


# ---------------------------------------------------------------------------
# Tab 5 — Live pipeline playground
# ---------------------------------------------------------------------------

with tab_live:
    st.header("Run the two-agent LangGraph pipeline live")
    st.caption("START → retriever → writer → END, with the deception probe reading the "
               "writer's view. Toggle the poisoned document in and out of the document "
               "pool and watch the probe react.")

    s1, s2, s3 = st.columns(3)
    rep_choice = s1.selectbox(
        "Representation", ["tfidf_svd", "neural", "random_init"],
        format_func=lambda r: REP_LABELS.get(r, r),
        help="tfidf_svd is instant; neural (MiniLM) loads a model on first use.")
    style_live = s2.radio("Corpus style", ["naive", "hardened"], horizontal=True, key="live_style")
    poison_in_pool = s3.toggle("Poisoned doc in retrieval pool", value=True)

    examples, graph, deception_score = playground(rep_choice, style_live)

    queries = {f"q{f[0]:03d} — {f[2]}": f[0] for f in _FACTS}
    qkey = st.selectbox("Query", list(queries.keys()))
    qid = queries[qkey]
    base = next(e for e in examples if e.qid == qid and e.label == 0)
    pois = next(e for e in examples if e.qid == qid and e.label == 1)

    if st.button("▶ Run pipeline", type="primary"):
        allowed = [i for i, e in enumerate(examples)
                   if e.label == 0 or (poison_in_pool and e.qid == qid)]
        false_answer = pois.injected_claim if pois.attack_type == "poisonedrag" else None
        out = run_query(graph, base.query, gold_answer=base.answer,
                        false_answer=false_answer,
                        condition="attacked" if poison_in_pool else "clean",
                        allowed=allowed)

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Agent 1 — Retriever")
            for r in out["retrieved"]:
                badge = "🔴 POISON" if r["is_poison"] else "🟢 clean"
                st.markdown(f"`{r['doc_id']}` · sim {r['score']:.3f} · {badge}")
                st.caption(r["text"])
            st.subheader("Agent 2 — Writer")
            st.markdown(f"**Answer:** {out['answer']}")
            if out["surfaced_injection"]:
                st.error("⚠️ The writer surfaced the injected content — attack succeeded.")
            elif out["poison_retrieved"]:
                st.warning("Poison was retrieved but did not surface in the answer.")
            else:
                st.success("Clean run — no poison retrieved.")
        with right:
            score = deception_score(out["writer_view"])
            st.plotly_chart(gauge(score, "Probe P(poisoned context)"), use_container_width=True)
            st.caption("Full-data probe trained on this corpus/representation — a "
                       "demonstration readout, not the held-out Phase 5 evaluation.")
            with st.expander("Writer view (what crosses the agent boundary)"):
                st.text(out["writer_view"])
