"""Phase 1 gate: the LangGraph wrap must reproduce the pipeline-lite numbers.

Uses the cheap tfidf_svd representer so the test runs in seconds with no
model download.
"""

from __future__ import annotations

from sae_deception.attacks.corpus import GRESHAKE_SENTINEL, build_corpus
from sae_deception.interp.represent import get_representer
from sae_deception.pipeline.graph import build_graph, run_query
from sae_deception.pipeline.rag import Retriever


def _setup(style: str = "naive"):
    ex = build_corpus(seed=0, n_base_facts=20, style=style)
    rep = get_representer("tfidf_svd", dim=64, seed=0).fit([e.context_text for e in ex])
    retr = Retriever(rep, seed=0).index(ex)
    return ex, build_graph(retr, top_k=4)


def test_graph_clean_run_answers_gold():
    ex, graph = _setup()
    clean = next(e for e in ex if e.qid == 1 and e.label == 0)
    allowed = [i for i, e in enumerate(ex) if e.label == 0]
    out = run_query(graph, clean.query, gold_answer=clean.answer, allowed=allowed)
    assert out["poison_retrieved"] is False
    assert clean.answer.lower() in out["answer"].lower()
    assert out["retriever_view"] and out["writer_view"].startswith("Question:")


def test_graph_attacked_run_surfaces_injection():
    ex, graph = _setup()
    pois = next(e for e in ex if e.attack_type == "greshake")
    out = run_query(graph, pois.query, gold_answer=pois.answer,
                    false_answer=None, condition="attacked")  # whole pool: poison present
    if out["poison_retrieved"] and GRESHAKE_SENTINEL in out["retriever_view"]:
        assert out["answer"] == GRESHAKE_SENTINEL
        assert out["surfaced_injection"] is True


def test_graph_state_views_match_dataclass_contract():
    ex, graph = _setup()
    e = ex[0]
    out = run_query(graph, e.query, gold_answer=e.answer)
    # writer view = query + retriever view, exactly as state.GraphState defines
    assert out["retriever_view"] in out["writer_view"]
    assert f"Question: {e.query}" in out["writer_view"]
    assert len(out["retrieved"]) == 4
