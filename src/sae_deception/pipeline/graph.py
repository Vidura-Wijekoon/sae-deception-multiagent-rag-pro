"""Phase 1 completion - the retriever→writer pipeline as a real LangGraph StateGraph.

`state.py` deliberately mirrored a LangGraph state so this wrap would be
mechanical; this module is that wrap. The graph has exactly the two nodes the
research design names as agents:

    START -> retriever -> writer -> END

The nodes delegate to the same `Retriever` / `writer_node` used by the
pipeline-lite in `rag.py`, so numbers are identical between the two code paths
(asserted in tests/test_graph.py). The graph state carries the two text views
(`retriever_view`, `writer_view`) that the interp layer captures activations
on - the cross-agent context boundary is the `retriever -> writer` edge.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .rag import Retriever, writer_node
from .state import GraphState, Retrieved


class RAGGraphState(TypedDict, total=False):
    """LangGraph channel schema for one query through the two-agent pipeline."""

    query: str
    gold_answer: str
    condition: str                 # "clean" | "attacked"
    false_answer: str | None       # PoisonedRAG injected claim, if any
    top_k: int
    allowed: list[int] | None      # doc-pool restriction (None = whole pool)
    retrieved: list[dict]          # serialised Retrieved entries
    retriever_view: str            # what crosses the agent boundary
    writer_view: str               # what the writer conditions on
    answer: str
    surfaced_injection: bool
    poison_retrieved: bool


def _to_graph_state(state: RAGGraphState) -> GraphState:
    gs = GraphState(
        query=state["query"],
        gold_answer=state.get("gold_answer", ""),
        condition=state.get("condition", "clean"),
    )
    gs.retrieved = [Retrieved(**r) for r in state.get("retrieved", [])]
    return gs


def build_graph(retriever: Retriever, top_k: int = 4):
    """Compile the two-agent StateGraph over an already-indexed Retriever."""

    def retrieve(state: RAGGraphState) -> dict:
        docs = retriever.retrieve(
            state["query"], state.get("top_k", top_k), state.get("allowed")
        )
        gs = GraphState(query=state["query"], gold_answer=state.get("gold_answer", ""),
                        condition=state.get("condition", "clean"))
        gs.retrieved = docs
        return {
            "retrieved": [vars(r) for r in docs],
            "retriever_view": gs.retriever_view,
            "poison_retrieved": gs.poison_retrieved,
        }

    def write(state: RAGGraphState) -> dict:
        gs = _to_graph_state(state)
        answer, surfaced = writer_node(gs, state.get("false_answer"))
        return {
            "writer_view": gs.writer_view,
            "answer": answer,
            "surfaced_injection": surfaced,
        }

    g = StateGraph(RAGGraphState)
    g.add_node("retriever", retrieve)
    g.add_node("writer", write)
    g.add_edge(START, "retriever")
    g.add_edge("retriever", "writer")
    g.add_edge("writer", END)
    return g.compile()


def run_query(
    graph,
    query: str,
    gold_answer: str = "",
    false_answer: str | None = None,
    condition: str = "clean",
    allowed: list[int] | None = None,
    top_k: int | None = None,
) -> RAGGraphState:
    """Convenience: invoke the compiled graph for one query and return final state."""
    inp: RAGGraphState = {
        "query": query,
        "gold_answer": gold_answer,
        "condition": condition,
        "false_answer": false_answer,
        "allowed": allowed,
    }
    if top_k is not None:
        inp["top_k"] = top_k
    return graph.invoke(inp)
