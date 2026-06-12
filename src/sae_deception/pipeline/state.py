"""Shared graph state for the two-agent RAG pipeline.

Deliberately a plain dataclass rather than a LangGraph `TypedDict` for the
de-risk run - the structure mirrors what a LangGraph `StateGraph` would carry
(Retriever node -> Writer node), so porting to real LangGraph later is a
mechanical wrap. The two text views this exposes (`retriever_view`,
`writer_view`) are exactly what the interp layer captures activations on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Retrieved:
    doc_id: str
    text: str
    score: float
    is_poison: bool


@dataclass
class GraphState:
    query: str
    gold_answer: str
    condition: str                       # "clean" | "attacked"
    retrieved: list[Retrieved] = field(default_factory=list)

    @property
    def retriever_view(self) -> str:
        """What the Retriever emits across the context boundary: top-k docs."""
        return "\n\n".join(r.text for r in self.retrieved)

    @property
    def writer_view(self) -> str:
        """What the Writer conditions on: the query plus the retrieved context."""
        ctx = self.retriever_view
        return f"Question: {self.query}\n\nContext:\n{ctx}\n\nAnswer:"

    @property
    def poison_retrieved(self) -> bool:
        return any(r.is_poison for r in self.retrieved)
