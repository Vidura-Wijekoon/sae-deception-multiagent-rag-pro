# sae-deception-multiagent-rag

**Probing for deception in multi-agent RAG pipelines using sparse autoencoders, linear probes, and attribution graphs.**

> Status: **Research preview, de-risk phase.** Not for production use. Results may be null, partial, or revised. See [Phase 7 — De-Risk Review](#roadmap) for milestone definitions.

---

## The one-paragraph version

We can't see inside multi-agent LLM systems when they're behaving badly. If one agent manipulates another through a shared context window — deliberately, or through indirect prompt injection in a retrieved document — there is no reliable mechanism today for catching it before harmful output appears. This project takes three single-model interpretability tools — sparse autoencoders (SAEs), linear probes, and attribution graphs — and instruments them across a LangGraph retrieval-augmented generation (RAG) pipeline. The goal is to test whether deceptive or adversarial activation patterns are detectable *between* agents, not just *within* one. The project is anchored on Sharkey et al. (2025), *Open Problems in Mechanistic Interpretability*, which explicitly names multi-agent and multi-model settings as an open problem.

---

## Why this matters

Agentic LLM deployments are scaling faster than the interpretability tooling that should oversee them. The 2025 mechanistic-interpretability consensus paper ([Sharkey et al., 2025](https://arxiv.org/abs/2501.16496)) maps the tools we have for seeing inside single models — sparse autoencoders, attribution graphs, linear probes — and flags multi-agent settings as an unsolved gap. Concurrently, attack research has demonstrated that retrieval-augmented agents are vulnerable to indirect prompt injection ([Greshake et al., 2023](https://arxiv.org/abs/2302.12173)), knowledge-base poisoning ([Zou et al., 2024 — PoisonedRAG](https://arxiv.org/abs/2402.07867)), and memory backdoors ([Chen et al., 2024 — AgentPoison](https://arxiv.org/abs/2407.12784)) with attack-success rates above 80% at poison rates below 0.1%.

This project sits at the intersection of three Shallow Review 2025 agendas: [Sparse Coding](https://shallowreview.ai/White_box_safety/Sparse_Coding) (≈50–100 FTEs), [Lie and Deception Detectors](https://shallowreview.ai/White_box_safety/Lie_and_deception_detectors) (≈10–50 FTEs), and the under-resourced [Tools for Aligning Multiple AIs](https://shallowreview.ai/Multi_agent_first/Tools_for_aligning_multiple_AIs) (≈10–15 FTEs). The contribution is the bridge.

---

## The three Directions

### Direction 1 — Cross-agent activation transfer as a deception signal

> *If an adversarial feature activates in Agent A and a structurally similar feature appears in Agent B's next forward pass, that pattern of transfer across the context boundary may be a more reliable deception signal than any single-agent probe.*

**Hypothesis.** Catching cross-agent feature transfer would give us an interpretability-based tripwire for agent collusion *before* it produces harmful outputs.

**Method.** Train linear probes on SAE features from both the retriever and the writer in a LangGraph RAG pipeline. Compare (a) writer-only probe, (b) retriever-only probe, (c) combined probe. Identify features in the top-discriminative set for both agents; run causal ablations on those features to test for a transfer mechanism.

**Falsifier.** Combined-probe AUROC indistinguishable from writer-only AUROC across at least two layer combinations.

**Closest prior art.** [Marks et al., 2024 — Sparse Feature Circuits](https://arxiv.org/abs/2403.19647); [Templeton et al., 2024 — Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity).

### Direction 2 — Does retrieval amplify or suppress adversarial features?

> *The retrieval step is the most under-examined attack surface in a RAG pipeline. Injected documents could either activate latent deceptive features or drown them out.*

**Hypothesis.** Retrieval either acts as an *amplifier* (adversarial content in retrieved docs lights up downstream SAE features the model would never produce on its own) or a *filter* (the retrieved noise drowns out adversarial structure). Telling the two apart is the first step to understanding whether retrieval is a vulnerability or a natural defense.

**Method.** Measure top-feature activations in the writer across three conditions: (a) no context, (b) clean retrieved context, (c) poisoned retrieved context. Compute amplification ratios per feature.

**Falsifier.** Amplification ratios concentrated near 1.0 across all feature axes (i.e. retrieval is neither amplifier nor filter — it's neutral).

**Closest prior art.** [Zou et al., 2024 — PoisonedRAG](https://arxiv.org/abs/2402.07867); [Chen et al., 2024 — AgentPoison](https://arxiv.org/abs/2407.12784).

### Direction 3 — Generalising probes across model families

> *SAE probes trained on a LangGraph pipeline running Gemma may not transfer to one running Llama or Mistral.*

**Hypothesis.** If probes generalise across model families, deceptive representations are likely a universal property of the multi-agent setting. If they don't, deception is architecture-specific — with serious implications for any cross-model safety monitoring system.

**Method.** Train SAE probes on Gemma 2-2B + Gemma Scope. Evaluate on (a) Llama 3.2-3B + community SAEs, (b) Mistral-7B + Llama Scope analogues, in identical LangGraph pipelines and on identical attack sets.

**Falsifier.** Cross-architecture transfer AUROC drops to chance.

**Closest prior art.** [MacDiarmid et al., 2024 — Simple Probes Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents).

---

## Load-bearing assumptions

Spelled out so you can falsify them rather than discover them empirically:

1. **Deception leaves a consistent latent fingerprint across model layers.** If it doesn't, SAE probes won't generalise. Falsifier: flat per-layer AUROC curve in Phase 5, or unstable top features across SAE seeds (a la [Paulo & Belrose, 2025](https://arxiv.org/abs/2501.16615)).
2. **Attribution graphs can be meaningfully extended across agent boundaries.** If cross-agent activations are too noisy or too text-mediated, tracing collapses. Falsifier: Direction 1 null result.
3. **LangGraph is a representative architecture.** If findings don't transfer to AutoGen or custom scaffolds, the project's scope is narrower than advertised. Mitigation in extended-project mode: replicate on at least one alternative scaffold before publishing.

---

## Repo layout

```
sae-deception-multiagent-rag/
├── README.md                  (you are here)
├── LICENSE                    (MIT)
├── CITATION.cff
├── .gitignore
├── pyproject.toml             (added in Phase 0 step 3)
├── configs/
│   └── default.yaml           (added in Phase 1 step 17)
├── src/
│   ├── pipeline/              LangGraph nodes (retriever, writer, graph state)
│   ├── probes/                Linear + SAE probe training and eval
│   ├── attacks/               PoisonedRAG + Greshake attack adapters
│   └── interp/                Activation capture, SAE wrappers, auto-interp glue
├── experiments/
│   └── YYMMDD_<name>/         One folder per experiment. Notebooks live here.
├── notebooks/                 Exploratory only. Move to experiments/ when shareable.
├── data/                      Gitignored. Raw activations, attack corpora, eval outputs.
├── docs/
│   └── literature_survey.md   Mirror of the Word doc (extracted markdown)
└── scripts/
    └── init_repo.sh           Run once to bootstrap git + first commit
```

Sub-package boundaries are loose during the de-risk phase. Refactor when (and only when) you transition to extended-project mode.

---

## Roadmap

The full plan lives in `docs/build_guide.md` (80 steps across 8 phases). Phase gates:

| Phase | Gate | Status |
|---|---|---|
| 0 | Environment + Sharkey/Ameisen/Lindsey read | not started |
| 1 | Working LangGraph RAG, 50 queries, caching | not started |
| 2 | Activations captured per agent per query | not started |
| 3 | Gemma Scope SAE attached, top features labeled | not started |
| 4 | 120-example attack set with measured success rate | not started |
| 5 | Per-layer AUROC: raw vs SAE vs random-init control | not started |
| 6 | Cross-agent transfer signal: correlational + causal | not started |
| 7 | 1-page summary + cohort feedback + extended-mode decision | not started |

---

## Quickstart

> Needs Python 3.11+ and a GPU with ≥16 GB VRAM (Gemma 2-2B in float16 fits on most cards from L4 / RTX 4080 up).

```bash
# 1. Clone
git clone git@github.com:<your-handle>/sae-deception-multiagent-rag.git
cd sae-deception-multiagent-rag

# 2. Env (pyproject.toml lands in Phase 0 step 3)
conda create -n sae-deception python=3.11 -y
conda activate sae-deception
pip install -e ".[dev]"        # once pyproject.toml exists

# 3. Smoke test
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')"
```

Detailed step-by-step instructions are in `docs/build_guide.md`.

---

## Reproducibility

- Every run writes a `run_manifest.json` with: git commit hash, config hash, model name + revision, SAE name + revision, seeds, host info.
- All LLM calls are cached to `data/cache/` (diskcache, keyed on prompt + model + seed). Re-runs of identical experiments hit the cache.
- The attack corpora are versioned in `data/attacks/manifest.jsonl` with hashes per example.
- Random seeds are set for `torch`, `numpy`, `random`, FAISS, and the tokenizer.

If a result is not reproducible from a single `run_pipeline.py --config <yaml>` invocation, treat that as a bug and fix before reporting.

---

## How to cite

Until there's a paper, please cite the repo via `CITATION.cff` or the suggested BibTeX below:

```bibtex
@misc{vidura2026saedeception,
  author       = {Vidura},
  title        = {sae-deception-multiagent-rag: Probing for Deception in
                  Multi-Agent RAG Pipelines Using Sparse Autoencoders},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-handle>/sae-deception-multiagent-rag}},
  note         = {BlueDot Impact Technical AI Safety Project. Research preview.},
}
```

---

## Acknowledgements

- **BlueDot Impact** Technical AI Safety Project, which scoped and funded this work.
- **Lee Sharkey** and the 30+ co-authors of *Open Problems in Mechanistic Interpretability*, whose Section 7 names the problem this project attacks.
- **Tom Lieberum and the Gemma Scope team** for releasing 400+ SAEs that make this project tractable in 30 hours rather than 6 months.
- **Jack Lindsey, Emmanuel Ameisen, and the Anthropic Circuit Tracing team** for the attribution-graph methodology this project lifts to the multi-agent setting.
- The **EleutherAI / delphi (sae-auto-interp)** maintainers for the open auto-interp pipeline.

Mistakes are mine, not theirs.

---

## License

MIT — see [LICENSE](./LICENSE). Choosing a permissive license because safety research benefits from being downstream-buildable; if you want to remix, fork, replicate, or extend, please do.

---

## Contact

Vidura — `businessaividura@viduraaitech.space`

Issues: [GitHub Issues](https://github.com/<your-handle>/sae-deception-multiagent-rag/issues) (preferred for technical questions and bug reports).
