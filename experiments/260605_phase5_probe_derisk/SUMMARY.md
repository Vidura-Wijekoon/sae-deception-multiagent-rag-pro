# De-risk summary - 1 page (2026-06-05)

**Question for this run:** before spending GPU hours on Gemma 2-2B + Gemma Scope, does the cross-agent deception-probe *method* hold up, and which assumptions break? Run on a CPU proxy (MiniLM embeddings as stand-in activations); the representation backend is the only thing that swaps for the real model.

**What ran (real, reproducible):** a 120-example PoisonedRAG + Greshake attack corpus (two styles), a 2-agent retriever→writer RAG pipeline, and writer/retriever/combined linear probes with 5-fold AUROC, 2000× bootstrap CIs, a random-init control (A1.1) and a shuffled-retriever control (A3.6) - across three representations. One command: `python scripts/run_experiment.py`. 5/5 unit tests pass; all numbers reproduce bit-for-bit.

## Three findings that change the GPU plan

1. **The "easy" version of the task is fake.** On naive (boilerplate-heavy) attacks, a representation with **no trained parameters** scores **AUROC 1.000** - the label is decodable from surface text alone. So a high SAE-probe AUROC would prove nothing about deception. → *Always report the random-init Gemma control beside every number; never headline naive attacks.*

2. **A frozen probe cannot catch factual poisoning.** When the attack is hardened to a single-token lie (Paris→Lyon), MiniLM drops to **0.14** on PoisonedRAG - below chance - and **0.50 [0.39, 0.60]** overall. There is no surface cue for *which* claim is the lie. → *Detecting PoisonedRAG needs the writer's generative behaviour or internal/causal signals - budget the Phase-6 ablation and attack-success labels; don't expect a linear probe on frozen features to do it.* Indirect prompt injection (Greshake) is different: it stays detectable (0.86) because an injected instruction is intrinsically foreign text.

3. **"Cross-agent" adds nothing - correlationally.** Combined (writer+retriever) never beat writer-only (Δ ≤ +0.05, all CIs overlapping); shuffling the retriever barely moved it. In a pipeline the two agents share the poisoned text, so a shared feature is not transfer (the A3.2 confound, shown concretely). → *Direction 1's claim must rest on the causal ablation, not on combined-vs-writer AUROC. That harness is now built and validated.*

Supporting: Phase-4 attack-success gate **PASS** (poison retrieval 1.00, worst-case reader); Direction-2 retrieval amplification ≈ **neutral** (median ratio 0.99).

## Go-forward decision

Proceed to the GPU run, with three method changes forced by the above: **(a)** use the hardened corpus + attack-*success* labels for headline results; **(b)** ship a random-init Gemma control with every AUROC; **(c)** treat the cross-agent result as a *causal-ablation* question (Phase 6), keeping the correlational probe only as a baseline. Separate "injection detection" from "factual-deception detection" in the writeup - they are not the same problem.

**Cost of this de-risk:** a few CPU-minutes. **Value:** it killed the naive-label and correlational-transfer framings *before* they consumed GPU time, and told us exactly which labels and controls the real experiment needs.

*Caveats: MiniLM ≠ Gemma residual stream ≠ SAE features; attack-success used a worst-case reader; corpus is 60 facts × 2 templates. These results bound the method, not the phenomenon. Full detail + figures: `results.md`.*
