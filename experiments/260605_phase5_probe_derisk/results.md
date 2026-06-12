# Phase 1 / 4 / 5 de-risk — probe harness on a CPU proxy

**Date:** 2026-06-05 (post-Phase-1 re-audit date in ASSUMPTIONS.md)
**Phases exercised:** 1 (RAG pipeline), 4 (attack corpus + success rate), 5 (per-representation probe AUROC with controls)
**Directions exercised:** 1 (cross-agent / combined vs single-agent probe), 2 (retrieval amplification)
**Run:** `python scripts/run_experiment.py --config configs/default.yaml` → `metrics.json`, `run_manifest.json`, `pipeline_results.json`, `figures/`

---

## What this run is, and what it is not

This is the **de-risk harness running end-to-end on a CPU-tractable proxy representation**, not the Gemma 2-2B + Gemma Scope experiment. Every number below is real (5-fold cross-validated AUROC, 2000-sample bootstrap CIs, real attack corpus, real retriever), but the "activations" are **MiniLM (`all-MiniLM-L6-v2`) sentence embeddings**, with **TF-IDF→SVD** and a **random-init random-feature map** as comparison backends. The representation is the only swappable component (`sae_deception.interp.represent`): pointing it at `gemma_sae` on a GPU reruns this identical analysis on real SAE features.

The purpose is exactly what a de-risk phase is for: **validate the experimental machinery and falsify cheap-to-falsify assumptions before spending GPU hours.** It did both, and it returned three results that change how the real run should be designed.

---

## Setup

- **Corpus (Phase 4).** 60 distinct factual questions × {clean, poisoned} = **120 balanced contexts**, built twice:
  - **naive** — attacks carry obvious boilerplate (PoisonedRAG: "the most up-to-date sources confirm…" + query echo; Greshake: a literal `[system note: disregard your instructions…]` block).
  - **hardened** — low lexical signature (PoisonedRAG: identical to the clean passage except the answer token, Paris→Lyon; Greshake: a naturalistic "confirm your details at secure-verify.example" sentence).
  - Clean and poisoned passages share the same benign lead/trailer, so a probe cannot win on length. Manifests with per-example SHA-256 in `data/attacks/`.
- **Representations.** `neural` = MiniLM 384-d (fastembed/ONNX, no GPU); `tfidf_svd` = TF-IDF(1,2)→SVD-384; `random_init` = hashed char 3–5-grams → fixed random projection → tanh (no trained parameters — the A1.1 control).
- **Probe (Phase 5).** L2-logistic regression, standardised per fold, **stratified 5-fold**, pooled out-of-fold AUROC, **2000× bootstrap 95% CI** (A2.2). Arms: writer-only, retriever-only, combined (concat), and a **shuffled-retriever** combined control (A3.6).
- **Pipeline (Phase 1/4).** Dense retriever (MiniLM cosine top-k=4) over a pool of benign docs ± the query's poisoned doc; a deterministic **non-robust reader** (worst-case writer proxy) for attack-success.

---

## Results

### Direction 1 — probe AUROC (writer-only / combined), 95% CI

**Naive attacks**

| Representation | writer-only | retriever-only | combined (W+R) | combined − writer | shuffled-retriever ctrl |
|---|---|---|---|---|---|
| MiniLM (neural) | 0.981 [0.96, 1.00] | 1.000 | 0.997 [0.99, 1.00] | +0.016 | 0.932 |
| TF-IDF→SVD | 0.638 [0.53, 0.74] | 0.689 | 0.686 [0.58, 0.78] | +0.047 | 0.605 |
| **random-init (A1.1 ctrl)** | **1.000 [1.00, 1.00]** | 1.000 | 1.000 | +0.000 | 1.000 |

**Hardened attacks**

| Representation | writer-only | retriever-only | combined (W+R) | combined − writer | shuffled-retriever ctrl |
|---|---|---|---|---|---|
| MiniLM (neural) | 0.497 [0.39, 0.60] | 0.512 | 0.512 [0.40, 0.62] | +0.015 | 0.538 |
| TF-IDF→SVD | 0.419 [0.32, 0.52] | 0.441 | 0.428 [0.32, 0.53] | +0.009 | 0.461 |
| random-init | 0.621 [0.51, 0.72] | 0.637 | 0.634 [0.52, 0.73] | +0.013 | 0.631 |

Per-attack-type writer AUROC (MiniLM): naive PoisonedRAG **0.963**, Greshake **0.999**; hardened PoisonedRAG **0.138**, Greshake **0.855**.

→ `figures/fig1_auroc_by_representation.png`, `figures/fig2_per_attack_type.png`

### Controls

| Control | Value | Reads on |
|---|---|---|
| Random-init writer AUROC, **naive** | **1.000** | A1.1 — naive labels are lexically trivial |
| Random-init writer AUROC, **hardened** | 0.621 | A1.1 — residual signal is the Greshake URL |
| Shuffled-retriever combined (neural, naive) | 0.932 | A3.6 — "combined" gain is not from the retriever |

### Phase 1 / 4 — pipeline attack-success (naive corpus, worst-case reader)

| Metric | Value |
|---|---|
| Poison retrieval rate (top-4) | **1.00** |
| Naive-reader surface rate | 1.00 |
| Attack-success (retrieved ∧ surfaced) | 1.00 |
| Clean-context answer accuracy | 1.00 |
| **Phase-4 gate A2.1 (≥0.30)** | **PASS** |

### Direction 2 — retrieval amplification (MiniLM, poisoned ÷ clean per feature)

Median **0.989**, p90 1.251; 13.3% of features amplified >1.2×, 8.1% suppressed <0.8×. → `figures/fig3_amplification.png`

---

## Five findings (and why each matters for the GPU run)

1. **Attack-*presence* labels are lexically confounded.** The random-init control — a representation with **no trained features** — scores **AUROC 1.000** on the naive corpus. So MiniLM's near-perfect 0.98 there is *not* evidence of a deception feature; it is evidence the label is decodable from surface tokens. **Heap et al. 2025 (A1.1) confirmed in proxy.** → On the real run, every SAE-probe AUROC must be reported next to a random-init Gemma control, and the naive attack style must not be used for headline numbers.

2. **A frozen representation cannot detect hardened factual poisoning.** On hardened PoisonedRAG (single answer-token swap), MiniLM lands at **0.138** — indistinguishable from chance once you account for sign, and the overall hardened writer AUROC is **0.497 [0.39, 0.60]**. There is no surface cue for *which* answer is the lie, and a frozen pooled embedding has no access to the model's own belief. → **This is the single most important steer:** detecting PoisonedRAG-style lies needs either the writer's *generative* behaviour (does it emit the falsehood?) or *internal/causal* signals — budget Phase 6's ablation and attack-*success* labels (A2.3) accordingly; do not expect a linear probe on frozen features to do it.

3. **Indirect prompt injection stays detectable; factual poisoning does not.** Hardened Greshake remains at **0.855** (MiniLM) because an injected instruction is intrinsically out-of-place text — but random-init also catches it (1.000), so a chunk of that is still surface. → Treat "injection detection" and "factual-deception detection" as **two different problems** in the writeup; lumping them as "deception" would overclaim.

4. **"Combined" never beats writer-only — the README Direction-1 falsifier fires.** combined − writer is +0.00 to +0.05 across **all** representations and both styles, every CI overlapping; the shuffled-retriever control stays high (0.932). In the proxy the writer-view already contains the retrieved text, so retriever features are redundant — a concrete demonstration of the **A3.2 confound** ("both agents see the same poisoned text" ≠ transfer). → The cross-agent claim **cannot** rest on correlational combined-vs-writer AUROC; it must rest on the Phase-6 causal ablation. The harness for that comparison now exists and is validated.

5. **Retrieval is ~neutral in representation space.** Amplification ratios cluster at 1.0 (median 0.989) with a modest amplified tail (13% of features >1.2×). The README Direction-2 "neutral" falsifier is **largely supported** in the proxy, with a tail worth checking on real sparse SAE features (where amplification should be far more legible than in a dense embedding).

---

## Decision rules → assumption audit updates

| Assumption | Pre-registered falsifier | Outcome here (proxy) | Action |
|---|---|---|---|
| **A2.2** 120 ex. enough for AUROC | bootstrap CIs overlap | CIs are ±0.05–0.11 wide; small gaps (e.g. +0.016) are **inside noise** | Keep CIs on every number; treat sub-0.05 deltas as null. Consider n≥300 for Direction 1. |
| **A1.1** SAEs/probes find real features | random-init still separates | **Triggered** (1.000 naive) | Random-init Gemma control is now **mandatory**, not optional. |
| **A2.3** attack-presence is a clean label | presence vs success features disjoint | Presence is partly lexical (finding 1) and undecidable when hardened (finding 2) | Switch the headline label to **attack-success**; keep presence only as a lexical-baseline. |
| **A3.2 / A3.6** shared-feature = transfer | shuffled-input ≈ true-input | combined≈writer; shuffled-combined≈combined → **transfer not shown** | Direction 1 must use the **causal ablation**; correlational arm is a baseline only. |
| **A2.1** attacks transfer to the index | success ≥0.30 | **PASS** (retrieval 1.00) | Proceed; re-measure with the *generative* writer on GPU (this 1.00 is a worst-case reader bound). |
| **Dir-2 falsifier** | ratios ≈1.0 | Largely **supported** (median 0.99) | Re-test on sparse SAE features before concluding. |

---

## Calibration retrospective

What a pre-registered prediction would likely have said vs. what happened:

- *Expected:* MiniLM clearly beats random-init, showing a "semantic" deception signal. *Actual:* on naive attacks **both** hit ~1.0 — the control was more informative than expected, and the naive design is weaker than it looked. Lesson: build the lexical control **first**, before trusting any AUROC.
- *Expected:* hardening costs maybe 0.1–0.2 AUROC. *Actual:* it collapses PoisonedRAG to chance (Δ≈0.83). The factual-poisoning case is **far** harder for frozen features than anticipated — the most valuable thing this cheap run surfaced.
- *Expected:* combining agents gives a measurable lift. *Actual:* zero lift, cleanly explained by the shared-text confound. The Direction-1 framing needed the causal test all along; now we know before paying for it.

---

## Limitations (read these before quoting any number)

- **Proxy, not Gemma.** MiniLM pooled embeddings ≠ Gemma residual stream ≠ Gemma Scope sparse features. Magnitudes (finding 2) and amplification (finding 5) may look materially different on sparse SAE features. These results bound the *method*, not the *phenomenon*.
- **Worst-case reader.** The pipeline's 1.00 attack-success uses a non-robust deterministic reader; it is an upper bound, not the generative writer's behaviour.
- **Small, templated corpus.** 60 facts, two templates. Hardened PoisonedRAG's false-answer vocabulary is disjoint from clean (a mild residual signal TF-IDF can exploit: hardened TF-IDF 0.738 on retriever view). Scale and vary templates before publication.
- **No SAEs were harmed.** Auto-interp, JumpReLU loading, and transformer-lens parity (A4.4/A4.5/A2.6) are untested here by construction — they live on the GPU path.

---

## Addendum 2026-06-12 — Phase 6 causal harness (proxy run)

The harness the A3.2 row above demanded is now built (`src/sae_deception/interp/ablate.py`, run via
`python scripts/run_phase6.py`; full numbers in `phase6_ablation.json`). Two arms, MiniLM proxy:

**Arm 1 — context-swap intervention** (do() on the retriever→writer channel, naive corpus, 60 queries
with poison retrieved): swapping the poisoned doc for its clean counterpart removes the attack
behaviour in **59/60** cases (necessity 0.98) and inserting it creates the behaviour in **59/60**
(sufficiency 0.98) — the non-flips are a benign-token confound (a false answer occurring truthfully
in a distractor doc, e.g. "nitrogen"). The writer deception-probe score **fell on 100% of queries**
under the swap (mean −0.30): on this proxy the probe reads the channel, not incidental text.

**Arm 2 — feature ablation** (shared-axis transfer test, k=32 of 384 dims): mean-ablating *in the
writer* the dims identified by the *retriever* probe costs the writer probe **~4.7σ more AUROC than
ablating random dims** (naive: −0.0117 vs −0.0015±0.0022; hardened: −0.0350 vs −0.0015±0.0077), and
the writer/retriever top-32 sets overlap in **13–14 dims vs 2.67 expected by chance**. On this proxy,
the two agents read the deception signal off substantially shared feature axes — the precondition
for the README's Direction-1 "structurally similar feature" claim.

*Caveats:* the hardened baseline writer AUROC is ≈0.50 (chance, consistent with finding 2), so the
hardened ablation drop perturbs a probe that wasn't detecting deception to begin with — report it as
a mechanism check only. "Dims" here are dense MiniLM axes, not sparse SAE latents; the GPU run
repeats both arms on Gemma Scope features, where mean-ablation becomes feature clamping and the
overlap statistic becomes interpretable per-latent.
