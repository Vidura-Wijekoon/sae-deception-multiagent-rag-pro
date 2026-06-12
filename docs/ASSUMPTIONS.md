# Assumptions audit

26 load-bearing assumptions across 5 categories. Each entry:

- **Claim** - what we are assuming
- **Why it might be false** - concrete prior art or mechanism
- **Falsifier** - the cheapest check that would tell us we're wrong
- **Impact if wrong** - does the whole project die or just one Direction?

Date: 2026-05-29. Re-audit at end of every phase.

## The top 5 (project-threatening if false)

Ranked by P(wrong) × project-level impact. These are the five that determine whether the project produces a real result vs a hedged null vs nothing publishable.

1. **A3.2 - A feature firing in both agents implies cross-agent transfer.** Kills Direction 1's main claim if false. Test: Phase 6 step 70 causal ablation.
2. **A1.5 - Attribution graphs extend across agent boundaries.** Kills Direction 1's methodological contribution if false. Test: 2-layer hand traced attribution graph on 3 examples, Phase 6 step 68.
3. **A2.2 - 120 examples is enough for AUROC precision.** Invalidates every quantitative claim if false. Test: bootstrap CI reporting from day one of Phase 5.
4. **A2.4 - Wikipedia is a representative retrieval domain.** Confounds Direction 2 (and partially Direction 1) if false. Test: 20-query no-context ablation, ~30 minutes, before building the FAISS index.
5. **A1.1 - SAEs find features that genuinely exist in the model.** Foundational; invalidates SAE story if false. Test: Phase 5 step 61 random-init control.

---

## Re-audit 2026-06-05 - post-Phase-1 de-risk (CPU proxy)

First empirical pass, run on a MiniLM proxy (not Gemma+SAE). Full writeup: `experiments/260605_phase5_probe_derisk/results.md`. Updates to the top-5 and related assumptions:

- **A1.1 - TRIGGERED (proxy).** The random-init representation scored **AUROC 1.000** on the naive attack labels: an untrained feature map separates poisoned from clean perfectly, so high AUROC there reflects lexical structure, not learned features. Action: the random-init *Gemma* control is now mandatory and must sit beside every reported SAE-probe number, and naive-style attacks are barred from headline results.
- **A2.3 - WEAKENED.** Attack-*presence* is a poor label: it is lexically decodable (above) and, once hardening removes the surface cue, *undecidable* from a frozen representation (hardened PoisonedRAG AUROC 0.14 ≈ chance). Action: switch the headline label to attack-*success*.
- **A3.2 / A3.6 - transfer NOT shown (as expected).** Combined (writer+retriever) never beat writer-only (Δ = +0.00…+0.05, all CIs overlapping) and the shuffled-retriever control stayed high (0.93). In the proxy the two agents share the poisoned text, so "shared feature" ≠ "transfer". Action: Direction 1 must rest on the Phase-6 causal ablation; the correlational arm is only a baseline.
- **A2.2 - holds, and bites.** Bootstrap 95% CIs are ±0.05-0.11 wide at n=120; sub-0.05 AUROC gaps are inside noise. Keep CIs on everything; consider n≥300 for Direction 1.
- **A2.1 - PASS (worst-case).** Poison retrieval rate 1.00 (gate ≥0.30). Caveat: measured with a non-robust reader proxy; re-measure with the generative Gemma writer.
- **Direction-2 "neutral" falsifier - largely supported (proxy).** Amplification ratios cluster at 1.0 (median 0.99); re-test on sparse SAE features before concluding.

---

## Category 1 - Existing work

### A1.1 SAEs find features that genuinely exist in the model

**Why it might be false.** [Heap et al. 2025](https://arxiv.org/abs/2501.17727) showed SAEs produce "interpretable" features on randomly-initialised transformers - the interpretability may partly be an artefact of the data plus the SAE inductive bias.

**Falsifier.** Phase 5 step 61 - run the entire SAE-probe pipeline on a random-init Gemma 2-2B. If our deception-correlated features still appear, the result isn't about the trained model.

**Impact if wrong.** Direction 1's central methodological claim collapses. We pivot to comparing raw-activation probes against SAE probes head-to-head and reporting the delta honestly.

### A1.2 SAE features are stable across training seeds

**Why it might be false.** [Paulo & Belrose 2025](https://arxiv.org/abs/2501.16615): two SAEs trained on identical data with different seeds learn substantially different features. "The deception feature" may not be a stable concept.

**Falsifier.** If `sae-lens` exposes two seeds for the chosen Gemma Scope layer, encode the same activation through both and measure top-feature overlap. Aim for ≥30% overlap in top-30 features.

**Impact if wrong.** Qualitative claims about specific features ("feature 4287 fires on hidden-instruction-following") need to be downgraded to "an SAE feature with description X fires." Probe-level AUROC results are still valid.

### A1.3 The Sharkey+ 2025 "multi-agent settings are an open problem" framing is still accurate

**Why it might be false.** Lindsey/Ameisen team at Anthropic is actively pushing on attribution graphs at scale and could release a multi-agent extension before we do. Apollo and Goodfire are also in this space.

**Falsifier.** Set Google Scholar alerts for Lindsey, Ameisen, Marks, Heimersheim, Cunningham. Quarterly arxiv-sanity search for `("multi-agent" OR "cross-agent") AND ("sparse autoencoder" OR "attribution graph")`.

**Impact if wrong.** Reframe as "replication and extension" rather than "first." Still publishable but smaller claim.

### A1.4 MacDiarmid+ 2024's AUROC > 0.99 is a real ceiling, not an artefact of the sleeper-agent setup

**Why it might be false.** Their sleeper-agent task is unusually clean (binary backdoor with a hard trigger). [Goldowsky-Dill+ 2025](https://www.lesswrong.com/posts/9pGbTz6c78PGwJein/detecting-strategic-deception-using-linear-probes) at AUROC ~0.95 on insider-trading deception is a more realistic baseline.

**Falsifier.** Replicate MacDiarmid on the public sleeper-agent weights before treating 0.99 as the bar to beat.

**Impact if wrong.** Lower the bar from "beat 0.99" to "match or beat 0.95." Less heroic but still meaningful.

### A1.5 Attribution graphs (Ameisen+ 2025) extend meaningfully across agent boundaries

**Why it might be false.** Attribution graphs are causally meaningful within one forward pass. Across agents, the causal chain runs through a discrete bottleneck - the tokens that Agent A emits. The "graph" between agents may degenerate to text overlap.

**Falsifier.** Phase 6 step 70 - ablation experiment. If ablating a candidate transfer feature in the Writer does not reduce attack-success, the cross-agent claim is about correlation, not causation.

**Impact if wrong.** Direction 1's methodological contribution shrinks. We can still report a correlational result, but the "attribution graph extension" framing has to go.

---

## Category 2 - Data

### A2.1 PoisonedRAG + Greshake attacks transfer to a 200-paragraph Wikipedia index

**Why it might be false.** PoisonedRAG was tuned for Natural Questions with a different retriever. Attack-success rates may drop to noise on our toy corpus.

**Falsifier.** Phase 4 step 49 - measure attack-success rate. Threshold: must be ≥30% before proceeding.

**Impact if wrong.** Switch to GCG-tuned poison strings from the PoisonedRAG repo, or scale the corpus to 2,000+ paragraphs.

### A2.2 120 examples is enough to estimate AUROC precisely

**Why it might be false.** Standard error on AUROC with 120 binary examples is roughly ±0.05. A claim like "SAE probe beats raw probe by 0.02" is below detection threshold.

**Falsifier.** Compute bootstrap 95% CIs on every reported AUROC. If they overlap, we don't have a finding.

**Impact if wrong.** Scale to 500-1000 examples. Eats into Direction 2/3 time budget.

### A2.3 "Attack-presence" is a clean training label for the probe

**Why it might be false.** An attack can be present but unsuccessful (the model resists). Training on attack-presence may surface features unrelated to deception.

**Falsifier.** Phase 5 step 53 - train two probes (attack-presence vs attack-success) and compare top features. If the feature sets are disjoint, the labels are measuring different things.

**Impact if wrong.** Reframe the result from "detecting deception" to "detecting attempted manipulation of context."

### A2.4 Wikipedia is a representative retrieval domain

**Why it might be false.** Wikipedia is heavily in Gemma's pretraining. The model may answer correctly from memorisation regardless of retrieval. Features fired by retrieval may differ from features fired by recall.

**Falsifier.** No-context ablation. If Gemma answers ≥80% of clean queries correctly with empty context, the RAG-vs-no-RAG signal is contaminated.

**Impact if wrong.** Direction 2 (does retrieval amplify or suppress) is partially confounded. Mitigation: use a non-Wikipedia corpus (e.g., recent CommonCrawl post-cutoff) for the retrieval index.

### A2.5 Auto-interp labels are accurate enough to trust qualitatively

**Why it might be false.** Bills et al. 2023 reported explainer scores averaging 0.3-0.5. Top labels may be plausible-sounding but wrong on closer inspection.

**Falsifier.** Hand-label 20 top-weighted SAE features; compute agreement with auto-interp labels.

**Impact if wrong.** Qualitative readings of "deception-adjacent features" become unreliable. Only the AUROC numbers stay trustworthy.

### A2.6 Activation captures faithfully represent what the model sees in production

**Why it might be false.** `transformer-lens` may apply different default settings than HuggingFace (e.g., RMSNorm fold, attention mask handling). Captured activations may not match the HF model's outputs to float16 precision.

**Falsifier.** Phase 2 step 23 - verify HF and TL outputs match on 5 fixed prompts. Tolerate < 1e-3 mean absolute error in logits.

**Impact if wrong.** Either fix `transformer-lens` (high time cost) or implement hooks on the HF model directly.

---

## Category 3 - Method

### A3.1 Linear probes on the last-token residual stream capture the relevant signal

**Why it might be false.** Goldowsky-Dill+ uses last-token; MacDiarmid uses mean-pooled. The choice can swing AUROC by ±0.05.

**Falsifier.** Phase 5 step 53 - sweep over (last-token, mean-pool, max-pool, attention-pool) on layer 12.

**Impact if wrong.** Pick the best pooling and re-run. Modest budget hit.

### A3.2 A feature active in both Retriever and Writer implies cross-agent transfer

**Why it might be false.** Both agents may activate the same feature because they see the same poisoned text - not because of any cross-agent mechanism. Coincidence, not transfer.

**Falsifier.** Phase 6 step 70 - causal ablation. If ablating in the Writer drops attack-success and ablating in the Retriever does not (or vice versa), the feature isn't a transfer feature; it's an independently triggered shared feature.

**Impact if wrong.** Direction 1's central claim degrades from "cross-agent transfer" to "shared text triggers shared features." Reframe accordingly.

### A3.3 A two-agent LangGraph is a meaningful proxy for "multi-agent"

**Why it might be false.** Two agents is a pipeline. Collusion phenomena studied in [Ren et al. 2025](https://arxiv.org/abs/2507.14660) emerge in 5+ agent populations with feedback loops. Two agents may not exhibit the dynamic of interest.

**Falsifier.** After de-risk, test a three-agent graph (Retriever → Critic → Writer) and check whether the signal persists. If the answer changes qualitatively, "multi-agent" is overclaim.

**Impact if wrong.** Reframe the project as "cross-component interpretability in RAG agents" rather than "multi-agent." Still publishable, smaller scope.

### A3.4 A single mid-layer SAE (layer 12) captures the deception signal

**Why it might be false.** Templeton+ 2024 found their highest-quality deception features in late layers of Sonnet. Layer 12 of Gemma 2-2B may miss them.

**Falsifier.** Phase 5 step 57 - sweep layers 6, 12, 18, 24 with per-layer AUROC.

**Impact if wrong.** Use multiple layers' SAEs simultaneously. Adds complexity but no extra training.

### A3.5 JumpReLU SAEs (Gemma Scope) are comparable to gated/TopK SAEs in other literature

**Why it might be false.** JumpReLU trades reconstruction quality and sparsity differently than other architectures. Cross-paper feature comparisons may be apples-to-oranges.

**Falsifier.** For one layer, also load a TopK SAE (Karvonen+ 2025 has them in SAEBench) and compare the top discriminative features.

**Impact if wrong.** Caveat cross-paper comparisons in the writeup.

### A3.6 The cross-agent probe AUROC and the within-agent probe AUROC are meaningfully comparable

**Why it might be false.** The two probes are trained on different input dimensions (Retriever's hidden state vs Writer's). Differences in baseline difficulty may swamp the "cross-agent adds info" signal.

**Falsifier.** Train a probe on shuffled-Retriever features as a negative control. If shuffled-input AUROC ≈ true-input AUROC, the cross-agent claim is dead.

**Impact if wrong.** The Direction 1 comparison framework needs redesign. Substantial re-work.

---

## Category 4 - Tools and resources

### A4.1 30 focused hours is enough to reach a publishable result OR null

**Why it might be false.** The 80-step build guide alone estimates ~25-30 hours of focused work; the writeup is another ~10. The first time through any new toolchain (transformer-lens, sae-lens, delphi) costs 2-3× the steady-state estimate.

**Falsifier.** Check pace at hour 12 (should be finishing Phase 3) and hour 20 (should be finishing Phase 5). If off-pace by > 4 hours, drop Direction 2 and 3.

**Impact if wrong.** Direction 1 ships alone. Directions 2 and 3 become "future work."

### A4.2 A single GPU with ≥16 GB VRAM handles Gemma 2-2B + Gemma Scope SAE + activation capture

**Why it might be false.** Gemma 2-2B in float16 is ~5 GB; a layer-12 SAE adds ~1 GB; capturing activations at 6 layers × 1024 tokens × 2304 dims ≈ 30 MB per forward pass, accumulated across batches. Peak VRAM can exceed 16 GB at long contexts.

**Falsifier.** Phase 2 - measure peak VRAM at 1024-token context. If > 14 GB, reduce batch size to 1 or move to a 24 GB card.

**Impact if wrong.** Switch to int8 quantization (may break SAE assumptions) or rent a 24 GB GPU on RunPod / Lambda.

### A4.3 LangGraph's API is stable enough to depend on for 30 hours of work

**Why it might be false.** LangGraph is pre-1.0 with breaking changes between minor versions.

**Falsifier.** Pin to one specific LangGraph version in `requirements.txt` and document it in `experiments/env_pins.txt`.

**Impact if wrong.** Re-write pipeline glue mid-project. ~2-hour cost.

### A4.4 `transformer-lens` supports Gemma 2 cleanly

**Why it might be false.** Gemma 2 uses GQA + RMSNorm + sliding-window attention; transformer-lens's coverage of Gemma 2 may have edge cases.

**Falsifier.** Phase 2 step 23 - HF-vs-TL output parity check on 5 prompts.

**Impact if wrong.** Hook directly on the HF model (more verbose but more reliable).

### A4.5 `sae-lens` correctly handles Gemma Scope's JumpReLU SAEs

**Why it might be false.** `sae-lens`'s JumpReLU loader may have stale assumptions; round-trip reconstruction MSE may not match Gemma Scope's published number.

**Falsifier.** Phase 3 step 34 - round-trip an activation, compare MSE to the model card.

**Impact if wrong.** Load SAE weights manually and re-implement the encoder. ~4 hour cost.

### A4.6 EleutherAI delphi (sae-auto-interp) runs in the available time budget

**Why it might be false.** EleutherAI's blog reports auto-interp on 1.5M GPT-2 features cost $1300 in Llama 3.1 API calls. We only need ~100 features, but each call is 5-30 seconds wall-clock.

**Falsifier.** Phase 3 step 36 - time auto-interp on 5 features end-to-end. If > 10 minutes for 5, scale plan accordingly.

**Impact if wrong.** Skip auto-interp; hand-label 30 features. Costs qualitative depth but not the quantitative AUROC results.

### A4.7 Disk space is sufficient for stored activations and SAE encodings

**Why it might be false.** 50 queries × 2 agents × 6 layers × 500 tokens × 2304 dims × float16 ≈ 700 MB. At 5,000 queries (if we scale): 70 GB.

**Falsifier.** Phase 2 step 28 - measure and project. The V: drive may have hard limits.

**Impact if wrong.** Sub-sample tokens (every 4th). Store on a local SSD rather than the synced V: drive.

---

## Category 5 - Impact

### A5.1 A positive result would change practice

**Why it might be false.** Linear probes ≥0.95 AUROC are already deployable cheaply. The engineering overhead of running SAEs inline in production may make our probe unusable even if it wins on AUROC by 0.05.

**Falsifier.** After Phase 6, calculate the inference-time overhead of SAE encoding on every Writer call. If overhead > 30% of model latency, the result is offline-only.

**Impact if wrong.** Reframe as a tool for offline auditing rather than runtime monitoring. Still useful, narrower deployment story.

### A5.2 A null result is still publishable

**Why it might be false.** Null results in ML are hard to publish at top venues. LessWrong / Alignment Forum accept them but the academic recognition is lower.

**Falsifier.** Search LessWrong + AF + arxiv for "negative results SAE deception" - count posts published in the last year. If < 5, our null result has a place.

**Impact if wrong.** Reframe as a methodology contribution: "Here's the correct way to test cross-agent transfer; here's what we learn from it failing."

### A5.3 An audience for this work exists

**Why it might be false.** Multi-agent + mech-interp is the smallest of the three Shallow Review 2025 agendas we're bridging - only 10-15 FTEs. The natural audience may be 50 people total.

**Falsifier.** Post the Phase 0 framing on LessWrong / Alignment Forum and measure engagement (karma, comments, citations within a month).

**Impact if wrong.** Lean the writeup more on the mech-interp angle (broader audience) and less on the multi-agent angle.

### A5.4 No competing lab releases the same result in the next 8 weeks

**Why it might be false.** Anthropic Lindsey/Ameisen, Apollo Hobbhahn/Heimersheim, Goodfire - all are pushing on attribution graphs and SAE probes for deception at scale and have more resources than we do.

**Falsifier.** Set Google Scholar alerts for those names. Check arxiv weekly under cs.CL and cs.AI.

**Impact if wrong.** Pivot to "replication on open-weight Gemma" - still valuable, smaller-claim.

### A5.5 The BlueDot Impact cohort and mentor network are the right feedback channel

**Why it might be false.** The BlueDot reviewer network is broad (covers all AI safety) rather than deep on mech-interp specifically. Detailed methodological feedback may need to come from elsewhere.

**Falsifier.** After Phase 7 step 75, get feedback on the 1-page summary from one BlueDot reviewer AND one mech-interp practitioner (e.g., a Neuronpedia / Apollo / Goodfire DM). Compare quality of feedback.

**Impact if wrong.** Cross-post the writeup to the EleutherAI Discord interpretability channel for additional feedback.

---

## How to use this document

- **Pre-Phase-0**: re-read this once, take a single page of notes on which assumptions feel weakest. Those are your week-1 priorities.
- **End of each phase**: open this file, mark assumptions that the phase's results have updated. Convert any falsified ones into followup actions.
- **Pre-publication**: every claim in the writeup should map to an assumption that survived. Anything that depends on a falsified assumption must be removed or hedged.

Re-audit dates: 2026-06-05 (post-Phase-1), 2026-06-12 (post-Phase-3), 2026-06-19 (post-Phase-5), 2026-06-26 (post-Phase-7).
