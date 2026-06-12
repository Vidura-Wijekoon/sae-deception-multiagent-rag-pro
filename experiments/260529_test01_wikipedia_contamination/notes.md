# Test 01 — Wikipedia retrieval contamination

**Date:** 2026-05-29
**Phase:** 0 (env setup done; pipeline not yet built)
**Tests:** Assumption [A2.4 — Wikipedia is a representative retrieval domain](../../docs/ASSUMPTIONS.md)
**Cost estimate:** 30 minutes (5 min script time + 25 min reading results and writing up)
**Pre-registered prediction:** see below — written before running the test.

---

## Why this test, and not one of the other top-5 cruxes?

The top 5 project-threatening assumptions, scored on **stakes × uncertainty × testability-right-now**:

| # | Assumption | Stakes | Uncertainty | Testable at Phase 0? | First-test candidate? |
|---|---|---|---|---|---|
| A3.2 | Shared feature = transfer | very high | high | No — needs Phase 6 pipeline | No |
| A1.5 | Attribution graphs across agents | very high | high | No — needs Phase 6 pipeline | No |
| A2.2 | 120 examples enough for AUROC | high | low (it's statistics) | N/A — methodology commitment, not a test | No |
| A2.4 | Wikipedia is a representative corpus | high | medium-high | **Yes — needs only the model** | **Yes** |
| A1.1 | SAEs find real features | high | medium | Partially — needs SAE loaded (~1 hour setup) | Maybe later this week |

A2.4 is the only top-5 crux I can test in 30 minutes with what's already installed (Gemma 2-2B). Of the cheap-to-test options, it's also the one whose answer **blocks** the next phase: Phase 1 step 10 picks the retrieval corpus. If Wikipedia is contaminated, I need to swap it out *before* building the FAISS index, not after.

The other top-5 cruxes either need the pipeline I'm about to build (A3.2, A1.5) or are methodological commitments rather than empirical tests (A2.2). A1.1 (SAEs on random-init) is a strong second choice but requires loading an SAE first, which adds an hour and doesn't block anything I'm doing this week.

**Decision: test A2.4 first.**

---

## What I'm testing (operationalised)

**Hypothesis:** Gemma 2-2B can answer most PoisonedRAG-style factual questions WITHOUT retrieved context, because Wikipedia is heavily in pretraining.

**Concrete test:**

1. Run 20 hand-picked factual questions through Gemma 2-2B with empty context (no retrieved documents, no system prompt steering it to use external knowledge).
2. Substring-match the generation against a gold answer (case-insensitive).
3. Report per-tier accuracy (famous / general / long-tail) and overall.

**Why this is the right operationalisation:**

- 20 questions is enough to *direction* the decision (≥80%, 50–79%, <50% bands are wide enough that 20-question noise stays inside the bands).
- Substring match is a noisy classifier but biased toward false-positives (gives Gemma every benefit of the doubt). If even that lenient measure says "Gemma can answer most without context," Wikipedia is contaminated.
- Questions span three difficulty tiers so I can see WHERE the contamination concentrates — useful for designing the Phase 4 attack set.

---

## Predictions (pre-registered, written before running)

### Overall accuracy

**Point estimate: 11 / 20 = 55%.**

**Reasoning:**

- Gemma 2-2B is small but pretrained on a Wikipedia-heavy corpus.
- The PoisonedRAG paper (Zou et al. 2024 Table 3) reports Llama 2-7B at ~30% no-context accuracy on Natural Questions. Gemma 2-2B is smaller but newer with better data — I expect a similar order of magnitude.
- My questions are biased toward shorter, more famous facts than the NQ distribution (because I picked from memory), so I expect my number to be HIGHER than the PoisonedRAG benchmark by roughly 15–25 percentage points.
- 30% + 25% ≈ 55%.

### Per-tier predictions

| Tier | n | Predicted correct | Reasoning |
|---|---|---|---|
| Famous facts (capital of France, etc.) | 5 | 5 (100%) | These are in every training corpus. Anything less than 5/5 would mean the model is broken, not contaminated. |
| General knowledge (Mona Lisa, longest river, etc.) | 10 | 5–6 (50–60%) | Gemma 2-2B is small; some general-knowledge answers will be confused or partial. |
| Long-tail (Burkina Faso capital, 2019 Nobel Lit, etc.) | 5 | 0–2 (0–40%) | These are the hardest. A 2B model often hallucinates here. |

### Confidence

Medium. I'd be surprised by < 8 or > 16 out of 20 overall, but anything in 8–16 would feel within model-noise.

### What would surprise me

- **Overall ≥ 17/20.** Would mean Gemma 2-2B is much more knowledge-dense than I think.
- **Overall ≤ 5/20.** Would mean it's much weaker than I think OR the substring-match judge is too strict.
- **Long-tail tier ≥ 4/5.** Would mean my "long tail" picks aren't actually long-tail — Gemma already knows them.
- **Famous-fact tier ≤ 3/5.** Would mean something's wrong with the prompt format or generation parameters, not with Gemma.

---

## Decision rule (what would change course)

| Overall accuracy | Action |
|---|---|
| **≥ 80%** | Wikipedia is too contaminated. **Swap retrieval corpus** before Phase 1 step 10. Candidates: post-cutoff CommonCrawl, recent news (BBC + AP from 2025), or a synthetic domain (made-up company knowledge bases). This is a 4-hour redesign of Phase 1. |
| **50–79%** | Mixed. Stick with Wikipedia but **filter the question set in Phase 4**: only use questions Gemma gets wrong without context. Reduces the usable attack-set size by 50–70%, so I'll need to over-collect (200 candidates → 120 usable). |
| **< 50%** | Wikipedia is fine. Direction 2's contamination confound is minor. Proceed with Phase 1 as planned. Note the result in the writeup as a methodological strength. |

A second, lower-priority signal I'll watch: per-tier accuracy. If famous-tier ≥ 80% and long-tail ≥ 50%, the contamination is broad and the swap is required. If only famous-tier is high and the rest is < 30%, the contamination is concentrated and I can mitigate by filtering rather than swapping.

---

## What I'll learn either way (the test pays for itself)

Even if my prediction is exactly right (55%), I learn:

1. *Where* the contamination concentrates (which question types are unsafe to use).
2. A calibration data point on my own intuition for what a 2B model knows.
3. A reusable baseline number for the writeup (no-context accuracy ≈ retrieval-floor for any attack-success measurement).

---

## Results

(To fill in after running. Compare to predictions, write up what surprised me, and the actual action taken.)

```
[run the script, paste results here]
```

### Calibration retrospective

(Fill in after results. Per the methodology in TurnTrout's predictions-for-shard-theory post: where was the prediction off? What does that tell me about my prior for this domain?)

---

## Next test (after this one)

Conditional on the result:

- If accuracy ≥ 80%: Test 02 becomes "compare Wikipedia vs CommonCrawl-2025 no-context accuracy" — picking the swap target.
- If accuracy 50–79%: Test 02 becomes "what's the cheapest classifier that filters NQ questions to 'Gemma-doesn't-know'?"
- If accuracy < 50%: Test 02 becomes A1.1 (SAEs on random-init Gemma) — the next-highest-stakes crux.
