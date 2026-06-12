# Epistemic status — what this de-risk does and doesn't establish

A de-risk's job is to buy **certainty about the machinery and the method cheaply**, while deliberately *not* buying certainty about the phenomenon. The two design choices that make it cheap — a **synthetic corpus** and a **proxy representation** (MiniLM instead of Gemma + Gemma Scope) — are also the two main reasons to hold the numbers loosely. This note separates what I would stake a claim on from what I would not, and is explicit about where the synthetic data helps and where it misleads.

## The single most useful distinction: measurements vs. structural lessons

Every result here is one of two things, and they carry very different confidence:

- A **measurement** — a specific AUROC on this corpus, with this representation. Its *magnitude* is bound to the synthetic data and the proxy, so its external validity is low.
- A **structural lesson** — an argument about the *logic* of the setup that the measurement surfaced. These transfer to the real run with much higher confidence, because they are about the design, not the number.

| Result | Measurement (low transfer) | Structural lesson (high transfer) |
|---|---|---|
| Random-init scores AUROC 1.000 on naive attacks | the 1.000 is corpus-specific | attack-*presence* labels can be lexically trivial → a lexical/random-init control is mandatory beside every number |
| Hardened PoisonedRAG ≈ 0.17 (below chance) | the 0.17 is proxy-specific | a frozen representation of *text* cannot know which plausible answer is the lie → factual-deception detection needs the model's generative commitment or internal/causal signals |
| Combined − writer ≈ +0.01 | the +0.01 is proxy-specific | when two agents share the poisoned text, a shared feature ≠ transfer → Direction 1 needs a causal ablation, not a correlational probe |
| Amplification median ≈ 0.99 | likely an artifact of dense, L2-normalised embeddings | "is retrieval an amplifier?" is only legible on *sparse* features — defer the claim |

The headline of this milestone is a set of structural lessons. The measurements are scaffolding that produced them.

## What I am confident about

**The engineering.** The pipeline runs end-to-end with one command, the corpus is deterministic and hash-versioned, probes and bootstraps are seeded, 5 unit tests pass, and a re-run reproduces every number bit-for-bit. A 10-seed robustness pass confirmed the key quantities are stable (e.g. hardened PoisonedRAG 0.17 ± 0.03, not a single-seed fluke). This is close to as certain as research code gets.

**The descriptive results — as descriptions.** On *this* corpus, with *this* representation, the numbers are what they are and they replicate. Stated as "on a templated 120-example set, a random-init feature map separates the naive attacks perfectly," they are true with high confidence.

**The structural lessons** in the table above. These are arguments, not artifacts of one dataset, so I expect them to survive contact with the real model.

## What I am uncertain about

### Synthetic data — the largest source of uncertainty

The corpus is fully synthetic: 60 hand-written facts, two passage templates, programmatically combined into clean/poisoned pairs. That is a deliberate trade, and it cuts both ways.

Why it limits confidence: the construction creates **lexical artifacts** that a real corpus would not have. The false answers form a vocabulary disjoint from the clean passages ("Lyon" never appears in a genuine France passage), the Greshake payload is a single constant sentinel string, and every passage shares the same length and register. So a probe can "detect" an attack for reasons that have nothing to do with deception — which is precisely why the naive result is uninterpretable, and why even the hardened result may carry residual template signal. Real PoisonedRAG passages are LLM-generated, fluent, varied, and embedded in long multi-topic retrieved documents; real factual poisoning is rarely a clean single-token swap. **The magnitude of every AUROC in this run is corpus-specific and should not be quoted as a property of the phenomenon.** With only 120 examples, the bootstrap CIs are ±0.05–0.11 (assumption A2.2), so small effects are invisible regardless.

Why it was still the right choice: synthetic data is *controlled, balanced, deterministic, versioned, and free*. It let me isolate exactly one variable — lexical signature — by building a naive and a hardened version of the same facts, which is what exposed the confound. Synthetic data is the correct instrument for **validating a harness and falsifying cheap assumptions**; it is the wrong instrument for **final quantitative claims**. The error would be to confuse the two — to report "MiniLM detects injection at 0.83" as a finding rather than as a property of the test set. I am not making that error, and the GPU run must not either.

### Proxy representation

MiniLM pooled embeddings are not the Gemma residual stream and certainly not Gemma Scope's sparse features. The amplification result is most likely a property of dense normalised vectors rather than of retrieval. The factual-poisoning negative is the interesting mixed case: **part of it is representation-specific** (a different model might encode factuality differently — uncertain) and **part of it is structural** (no frozen representation of the text alone can know which answer is true — more certain). I am confident about the structural half and genuinely uncertain about the representation-specific half; that uncertainty is the main thing the GPU run resolves.

### Worst-case reader

The pipeline's attack-success of 1.00 uses a deterministic non-robust reader. It is an upper bound, not the generative writer's real behaviour. As a real attack-success rate it is close to uninformative.

### The actual scientific question

The de-risk **barely moves my belief** about whether cross-agent deception is detectable in real multi-agent RAG. It tells me the *correlational* method can't answer it and which labels/controls are needed — but the phenomenon itself stays open until the causal ablation runs on Gemma. External validity to other scaffolds (AutoGen), other model families (Direction 3), and 5+ agent dynamics is entirely untouched.

## Calibration note

I was initially **over-confident** that a real embedding (MiniLM) would beat the untrained control — it did not, on the naive set, which made the control more informative than expected. I was initially **under-confident** about the below-chance hardened result, flagging 0.14 as "probably noise"; the 10-seed pass (0.17 ± 0.03) upgraded it to a real, reproducible anti-signal. Lesson banked: build the control first, and check seed-stability before deciding whether a surprising number is signal or noise.

## Bottom line

I am confident in the harness, the method, and the structural lessons. I am uncertain about every magnitude and about the phenomenon itself. That split is not a weakness of the run — it is the *point* of it: the synthetic-corpus + proxy-model design was chosen to make the certain things cheap and to defer the uncertain things to the GPU experiment, with the controls in place so the deferred questions get an honest answer when they're finally asked.
