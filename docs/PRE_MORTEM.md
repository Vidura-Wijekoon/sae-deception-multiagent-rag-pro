# Project pre-mortem

> Imagine you didn't submit the project by the end of the sprint. What's the most likely reason why? What can you do now to prevent that?

Date written: **2026-05-29** (Phase 0, before any experiments).
Imagined failure date: **2026-06-26** (end of sprint).

This document is **different from `ASSUMPTIONS.md`**. The assumptions audit is about *empirical* failures — claims that turn out false. This pre-mortem is about *process* failures — reasons the project doesn't ship even when the empirical work is fine. They overlap less than people expect; most papers that don't ship aren't killed by null results, they're killed by time, scope, and discipline.

---

## The imagined failure (one paragraph, written in past tense)

> It's 2026-06-26. The BlueDot sprint ended yesterday. I didn't submit. The repo has 47 commits, two working notebooks, a 60% finished LangGraph pipeline, and a directory called `experiments/260618_layer_sweep_v3` with TODO notes I never came back to. The literature survey doc is the most polished thing I produced. I have a draft LessWrong post sitting in `docs/draft_post.md` that's 800 words of intro and no results. The most embarrassing detail: I never actually ran a probe. I spent week 1 setting up the env, week 2 fighting transformer-lens compatibility with Gemma 2, week 3 redesigning the LangGraph pipeline twice, and the last weekend was lost to a `sae-lens` API change I didn't catch. The substance never happened.

That's the failure mode I'm pre-empting. The seven specific paths to that outcome are below.

---

## The seven most likely failure paths, ranked by probability

For each: what it looks like, the **early signal** that tells me it's happening, the **mitigation** I can lock in now, and the **fail-fast pivot** if it happens anyway.

### F1. Toolchain rabbit hole eats week 1

**The story.** I spend three days debugging `transformer-lens` vs HuggingFace output parity on Gemma 2. The fix turns out to be a one-line config change but I find it on day three, not day one. Phase 1 starts a week late.

**Probability:** 60%. This is the modal failure mode for any "fork a research framework" project. Pre-1.0 ML libraries break in unobvious ways.

**Early signal.** End of day 2 of Phase 2, the HF↔TL parity check still fails. By default I would push through; the rule is: if it's not passing by end of day 2, stop debugging it directly and switch to plan B.

**Mitigation (commit to now).**
- Hard time-box Phase 2 (activation capture) at **8 hours total wall-clock**. After 8 hours, if `transformer-lens` isn't working, abandon it and use raw HuggingFace forward hooks (more verbose but stable).
- Before starting Phase 2, write a one-paragraph "what would 'good enough' look like" — set the bar at "I can extract a residual-stream tensor for an arbitrary token in either agent," not "the activations match HF to float16 precision."

**Fail-fast pivot.** Bypass `transformer-lens` entirely. Hook on the HuggingFace model with `register_forward_hook`. ~3 hours of work, fully under my control, no compatibility surprises.

---

### F2. Pipeline over-engineering eats week 2

**The story.** I keep "improving" the LangGraph code — generic node decorators, fancy state types, retry logic. Two weeks in, the pipeline is elegant but I still haven't run a probe.

**Probability:** 45%. Especially likely because I have a software-engineering background and the LangGraph API is fun to play with.

**Early signal.** End of Phase 1 (day 5–6), `run_pipeline.py` has more than 250 lines of Python or imports more than 10 internal modules. If yes, I am gold-plating.

**Mitigation.**
- The Phase 1 pipeline ships when it can produce a generation for one query in <60 seconds. Not before. Not after.
- Forbid abstractions until they are duplicated three times. Copy-paste is allowed during de-risk; refactoring is forbidden until Phase 7 (extended mode).
- Set a daily "is this getting me closer to running a probe?" question at end of day. If two consecutive days are "no," stop and reduce scope.

**Fail-fast pivot.** Throw away the pipeline code, write a 50-line script that calls Gemma twice in sequence (faking the "retriever" and "writer" steps with hand-built prompts), and run probes on those. Direction 1 with a fake pipeline beats Direction 1 with a beautiful pipeline that never runs.

---

### F3. The writeup never gets written

**The story.** I have results by week 3 but tell myself "one more experiment" and never start the LessWrong post. End of sprint arrives with two paragraphs of intro and no submission.

**Probability:** 50%. The single most common failure mode for research projects, especially when the result is null or partial.

**Early signal.** Today is 2026-06-15 and I have not yet written a single bullet of the writeup. Or: I have started 3 experiments since last writing in the doc.

**Mitigation.**
- **Write the writeup outline today, before any experiments run.** Bullet points, no prose. Sections: motivation, method, results (with empty placeholders), interpretation, what would change my mind.
- Commit to a "write 200 words per phase" rule. End of each Phase N, write 200 words into `docs/draft_post.md` describing what happened. By end of Phase 6 that's 1,200+ words of raw material.
- The LessWrong post does not need new framing — copy from the literature survey doc directly.
- **Publish the writeup, even if results are null.** A clean negative result with a clear setup is more valuable than no result at all.

**Fail-fast pivot.** If by 2026-06-19 the writeup is still under 500 words, freeze experimentation and spend the remaining time only on writing. A 1,200-word "here's what I tried and what I learned, including what didn't work" post is publishable. A perfect experiment with no writeup is not.

---

### F4. Scope creep across the three Directions

**The story.** I try to make progress on all three Directions simultaneously instead of completing Direction 1 first. End of sprint: 30% done on three things, 0% done on anything.

**Probability:** 40%. Project doc already names three Directions, which is itself a scope-creep risk.

**Early signal.** I find myself reading Llama 2 SAE papers (Direction 3) before Direction 1's first AUROC number is on disk.

**Mitigation.**
- **Direction 1 ships first.** No work on Direction 2 or 3 until Direction 1 has a per-layer AUROC plot saved as a PNG. This is non-negotiable.
- Move Direction 2 and 3 into a "Future work" section in the README NOW so they have a home and stop appearing in my todo list.

**Fail-fast pivot.** If by end of week 3 (2026-06-19) Direction 1 doesn't have an AUROC plot, drop Direction 2 and 3 from the writeup entirely. Re-frame as "we tested cross-agent transfer for prompt injection; here is what we found." Smaller scope, real result.

---

### F5. Data prep eats all the time

**The story.** Phase 4 (attack set assembly) takes 12 hours instead of 4. PoisonedRAG attacks don't transfer to my 200-paragraph index, I hand-craft replacements, they don't work either, I keep iterating on attack quality instead of moving on.

**Probability:** 35%. This is the Phase 4 risk explicit in the build guide.

**Early signal.** End of day 12 (mid-week 3), attack-success rate is still < 10%.

**Mitigation.**
- Test 01 (Wikipedia contamination — being run today) gives an early read on whether the corpus choice is sound. If it requires a swap, do the swap before building the FAISS index.
- Time-box Phase 4 at **6 hours**. If after 6 hours attack-success is < 30%, accept whatever you have and move on. Document the lower bound and note it as a limitation.
- Use PoisonedRAG's pre-tuned GCG strings rather than hand-crafting attacks.

**Fail-fast pivot.** If attacks don't fire on Gemma 2-2B, switch to the Hubinger sleeper-agent weights (Mistral-7B variant). The attack is "the trigger token is in context" rather than "this poisoned doc was retrieved" — different threat model but a working pipeline.

---

### F6. The result is "nothing happens" and I lose motivation

**The story.** Phase 5 produces AUROC ≈ 0.55 for SAE probes (barely above chance). Direction 1 looks dead. I lose enthusiasm and don't push through to Phase 6 or the writeup.

**Probability:** 30%. Genuinely possible. Null results are demotivating in a way that's irrational but real.

**Early signal.** Phase 5 results land. I feel embarrassed to share them with the BlueDot cohort. I start "taking a break" from the project for "just a few days."

**Mitigation.**
- Pre-commit to publishing the result regardless of direction. Write this commit into the LessWrong draft NOW, in the first paragraph: "I expect to publish whatever I find, including null results. Here is why."
- The MacDiarmid+ 2024 and Heap+ 2025 papers both lean on negative-results framing — read them before Phase 5 to internalise that null results are publishable and useful.
- Tell my BlueDot mentor in week 1 that I plan to publish either way. The social commitment makes backsliding harder.

**Fail-fast pivot.** Reframe the project as a methodology contribution. "We applied SAE probes to a multi-agent RAG setting. They didn't find a useful signal. Here is what we learn from the failure, and what to try instead." This is genuinely useful for the field.

---

### F7. External blockers (mentor unavailable, hardware fails, life happens)

**The story.** GPU dies at hour 15. Or mentor is on vacation when I'm stuck on the most important question. Or a work crunch eats a week.

**Probability:** 25%. Smaller per-event probability but the menu is long.

**Early signal.** Any single day where I spend more than 4 hours blocked on something I can't unblock myself.

**Mitigation.**
- **GPU plan B** — set up a RunPod account ahead of time. Test it once. ~30 minutes; pays off if hardware fails.
- **Mentor backup** — identify two non-mentor channels for technical questions: the EleutherAI Discord interpretability channel, and the Open Source Mech Interp Slack. Join both this week.
- **Schedule buffer** — block 4 hours of "buffer time" per week that's unscheduled, to absorb life events.
- **Cloud backup** — `git push` at end of every day. Activations are gitignored but should be backed up to a cloud drive nightly via a cron job.

**Fail-fast pivot.** If life genuinely eats a week, cut Directions 2 and 3 immediately and salvage Direction 1 from whatever exists.

---

## Weekly fail-fast schedule

Each Friday, 30 minutes. Open `docs/PRE_MORTEM.md`, run through the checklist. If any "yes" answer, the corresponding mitigation triggers.

### Week 1 (2026-05-29 → 2026-06-05)

- [ ] Env installed + smoke test passing? *If no by Tuesday, contact mentor.*
- [ ] Test 01 (Wikipedia contamination) run and notes written? *If no, the predict-before-test discipline isn't being followed.*
- [ ] LessWrong draft has at least an outline? *If no, F3 risk is building.*
- [ ] RunPod account configured as backup? *Mitigation for F7.*
- [ ] Mentor briefed on "I'll publish either way" commitment? *Mitigation for F6.*

### Week 2 (2026-06-05 → 2026-06-12)

- [ ] Phase 1 (LangGraph RAG) shipping in <60s per query? *If no by Tuesday, F2 (over-engineering) is happening.*
- [ ] Phase 2 (activation capture) working? *If still debugging by Thursday, switch to raw HF hooks per F1.*
- [ ] Daily commit happening? *If three days in a row with no commit, F2 or F1 is happening.*
- [ ] Have I written into `docs/draft_post.md` this week? *200-word minimum.*

### Week 3 (2026-06-12 → 2026-06-19)

- [ ] Phase 4 attack-success rate ≥ 30%? *If no by Wednesday, accept and move on (F5).*
- [ ] Phase 5 has at least a first AUROC number? *If no by Friday, drop Directions 2 and 3 (F4).*
- [ ] LessWrong draft over 500 words? *If no, freeze experiments per F3.*
- [ ] Reading any paper not directly load-bearing for this week's work? *If yes, that's F4 (scope creep).*

### Week 4 (2026-06-19 → 2026-06-26)

- [ ] **No new experiments.** Only writeup work. (Hard rule, no exceptions.)
- [ ] LessWrong post drafted by Tuesday?
- [ ] Mentor review by Thursday?
- [ ] Submitted by Friday end of day?

---

## Minimum shippable artefact (the safety net)

If absolutely everything goes wrong — empirical results are null, pipeline never quite works end-to-end, two Directions get dropped — what do I still ship?

**The minimum.** A LessWrong / Alignment Forum post titled something like *"What I learned trying to probe deception in multi-agent RAG with SAEs"*. Roughly 1,500 words. Sections:

1. The motivation (lifted directly from the literature survey, no new writing).
2. The setup (whatever pipeline did work, even if degraded).
3. What I tried and what happened — one paragraph per attempted experiment, including the failures.
4. What I learned about the methodology (whether the approach is viable, what the obstacles are, what I'd try differently).
5. The code (link to the repo, even if incomplete).

**This counts as shipping.** Negative results with a clean writeup are more valuable to the field than no writeup. The `docs/draft_post.md` file gets opened today; the outline gets bullets today. Even with zero further experiments, the minimum shippable artefact is achievable.

---

## Pre-commitments (locked in today)

These are decisions I'm making **now** so that future-me, under pressure, can't backslide.

1. **I will publish the writeup by 2026-06-26 regardless of result direction.** Including if the result is "everything was null."
2. **I will keep the experimental log in `experiments/YYMMDD_<name>/notes.md` form, one folder per experiment, with predict-before-run discipline.** No experiment runs without a prediction written first.
3. **I will commit and push at end of every working day,** even if the commit message is "WIP nothing works."
4. **I will not work on Direction 2 or 3 until Direction 1 has a per-layer AUROC PNG saved to disk.**
5. **I will start writing `docs/draft_post.md` today.** Bullet outline, no prose. 200 words of substance added per phase.
6. **I will share the Phase 0 framing on LessWrong as a "research proposal" post in week 1** for early feedback and to lock in the social commitment.
7. **I will time-box rabbit holes at 8 hours.** After 8 hours stuck on one thing, switch to plan B (documented in F1–F7 above).

Re-read this document at the start of each weekly fail-fast review.
