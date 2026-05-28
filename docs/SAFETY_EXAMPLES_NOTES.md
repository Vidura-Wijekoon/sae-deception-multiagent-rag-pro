# Notes on the safety-examples lineage

This file records what we took from
[`safety-research/safety-examples`](https://github.com/safety-research/safety-examples)
when scaffolding this repo, what we left out, and the reasoning — so future-you
(and reviewers) can re-evaluate the decisions without re-reading the upstream
repo.

Upstream commit at time of fork: `main` as of 2026-05-28.
Upstream release tag we tracked: **v1.0.0** (8 May 2025).

---

## What we cherry-picked (the load-bearing pieces)

| File | Why | Local change |
|---|---|---|
| `.pre-commit-config.yaml` | Project lead Ethan Perez's MATS workflow notes explicitly call this out as load-bearing for catching bugs that would otherwise burn days of compute. | `no-commit-to-branch --branch=main` hook is **commented out** during de-risk (solo dev, no PR workflow). Re-enable in extended-project mode. |
| `pyproject.toml` | Their ruff + black + setuptools configuration is well-tuned to the safety-tooling style. | Renamed `examples` → `sae_deception`, switched to **src/ layout** (`src/sae_deception/`), added `[project.optional-dependencies.dev]`, added pytest config, added `setuptools_scm` for git-tag versioning. |
| `Makefile` | The `make hooks` target gets the pre-commit + pre-push + post-checkout hooks installed in one shot. | Extended with `make install / dev / test / format / lint / smoke / clean` for project-specific conveniences. The Phase 0 step 5 smoke test is now `make smoke`. |
| `.gitmodules` | Wires in `safety-tooling` as a submodule (LLM API + caching + prompt templating + env setup). | Switched the URL from `git@github.com:...` (SSH) to `https://github.com/...` so users without SSH keys can clone. |
| `requirements.txt` | Their file is essentially empty (`# install safety-tooling first, then add any other dependencies here`). | We wrote our own, pinned to the Phase 0 step 3 dependency list: torch, transformers, accelerate, transformer-lens, sae-lens, langgraph, langchain-core, faiss-cpu, datasets, wandb, diskcache, simple-parsing. Dev tools live in `pyproject.toml`. |

---

## What we left out

| File / dir | Why we skipped |
|---|---|
| `examples/` | Their examples are jailbreaking-focused (Best-of-N, PAIR attack, HarmBench-style classifiers, MMLU evals). None of these match this project's threat model. Worth a one-time read of `examples/inference/get_responses.py` to see their LLM call + caching pattern, but we don't need the code. |
| `experiments/examples/241223_running_examples/` | Concrete experiment scaffold for the jailbreaking examples. Useful as a *template* for our own `experiments/YYMMDD_<name>/` folders but not as code to keep. |
| `prompts/` | Jinja templates for the jailbreaking examples. Re-build our own as needed in `prompts/` later. |
| `LICENSE` | We have our own MIT LICENSE with Vidura as copyright holder. |
| `README.md` | We have our own. |
| `.vscode/` | Their VS Code settings are personal preference. Skip; add your own when needed. |

---

## Why we did NOT fork upstream

The upstream README recommends forking the repo as a template. We chose
**not to fork** for two reasons:

1. **The repo already exists.** `sae-deception-multiagent-rag` was created
   on GitHub before this step. Forking now would mean either renaming the
   existing repo (loses any incoming links and watchers) or maintaining two
   parallel repos.
2. **The four load-bearing files are easy to copy.** Forking buys you a
   git ancestry link to upstream — useful if you intend to upstream PRs
   against the example code, irrelevant if (like us) you only depend on
   the submodule and the config files.

If we ever want to upstream a feature into `safety-tooling`, we do that
**through the submodule itself** — `cd safety-tooling && git checkout -b
my-feature && ...`. The submodule is a real git repo, not a snapshot.

---

## How to update from upstream later

`safety-examples` is a moving target. To re-sync the config files after the
upstream evolves:

```bash
# Pick a date / tag to compare against:
UPSTREAM_TAG=v1.1.0   # whatever the new release is

# Diff our local files against upstream:
for f in .pre-commit-config.yaml pyproject.toml Makefile; do
  curl -s "https://raw.githubusercontent.com/safety-research/safety-examples/${UPSTREAM_TAG}/${f}" | \
    diff - "$f" | head -50
done
```

Then cherry-pick by hand. Don't do a wholesale overwrite — our local
deviations (renamed package, commented-out branch hook, src/ layout, our
dependency pins) need to survive.

---

## How to update the submodule

```bash
cd safety-tooling
git fetch
git checkout main
git pull
cd ..
git add safety-tooling
git commit -m "chore: bump safety-tooling to <new-sha>"
```

Pin to a specific SHA (not `main`) once you have results worth reproducing.

---

## Reference

- safety-examples: https://github.com/safety-research/safety-examples
- safety-tooling: https://github.com/safety-research/safety-tooling
- Ethan Perez et al.'s workflow notes (the source of these patterns):
  Part 4 of the BlueDot Impact "30hr Open Weight Safety Projects" guide.
