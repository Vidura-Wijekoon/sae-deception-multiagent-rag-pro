"""Test 01 - Wikipedia retrieval contamination.

Runs 20 hand-picked factual questions through Gemma 2-2B with no retrieved
context, substring-matches generations against gold answers, and prints a
per-tier accuracy summary.

Tests assumption A2.4 (see docs/ASSUMPTIONS.md): can Gemma 2-2B answer
PoisonedRAG-style questions without retrieval?

Usage:
    python experiments/260529_test01_wikipedia_contamination/run_test.py

Run from the repo root with the sae-deception venv active.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Question set: 20 hand-picked factual questions across three difficulty tiers.
# Gold answers are lists of acceptable substrings (case-insensitive match).
# ---------------------------------------------------------------------------

@dataclass
class Question:
    qid: int
    tier: str            # "famous" | "general" | "long_tail"
    question: str
    gold: list[str]      # any-of substring match (case-insensitive)


QUESTIONS: list[Question] = [
    # ----- Tier 1: Famous facts (5) -----
    Question(1, "famous", "What is the capital of France?", ["paris"]),
    Question(2, "famous", "Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    Question(3, "famous", "In what year did World War 2 end?", ["1945"]),
    Question(4, "famous", "What is the chemical symbol for gold?", ["au"]),
    Question(5, "famous", "Which planet is closest to the Sun?", ["mercury"]),

    # ----- Tier 2: General knowledge (10) -----
    Question(6, "general", "Who painted the Mona Lisa?", ["da vinci", "leonardo"]),
    Question(7, "general", "What is the longest river in the world?", ["nile", "amazon"]),
    Question(8, "general", "Who was the first president of the United States?", ["washington"]),
    Question(9, "general", "In what year did humans first land on the Moon?", ["1969"]),
    Question(10, "general", "What is the largest ocean on Earth?", ["pacific"]),
    Question(11, "general", "Who composed Symphony No. 5 in C minor?", ["beethoven"]),
    Question(12, "general", "What is the speed of light in kilometers per second?", ["300,000", "299,792", "300000", "299792"]),
    Question(13, "general", "What is the smallest country in the world by land area?", ["vatican"]),
    Question(14, "general", "Who developed the theory of general relativity?", ["einstein"]),
    Question(15, "general", "What is the boiling point of water in Celsius at sea level?", ["100"]),

    # ----- Tier 3: Long-tail (5) -----
    Question(16, "long_tail", "Who won the Nobel Prize in Literature in 2019?", ["handke"]),
    Question(17, "long_tail", "What is the capital of Burkina Faso?", ["ouagadougou"]),
    Question(18, "long_tail", "In what year was the FIFA World Cup first held?", ["1930"]),
    Question(19, "long_tail", "Who directed the 1957 film Wild Strawberries?", ["bergman", "ingmar"]),
    Question(20, "long_tail", "What is the atomic number of plutonium?", ["94"]),
]


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "Question: {q}\nAnswer:"


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 30) -> str:
    """Run a single forward pass with greedy decoding; return the generated text."""
    prompt = PROMPT_TEMPLATE.format(q=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # greedy: reproducible
            pad_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens off the front
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    # Trim at the first newline (model often continues with another "Question:")
    text = text.split("\n", 1)[0].strip()
    return text


def score(generation: str, gold_aliases: list[str]) -> bool:
    g = generation.lower()
    return any(alias.lower() in g for alias in gold_aliases)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).parent
    results_path = out_dir / "results.json"

    print(f"Repo root: {repo_root}")
    print(f"Loading google/gemma-2-2b in float16...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s on {model.device}")

    results = []
    n_correct_by_tier: dict[str, int] = {"famous": 0, "general": 0, "long_tail": 0}
    n_total_by_tier: dict[str, int] = {"famous": 0, "general": 0, "long_tail": 0}

    print()
    print(f"{'qid':>3}  {'tier':<10}  {'OK':<3}  question / generation / gold")
    print("-" * 100)

    for q in QUESTIONS:
        gen = generate_answer(model, tokenizer, q.question)
        correct = score(gen, q.gold)
        n_total_by_tier[q.tier] += 1
        if correct:
            n_correct_by_tier[q.tier] += 1
        tag = "OK " if correct else "MISS"
        print(f"{q.qid:>3}  {q.tier:<10}  {tag:<3}  {q.question}")
        print(f"{'':>3}  {'':<10}  {'':<3}  -> gen:  {gen!r}")
        print(f"{'':>3}  {'':<10}  {'':<3}  -> gold: {q.gold}")
        results.append({
            "qid": q.qid,
            "tier": q.tier,
            "question": q.question,
            "gold": q.gold,
            "generation": gen,
            "correct": correct,
        })

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_correct = sum(n_correct_by_tier.values())
    total = sum(n_total_by_tier.values())
    for tier in ("famous", "general", "long_tail"):
        c, n = n_correct_by_tier[tier], n_total_by_tier[tier]
        pct = 100.0 * c / n if n else 0
        print(f"  {tier:<10}  {c}/{n}   ({pct:.0f}%)")
    overall_pct = 100.0 * total_correct / total
    print(f"  {'OVERALL':<10}  {total_correct}/{total}   ({overall_pct:.0f}%)")

    print()
    print("Decision (per the pre-registered rule):")
    if overall_pct >= 80:
        print("  >= 80%: Wikipedia is contaminated. SWAP retrieval corpus before Phase 1.")
    elif overall_pct >= 50:
        print("  50-79%: Mixed. Filter Phase 4 question set to Gemma-doesn't-know subset.")
    else:
        print("  < 50%: Wikipedia is fine. Proceed with Phase 1 as planned.")

    payload = {
        "test_id": "260529_test01_wikipedia_contamination",
        "model": "google/gemma-2-2b",
        "prompt_template": PROMPT_TEMPLATE,
        "generation_config": {"max_new_tokens": 30, "do_sample": False},
        "summary": {
            "overall_correct": total_correct,
            "overall_total": total,
            "overall_pct": overall_pct,
            "by_tier": {
                tier: {
                    "correct": n_correct_by_tier[tier],
                    "total": n_total_by_tier[tier],
                    "pct": 100.0 * n_correct_by_tier[tier] / n_total_by_tier[tier]
                    if n_total_by_tier[tier] else 0,
                }
                for tier in ("famous", "general", "long_tail")
            },
        },
        "per_question": results,
    }
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {results_path}")
    print("Paste the SUMMARY block into notes.md > Results section,")
    print("then write the calibration retrospective.")


if __name__ == "__main__":
    main()
