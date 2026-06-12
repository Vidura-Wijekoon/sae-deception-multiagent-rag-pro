"""Single-command entry point for the de-risk experiment (README "Reproducibility").

    python scripts/run_experiment.py --config configs/default.yaml

Loads the YAML config, runs the full Phase 1/4/5 experiment, and writes
metrics.json + run_manifest.json + pipeline_results.json into the experiment
directory. Adds src/ to the path so it works without `pip install -e .`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from sae_deception.config import ExperimentConfig  # noqa: E402
from sae_deception.experiment import run_experiment  # noqa: E402


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    # tuples for the dataclass fields that expect them
    for k in ("attack_mix", "gemma_layers"):
        if k in raw and isinstance(raw[k], list):
            raw[k] = tuple(raw[k])
    fields = ExperimentConfig.__dataclass_fields__
    return ExperimentConfig(**{k: v for k, v in raw.items() if k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--representation", default=None, help="override config representation")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.representation:
        cfg.representation = args.representation

    print(f"[run] experiment={cfg.experiment_id} representation={cfg.representation} "
          f"seed={cfg.seed} config_hash={cfg.config_hash()}")
    metrics = run_experiment(cfg, REPO_ROOT)

    p = metrics["pipeline_phase1_4"]
    print(f"[phase1/4] poison_retrieval_rate={p['poison_retrieval_rate']} "
          f"attack_success_rate={p['attack_success_rate']} gate_pass={p['phase4_gate_A2_1_pass']}")
    for rep, d in metrics["direction1_probes"].items():
        w, c = d["writer_only"], d["combined"]
        print(f"[phase5:{rep}] writer={w['auroc']} [{w['ci95_low']},{w['ci95_high']}]  "
              f"combined={c['auroc']} [{c['ci95_low']},{c['ci95_high']}]  "
              f"delta={d['combined_minus_writer']}")
    print(f"[done] wrote metrics.json + run_manifest.json to experiments/{cfg.experiment_id}/")


if __name__ == "__main__":
    main()
