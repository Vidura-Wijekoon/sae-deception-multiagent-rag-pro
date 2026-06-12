"""Experiment configuration and run manifests.

A single `ExperimentConfig` dataclass drives the whole de-risk run. The
`RunManifest` captures everything needed to reproduce a result: git commit,
config hash, representation backend + revision, seeds, and host info (this is
the `run_manifest.json` promised in the README "Reproducibility" section).

Nothing here imports torch / transformers, so it loads instantly on CPU.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All knobs for one probe de-risk run.

    The single most important field is `representation`. On CPU it is one of
    {"neural", "tfidf_svd", "random_init"}. On a GPU, point it at the real
    backend ("gemma_sae") and the rest of the pipeline is unchanged — that is
    the whole design goal of the de-risk phase.
    """

    # --- identity ---
    experiment_id: str = "260605_phase5_probe_derisk"
    seed: int = 0

    # --- corpus (Phase 4) ---
    n_base_facts: int = 60           # each fact -> 1 clean + 1 poisoned context
    poison_fraction: float = 0.5     # balanced by construction
    attack_mix: tuple[str, ...] = ("poisonedrag", "greshake")

    # --- representation (the swappable backend) ---
    representation: str = "neural"   # neural | tfidf_svd | random_init | gemma_sae
    neural_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    repr_dim: int = 384              # tfidf_svd / random_init target dimensionality

    # --- GPU backend (unused on CPU; documents the real target) ---
    gemma_model: str = "google/gemma-2-2b"
    gemma_sae_release: str = "gemma-scope-2b-pt-res-canonical"
    gemma_layers: tuple[int, ...] = (6, 12, 18, 24)
    pooling: str = "mean"            # last | mean | max

    # --- retrieval (Phase 1) ---
    top_k: int = 4

    # --- probes (Phase 5) ---
    n_folds: int = 5
    n_bootstrap: int = 2000
    probe_C: float = 1.0             # inverse L2 strength for logistic regression

    # --- output ---
    out_root: str = "experiments"

    def config_hash(self) -> str:
        """Stable hash of the config (excludes nothing — full provenance)."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def out_dir(self, repo_root: Path) -> Path:
        return repo_root / self.out_root / self.experiment_id


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    created_utc: str
    experiment_id: str
    config_hash: str
    representation: str
    representation_revision: str
    git_commit: str
    seeds: dict[str, int]
    host: dict[str, str]
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def build(cls, cfg: ExperimentConfig, repo_root: Path, repr_revision: str, notes: str = "") -> "RunManifest":
        return cls(
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            experiment_id=cfg.experiment_id,
            config_hash=cfg.config_hash(),
            representation=cfg.representation,
            representation_revision=repr_revision,
            git_commit=_git_commit(repo_root),
            seeds={"global": cfg.seed},
            host=_host_info(),
            config=asdict(cfg),
            notes=notes,
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))


# ---------------------------------------------------------------------------
# Determinism + provenance helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Seed every RNG we depend on (torch/faiss are seeded in their own modules
    when present). Keeps CPU runs bit-reproducible."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _host_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
