"""Representation backends - the swappable "activation" layer.

This is the one module you change to move from the CPU de-risk run to the real
Gemma 2-2B + Gemma Scope SAE run. Every backend implements the same tiny
interface:

    rep = get_representer(name, dim=..., seed=...)
    rep.fit(all_texts)            # no-op for frozen/stateless backends
    X = rep.encode(list_of_texts) # -> np.ndarray [n, d]
    rep.revision                  # provenance string for the manifest

CPU backends (real, runnable here):
  * "neural"      - MiniLM (all-MiniLM-L6-v2) sentence embeddings via fastembed
                    (ONNX, no torch). A real pretrained-transformer representation;
                    the closest cheap analog to a model's pooled residual stream.
  * "tfidf_svd"   - TF-IDF -> TruncatedSVD. A real, label-free classical
                    representation. Used as a representation-agnostic cross-check.
  * "random_init" - random feature map (hashed char n-grams -> fixed random
                    projection -> tanh). The control for ASSUMPTIONS A1.1: a
                    representation with NO trained features. If probes still
                    separate poisoned/clean here, the signal is lexically trivial.

GPU backend (stub; the actual research target):
  * "gemma_sae"   - wire transformer-lens activation capture + sae-lens Gemma
                    Scope encoding here. Raises a clear NotImplementedError that
                    explains exactly what to fill in.

The probe + experiment code never imports any of these directly - it only ever
sees `encode(...) -> ndarray`, which is what makes the swap a one-liner.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Representer(Protocol):
    revision: str

    def fit(self, texts: list[str]) -> "Representer": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Neural - real MiniLM embeddings (fastembed / ONNX, torch-free)
# ---------------------------------------------------------------------------

class NeuralRepresenter:
    """MiniLM sentence embeddings. Prefers fastembed (ONNX, torch-free); falls
    back to sentence-transformers when fastembed is absent. The revision string
    records which backend actually ran, so manifests stay honest. Both backends
    produce mean-pooled, L2-normalised MiniLM embeddings, so cached activations
    remain comparable across the two."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=model_name)
            self._backend = "fastembed"
        except ImportError:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._backend = "sentence-transformers"
        self.revision = f"{self._backend}::{model_name}"

    def fit(self, texts: list[str]) -> "NeuralRepresenter":
        return self  # frozen encoder

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._backend == "fastembed":
            return np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# TF-IDF -> SVD - real, label-free classical representation
# ---------------------------------------------------------------------------

class TfidfSvdRepresenter:
    def __init__(self, dim: int = 384, seed: int = 0):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.dim = dim
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self._svd = TruncatedSVD(n_components=dim, random_state=seed)
        self.revision = f"tfidf(1,2)->svd{dim}"
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfSvdRepresenter":
        X = self._tfidf.fit_transform(texts)
        # cap components at min(dim, n_features-1) for tiny corpora
        n_comp = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        self._svd.n_components = max(2, n_comp)
        self._svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdRepresenter.encode called before fit()")
        return self._svd.transform(self._tfidf.transform(texts)).astype(np.float32)


# ---------------------------------------------------------------------------
# Random-init - untrained random feature map (control for A1.1)
# ---------------------------------------------------------------------------

class RandomInitRepresenter:
    """Hashed char n-grams -> fixed random Gaussian projection -> tanh.

    A deterministic random feature map with no trained parameters: the cheap
    analog of running the whole pipeline on a randomly-initialised model
    (Heap et al. 2025). It still has lexical access to the text, so a *high*
    AUROC here means the attack signal is lexically trivial rather than a
    learned-feature phenomenon - exactly the thing the A1.1 control is for.
    """

    def __init__(self, dim: int = 384, seed: int = 0, n_hash: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.n_hash = n_hash
        self._hv = HashingVectorizer(
            n_features=n_hash, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False, norm="l2"
        )
        rng = np.random.RandomState(seed)
        self._proj = rng.normal(0.0, 1.0 / np.sqrt(n_hash), size=(n_hash, dim)).astype(np.float32)
        self.revision = f"randinit(char3-5,h{n_hash}->{dim},seed{seed})"

    def fit(self, texts: list[str]) -> "RandomInitRepresenter":
        return self  # stateless / untrained by design

    def encode(self, texts: list[str]) -> np.ndarray:
        H = self._hv.transform(texts)              # [n, n_hash] sparse, deterministic
        return np.tanh(H @ self._proj).astype(np.float32)


# ---------------------------------------------------------------------------
# Gemma + SAE - the real research target (GPU). Stub.
# ---------------------------------------------------------------------------

class GemmaSAERepresenter:
    """Placeholder for the GPU run. Filling this in is the entire bridge from
    de-risk to the real experiment, and it changes nothing downstream.

    Implementation sketch (Phase 2 + Phase 3 of the build guide):
        1. Load google/gemma-2-2b via transformer-lens (HookedTransformer).
        2. Run each text; cache the residual stream at cfg.gemma_layers
           (e.g. blocks.12.hook_resid_post).
        3. Pool over tokens per cfg.pooling (mean | last | max).
        4. Encode the pooled activation through the Gemma Scope JumpReLU SAE
           (sae-lens, release cfg.gemma_sae_release) to get sparse features.
        5. Return the SAE feature matrix [n, n_sae_features] (or the raw pooled
           residual for the raw-activation probe arm).
    """

    def __init__(self, **kwargs):
        self.revision = "gemma-sae::STUB"

    def fit(self, texts: list[str]) -> "GemmaSAERepresenter":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "GemmaSAERepresenter is a GPU-only stub. Wire transformer-lens + sae-lens "
            "here (see docstring). The CPU de-risk uses representation='neural'."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "neural": NeuralRepresenter,
    "tfidf_svd": TfidfSvdRepresenter,
    "random_init": RandomInitRepresenter,
    "gemma_sae": GemmaSAERepresenter,
}


def get_representer(name: str, *, dim: int = 384, seed: int = 0, neural_model: str | None = None) -> Representer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown representation '{name}'; choose from {sorted(_REGISTRY)}")
    if name == "neural":
        return NeuralRepresenter(neural_model or "sentence-transformers/all-MiniLM-L6-v2")
    if name == "tfidf_svd":
        return TfidfSvdRepresenter(dim=dim, seed=seed)
    if name == "random_init":
        return RandomInitRepresenter(dim=dim, seed=seed)
    return GemmaSAERepresenter()
