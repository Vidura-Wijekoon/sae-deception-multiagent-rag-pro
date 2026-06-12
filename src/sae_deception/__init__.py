"""sae_deception - probing for deception in multi-agent RAG pipelines.

De-risk phase package. The public surface is intentionally small while
sub-package boundaries are still loose (see README "Repo layout").

The representation backend is pluggable (`sae_deception.interp.represent`):
the CPU de-risk run uses real MiniLM / TF-IDF / random-init representations, and
the same probe + experiment code runs unchanged against Gemma 2-2B + Gemma Scope
SAE features on a GPU by swapping the `representation` field in the config.
"""

__version__ = "0.1.0"
