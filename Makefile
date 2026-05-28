# Makefile for sae-deception-multiagent-rag.
# `make hooks` is lifted from safety-research/safety-examples@v1.0.0.
# The other targets are project-specific conveniences.

.PHONY: help hooks install dev test format lint smoke clean

help:
	@echo "Available targets:"
	@echo "  make install     - editable install + safety-tooling submodule"
	@echo "  make dev         - install + dev extras + pre-commit hooks"
	@echo "  make hooks       - install pre-commit hooks (pre-commit + post-checkout + pre-push)"
	@echo "  make test        - run pytest in parallel"
	@echo "  make format      - run black + ruff --fix on everything"
	@echo "  make lint        - run ruff check (no fix)"
	@echo "  make smoke       - run the Phase 0 smoke test (loads Gemma 2-2B, single fwd pass)"
	@echo "  make clean       - remove caches and build artifacts"

# ----- Hook setup (from safety-examples) -----
hooks:
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push

# ----- Install -----
install:
	@echo ">>> Pulling safety-tooling submodule"
	git submodule update --init --recursive
	@echo ">>> Installing safety-tooling (editable)"
	pip install -e safety-tooling
	@echo ">>> Installing this project (editable)"
	pip install -e .

dev: install
	pip install -e ".[dev]"
	$(MAKE) hooks

# ----- Test -----
test:
	pytest -n auto

# ----- Format / lint -----
format:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .

# ----- Phase 0 smoke test (build guide step 5) -----
# Loads Gemma 2-2B in float16, runs one forward pass on "Hello world",
# prints tokens-per-second. If this works, your GPU/env setup is fine.
smoke:
	python -c "import time, torch; \
from transformers import AutoModelForCausalLM, AutoTokenizer; \
tok = AutoTokenizer.from_pretrained('google/gemma-2-2b'); \
m = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b', torch_dtype=torch.float16, device_map='auto'); \
x = tok('Hello world', return_tensors='pt').to(m.device); \
t0 = time.time(); out = m.generate(**x, max_new_tokens=20); dt = time.time() - t0; \
print(f'OK - {out.shape[1]/dt:.1f} tok/s on {m.device}')"

# ----- Clean -----
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf build dist *.egg-info src/*.egg-info
	find . -name "*.pyc" -delete
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
