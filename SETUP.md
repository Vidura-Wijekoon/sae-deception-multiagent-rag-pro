# Development Environment Setup

## Quick Start

### 1. Activate the Virtual Environment
Every new terminal session needs to activate the venv:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Verify Setup (Smoke Test)
Load Gemma 2-2B, run a single forward pass, and measure throughput:

```powershell
python -c "import time, torch; from transformers import AutoModelForCausalLM, AutoTokenizer; tok = AutoTokenizer.from_pretrained('google/gemma-2-2b'); m = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b', torch_dtype=torch.float16, device_map='auto'); x = tok('Hello world', return_tensors='pt').to(m.device); t0 = time.time(); out = m.generate(**x, max_new_tokens=20); dt = time.time() - t0; print(f'OK - {out.shape[1]/dt:.1f} tok/s on {m.device}')"
```

**Expected output:** `OK - XXX tok/s on cuda:0` (where XXX is tokens/second, typically 20-50 for Gemma 2-2B)

### 3. Installing More Dev Tools
Add pre-commit hooks (runs linters + formatters automatically):

```powershell
pre-commit install --overwrite --install-hooks
```

## Environment Details

- **Type:** Python 3.11 venv (Windows)
- **Location:** `.venv/`
- **PyTorch:** CUDA 12.1 (`torch>=2.3`)
- **Key packages:**
  - `transformers>=4.44` - LLM inference
  - `sae-lens>=4.0` - Sparse Autoencoders
  - `langgraph>=0.2` - Multi-agent orchestration
  - `faiss-cpu>=1.8` - Vector search
  - `datasets>=3.0` - Data loading

## Common Commands (from Makefile)

After activating the venv, you can run:

```powershell
# Format code (black + ruff)
python -m black .
python -m ruff check --fix .

# Lint (no changes)
python -m ruff check .

# Run tests in parallel
pytest -n auto

# Run the smoke test (same as above)
python -m transformers
```

## Troubleshooting

**Q: "No module named 'torch'" after activating venv?**  
A: The torch installation is still running. Wait for it to complete, then open a new terminal and activate the venv again.

**Q: Commands like `ruff`, `black`, `pre-commit` not found?**  
A: Make sure you've activated the venv (see step 1).

**Q: Want to use this project in VS Code notebooks?**  
A: Select the kernel at `.venv\Scripts\python.exe` when opening a Jupyter notebook.

## Next Steps

1. ✅ Repository initialized and pushed to GitHub
2. ✅ Python environment created and dependencies installed
3. ⏳ **Smoke test pending** (torch still downloading)
4. → Once smoke test passes, proceed with Phase 0 step 2 of the build guide
