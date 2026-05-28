#!/usr/bin/env bash
# init_repo.sh — one-shot bootstrap for sae-deception-multiagent-rag.
#
# What this does:
#   1. Creates the empty directory skeleton from Phase 0 step 6 of the build guide
#   2. Adds .gitkeep so empty dirs survive the first commit
#   3. Runs `git init`, sets up main as the default branch, and stages everything
#   4. Makes the first commit
#   5. Prints the next commands to create the GitHub repo via the `gh` CLI
#
# Safe to run multiple times — mkdir -p and `git init` are idempotent.
#
# Usage:
#   bash scripts/init_repo.sh
#
# Prerequisites (suggested):
#   - git ≥ 2.30
#   - gh ≥ 2.0 (optional, only for the last step)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Working from: $REPO_ROOT"

# ----- 1. Create directory skeleton --------------------------------
echo "==> Creating directory skeleton"
DIRS=(
  "src/pipeline"
  "src/probes"
  "src/attacks"
  "src/interp"
  "configs"
  "experiments"
  "notebooks"
  "data"
  "data/cache"
  "data/attacks"
  "data/acts"
  "data/feature_labels"
  "docs"
  "scripts"
  "tests"
)
for d in "${DIRS[@]}"; do
  mkdir -p "$d"
  if [ ! -f "$d/.gitkeep" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
    touch "$d/.gitkeep"
  fi
done

# ----- 2. Stub Python packages -------------------------------------
echo "==> Creating empty __init__.py for src subpackages"
for pkg in pipeline probes attacks interp; do
  touch "src/$pkg/__init__.py"
done
touch "src/__init__.py"

# ----- 3. Git init -------------------------------------------------
if [ ! -d .git ]; then
  echo "==> Running git init"
  git init -b main
else
  echo "==> .git already exists, skipping git init"
fi

# ----- 4. First commit ---------------------------------------------
git add -A
if git diff --cached --quiet; then
  echo "==> Nothing to commit. Working tree clean."
else
  echo "==> Making initial commit"
  git commit -m "chore: initial scaffold

- README.md naming the project and the three Directions
- .gitignore (Python/ML defaults)
- LICENSE (MIT)
- CITATION.cff
- empty directory skeleton (src/{pipeline,probes,attacks,interp}, configs,
  experiments, notebooks, data, docs, scripts, tests)
- scripts/init_repo.sh (this script)

Refs: Phase 0 step 1 of the de-risk build guide."
fi

# ----- 5. Next steps -----------------------------------------------
cat <<'EOF'

==> Done. Next steps:

   1. Verify the commit:
        git log --oneline -1

   2. (Optional) Create the GitHub repo via the gh CLI:
        gh repo create sae-deception-multiagent-rag --private --source=. --remote=origin --push

      Or do it manually on github.com, then:
        git remote add origin git@github.com:<your-handle>/sae-deception-multiagent-rag.git
        git push -u origin main

   3. Confirm Phase 0 step 1 is complete in your tracker, and move on to step 2.

EOF
