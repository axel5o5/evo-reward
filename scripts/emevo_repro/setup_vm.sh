#!/usr/bin/env bash
# Set up emevo reproduction environment on the GCE L4 VM.
# Run ONCE. Idempotent — safe to re-run.
#
# Installs:
#   - uv (Python package manager)
#   - emevo @ gecco2026 branch (tip: f87a880)
#   - CUDA12 JAX + deps via `uv sync --extra=cuda12 --extra=analysis`
#
# Result: ~/emevo_repro/emevo/ with a working .venv that can run cf_predator.py.
# Our existing ~/evo-env for our code is untouched (separate venv).

set -euo pipefail

EMEVO_REPO="https://github.com/oist/emevo.git"
EMEVO_BRANCH="gecco2026"
# Pin commit for reproducibility. HEAD as of 2026-04-24.
# If reproduction fails, fall back to a777689 (closer to arxiv v2 submission).
EMEVO_COMMIT="f87a880e539f41d5a0ff0a85115930465fa87bcb"

WORKDIR="$HOME/emevo_repro"
EMEVO_DIR="$WORKDIR/emevo"

echo "[1/5] Ensure uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
uv --version

echo "[2/5] Clone emevo ($EMEVO_BRANCH @ $EMEVO_COMMIT)"
mkdir -p "$WORKDIR"
if [ ! -d "$EMEVO_DIR/.git" ]; then
  git clone --branch "$EMEVO_BRANCH" "$EMEVO_REPO" "$EMEVO_DIR"
fi
cd "$EMEVO_DIR"
git fetch origin
git checkout "$EMEVO_COMMIT"
echo "At commit: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"

echo "[3/5] uv sync (cuda12 + analysis)"
uv sync --extra=cuda12 --extra=analysis

echo "[4/5] CUDA smoke check"
uv run python -c "
import jax
print('jax version:', jax.__version__)
print('devices:', jax.devices())
assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices()), 'No GPU visible to JAX'
print('GPU OK')
"

echo "[5/5] emevo import smoke"
uv run python -c "
from emevo import make
import dataclasses
from experiments.cf_predator import CfConfigWithPredator
from serde import toml
cfg_path = 'config/env/20251001-predator-default.toml'
with open(cfg_path) as f:
    cfg = toml.from_toml(CfConfigWithPredator, f.read())
env = make('CircleForaging-v2', **dataclasses.asdict(cfg))
print('env constructed:', type(env).__name__)
print('n_max_agents:', cfg.n_max_agents, 'n_max_predators:', cfg.n_max_predators)
"

echo
echo "=== Setup complete ==="
echo "Emevo path: $EMEVO_DIR"
echo "Next step: bash ~/evo-reward/scripts/emevo_repro/launch_smoke.sh"
