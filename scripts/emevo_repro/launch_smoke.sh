#!/usr/bin/env bash
# Launch emevo reproduction smoke run in a tmux session.
#
# Config: paper-default (20251001-predator-default.toml — medium mouth, n=0.5 food regen).
# Seed: 0 (matches our own d31d baseline for direct comparison).
# Steps: 512,000 (500 rollouts × 1024 steps). ~5h on L4 at ~27 sps.
# Cost: ~$6 at $1.20/hr.
#
# Outputs: ~/emevo_repro/logs/smoke_seed0/
#   - reward.parquet (per-agent weights over time)
#   - log.parquet    (birth/death events)
#   - foodlog.parquet (feeding events)
#   - state/         (checkpoints every 1000 steps)
#
# Resume behavior: if tmux session already exists, attaches instead of re-launching.

set -euo pipefail

EMEVO_DIR="$HOME/emevo_repro/emevo"
LOG_ROOT="$HOME/emevo_repro/logs/smoke_seed0"
TMUX_SESSION="emevo-smoke"
SEED=0
N_TOTAL_STEPS=$((1024 * 500))  # 512,000 steps
#
# Config: 20251122-predator-square.toml (960x960 square, +28% area vs rectangular).
# The earlier 20251001-predator-default.toml (1200x600 rect) is explicitly marked
# "unused now" by the authors (commit fd09012, 2026-04-10). The square variant
# matches the paper's figures and is the intended canonical config.
CONFIG="$EMEVO_DIR/config/env/20251122-predator-square.toml"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session '$TMUX_SESSION' already exists. Attach with: tmux attach -t $TMUX_SESSION"
  exit 0
fi

mkdir -p "$LOG_ROOT"

# Build the command. Using the paper-default config. n_total_steps overridden.
# log_interval=1000 matches our progress cadence. force_gpu=True asserts CUDA.
CMD="cd $EMEVO_DIR && uv run python experiments/cf_predator.py evolve \
  --seed $SEED \
  --n-total-steps $N_TOTAL_STEPS \
  --cfconfig-path $CONFIG \
  --logdir $LOG_ROOT \
  --log-interval 10 \
  --savestate-interval 10000 \
  --log-mode reward-log-state \
  --debug-print \
  --measure-time \
  2>&1 | tee $HOME/emevo_smoke.log"

echo "Launching emevo smoke (seed=$SEED, $N_TOTAL_STEPS steps) in tmux '$TMUX_SESSION'"
tmux new-session -d -s "$TMUX_SESSION" "$CMD"

echo
echo "=== Launched ==="
echo "Attach:    tmux attach -t $TMUX_SESSION"
echo "Tail log:  tail -f ~/emevo_smoke.log"
echo "Outputs:   $LOG_ROOT"
echo
echo "Expected duration: ~5h at 27 sps. First 10K steps ~6min."
