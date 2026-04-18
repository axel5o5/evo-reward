# Local Setup — Mac and Raspberry Pi

This runbook covers running evo-reward on consumer hardware instead of
a cloud GPU. The Mac is a realistic venue for full Phase 1a runs
(slower than an L4 but free). The Raspberry Pi is only for validation
that the code is portable to ARM CPU — it would take months to finish
a full run and is not a serious training target.

For the GCP L4 path, see [gcp-setup.md](gcp-setup.md).

---

## Prerequisites (both platforms)

Python 3.10+ and pip. On macOS:

```
brew install python@3.10
```

On Raspberry Pi OS (Debian-based, 64-bit):

```
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git build-essential
```

---

## Mac setup (M-series or Intel)

This is the path you'll use if you're running Phase 1a locally to avoid
cloud costs.

### Install

```
git clone <repo-url>   # or use your existing local clone
cd evo-reward
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U "jax[cpu]"
pip install -r requirements.txt
```

### Verify

```
pytest -x -q
```

Expect `113 passed, 3 skipped` in about 45 seconds on an M-series Mac
(somewhat longer on Intel).

Confirm JAX sees the Mac:

```
python -c "import jax; print(jax.devices())"
```

On Apple Silicon CPU: `[CpuDevice(id=0)]`.

### Optional: JAX Metal for Apple Silicon

`jax-metal` uses the Mac's GPU via Metal, which is faster than CPU but
has known compatibility gaps. As of 2026 it supports most basic ops but
not every primitive the sim uses, so the code may fall back to CPU
silently. Only try this if you're comfortable debugging missing-op
errors.

```
pip install jax-metal
```

If it works, `jax.devices()` will show `[METAL(id=0)]`. If the sim
crashes with `NotImplementedError` on a specific lax op, uninstall and
stick with `jax[cpu]`.

### Run Phase 1a locally

The Mac won't hit L4 throughput (~1400 steps/s); expect 100–300 steps/s
depending on your chip. A 10.24M-step run translates to roughly:

| Chip          | Approximate wall clock |
| ------------- | ---------------------- |
| M1            | 4–7 days               |
| M2 / M3       | 3–5 days               |
| M4 Pro / Max  | 2–3 days               |

Run inside a `tmux` so the process survives lid-close, SSH disconnects,
terminal crashes:

```
brew install tmux
tmux new -s phase1a
source .venv/bin/activate
python scripts/run_experiment_jax.py \
  --config configs/baseline_faithful.yaml \
  --seed 0
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t phase1a`.

### Recovering from interruption

The runner checkpoints every 100K steps (~30–100MB each compressed,
grows as rollout buffers fill up; last 3 kept) into
`results/baseline_faithful/seed_0/checkpoints/`. If the process
dies — for any reason — you can resume from the latest checkpoint:

```
python scripts/run_experiment_jax.py \
  --config configs/baseline_faithful.yaml \
  --seed 0 \
  --resume
```

This is bit-identical to an uninterrupted run (verified by
`tests/test_checkpoint_jax.py::TestResumeDeterminism`). The trade-off
is up to 100K steps of work lost — a few minutes to a few hours of
wall-clock depending on your chip.

If you want to pin to a specific checkpoint:

```
python scripts/run_experiment_jax.py \
  --config configs/baseline_faithful.yaml \
  --seed 0 \
  --resume-from results/baseline_faithful/seed_0/checkpoints/step_05000000.npz
```

Without `--resume`, the runner refuses to start if checkpoints already
exist — this prevents accidental overwrite of an in-progress run.

---

## Raspberry Pi setup (smoke test only)

Use this path only to verify that the code is portable to ARM CPU. Do
not attempt a full Phase 1a run — the Pi is not fast enough.

Tested target: Raspberry Pi 5 (8GB), 64-bit Raspberry Pi OS.

### Install Rust for phyjax2d

`phyjax2d` has a Rust-backed component that requires a local toolchain
on ARM since prebuilt wheels don't always cover Pi:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Install Python deps

```
git clone <repo-url>
cd evo-reward
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U jax   # plain jax — no GPU wheels on Pi
pip install -r requirements.txt
```

The `phyjax2d` install will take several minutes on Pi 5 (Rust compiles
from source). If it fails, ensure Rust is on your PATH:
`rustc --version` should print a version.

### Validate

Only the test suite is a realistic target. Skip the full runner.

```
pytest tests/test_checkpoint_jax.py -v
```

Expected: 9 passed. Total wall-clock on Pi 5: roughly 2–5 minutes.

Then the full suite as a broader smoke test:

```
pytest -x -q
```

Expected: 113 passed, 3 skipped. Wall-clock: 5–15 minutes on Pi 5.

If all tests pass, the code is verified portable to ARM. This is the
Pi's only validation role — don't try to train.

---

## Troubleshooting

**`ImportError: No module named 'phyjax2d'`** — pip install failed
silently. On Pi, run `pip install --no-binary :all: phyjax2d` to force
a source build and get a clearer error trace.

**`RuntimeError: Backend 'metal' ... failed to initialize`** — `jax-metal`
is installed but the op graph uses something Metal doesn't implement.
`pip uninstall jax-metal && pip install -U "jax[cpu]"`.

**Tests pass but `run_experiment_jax.py` is wildly slow** — Python may
be falling back to 32-bit NumPy on an old libopenblas build. On Mac:
`brew reinstall openblas`. On Pi: `sudo apt-get install
libopenblas-dev` and reinstall numpy.

**`Checkpoints already exist`** error on fresh-looking run — you've
started this seed before. Either pass `--resume`, or delete the
`results/<exp>/seed_<N>/checkpoints/` directory and start over.
