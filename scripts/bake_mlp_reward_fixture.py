"""bake_mlp_reward_fixture.py
---------------------------
Generate a JSON fixture of MLP reward genomes for the dashboard's
RewardLandscape spike. Numpy-only (no JAX/Flax dependency) so it runs
in <1s on any machine.

The fixtures are *synthetic* — they do not come from a trained run,
because Axis 1 isn't wired through the JAX runner yet (see jax_sim.py:263).
They exist solely to validate the visualization pipeline before we invest
in actually plumbing MLP reward through the simulator.

Architecture matches src/reward.py::RewardMLP exactly:
  input(4) -> Dense(8, tanh) -> Dense(8, tanh) -> Dense(1, linear)
Total params: 121 (= 40 + 72 + 9).

Output: dashboard/site/public/fixtures/mlp_reward_examples.json
        (committed; the dashboard fetches it at runtime)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "dashboard" / "site" / "public" / "fixtures" / "mlp_reward_examples.json"

INPUT_DIM = 4
HIDDEN = 8
INPUT_LABELS = ["n_eaten", "motor_norm", "s_prey", "s_pred"]


def lecun_normal(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0]
    return rng.normal(0, np.sqrt(1.0 / fan_in), size=shape).astype(np.float32)


def random_genome(seed: int, scale: float = 1.0) -> dict:
    """Random init mimicking Flax's default (lecun_normal kernel, zero bias).

    `scale` multiplies the kernels — at scale=1.0 we get a freshly initialized
    network; higher values approximate post-mutation drift.
    """
    rng = np.random.default_rng(seed)
    return {
        "Dense_0": {
            "kernel": lecun_normal((INPUT_DIM, HIDDEN), rng) * scale,
            "bias": np.zeros(HIDDEN, dtype=np.float32),
        },
        "Dense_1": {
            "kernel": lecun_normal((HIDDEN, HIDDEN), rng) * scale,
            "bias": np.zeros(HIDDEN, dtype=np.float32),
        },
        "Dense_2": {
            "kernel": lecun_normal((HIDDEN, 1), rng) * scale,
            "bias": np.zeros(1, dtype=np.float32),
        },
    }


def linear_equivalent_genome(target_w: np.ndarray) -> dict:
    """Construct a genome that approximates a linear reward function r = target_w · s.

    Trick: route each input dimension through one hidden unit with a small kernel
    so tanh stays in its linear regime (tanh(x) ~ x for |x| << 1). Use unit second
    layer + output kernel = target_w (rescaled to undo the small-kernel attenuation).

    Result: the MLP is approximately linear with weights `target_w`, deviating
    only when stimuli grow large enough to push tanh out of its linear regime.
    Useful as a "what would an evolved-linear-equivalent MLP look like" baseline.
    """
    assert target_w.shape == (INPUT_DIM,)
    w0 = np.zeros((INPUT_DIM, HIDDEN), dtype=np.float32)
    # Route dim i to hidden unit i with a small weight; remaining hidden units idle.
    eps = 0.1
    for i in range(INPUT_DIM):
        w0[i, i] = eps
    w1 = np.zeros((HIDDEN, HIDDEN), dtype=np.float32)
    # Identity routing (same trick: small weight, stays linear)
    for i in range(HIDDEN):
        w1[i, i] = 1.0
    # Output kernel applies target weights, rescaled by 1/eps to undo attenuation
    w2 = np.zeros((HIDDEN, 1), dtype=np.float32)
    for i in range(INPUT_DIM):
        w2[i, 0] = target_w[i] / eps
    return {
        "Dense_0": {"kernel": w0, "bias": np.zeros(HIDDEN, dtype=np.float32)},
        "Dense_1": {"kernel": w1, "bias": np.zeros(HIDDEN, dtype=np.float32)},
        "Dense_2": {"kernel": w2, "bias": np.zeros(1, dtype=np.float32)},
    }


def threshold_fear_genome() -> dict:
    """Hand-crafted genome that exhibits nonlinear "threshold fear":
    reward is roughly flat when s_pred is low, then drops sharply once s_pred
    crosses a threshold. This is a function-class that linear genomes cannot
    represent — included to make the nonlinearity utilization metric have
    something visually striking to point at.

    Mechanism: route s_pred (input dim 3) into one hidden unit with a sharp
    bias so tanh saturates near a threshold. Output kernel applies a strong
    negative weight to that unit.
    """
    rng = np.random.default_rng(7)
    # Start from a small random baseline so other inputs have *some* effect
    w0 = lecun_normal((INPUT_DIM, HIDDEN), rng) * 0.2
    b0 = np.zeros(HIDDEN, dtype=np.float32)
    # Hidden unit 0 = "predator-near detector": large weight on s_pred, bias to
    # threshold near s_pred=0.5
    w0[:, 0] = 0.0
    w0[3, 0] = 8.0  # s_pred → hidden 0, sharp activation
    b0[0] = -3.0    # threshold: tanh(8 * s_pred - 3) ~ -1 below s_pred ~ 0.4
    w1 = np.eye(HIDDEN, dtype=np.float32)
    w2 = np.zeros((HIDDEN, 1), dtype=np.float32)
    # Big negative output weight on the detector unit; small weights on others
    w2[0, 0] = -2.5
    w2[1:, 0] = lecun_normal((HIDDEN - 1, 1), rng).flatten() * 0.3
    return {
        "Dense_0": {"kernel": w0, "bias": b0},
        "Dense_1": {"kernel": w1, "bias": np.zeros(HIDDEN, dtype=np.float32)},
        "Dense_2": {"kernel": w2, "bias": np.zeros(1, dtype=np.float32)},
    }


def serialize(g: dict) -> dict:
    return {
        layer: {k: v.tolist() for k, v in d.items()}
        for layer, d in g.items()
    }


def main() -> None:
    fixtures = [
        {
            "name": "founder",
            "description": "Random init at scale 1.0 (Flax default). Near-zero output, near-linear.",
            "params": serialize(random_genome(seed=0, scale=1.0)),
        },
        {
            "name": "drifted_seed42",
            "description": "Random init at scale 1.5 — proxy for ~tens of mutation steps from founder.",
            "params": serialize(random_genome(seed=42, scale=1.5)),
        },
        {
            "name": "linear_evolved",
            "description": "Hand-crafted to approximate linear weights [+1, -0.5, +0.5, -1.5] — what an evolved-linear-equivalent MLP would look like.",
            "params": serialize(linear_equivalent_genome(np.array([1.0, -0.5, 0.5, -1.5], dtype=np.float32))),
        },
        {
            "name": "threshold_fear",
            "description": "Nonlinear: reward flat when s_pred is low, sharp drop above s_pred~0.4. Function class linear genomes cannot represent.",
            "params": serialize(threshold_fear_genome()),
        },
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "version": 1,
                "arch": "RewardMLP",
                "input_dim": INPUT_DIM,
                "input_labels": INPUT_LABELS,
                "hidden_size": HIDDEN,
                "synthetic": True,
                "note": "These fixtures are NOT from a trained run — Axis 1 (MLP reward) is not yet wired through jax_sim. They demonstrate the visualization pipeline.",
                "fixtures": fixtures,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(fixtures)} fixtures to {OUT_PATH}")


if __name__ == "__main__":
    main()
