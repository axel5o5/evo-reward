"""
capacity_util.py
----------------
Axis 1 capacity utilization metric: measures how much of the MLP's
nonlinear capacity is being used by an evolved genome.

A genome that is functionally linear (tanh operating in linear regime)
will have near-zero residual. A genome exploiting nonlinearity will
show higher residual, indicating evolution found nonlinear reward
structure useful.
"""

import jax
import jax.numpy as jnp

from src.reward import compute_mlp_reward


def compute_reward_nonlinearity(genome, config, n_samples=1000):
    """Measure how nonlinear the MLP reward function is vs best linear fit.

    Samples random stimulus vectors, computes MLP output for each, fits
    the best affine approximation via least squares, and returns the
    relative residual (1 - R^2).

    Args:
        genome: Flax parameter PyTree from init_mlp_genome.
        config: dict with at least mlp_hidden_size.
        n_samples: number of random stimulus samples.

    Returns:
        float in [0, ~1]: 0 = perfectly linear, higher = more nonlinear.
        Zero residual means the genome is functionally linear despite
        having MLP architecture.
    """
    rng = jax.random.PRNGKey(42)  # fixed seed for reproducibility
    X = jax.random.uniform(rng, (n_samples, 4))

    # Vectorized MLP forward pass
    y = jax.vmap(lambda x: compute_mlp_reward(genome, x))(X)

    # Augmented design matrix for affine fit: [x1, x2, x3, x4, 1]
    ones = jnp.ones((n_samples, 1))
    X_aug = jnp.concatenate([X, ones], axis=1)

    # Least squares: y = X_aug @ beta
    beta, _, _, _ = jnp.linalg.lstsq(X_aug, y, rcond=None)
    y_hat = X_aug @ beta

    # Relative residual: 1 - R^2
    ss_res = jnp.sum((y - y_hat) ** 2)
    ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)

    # Guard against zero variance (constant output = trivially linear)
    nonlinearity = jnp.where(ss_tot > 1e-10, ss_res / ss_tot, 0.0)
    return float(nonlinearity)
