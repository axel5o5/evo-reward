"""
capacity_util.py
----------------
Capacity utilization metrics for extension axes.

Axis 1: Reward nonlinearity — measures how much of the MLP's nonlinear
  capacity is used by an evolved genome. Near zero = functionally linear.

Axis 2: Social observation utilization — estimates mutual information
  between the social observation block (conspecific heading/speed) and
  the agent's action. Near zero = agent ignores the social channel.
"""

import jax
import jax.numpy as jnp
import numpy as np

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


# ---------------------------------------------------------------------------
# Axis 2: Social observation utilization (mutual information)
# ---------------------------------------------------------------------------

def _binned_entropy(x, n_bins):
    """Compute Shannon entropy of a 1D array using quantile-based bins.

    Returns entropy in nats.
    """
    # Use quantile edges for adaptive binning
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    # Deduplicate edges (can happen with many identical values)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    counts, _ = np.histogram(x, bins=edges)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def _binned_joint_entropy(x, y, n_bins):
    """Compute joint Shannon entropy of two 1D arrays using quantile bins.

    Returns entropy in nats.
    """
    edges_x = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    edges_y = np.unique(np.percentile(y, np.linspace(0, 100, n_bins + 1)))
    if len(edges_x) < 2 or len(edges_y) < 2:
        return 0.0
    counts, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    probs = counts.ravel() / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def compute_social_obs_utilization(
    observations: np.ndarray,
    actions: np.ndarray,
    config: dict,
    n_bins: int = 10,
) -> float:
    """Estimate mutual information between social obs block and actions.

    Uses a binned histogram estimator:
      MI(X; Y) = H(X) + H(Y) - H(X, Y)

    Computes MI for each (social_dim, action_dim) pair independently,
    then returns the mean MI across all pairs.

    Args:
        observations: (T, obs_dim) trajectory observations, numpy array.
        actions: (T, 2) actions, numpy array.
        config: dict with n_social_neighbors.
        n_bins: number of quantile-based bins for MI estimation.

    Returns:
        Mean MI in nats across all (social_dim, action_dim) pairs.
        Near zero = agent ignores social channel.
    """
    n_neighbors = config.get("n_social_neighbors", 5)
    social_start = 205
    social_end = social_start + 2 * n_neighbors
    social_block = np.asarray(observations[:, social_start:social_end])
    actions = np.asarray(actions)

    n_social_dims = social_block.shape[1]
    n_action_dims = actions.shape[1]

    mi_values = []
    for i in range(n_social_dims):
        x = social_block[:, i]
        for j in range(n_action_dims):
            # Constant dimension (zero-padded) has zero MI with anything
            if np.std(x) < 1e-10:
                mi_values.append(0.0)
                continue
            y = actions[:, j]
            h_x = _binned_entropy(x, n_bins)
            h_y = _binned_entropy(y, n_bins)
            h_xy = _binned_joint_entropy(x, y, n_bins)
            mi = max(0.0, h_x + h_y - h_xy)  # clamp rounding errors
            mi_values.append(mi)

    return float(np.mean(mi_values)) if mi_values else 0.0
