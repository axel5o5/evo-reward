"""
evolution.py
------------
Mutation and offspring spawning for the evolutionary outer loop.

Mutation: Student's t(df=2, scale=0.4), clipped to +/-100.
This is NOT Gaussian, NOT Cauchy. t(df=2) has heavier tails than
Gaussian but lighter than Cauchy (df=1).

emevo source references:
  config/gops/20250805-mutation-t2-clip100.toml
  2025 paper Section 4: "Student's t-distribution with 2 degrees
  of freedom and a scale of 0.4"
"""

import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.stats import t as t_dist

from src.environment import AgentState


def mutate_genome(
    parent_genome: jnp.ndarray,
    rng_key,
) -> jnp.ndarray:
    """
    child = clip(parent + delta, -100, 100)
    delta ~ StudentT(df=2, scale=0.4), sampled independently per weight.

    Uses scipy.stats.t for correct heavy-tail distribution.
    JAX rng_key is used to derive a seed for scipy RNG.
    """
    seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31 - 1))
    rng = np.random.default_rng(seed)

    delta = t_dist(df=2, scale=0.4).rvs(size=4, random_state=rng).astype(np.float32)
    child = np.array(parent_genome) + delta
    child = np.clip(child, -100.0, 100.0)
    return jnp.array(child)


def mutate_mlp_genome(parent_genome, rng_key, config):
    """Mutate an MLP reward genome (Flax PyTree).

    Flatten to 1D array, apply Student's t(df=2) noise per weight,
    clip to ±mlp_weight_clip, unflatten back to PyTree.

    Uses scipy.stats.t (same pattern as mutate_genome).
    """
    flat, unflatten_fn = ravel_pytree(parent_genome)

    df = config.get("mutation_df", 2)
    scale = config["mlp_mutation_scale"]
    clip_val = config["mlp_weight_clip"]

    seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31 - 1))
    rng = np.random.default_rng(seed)

    delta = t_dist(df=df, scale=scale).rvs(
        size=flat.shape[0], random_state=rng
    ).astype(np.float32)

    child_flat = np.array(flat) + delta
    child_flat = np.clip(child_flat, -clip_val, clip_val)

    return unflatten_fn(jnp.array(child_flat))


def mutate_temporal_genome(parent_genome, rng_key, config):
    """Mutate a temporal reward genome (Flax PyTree).

    Same pattern as mutate_mlp_genome: flatten to 1D, apply Student's t(df=2)
    noise per weight, clip to ±temporal_weight_clip, unflatten.
    """
    flat, unflatten_fn = ravel_pytree(parent_genome)

    df = config.get("mutation_df", 2)
    scale = config["temporal_mutation_scale"]
    clip_val = config["temporal_weight_clip"]

    seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31 - 1))
    rng = np.random.default_rng(seed)

    delta = t_dist(df=df, scale=scale).rvs(
        size=flat.shape[0], random_state=rng
    ).astype(np.float32)

    child_flat = np.array(flat) + delta
    child_flat = np.clip(child_flat, -clip_val, clip_val)

    return unflatten_fn(jnp.array(child_flat))


def spawn_offspring(
    parent: AgentState,
    new_id: int,
    rng_key,
    config: dict,
) -> AgentState:
    """
    Create offspring AgentState:
    - Mutated genome from parent
    - Fresh random policy (NOT copied from parent)
    - energy = parent.energy * energy_share_ratio
    - position ~ N(parent.position, spawn_spread)
    - angle ~ Uniform[-pi, pi]
    """
    spawn_spread = config["spawn_spread"]
    world_size = config["world_size"]
    energy_share_ratio = config["energy_share_ratio"]

    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Position: Gaussian around parent, clamped to world
    parent_pos = parent.position
    if parent_pos is None:
        parent_pos = jnp.array([world_size / 2.0, world_size / 2.0])
    offset = jax.random.normal(k1, shape=(2,)) * spawn_spread
    child_pos = jnp.clip(parent_pos + offset, 0.0, float(world_size))

    # Angle: uniform [-pi, pi]
    child_angle = float(jax.random.uniform(k2, minval=-math.pi, maxval=math.pi))

    # Genome: mutated from parent (Student's t, NOT copied)
    parent_weights = parent.reward_weights
    if parent_weights is None:
        parent_weights = jnp.zeros(4)
    child_weights = mutate_genome(parent_weights, k3)

    # Fresh policy: None -- will be initialized by policy.py when needed.
    # NOT copied from parent.
    return AgentState(
        agent_id=new_id,
        species=parent.species,
        parent_id=parent.agent_id,
        position=child_pos,
        velocity=jnp.zeros(2),
        angle=child_angle,
        ang_vel=0.0,
        age=0,
        energy=parent.energy * energy_share_ratio,
        reward_weights=child_weights,
        policy_params=None,
        policy_opt_state=None,
        rollout=None,
    )
