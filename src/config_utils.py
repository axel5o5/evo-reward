"""Config-loading helpers shared across entry points.

Currently exposes one resolver that converts scale-relative parameter forms
into their absolute equivalents. Keep this module dependency-free (stdlib
only) so it can be imported from both run_experiment_jax.py and tests
without pulling in JAX/Flax.
"""

# Reference world side length for scale-relative params. Equal to paper
# Appendix A's 960×960 simulation domain. Density-normalized config keys
# (food_growth_rate_at_960sq, etc.) are interpreted as "value at this
# reference world size" and scaled by world area.
REFERENCE_WORLD_SIDE = 960.0


def resolve_scale_dependent_params(config: dict) -> dict:
    """Resolve scale-relative config keys into their absolute counterparts.

    Mutates and returns ``config``.

    Currently handles:

    * ``food_growth_rate_at_960sq`` → ``food_growth_rate``. The given value
      is taken to be the food/step rate at a 960×960 world. The effective
      rate at the actual ``world_size`` is scaled by the area ratio
      ``(world_size / 960)**2`` so food density per unit area stays paper-
      faithful. If both the scale-relative and absolute keys are present
      the function raises — pick one.

    Configs that only set the legacy absolute ``food_growth_rate`` key are
    untouched.
    """
    rel_key = "food_growth_rate_at_960sq"
    abs_key = "food_growth_rate"

    if rel_key in config:
        if abs_key in config:
            raise ValueError(
                f"Config defines both '{rel_key}' and '{abs_key}'. Pick one — "
                f"the scale-relative form auto-derives the absolute value."
            )
        world_size = config.get("world_size")
        if world_size is None:
            raise ValueError(
                f"Config uses '{rel_key}' but does not define 'world_size'."
            )
        ratio = float(world_size) / REFERENCE_WORLD_SIDE
        config[abs_key] = float(config[rel_key]) * (ratio ** 2)

    return config
