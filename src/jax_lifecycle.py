"""
jax_lifecycle.py
----------------
JIT-compiled energy updates, death/birth processing operating on SimState.

All functions are pure JAX — no Python loops or mutable objects.
Species-conditional parameters use jnp.where for branchless computation.
"""

import jax
import jax.numpy as jnp

from src.jax_evolution import spawn_offspring_jax


# ---------------------------------------------------------------------------
# Energy updates
# ---------------------------------------------------------------------------

def update_energies_jax(sim_state, prey_n_eaten, pred_caught_energy, pred_n_catches,
                        all_actions, config, pred_count=None, prey_count=None):
    """Vectorized energy update for all agents. Pure JAX.

    Optionally applies density-dependent metabolism (DDM): when species
    population is low, the passive metabolic cost is scaled down by the
    same squared-saturation curve as DDB. Lone survivors hold their energy
    longer while waiting to feed. Applied to both species symmetrically
    (predators via d_b, prey via c_b), each with its own threshold.
    See findings.md §15.

    Args:
        sim_state: SimState
        prey_n_eaten: (max_agents,) int32 — food eaten per prey
        pred_caught_energy: (max_agents,) float32 — sum of caught prey
            energies per predator (shared credit; D28). eta applied here.
        pred_n_catches: (max_agents,) int32 — catches per predator
        pred_count: scalar int — current predator population, required
            when stability_mechanism=="ddb_ddm" or "ddm".
        prey_count: scalar int — current prey population, required when
            stability_mechanism=="ddb_ddm" or "ddm".
        all_actions: (max_agents, 2) — actions taken (sigmoid-scaled)
        config: dict
    """
    e_food = config["prey_e_food"]
    c_b = config["prey_c_b"]
    c_a = config["prey_c_a"]
    d_b = config["predator_d_b"]
    d_a = config["predator_d_a"]
    eta = config["predator_eta"]
    cap = config.get("energy_capacity", 1000.0)

    # D24: emevo's energy cost is `d_a · ||f_applied|| + d_b`, where f_applied
    # is the post-act_ratio scaled force. For predators, act_ratio=(pred_r/prey_r)²
    # ≈ 1.96, so using the raw action here undercharges them ~49%. Matching
    # emevo's `force_norm = sqrt(f1_raw² + f2_raw²)` after scaling.
    scaled_actions = all_actions * sim_state.act_ratio
    action_norms = jnp.linalg.norm(scaled_actions, axis=1)  # (max_agents,)

    # Prey food gain
    prey_gain = prey_n_eaten.astype(jnp.float32) * e_food

    # Predator food gain: eta * sum-of-caught-prey-energies, where the sum
    # was computed upstream without nearest-pred dedup (D28 shared-credit).
    pred_gain = eta * pred_caught_energy  # (max_agents,)

    # Density-dependent metabolism: scale species' passive cost (d_b for
    # predators, c_b for prey) down when own-species pop is low. Mirrors DDB
    # shape: f(N) = max(floor, N^2 / (N^2 + threshold^2)).
    # Same threshold/floor as DDB by default but configurable separately.
    # Symmetric across species (§15.22): each species has its own DDM threshold.
    stability = config.get("stability_mechanism", "none")
    apply_ddm = stability in ("ddm", "ddb_ddm")
    if apply_ddm:
        if pred_count is None or prey_count is None:
            raise ValueError(
                f"stability_mechanism={stability!r} requires pred_count and "
                "prey_count to be passed to update_energies_jax."
            )
        # New names (§15.23) — fall back to legacy ddm_*/ddb_* keys for
        # backward compat with older configs and checkpoints.
        ddm_floor = float(config.get(
            "density_factor_floor",
            config.get("ddm_floor", config.get("ddb_floor", 0.3))
        ))
        ddm_pred_threshold = float(config.get(
            "density_metabolism_threshold_pred",
            config.get(
                "ddm_pred_threshold",
                config.get(
                    "density_breeding_threshold_pred",
                    config.get("ddb_pred_threshold", 8.0),
                ),
            )
        ))
        ddm_prey_threshold = float(config.get(
            "density_metabolism_threshold_prey",
            config.get(
                "ddm_prey_threshold",
                config.get(
                    "density_breeding_threshold_prey",
                    config.get("ddb_prey_threshold", ddm_pred_threshold * 10.0),
                ),
            )
        ))
        ddm_factor_pred = _ddb_factor(pred_count, ddm_pred_threshold, ddm_floor)
        ddm_factor_prey = _ddb_factor(prey_count, ddm_prey_threshold, ddm_floor)
        d_b_eff = d_b * ddm_factor_pred
        c_b_eff = c_b * ddm_factor_prey
    else:
        d_b_eff = d_b
        c_b_eff = c_b

    # Species-conditional update
    is_prey = sim_state.species == 0
    food_gain = jnp.where(is_prey, prey_gain, pred_gain)
    cost_a = jnp.where(is_prey, c_a, d_a)
    cost_b = jnp.where(is_prey, c_b_eff, d_b_eff)

    delta = food_gain - cost_a * action_norms - cost_b
    new_energies = jnp.minimum(sim_state.energies + delta, cap)

    # Only update active agents
    new_energies = jnp.where(sim_state.is_active, new_energies, sim_state.energies)
    new_ages = jnp.where(sim_state.is_active, sim_state.ages + 1, sim_state.ages)

    return sim_state.replace(energies=new_energies, ages=new_ages)


# ---------------------------------------------------------------------------
# Death and birth processing
# ---------------------------------------------------------------------------

def _batch_hazard_prob_jax(ages, energies, species, config):
    """Vectorized hazard probability. Pure JAX."""
    kappa_h = config["kappa_h"]
    alpha_e = config["alpha_e"]
    beta_h = config["beta_h"]

    alpha_t = jnp.where(species == 0, config["alpha_t_prey"], config["alpha_t_pred"])
    beta_t = jnp.where(species == 0, config["beta_t_prey"], config["beta_t_pred"])

    energy_term = 1.0 - 1.0 / (1.0 + alpha_e * jnp.exp(jnp.clip(-beta_h * energies, -700, 700)))
    age_term = alpha_t * jnp.exp(jnp.clip(beta_t * ages.astype(jnp.float32), -700, 700))
    h = kappa_h * energy_term * age_term
    return jnp.clip(h, 0.0, 1.0)


def _ddb_factor(N, threshold, floor):
    """Density-dependent breeding scaling factor (Option B: squared saturation).

    f(N) = max(floor, N^2 / (N^2 + threshold^2))
    Returns 0.5 at N=threshold, ~0.8 at N=2*threshold, ~0.94 at N=4*threshold.
    Floor caps the minimum factor when N is at zero.
    """
    N = N.astype(jnp.float32)
    smooth = (N * N) / (N * N + threshold * threshold)
    return jnp.maximum(floor, smooth)


def _batch_birth_prob_jax(energies, species, config,
                          prey_count=None, pred_count=None,
                          is_active=None):
    """Vectorized birth probability. Pure JAX.

    Optionally applies density-dependent breeding (DDB): when species
    population is low, the energy threshold (zeta) for breeding is scaled
    down by a squared-saturation factor, making breeding easier at deep
    bottlenecks.

    Also scales the breeding *rate* (kappa_b) upward by the inverse factor,
    so a lone survivor not only breeds at low energy but breeds *frequently*.
    The natural floor is `factor >= kappa_b`, which keeps `P_birth ≤ 1` for
    any sensible config (cap of 1/kappa_b ≈ 1000× at integer pops). The
    legacy `ddb_max_boost` config knob has been removed (§15.22) — it was
    an arbitrary cap on the smooth `1/factor` curve, not a biologically
    meaningful constraint. See findings.md §15.

    Boost distribution control (findings.md §15.14, §15.16):
        Two ways to set the distribution; alpha takes precedence if set:

        ddb_boost_distribution_alpha: float in [0, 1]
            Continuous tuning. Internally converted to a power-law
            exponent k = alpha / max(1-alpha, 1e-3), and per-agent
            share is share_i ∝ energy_i^k (within species, normalized).
              alpha=0   → k=0   → uniform share (every agent same boost)
              alpha=0.5 → k=1   → linear share (proportional to energy)
              alpha=1   → k≫1  → winner-take-all (only top hunter breeds)
            Strict selection-aligned scaffolds use higher alpha; broader
            (more inclusive) scaffolds use lower alpha. Default: 0.0.

        ddb_boost_distribution: string (legacy / shorthand)
            "uniform"          → alpha=0.0
            "energy_weighted"  → alpha=0.5
            Used only if ddb_boost_distribution_alpha is not set.

    `is_active` is required whenever the effective alpha > 0 (so the
    energy denominator excludes dead/inactive slots).
    """
    kappa_b = config["kappa_b"]
    beta_b = config["beta_b"]
    zeta_prey = config["zeta_b_prey"]
    zeta_pred = config["zeta_b_pred"]

    kappa_b_eff_all = jnp.full_like(energies, kappa_b)

    stability = config.get("stability_mechanism", "none")
    apply_ddb = stability in ("ddb", "ddb_ddm")
    if apply_ddb:
        if prey_count is None or pred_count is None:
            raise ValueError(
                f"stability_mechanism={stability!r} requires prey_count and pred_count "
                "to be passed to _batch_birth_prob_jax."
            )
        # New names (§15.23) — fall back to legacy ddb_* keys for backward
        # compat with older configs and checkpoints.
        floor = float(config.get(
            "density_factor_floor", config.get("ddb_floor", 0.3)
        ))
        prey_threshold = float(config.get(
            "density_breeding_threshold_prey",
            config.get("ddb_prey_threshold", 30.0)
        ))
        pred_threshold = float(config.get(
            "density_breeding_threshold_pred",
            config.get("ddb_pred_threshold", 5.0)
        ))
        if "ddb_max_boost" in config:
            # Legacy knob — silently ignored as of §15.22. Logging would
            # require a host-side print which doesn't trace under jit, so we
            # rely on the deprecation note in findings.md and the configs.
            pass

        # Resolve effective alpha. Explicit float wins; else map legacy string.
        if "breeding_share_alpha" in config:
            alpha = float(config["breeding_share_alpha"])
        elif "ddb_boost_distribution_alpha" in config:
            alpha = float(config["ddb_boost_distribution_alpha"])
        else:
            boost_dist = config.get("ddb_boost_distribution", "uniform")
            alpha = {"uniform": 0.0, "energy_weighted": 0.5}.get(boost_dist, 0.0)
        alpha = max(0.0, min(1.0, alpha))

        prey_factor = _ddb_factor(prey_count, prey_threshold, floor)
        pred_factor = _ddb_factor(pred_count, pred_threshold, floor)
        zeta_prey = zeta_prey * prey_factor
        zeta_pred = zeta_pred * pred_factor

        # Rate boost: 1/factor with a natural floor at kappa_b that keeps
        # P_birth = kappa_eff/(1+exp(...)) ≤ 1 (since denom ≥ 1). The floor
        # only bites at sub-integer populations (factor ≥ kappa_b at any
        # pop ≥ 1 with sane T values), so for all real states this is the
        # smooth `1/factor` curve. (Removed the legacy `max_boost` cap in
        # §15.22 — it was an arbitrary clip on the recovery curve.)
        prey_boost_uniform = 1.0 / jnp.maximum(prey_factor, kappa_b)
        pred_boost_uniform = 1.0 / jnp.maximum(pred_factor, kappa_b)

        if alpha <= 0.0:
            # Pure uniform — no per-agent redistribution. No is_active needed.
            kappa_b_prey = kappa_b * prey_boost_uniform
            kappa_b_pred = kappa_b * pred_boost_uniform
            kappa_b_eff_all = jnp.where(species == 0, kappa_b_prey, kappa_b_pred)
        else:
            if is_active is None:
                raise ValueError(
                    "ddb_boost_distribution_alpha > 0 (or 'energy_weighted') "
                    "requires is_active to be passed to _batch_birth_prob_jax."
                )
            # Power-law share: share_i ∝ energy_i^k where k = alpha/(1-alpha).
            # Computed via masked log-softmax for numerical stability.
            # Total budget per species preserved: Σ shares = 1, so
            # Σ (N * boost_uniform * share_i) = N * boost_uniform.
            k = alpha / max(1.0 - alpha, 1e-3)
            prey_active = is_active & (species == 0)
            pred_active = is_active & (species == 1)

            log_e = jnp.log(jnp.maximum(energies, 1e-9))
            # Weighted log-energy; mask inactive/wrong-species to -inf so
            # softmax assigns ~0 weight there.
            logits_prey = jnp.where(prey_active, k * log_e, -jnp.inf)
            logits_pred = jnp.where(pred_active, k * log_e, -jnp.inf)
            # Stable softmax (subtract max within each species' active set).
            shares_prey = jax.nn.softmax(logits_prey)
            shares_pred = jax.nn.softmax(logits_pred)

            prey_count_f = prey_count.astype(jnp.float32)
            pred_count_f = pred_count.astype(jnp.float32)
            prey_agent_boost = prey_boost_uniform * prey_count_f * shares_prey
            pred_agent_boost = pred_boost_uniform * pred_count_f * shares_pred
            kappa_b_eff_all = kappa_b * jnp.where(
                species == 0, prey_agent_boost, pred_agent_boost
            )

    zeta = jnp.where(species == 0, zeta_prey, zeta_pred)
    exponent = jnp.clip(zeta - beta_b * energies, -700, 700)
    p_standard = kappa_b_eff_all / (1.0 + jnp.exp(exponent))

    # Emergency breeding clause: when species count drops below a critical
    # threshold, override the energy gate so a starving low-N cohort can
    # still reproduce. Linear ramp from kappa_b at N=0 down to 0 at N>=N_em.
    # Disabled when N_em <= 0.
    n_emerg_prey = float(config.get("emergency_breeding_n_prey", 0.0))
    n_emerg_pred = float(config.get("emergency_breeding_n_pred", 0.0))
    if (n_emerg_prey > 0.0 or n_emerg_pred > 0.0) and (
        prey_count is not None and pred_count is not None
    ):
        prey_cf = prey_count.astype(jnp.float32)
        pred_cf = pred_count.astype(jnp.float32)
        emerg_prey = (
            jnp.clip(1.0 - prey_cf / max(n_emerg_prey, 1e-9), 0.0, 1.0)
            if n_emerg_prey > 0.0 else jnp.float32(0.0)
        )
        emerg_pred = (
            jnp.clip(1.0 - pred_cf / max(n_emerg_pred, 1e-9), 0.0, 1.0)
            if n_emerg_pred > 0.0 else jnp.float32(0.0)
        )
        emerg_factor = jnp.where(species == 0, emerg_prey, emerg_pred)
        p_emergency = kappa_b * emerg_factor
        return jnp.maximum(p_standard, p_emergency)
    return p_standard


def _write_death_ages_jax(ring_prey, ring_pred, idx_prey, idx_pred,
                           dead_mask, species, ages):
    """Append ages of just-died agents into per-species ring buffers (v10).

    Pure JAX. Iterates over agent slots via lax.scan; for each dying
    slot, writes age into the appropriate species ring at idx % ring_size
    and advances that species' index. Slots that aren't dying contribute
    a no-op (the ring is rewritten with its own value at the same index).
    """
    ring_size = ring_prey.shape[0]
    n_slots = dead_mask.shape[0]

    def step_fn(carry, slot):
        rp, rd, ip, id_ = carry
        died = dead_mask[slot]
        is_prey = species[slot] == 0
        is_pred = species[slot] == 1
        age = ages[slot]
        write_prey = died & is_prey
        write_pred = died & is_pred

        wi_prey = ip % ring_size
        wi_pred = id_ % ring_size

        # No-op write: replace target cell with itself when not writing.
        new_prey_val = jnp.where(write_prey, age, rp[wi_prey])
        new_pred_val = jnp.where(write_pred, age, rd[wi_pred])

        rp = rp.at[wi_prey].set(new_prey_val)
        rd = rd.at[wi_pred].set(new_pred_val)
        ip = ip + jnp.where(write_prey, 1, 0)
        id_ = id_ + jnp.where(write_pred, 1, 0)
        return (rp, rd, ip, id_), None

    (rp, rd, ip, id_), _ = jax.lax.scan(
        step_fn, (ring_prey, ring_pred, idx_prey, idx_pred), jnp.arange(n_slots),
    )
    return rp, rd, ip, id_


def process_births_and_deaths_jax(sim_state, config, rollout_ptrs_for_done=None):
    """Process deaths and births for all agents. Pure JAX, JIT-compatible.

    Deaths: energy < 0 OR random < hazard_prob.
    Births: random < birth_prob AND under population cap.
    Uses lax.scan for births (fixed max_births_per_step iterations).

    Args:
        sim_state: SimState before births/deaths.
        config: dict.
        rollout_ptrs_for_done: (max_agents,) int32 | None. If provided, marks
            rollout_dones[slot, rollout_ptrs_for_done[slot]] = True for any
            agent dying from hazard/starvation this step (D23). Caller should
            pass the safe_ptrs used to write this step's reward/done, so the
            terminal flag lands on the same rollout slot that has the last
            obs/action/reward of the dying agent.
    """
    max_births_per_step = 20  # fixed scan length
    max_agents = sim_state.is_active.shape[0]
    energy_share_ratio = config["energy_share_ratio"]
    prey_cap = config.get("prey_cap", 450)
    predator_cap = config.get("predator_cap", 50)

    # --- Death processing ---
    rng, death_key = jax.random.split(sim_state.rng_key)
    death_randoms = jax.random.uniform(death_key, shape=(max_agents,))

    h_all = _batch_hazard_prob_jax(sim_state.ages, sim_state.energies, sim_state.species, config)
    dead_mask = sim_state.is_active & ((sim_state.energies < 0) | (death_randoms < h_all))

    # Deactivate dead agents
    new_is_active = sim_state.is_active & ~dead_mask

    # Zero dead agents' rollout pointers and velocities
    new_rollout_ptrs = jnp.where(dead_mask, 0, sim_state.rollout_ptrs)

    # Update physics is_active
    circle = sim_state.phyjax_stated.get("circle")
    import phyjax2d as pj
    new_phys_active = circle.is_active & ~dead_mask
    new_vel_xy = jnp.where(dead_mask[:, None], 0.0, circle.v.xy)
    new_vel_ang = jnp.where(dead_mask, 0.0, circle.v.angle)
    circle = circle.replace(
        v=pj.Velocity(angle=new_vel_ang, xy=new_vel_xy),
        is_active=new_phys_active,
    )
    new_stated = sim_state.phyjax_stated.replace(circle=circle)

    # D21: bump cumulative death counter for hazard/starvation deaths here.
    # (D20 catch-deaths are counted separately in sim_step_core.)
    deaths_this_step = jnp.sum(dead_mask.astype(jnp.int32))

    # v10 death-age ring: capture ages of just-died agents into per-species
    # ring buffers BEFORE deactivation. Reading live ages has inspection
    # bias (survivors skew long); this gives a true window of death-age
    # distribution to log against. Skipped when ring_size==1 (sentinel
    # for "disabled" since we still need a tracked field for checkpoint
    # shape compatibility).
    ring_size = int(sim_state.death_age_ring_prey.shape[0])
    if ring_size > 1:
        ring_prey, ring_pred, idx_prey, idx_pred = _write_death_ages_jax(
            sim_state.death_age_ring_prey,
            sim_state.death_age_ring_pred,
            sim_state.death_age_idx_prey,
            sim_state.death_age_idx_pred,
            dead_mask,
            sim_state.species,
            sim_state.ages,
        )
    else:
        ring_prey = sim_state.death_age_ring_prey
        ring_pred = sim_state.death_age_ring_pred
        idx_prey = sim_state.death_age_idx_prey
        idx_pred = sim_state.death_age_idx_pred

    # D23: mark hazard-dead agents' last rollout slot as terminal. The caller
    # (sim_step_core) passes safe_ptrs from the pre-step write, which is the
    # position where this dying agent's last obs/action/reward landed. We
    # OR-merge (via at[].max()) to preserve any prey_caught_mask=True flags
    # already written in sim_step_core step 6.
    if rollout_ptrs_for_done is not None:
        agent_idx = jnp.arange(max_agents)
        new_rollout_dones = sim_state.rollout_dones.at[
            agent_idx, rollout_ptrs_for_done
        ].max(dead_mask)
    else:
        new_rollout_dones = sim_state.rollout_dones

    sim_state = sim_state.replace(
        is_active=new_is_active,
        rollout_ptrs=new_rollout_ptrs,
        phyjax_stated=new_stated,
        cum_deaths=sim_state.cum_deaths + deaths_this_step,
        rollout_dones=new_rollout_dones,
        death_age_ring_prey=ring_prey,
        death_age_ring_pred=ring_pred,
        death_age_idx_prey=idx_prey,
        death_age_idx_pred=idx_pred,
    )

    # --- Birth processing ---
    rng, birth_key = jax.random.split(rng)
    birth_randoms = jax.random.uniform(birth_key, shape=(max_agents,))

    # Compute post-death populations early — used both by DDB (if enabled) to
    # scale the breeding threshold, and by the population caps below.
    prey_count = jnp.sum(new_is_active & (sim_state.species == 0))
    pred_count = jnp.sum(new_is_active & (sim_state.species == 1))

    b_all = _batch_birth_prob_jax(
        sim_state.energies, sim_state.species, config,
        prey_count=prey_count, pred_count=pred_count,
        is_active=new_is_active,
    )
    wants_birth = sim_state.is_active & (birth_randoms < b_all)

    prey_under_cap = prey_count < prey_cap
    pred_under_cap = pred_count < predator_cap

    # Filter by cap: prey parents only if under prey cap, etc.
    can_birth = wants_birth & jnp.where(
        sim_state.species == 0, prey_under_cap, pred_under_cap
    )

    # Collect parent slots that will birth (take first max_births_per_step).
    # We then pad out to max_births_per_step with the max_agents sentinel so
    # the lax.scan below always gets a fixed-length input even for tiny
    # configs where max_agents < max_births_per_step (e.g. unit tests).
    birth_indices = jnp.where(can_birth, jnp.arange(max_agents), max_agents)
    sorted_birth = jnp.sort(birth_indices)  # valid births first, max_agents padding at end
    pad_tail = jnp.full(max_births_per_step, max_agents, dtype=jnp.int32)
    sorted_birth = jnp.concatenate([sorted_birth, pad_tail])
    parent_slots = sorted_birth[:max_births_per_step]

    # lax.scan over potential births
    rng, spawn_rng = jax.random.split(rng)
    spawn_keys = jax.random.split(spawn_rng, max_births_per_step)

    def do_one_birth(carry, inputs):
        state, prey_ct, pred_ct = carry
        parent_slot, spawn_key = inputs

        is_valid = parent_slot < max_agents
        parent_species = state.species[parent_slot]

        # Check cap at this point (caps may have been reached by earlier births in scan)
        still_under_cap = jnp.where(
            parent_species == 0,
            prey_ct < prey_cap,
            pred_ct < predator_cap,
        )
        should_spawn = is_valid & still_under_cap & state.is_active[parent_slot]

        # Find first free slot WITHIN the parent's species range (D19 fix).
        # Prey slots are [0, prey_cap); predator slots are [prey_cap, max_agents).
        # Physics body radii are bound to slot index at builder time, so an
        # offspring must be spawned in a slot whose physics body matches its
        # species — otherwise it ends up in a wrong-sized body.
        slot_idx = jnp.arange(max_agents)
        in_species_range = jnp.where(
            parent_species == 0,
            slot_idx < prey_cap,
            slot_idx >= prey_cap,
        )
        inactive_slots = jnp.where(
            ~state.is_active & in_species_range,
            slot_idx,
            max_agents,
        )
        first_free = jnp.min(inactive_slots)
        has_slot = first_free < max_agents

        do_spawn = should_spawn & has_slot

        # Spawn offspring (always executes due to JIT, but result discarded if !do_spawn)
        new_state = spawn_offspring_jax(state, parent_slot, first_free, spawn_key, config)

        # Parent energy reduction
        parent_energy = state.energies[parent_slot]
        new_parent_energy = parent_energy - parent_energy * energy_share_ratio
        new_state = new_state.replace(
            energies=new_state.energies.at[parent_slot].set(
                jnp.where(do_spawn, new_parent_energy, state.energies[parent_slot])
            )
        )

        # Select between spawned and original state
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_spawn, new, old) if hasattr(new, 'shape') else new,
            new_state, state,
        )

        # Update counters
        prey_ct = prey_ct + jnp.where(do_spawn & (parent_species == 0), 1, 0)
        pred_ct = pred_ct + jnp.where(do_spawn & (parent_species == 1), 1, 0)

        return (state, prey_ct, pred_ct), None

    (sim_state, _, _), _ = jax.lax.scan(
        do_one_birth,
        (sim_state, prey_count, pred_count),
        (parent_slots, spawn_keys),
    )

    sim_state = sim_state.replace(rng_key=rng)
    return sim_state
