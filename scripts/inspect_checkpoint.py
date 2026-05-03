"""Inspect one or two saved checkpoints — population, weights, residual MLP, cohort survival.

Usage:
    python scripts/inspect_checkpoint.py <ckpt_path>
    python scripts/inspect_checkpoint.py <old_ckpt> <new_ckpt>

The single-checkpoint mode prints a snapshot summary. The two-checkpoint mode
adds a diff: prey-fear drift, residual L1 evolution, and predator cohort
survival between the two saves.

Backward-compat: v8-era checkpoints were saved before v10 added the four
death-age ring fields to SimState. We strip those leaves from the v10 template
when unflattening so the loader works against either era's saves. To customize
the config, edit CFG_PATH at the top.
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.jax_state import init_simstate
from src.config_utils import resolve_scale_dependent_params

CFG_PATH = "configs/axis1/med.yaml"
RING_FIELDS = {
    "death_age_ring_prey", "death_age_ring_pred",
    "death_age_idx_prey", "death_age_idx_pred",
}


def make_loader():
    with open(CFG_PATH) as f:
        config = yaml.safe_load(f)
    resolve_scale_dependent_params(config)
    template = init_simstate(config, jax.random.PRNGKey(config["seed"]))
    full_leaves, treedef = jtu.tree_flatten(template)
    paths = jtu.tree_flatten_with_path(template)[0]
    v8_idxs = [i for i, (p, _) in enumerate(paths)
               if getattr(p[0], "name", None) not in RING_FIELDS]

    def load(path):
        d = np.load(path, allow_pickle=False)
        n = sum(1 for k in d.files if k.startswith("leaf_"))
        loaded = [d[f"leaf_{i}"] for i in range(n)]
        new = list(full_leaves)
        if n == len(full_leaves):
            new = loaded
        else:
            assert n == len(v8_idxs), (
                f"Unexpected leaf count {n}; expected {len(full_leaves)} (v10) "
                f"or {len(v8_idxs)} (v8)"
            )
            for v8i, leaf in enumerate(loaded):
                new[v8_idxs[v8i]] = leaf
        return jtu.tree_unflatten(treedef, [jnp.asarray(l) for l in new])
    return load


def residual_l1_per_agent(s):
    leaves = jtu.tree_leaves(s.reward_mlp_params)
    if not leaves:
        return None
    n_agents = int(np.asarray(s.is_active).shape[0])
    sums = np.zeros(n_agents)
    for leaf in leaves:
        flat = np.asarray(leaf).reshape(leaf.shape[0], -1)
        sums += np.abs(flat).sum(axis=1)
    return sums


def summarize(s, label):
    is_active = np.asarray(s.is_active)
    species = np.asarray(s.species)
    ages = np.asarray(s.ages)
    rw = np.asarray(s.reward_weights)
    energies = np.asarray(s.energies)

    print(f"\n=== {label} ===")
    print(f"  step={int(s.step):,}  next_id={int(s.next_agent_id):,}  "
          f"cum_catches={int(s.cum_catches):,}  "
          f"cum_deaths={int(s.cum_deaths):,}  "
          f"cum_feedings={int(s.cum_feedings):,}")

    l1 = residual_l1_per_agent(s)

    for sp_name, sp in [("Predators", 1), ("Prey", 0)]:
        m = is_active & (species == sp)
        n = int(m.sum())
        if n == 0:
            print(f"  {sp_name}: n=0")
            continue
        a = ages[m]
        e = energies[m]
        w = rw[m]
        print(f"\n  {sp_name} (n={n})")
        print(f"    age:    median={int(np.median(a)):>7,d}  "
              f"p75={int(np.percentile(a,75)):>7,d}  max={int(a.max()):>7,d}")
        print(f"    energy: min={e.min():>5.1f}  mean={e.mean():>5.1f}  "
              f"max={e.max():>5.1f}")
        for j, name in enumerate(["w_eat ", "w_act ", "w_prey", "w_pred"]):
            print(f"    {name}: mean={w[:,j].mean():+.3f}  "
                  f"std={w[:,j].std():.3f}")
        if l1 is not None:
            l1_sp = l1[m]
            print(f"    residual L1: mean={l1_sp.mean():.3f}  "
                  f"median={np.median(l1_sp):.3f}  max={l1_sp.max():.3f}")


def diff(s_old, s_new, label_old="OLD", label_new="NEW"):
    print(f"\n=== Δ {label_old} → {label_new} ===")
    rw_old = np.asarray(s_old.reward_weights)
    rw_new = np.asarray(s_new.reward_weights)
    act_old = np.asarray(s_old.is_active)
    act_new = np.asarray(s_new.is_active)
    sp_old = np.asarray(s_old.species)
    sp_new = np.asarray(s_new.species)

    # Reward-weight drift per species
    for sp_name, sp in [("Pred", 1), ("Prey", 0)]:
        m_old = act_old & (sp_old == sp)
        m_new = act_new & (sp_new == sp)
        if m_old.sum() == 0 or m_new.sum() == 0:
            continue
        print(f"\n  {sp_name} reward-weight means:")
        for j, name in enumerate(["w_eat ", "w_act ", "w_prey", "w_pred"]):
            o = rw_old[m_old][:, j].mean()
            n = rw_new[m_new][:, j].mean()
            print(f"    {name}: {o:+.3f}  →  {n:+.3f}  (Δ {n-o:+.3f})")

    # Residual L1 drift per species
    l1_old = residual_l1_per_agent(s_old)
    l1_new = residual_l1_per_agent(s_new)
    if l1_old is not None and l1_new is not None:
        print(f"\n  Residual MLP L1 norm means:")
        for sp_name, sp in [("Pred", 1), ("Prey", 0)]:
            m_old = act_old & (sp_old == sp)
            m_new = act_new & (sp_new == sp)
            if m_old.sum() == 0 or m_new.sum() == 0:
                continue
            o = l1_old[m_old].mean()
            n = l1_new[m_new].mean()
            print(f"    {sp_name}: {o:.3f}  →  {n:.3f}  "
                  f"(×{n/max(o,1e-6):.2f})")

    # Predator cohort survival
    ids_old = np.asarray(s_old.agent_ids)
    ids_new = np.asarray(s_new.agent_ids)
    for sp_name, sp in [("Predators", 1), ("Prey", 0)]:
        m_old = act_old & (sp_old == sp)
        m_new = act_new & (sp_new == sp)
        ids_o = set(int(x) for x in ids_old[m_old])
        ids_n = set(int(x) for x in ids_new[m_new])
        survived = ids_o & ids_n
        died = ids_o - ids_n
        born = ids_n - ids_o
        n_old, n_new = len(ids_o), len(ids_n)
        if n_old == 0 and n_new == 0:
            continue
        sr = (100.0 * len(survived) / n_old) if n_old else 0.0
        print(f"\n  {sp_name} cohort:")
        print(f"    alive at {label_old}: {n_old}; alive at {label_new}: {n_new}")
        print(f"    survived both: {len(survived)} ({sr:.1f}% of {label_old})")
        print(f"    died in window: {len(died)};  "
              f"born in window (alive at {label_new}): {len(born)}")


def main(args):
    if len(args) not in (1, 2):
        print(__doc__)
        sys.exit(1)

    load = make_loader()
    if len(args) == 1:
        s = load(args[0])
        summarize(s, args[0])
        return

    s_old = load(args[0])
    s_new = load(args[1])
    label_old = Path(args[0]).name
    label_new = Path(args[1]).name
    summarize(s_old, label_old)
    summarize(s_new, label_new)
    diff(s_old, s_new, label_old, label_new)


if __name__ == "__main__":
    main(sys.argv[1:])
