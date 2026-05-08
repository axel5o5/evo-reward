import numpy as np, sys
sys.path.insert(0, ".")
from analysis.checkpoint_explorer import load

ckpt = "results/axis1_residual_reward_mlp_small/seed_0/2026-05-08T0407Z/checkpoints/step_02700000.npz"
state = load(ckpt, config="configs/axis1/small.yaml")
ages = np.asarray(state.ages)
energies = np.asarray(state.energies)
species = np.asarray(state.species)
active = np.asarray(state.is_active).astype(bool)
mask = active & (species == 1)
E = energies[mask]
n = mask.sum()

T_pred = 12.0
alpha = 0.75
kappa_b = 1e-3
beta_b = 0.4
zeta_pred = 100.0
bonus_global_pred = 5.0
bonus_emerg_pred = 50.0

factor = (n*n) / (n*n + T_pred*T_pred)
print("=== Scaffold at predator pop=%d (T_pred=%g) ===" % (n, T_pred))
print("DDB factor: %.3f -> breeding boost = %.2fx" % (factor, 1.0/factor))
print("zeta_eff = %.1f, E_cliff = zeta_eff/beta_b = %.1f" % (zeta_pred*factor, zeta_pred*factor/beta_b))

bonus = bonus_global_pred + bonus_emerg_pred * (1 - factor)
print("\n=== Birth-energy bonus per predator birth ===")
print("bonus = %.0f + %.0f * (1 - %.3f) = %.1f" % (bonus_global_pred, bonus_emerg_pred, factor, bonus))
print("(each parent + each child get +%.1f at every birth)" % bonus)

p_std = kappa_b / (1.0 + np.exp(zeta_pred*factor - beta_b * E))
print("\n=== Per-agent standard breeding rate (per step) ===")
print("median %.2e   max %.2e   min %.2e" % (np.median(p_std), p_std.max(), p_std.min()))

k = alpha / (1 - alpha)
log_e = np.log(np.maximum(E, 1e-9))
logits = k * log_e
logits = logits - logits.max()
shares = np.exp(logits) / np.exp(logits).sum()

print("\n=== Breeding share concentration (alpha=%.2f, k=%.0f) ===" % (alpha, k))
order = np.argsort(-E)
print("rank   E      share%   cum%")
cum = 0.0
for r, idx in enumerate(order):
    cum += shares[idx]
    print(" %2d  %5.1f   %5.1f%%   %5.1f%%" % (r+1, E[idx], 100*shares[idx], 100*cum))

shares_old = np.power(np.maximum(E, 1e-9), 1.0)
shares_old /= shares_old.sum()
print("\n=== Compare alpha=0.75 vs alpha=0.5 ===")
print("rank   alpha=0.75   alpha=0.5   ratio")
for r, idx in enumerate(order):
    ratio = shares[idx] / max(shares_old[idx], 1e-9)
    print(" %2d   %6.2f%%   %6.2f%%   %.2fx" % (r+1, 100*shares[idx], 100*shares_old[idx], ratio))
print("Top 3:    alpha=0.75 %.1f%% vs alpha=0.5 %.1f%%" % (100*shares[order[:3]].sum(), 100*shares_old[order[:3]].sum()))
print("Bottom 3: alpha=0.75 %.1f%% vs alpha=0.5 %.1f%%" % (100*shares[order[-3:]].sum(), 100*shares_old[order[-3:]].sum()))

# Total breeding boost analysis
prey_count_eff = 12  # n
boost_uniform = 1.0 / max(factor, kappa_b)
agent_boost = boost_uniform * n * shares
agent_kappa = kappa_b * agent_boost
total_p_std = p_std.sum() if False else (kappa_b * agent_boost / (1.0 + np.exp(zeta_pred*factor - beta_b * E)))
print("\n=== Per-agent breeding probability with scaffold (per step) ===")
print("rank   E      P_birth/step   E[births in 10K steps]")
for r, idx in enumerate(order):
    p = total_p_std[idx]
    expected_in_10k = 1 - (1-p)**10000 if p > 0 else 0
    print(" %2d  %5.1f   %.3e        %.2f" % (r+1, E[idx], p, expected_in_10k))
