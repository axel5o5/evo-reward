# configs/archive/

Configs that have been superseded, shelved, or deferred. Not actively run.
Kept here for git diffs, occasional comparison runs, and to preserve the
trail of what we tried.

For why each was moved here, see [docs/findings.md](../../docs/findings.md)
and [docs/CURRENT_STATE.md](../../docs/CURRENT_STATE.md).

## Superseded baselines (scaffold-tuning stepping stones)

| File | Outcome |
|---|---|
| [baseline_smol_ddb.yaml](baseline_smol_ddb.yaml) | Small-scale + DDB floor=0.3. Extinct ~80K. (findings §15.6) |
| [baseline_med_ddb.yaml](baseline_med_ddb.yaml) | Medium + DDB floor=0.3. Lone-survivor starvation ~100K. (findings §15.7-8) |
| [baseline_med_ddb_ddm.yaml](baseline_med_ddb_ddm.yaml) | Medium + DDB+DDM floor=0.3. 1.35M then trophic-collapse-via-herd. (findings §15.9-10) |
| [baseline_endpoint.yaml](baseline_endpoint.yaml) | Early endpoint-code parameterization (rectangular world, 9 tactile bins). Documented as fallback in emevo-diff.md; not the active replication target. |

## Superseded axis configs

| File | Why archived |
|---|---|
| [axis1_mlp_reward.yaml](axis1_mlp_reward.yaml) | Full-MLP-replacement reward genome. Bootstrap failure (extincted before evolution converged). Replaced by `axis1_residual.yaml` (residual design, zero-init MLP perturbation). (findings §11) |
| [axis2_aligned.yaml](axis2_aligned.yaml) | Earlier paper-scale variant. Replaced by `axis2_aligned_smol.yaml` (med-large scale + DDB+DDM scaffolds). |
| [axis2_social_obs.yaml](axis2_social_obs.yaml) | Slot-based social observation. Replaced by bin-aligned heading encoding (the active axis-2 mechanism). (findings §13) |

## Shelved combined experiments

The original 2×2 plan (linear/temporal × position-only/social) was scoped out in the 2026-04-29 strategic reset (findings §15.4).

| File | Status |
|---|---|
| [axis2_both_1.yaml](axis2_both_1.yaml) | 2×2 combined. Shelved. |
| [axis2_cross_1.yaml](axis2_cross_1.yaml) | 2×2 cross. Shelved. |

## Deferred (may revisit)

| File | Why deferred |
|---|---|
| [axis3_temporal_reward.yaml](axis3_temporal_reward.yaml) | 945-param temporal MLP has the same bootstrap problem as original axis-1; residual design would need redesigning for temporal context. (findings §15.4) |
| [axis4_lstm_policy.yaml](axis4_lstm_policy.yaml) | LSTM policy + truncated BPTT wired in commit 5e69965 but never exercised end-to-end. Lower-priority extension. (findings §15.4) |

## experiments/

[experiments/](experiments/) — early sweeps and ad-hoc tuning configs (`d28*`, `sweep_*`, `tune_eta_*`, `v8_no_cooldown`, `v9_slow_evolution`). All superseded by the post-§15 baseline + axis configs.
