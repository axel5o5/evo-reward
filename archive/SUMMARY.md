# Archive summary

Per-run roll-up of population and extinction stats from public replays.
Per-checkpoint detail lives in `runs/<exp>__seed_<N>__<tag>.json`.

| exp | seed | run_tag | final_step | ckpts | extinct | extinct@ | species | peak_prey | peak_pred |
|-----|------|---------|-----------:|------:|:-------:|---------:|:-------:|----------:|----------:|
| axis1_mlp_reward | 0 | 2026-04-28_axis1_mouth_smol_1M | 1,000,000 | 50 | yes | 370,001 | pred | 450 | 36 |
| axis1_mlp_reward | 0 | 2026-04-28_axis1_mouth_smol_1M_mut08 | 380,000 | 19 | yes | 290,001 | pred | 450 | 20 |
| axis1_mlp_reward | 0 | 2026-04-29_axis1_mouth_smol_1M_mut03 | 100,000 | 5 | no | — | none | 450 | 23 |
| axis1_residual | 0 | 2026-05-01T1646Z | 140,000 | 7 | no | — | none | 300 | 18 |
| axis1_residual | 0 | 2026-05-01T1800Z | 220,000 | 11 | no | — | none | 375 | 23 |
| axis1_residual | 0 | 2026-05-01T2019Z | 160,000 | 8 | yes | 90,001 | pred | 375 | 22 |
| axis2_aligned | 0 | 2026-04-30T1806Z | 1,500,000 | 58 | yes | 610,001 | pred | 450 | 23 |
| axis2_social_obs | 0 | 2026-04-28_axis2_mouth_smol_1M | 1,180,000 | 59 | yes | 1,090,001 | pred | 450 | 28 |
| axis2_social_obs | 1 | 2026-04-29_axis2_mouth_smol_2M_seed1 | 20,000 | 1 | no | — | none | 450 | 18 |
| baseline_faithful | 0 | 2026-04-21 | 100,000 | 1 | no | — | none | 450 | 50 |
| baseline_faithful | 0 | 2026-04-21T1935Z_post-d19 | 160,000 | 8 | no | — | none | 450 | 50 |
| baseline_faithful | 0 | 2026-04-21T2159Z_phase1a-v2 | 80,000 | 4 | no | — | none | 448 | 43 |
| baseline_faithful | 0 | 2026-04-21T2319Z_phase1a-v3 | 400,000 | 20 | yes | 70,001 | pred | 450 | 49 |
| baseline_faithful | 0 | 2026-04-21_pre_d18_fix | 1,800,000 | 17 | no | — | none | 450 | 50 |
| baseline_faithful | 0 | 2026-04-22T1417Z_phase1a-v4 | 60,000 | 3 | no | — | none | 450 | 42 |
| baseline_faithful | 0 | 2026-04-22T1546Z_phase1a-v5 | 680,000 | 34 | yes | 410,001 | pred | 450 | 50 |
| baseline_faithful | 0 | 2026-04-23T0400Z | 520,000 | 26 | yes | 70,001 | pred | 450 | 40 |
| baseline_faithful | 0 | 2026-04-23T1008Z | 320,000 | 16 | yes | 70,001 | pred | 450 | 41 |
| baseline_faithful | 0 | 2026-04-23_d19 | 220,000 | 11 | yes | 90,001 | pred | 450 | 38 |
| baseline_faithful | 0 | 2026-04-24_d28a | 80,000 | 4 | yes | 70,001 | pred | 450 | 42 |
| baseline_faithful | 0 | 2026-04-24_d28b | 80,000 | 4 | yes | 70,001 | pred | 450 | 36 |
| baseline_faithful | 0 | 2026-04-24_d30 | 220,000 | 11 | yes | 50,001 | pred | 450 | 42 |
| baseline_faithful | 0 | 2026-04-24_d31a | 80,000 | 4 | yes | 70,001 | pred | 450 | 38 |
| baseline_faithful | 0 | 2026-04-24_d31b | 80,000 | 4 | yes | 70,001 | pred | 450 | 37 |
| baseline_faithful | 0 | 2026-04-24_d31c | 80,000 | 4 | no | — | none | 450 | 35 |
| baseline_faithful | 0 | 2026-04-24_d31d | 80,000 | 4 | no | — | none | 450 | 39 |
| baseline_faithful | 1 | 2026-04-22T2328Z_phase1a-v7-seed1-sensor120 | 380,000 | 19 | yes | 110,001 | pred | 450 | 40 |
| baseline_faithful | 1 | 2026-04-24_d19 | 20,000 | 1 | no | — | none | 450 | 29 |
| baseline_med_ddb | 0 | 2026-04-30_baseline_med_ddb_2M | 120,000 | 6 | yes | 110,001 | pred | 300 | 16 |
| baseline_med_ddb_ddm | 0 | 2026-04-30_baseline_med_ddb_ddm_2M | 1,980,000 | 99 | yes | 1,350,001 | pred | 300 | 20 |
| baseline_smol_ddb | 0 | 2026-04-29_baseline_smol_ddb_2M | 2,000,000 | 100 | yes | 70,001 | pred | 200 | 14 |
| exp_sweep_mouth_smol | 0 | 2026-04-27_sweep_mouth_smol_1M | 1,000,000 | 50 | no | — | none | 450 | 28 |
| exp_sweep_mouth_smol | 1 | 2026-04-27_sweep_mouth_smol_1M_seed1 | 720,000 | 36 | no | — | none | 450 | 30 |
| exp_tune_eta_0.45 | 0 | 2026-04-24_tune_eta_045 | 140,000 | 7 | no | — | none | 450 | 24 |
| exp_tune_eta_0.50 | 0 | 2026-04-25_tune_eta_050 | 1,000,000 | 50 | yes | 670,001 | pred | 450 | 30 |
| exp_tune_eta_0.55 | 0 | 2026-04-25_tune_eta_055 | 880,000 | 44 | yes | 710,001 | pred | 450 | 50 |
| exp_v8_no_cooldown | 0 | 2026-04-23T1558Z_v8-no-cooldown-seed0 | 420,000 | 21 | yes | 210,001 | pred | 450 | 36 |
