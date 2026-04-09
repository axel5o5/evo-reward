STANDING CONVENTIONS (apply every session):

Git: After completing your task, stage all changed files and offer a commit with a
descriptive message: "[Phase X] short description". Do not commit while tests are failing.

Documentation: Any implementation decision not in interfaces.md or emevo-diff.md gets
documented there before committing. Code and doc changes commit together.

One thing at a time: Complete the stated task and pass the done-condition test before
doing anything else. Note but don't fix unrelated issues.

---

You are working on the evo-reward project. Read AGENTS.md first.

Session 1 is complete. The emevo gecco2026 branch has been audited and all 9 open
questions are resolved. Key confirmed values that changed significantly from our
initial assumptions:

- obs_dim = 205 (not 54): 32 sensors x 4 channels + 72 tactile bins + vx + vy + angle + ang_vel + energy
- 4 per-type sensor channels per proximity sensor: [prey, predator, food, wall]
- Sensor max range: 200 units (not 120)
- Initial energy: 100.0 for both species
- spawn_spread: 100.0 world units
- food_max: 600
- energy_share_ratio: 0.4
- Prey c_b = 1e-4, c_a = 2.5e-6 (code values, not paper Table 2 values)
- Action mapping: sigmoid scaling into [-20, 80], not hard clipping
- Network: 2 hidden layers (64, tanh), not 3
- No observation normalization

All of these are now in configs/baseline_faithful.yaml and docs/emevo-diff.md.
The observation vector layout in docs/interfaces.md has been rewritten.

Before implementing anything, read:
1. The updated docs/interfaces.md observation vector section -- confirm you understand
   the full 205-dim layout and the 4-channel sensor structure
2. The updated docs/emevo-diff.md -- understand all deviation entries D1-D10
3. The gecco2026 branch source for the sigmoid action mapping -- understand exactly
   how it maps raw network output to motor forces before writing policy.py

Your task this session:
1. Implement src/environment.py -- 2D world via phyjax2d, 4-channel proximity sensors
   (32 sensors x 4 object types), 72-bin tactile sensors, food dynamics (max=600),
   eating/capture detection. Match the confirmed 205-dim observation layout exactly.
2. Implement src/lifecycle.py -- energy updates using the CODE values (c_b=1e-4,
   c_a=2.5e-6 for prey), hazard h(t,e), birth b(e), process_births_and_deaths,
   food regeneration. Use energy_share_ratio=0.4, initial_energy=100.0.
3. Make pytest tests/test_components.py -k "lifecycle" pass.

Critical implementation notes:
- The reward equation's max_k(s_pred^k) is the MAX over the predator CHANNEL across
  all 32 sensors -- not the max over a mixed single channel. The channel separation
  matters for reward computation.
- Verify the sigmoid action mapping from emevo source before writing any motor output
  code. The derivative behavior at the action boundary is different from hard clipping
  and affects PPO convergence.
- spawn_spread=100 means offspring can spawn up to ~100 world units from parent
  in each dimension. This is large relative to the 960-unit world.

Done condition: pytest tests/test_components.py -k "lifecycle" passes with no failures.
Report: which emevo source files you referenced for environment.py, and confirm the
sigmoid action formula you're using.
