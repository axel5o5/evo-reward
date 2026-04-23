# Draft email to Kanagawa & Doya

**To:** yuji.kanagawa@oist.jp, doya@oist.jp
**Subject:** K&D 2025 replication — questions about sensor_length, predator_eat_interval, and seed variance

---

Dear Dr. Kanagawa and Dr. Doya,

I'm a student at Columbia's Applied Research in Learning lab working on a
close replication of "Evolution of Fear and Social Rewards in Prey-Predator
Relationship" (arXiv:2507.09992v2). We've followed the `gecco2026` branch of
`github.com/oist/emevo` closely — in particular `experiments/cf_predator.py`
and its defaulted env / bd / gops configs. Our implementation passes 217
unit tests, including assertions anchored against the paper's Tables 2-4,
Appendix A, and Figure 19 on the birth function.

Despite this, we've struggled to reproduce the stable predator-prey
oscillation that your Figure 6 depicts for the medium-mouth default
condition. Across multiple attempts we observe either (a) predator
over-success + prey crash + predator extinction on cycle 2 after prey fear
over-evolves, or (b) with `sensor_length=120` per Appendix A, predators
can't catch prey and extinct in cycle 1 before any fear evolution occurs.

Three specific questions where paper text and emevo code seem to disagree —
your guidance would save us substantial debugging time:

1. **`sensor_length`**: Paper Appendix A reads "proximity sensors with a
   maximum length of 120 units." The emevo env TOMLs we inspected
   (`20241212-predator.toml` and `20251122-predator-square.toml`) both set
   `sensor_length = 200.0`. Which value was used for the Figure 6 / Table 1
   experiments? We saw qualitatively different dynamics at each.

2. **`predator_eat_interval`**: Your `CircleForagingWithPredator` class has
   a `predator_eat_interval: int = 10` parameter implementing a per-predator
   catch cooldown, and the env TOMLs set it to 10. Paper Section 3 describes
   hunting as "initiating contact within a specific range (40 to 80)"
   without mentioning a cooldown. Was the cooldown active for the paper
   runs, and if so, what's the biological or algorithmic intuition
   (digestion time? prevent accidental over-catching?)?

3. **Mutation clip range**: Paper Section 4.2 states "reward weights are
   clipped into [−100, 100] after mutation for numerical stability." The
   gops TOML `20241010-mutation-t-2.toml` referenced by `cf_predator.py`
   uses `clip_min = -10.0, clip_max = 10.0`. Which was in effect for the
   paper's reported runs?

One related sanity check: **did any of seeds 1-5 exhibit predator
extinction** in your default medium-mouth runs, or was coexistence the
consistent outcome? We are seeing predator extinction reliably on seed 0
(and less reliably on seed 1), and we'd like to know whether that's within
normal seed variance we should simply repeat, or evidence of a remaining
bug in our code.

Thank you for sharing your work and the codebase. The paper is a
compelling read, and we've thoroughly enjoyed digging into the
coevolutionary dynamics it describes.

Best,
[your name]
Columbia Applied Research in Learning

---

*(Draft from 2026-04-23. Questions grounded in our `docs/emevo-diff.md`
D8/D18/D22/D26/D27 entries + `docs/phase1a-v5-analysis.md`.)*
