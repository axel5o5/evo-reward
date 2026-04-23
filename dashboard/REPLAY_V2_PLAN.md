Replay dashboard v2 — implementation plan

## Status

Order: 1.1 → 1.2 → 1.4 → 1.6 → 1.7 → 1.5 → 1.3 → Phase 2.

- [x] 1.1 URL state sync — shipped 73e2a75
- [x] 1.2 Selector revamp (ReplaySelector) — shipped 73e2a75
- [x] 1.4 Population mini-timeline (PopulationStrip) — shipped 3d285e7
- [x] 1.6 Click-agent panel lite (AgentInspector) — shipped 70fe1d0
- [x] 1.7 Canvas trail layer — shipped 70fe1d0
- [x] 1.5 "Interesting frames" chips — shipped 4fc2019
- [x] 1.3 Per-replay sparkline thumbnails — shipped 3fb3be1 (inlined into index.json instead of separate sparkline.json, see script header for rationale)
- [x] 2.1 Replay schema v2 — shipped b7ae35c (per-frame identity/lineage/phenotype/action; v1 replays still load)
- [x] 2.2 Full click-agent panel — shipped 5ad1846 (weight bars, action trace, parent-jump)
- [x] 2.3 Reward-weight histogram — shipped 54de062 (per-species overlaid, w_eat/w_act/w_prey/w_pred selector)
- [x] 2.4 Compare mode — shipped a41f66a (/replay/compare?a=…&b=… with shared scrubber)

v2 quantization — reward_weights/action go to int8 (scale 4/127 and 1/127),
agent_ids/parent_ids go to uint16 with a per-replay id_base offset. At
defaults this brings v2 replays from ~26 MB to ~14 MB (1.75× v1 vs 3.2×
unquantized). Falls back to int32 for ids if the window's id range exceeds
65534. Shipped before any v2 replays land on GCS so bucket stays tight.

Phase 2 deviations from plan:
  - reward_weights / parent_ids / ages are **per-frame** sections, not
    "static, written once". Slot reuse after death would otherwise
    misattribute the phenotype/parent to the previous occupant.
  - No separate generation-depth field — parent-jump traverses one hop
    at a time, and users can chain it. Pre-computing full lineage depth
    would require a second pass at load time for marginal UI value.
  - `action` is sampled from the last-written `rollout_actions` slot
    (`(ptrs - 1) % rollout_steps`). First frame after a PPO ptr reset
    reads the stale tail of the rollout buffer — documented artifact.

Existing v1 replays still render: phenotype/action blocks in the
inspector drop out; WeightHistogram shows a "phenotype not recorded"
notice. To populate the new payload, re-record with the current
recorder.

---

Two phases: Phase 1 is pure frontend (no replay schema change, all data already in the .bin). Phase 2 extends the recorder to unlock research payload. Do Phase 1 end-to-end before touching Phase 2 — you'll hit diminishing UX returns and want validation before re-recording replays.

Phase 1 — Frontend-only (no schema change)
Goal: the selector stops being friction, replays become URL-addressable and investigable, and clicking an agent shows what the current binary already knows.

1.1 URL state sync — Replay.tsx

Read ?tag=&exp=&seed=&step=&frame=&agent= from window.location.search on mount; fall back to first replay if unset.
On any change to selected / frameIdx / pinnedSlot, history.replaceState the new query string (debounce frame to ~200ms so scrubbing doesn't spam history).
Side benefit: localStorage "last opened" is now free (just cache the URL).
1.2 Selector revamp — new components/ReplaySelector.tsx, replace the <select> block in Replay.tsx

Three stacked chip rows: tag → exp → seed. Clicking a chip filters the rows below it.
Below the chips, a timeline strip: one dot per start_step for the current (tag, exp, seed) tuple, positioned along a horizontal axis scaled to the experiment's max trained step. Hover shows start_step + n_frames; click loads.
Keep the current grouped/tagOrder logic from Replay.tsx:105-131 — the data model is fine, it's only the rendering that changes.
1.3 Per-replay sparkline thumbnails — requires a small build-time step, not a schema change

Add scripts/gen_replay_thumbnails.py that reads each metrics.npz (if present alongside the replay directory) and emits a tiny sparkline.json with downsampled prey/pred counts over the window.
ReplaySelector renders a 60×18 inline SVG from that JSON next to each timeline dot.
If sparkline.json is missing, fall back to no thumbnail — graceful degradation so you can ship the selector without waiting on regeneration.
1.4 Population mini-timeline under the canvas — new components/PopulationStrip.tsx

Computed client-side from data.alive + data.species: one pass yields preyCount[f] and predCount[f] for all frames. Memoize per data instance.
Render as two overlaid SVG polylines across the full replay width; vertical playhead line at frameIdx.
Clicking the strip sets frameIdx (another way to scrub with context).
1.5 "Interesting frames" chips

Same precomputation pass as 1.4 also emits events: births = frames where alive[i] flipped 0→1, deaths = frames where 1→0, peak_pred, peak_prey, first_crossover (first frame where pred > prey or vice versa).
Render a single row of chips above the scrubber: "▶ birth ×12", "☠ death ×34", "↑ peak-pred @812", etc. Click a chip → sets frameIdx to next occurrence after current frame (wraps around).
1.6 Click-agent panel (lite) — new components/AgentInspector.tsx

Add pointer handling to ReplayCanvas: on click, hit-test against each alive agent (distance from click to pos[i] < radii[i]). On hit, emit onAgentPick(slotIdx).
Panel contents from data we already have:
slot, species, radius, current energy, heading (degrees), position
energy sparkline over all frames for that slot (grey where alive=0)
"distance travelled" = sum of per-frame position deltas while alive
pin toggle → when pinned, ReplayCanvas draws a trail (last N=50 frames of positions) and a highlight ring
Close button clears pin and URL ?agent=.
1.7 Canvas trail layer — extend ReplayCanvas.tsx

Add optional pinnedSlot?: number prop. In draw(), after the agent loop, if pinned + alive, draw a polyline through data.pos[f*N*2 + pinnedSlot*2 ..] for f ∈ [frameIdx-50, frameIdx], fading alpha by recency.
Phase 1 acceptance: you can paste a URL to a teammate, they land on the right frame with the right agent pinned; selector shows all your current runs without needing to read labels; clicking an agent reveals an energy sparkline; population strip makes it obvious where in an LV cycle you are.

Phase 2 — Schema extension for research payload
Only worth doing once Phase 1 is shipped and you know the inspector is where you spend time.

2.1 Extend the replay binary — replay_recorder.py + replayLoader.ts

Bump meta.json.version to 2.
Add sections (per frame unless noted):
agent_id — int32, shape (n_frames, max_agents) — stable identity across slot reuse
action — float32, shape (n_frames, max_agents, action_dim) — lets you distinguish "frozen" from "moving but ineffectual"
Add static sections (length = max_agents, written once):
parent_id — int32, -1 for founders
birth_step — int32
reward_weights — float32, shape (max_agents, n_weights) — the phenotype
replayLoader.ts gains matching fields on ReplayData and per-frame views. Gate on meta.version >= 2 so v1 replays still load (inspector panel shows "phenotype not recorded for this replay").
2.2 Full click-agent panel

Reward weights (w_eat, w_act, w_prey, w_pred) as labeled bars.
Lineage: parent_id clickable → jumps to the frame the parent was last alive and pins them. Generation depth computed by walking parent_id chain at load time.
Action trace: forward/turn sparklines aligned with the energy sparkline — you can visually correlate "energy dropped when action went to zero."
2.3 Reward-weight distribution histogram

New components/WeightHistogram.tsx: per-species histogram of w_pred (and any axis selector) over currently-alive agents at frameIdx.
Watching the mode shift across step_nums is the fear-evolution result — this is the scientific payoff of the whole page.
2.4 Compare mode

New route /replay/compare?a=...&b=... with two ReplayCanvas instances + one shared scrubber, one shared population strip per side.
Constrain to matched seed to keep the comparison honest.
Phase 2 acceptance: open any replay, click a predator during an LV crash, immediately see its evolved w_prey is near zero and its lineage goes back N generations. That's a research finding surfaced in one click.

Non-goals / deliberately deferred
Mouth cone / tactile bins / velocity arrows: you said the renderer is decent; skip these unless a specific debugging question resurfaces them.
GIF/MP4 export: nice-to-have, but scripts/replay_render.py already exists for offline rendering — not worth a frontend build.
Search/filter by outcome: depends on 1.3's sparkline data being present everywhere; revisit after regenerating thumbnails.
Order to implement
1.1 → 1.2 → 1.4 → 1.6 → 1.7 → 1.5 → 1.3 → Phase 2. Ship 1.1 and 1.2 as a first PR so the selector pain goes away immediately; the rest can follow independently.