// Precomputed per-replay summary stats. Consumed by PopulationStrip (the
// polylines) and by the "interesting frames" chip row — shared so we only
// walk data.alive once per replay load.

import { ReplayData } from "./replayLoader";

export interface ReplayStats {
  prey: Uint16Array;           // prey count per frame
  pred: Uint16Array;           // predator count per frame
  maxCount: number;            // max(prey|pred) across all frames, floored at 1
  births: Int32Array;          // frames with ≥1 prey/pred birth (any slot flipping 0→1)
  deaths: Int32Array;          // frames with ≥1 death
  peakPreyFrame: number;       // argmax prey
  peakPredFrame: number;       // argmax pred
  firstCrossoverFrame: number; // first frame where (pred > prey) state differs from frame 0; -1 if never
}

export function computeReplayStats(data: ReplayData): ReplayStats {
  const n = data.meta.n_frames;
  const N = data.meta.max_agents;
  const prey = new Uint16Array(n);
  const pred = new Uint16Array(n);
  const births: number[] = [];
  const deaths: number[] = [];
  let maxC = 1;
  let peakPrey = 0;
  let peakPred = 0;
  let preyMax = -1;
  let predMax = -1;

  for (let f = 0; f < n; f++) {
    const base = f * N;
    const prevBase = (f - 1) * N;
    let p = 0;
    let d = 0;
    let birthHere = false;
    let deathHere = false;
    for (let i = 0; i < N; i++) {
      const a = data.alive[base + i];
      if (a) {
        if (data.species[i] === 1) d++;
        else p++;
      }
      if (f > 0) {
        const prev = data.alive[prevBase + i];
        if (prev === 0 && a === 1) birthHere = true;
        if (prev === 1 && a === 0) deathHere = true;
      }
    }
    prey[f] = p;
    pred[f] = d;
    if (p > maxC) maxC = p;
    if (d > maxC) maxC = d;
    if (p > preyMax) { preyMax = p; peakPrey = f; }
    if (d > predMax) { predMax = d; peakPred = f; }
    if (birthHere) births.push(f);
    if (deathHere) deaths.push(f);
  }

  // Crossover: sign of (pred - prey). "First crossover" = first frame whose
  // sign differs from frame 0. If pred==prey at frame 0, take the first
  // frame with a non-zero sign instead.
  const sign0 = Math.sign(pred[0] - prey[0]);
  let crossover = -1;
  for (let f = 1; f < n; f++) {
    const s = Math.sign(pred[f] - prey[f]);
    if (sign0 === 0) {
      if (s !== 0) { crossover = f; break; }
    } else if (s !== 0 && s !== sign0) {
      crossover = f;
      break;
    }
  }

  return {
    prey,
    pred,
    maxCount: maxC,
    births: Int32Array.from(births),
    deaths: Int32Array.from(deaths),
    peakPreyFrame: peakPrey,
    peakPredFrame: peakPred,
    firstCrossoverFrame: crossover,
  };
}

// Next event at or after `frame`, wrapping to the start if none found.
// Returns -1 if the array is empty.
export function nextEventAfter(events: ArrayLike<number>, frame: number): number {
  if (events.length === 0) return -1;
  for (let i = 0; i < events.length; i++) {
    if (events[i] > frame) return events[i];
  }
  return events[0]; // wrap
}
