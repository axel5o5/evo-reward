import { useMemo } from "react";
import { ReplayData } from "../lib/replayLoader";

interface Props {
  data: ReplayData;
}

const COLOR_PREY = "#4ade80";
const COLOR_PRED = "#f87171";

interface BoxStats {
  n: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
}

function quantile(sorted: Int32Array, q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Collects ages across (frame × slot) cells where the agent is alive, split by
// species. We sample at most ~MAX_SAMPLES per species to keep the sort cheap
// on long replays — 200k cells × Int32 sort is ~10ms on a laptop, but a
// 10k-frame × 500-slot replay can hit 2.5M alive cells if pop is saturated.
//
// Sampling is uniform-stride (every k-th alive cell). This biases the sample
// toward agents present in many frames (long-lived ones contribute more
// cells), which is exactly the right weighting for "what age does an agent
// typically have at any given moment" — what the user actually sees in the
// canvas.
const MAX_SAMPLES_PER_SPECIES = 200_000;

function speciesStats(data: ReplayData, predator: boolean): BoxStats | null {
  if (!data.ages) return null;
  const n = data.meta.n_frames;
  const N = data.meta.max_agents;
  // First pass: count alive cells of this species so we can pick a stride.
  let alive = 0;
  for (let f = 0; f < n; f++) {
    const base = f * N;
    for (let s = 0; s < N; s++) {
      if (data.alive[base + s] === 0) continue;
      if ((data.species[s] === 1) !== predator) continue;
      alive++;
    }
  }
  if (alive === 0) return null;
  const stride = Math.max(1, Math.ceil(alive / MAX_SAMPLES_PER_SPECIES));
  const target = Math.ceil(alive / stride);
  const samples = new Int32Array(target);
  let idx = 0;
  let cursor = 0; // counts alive-of-species so we know when to sample
  let sum = 0;
  for (let f = 0; f < n && idx < target; f++) {
    const base = f * N;
    for (let s = 0; s < N && idx < target; s++) {
      if (data.alive[base + s] === 0) continue;
      if ((data.species[s] === 1) !== predator) continue;
      if (cursor % stride === 0) {
        const age = data.ages[base + s];
        samples[idx++] = age;
        sum += age;
      }
      cursor++;
    }
  }
  const usable = idx;
  const view =
    usable === samples.length ? samples : samples.subarray(0, usable);
  // typed arrays sort numerically by default
  const sorted = view.slice().sort();
  return {
    n: alive,
    min: sorted[0],
    q1: quantile(sorted, 0.25),
    median: quantile(sorted, 0.5),
    q3: quantile(sorted, 0.75),
    max: sorted[sorted.length - 1],
    mean: sum / usable,
  };
}

export default function AgeRangeBoxplot({ data }: Props) {
  const { prey, pred } = useMemo(
    () => ({
      prey: speciesStats(data, false),
      pred: speciesStats(data, true),
    }),
    [data],
  );

  if (!data.ages) {
    return null;
  }
  if (!prey && !pred) return null;

  const axisMax = Math.max(prey?.max ?? 0, pred?.max ?? 0, 1);

  return (
    <div className="w-full border border-gray-200 dark:border-gray-800 rounded p-2 bg-gray-50 dark:bg-gray-900/50">
      <div className="flex items-baseline justify-between mb-1.5">
        <div className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
          Age range (steps lived)
        </div>
        <div className="text-[10px] font-mono text-gray-400 dark:text-gray-500">
          across all alive agents over replay window
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        {prey && (
          <BoxRow label="prey" color={COLOR_PREY} stats={prey} axisMax={axisMax} />
        )}
        {pred && (
          <BoxRow label="pred" color={COLOR_PRED} stats={pred} axisMax={axisMax} />
        )}
      </div>
      <div className="mt-1.5 flex justify-between text-[9px] font-mono text-gray-400 dark:text-gray-500 px-[44px]">
        <span>0</span>
        <span>{axisMax.toLocaleString()} steps</span>
      </div>
    </div>
  );
}

function BoxRow({
  label,
  color,
  stats,
  axisMax,
}: {
  label: string;
  color: string;
  stats: BoxStats;
  axisMax: number;
}) {
  // Layout: 36px label · flex SVG track. The SVG is internally normalized
  // 0..1000 so the box scales proportionally to axisMax.
  const W = 1000;
  const H = 18;
  const x = (v: number) => (v / axisMax) * W;
  const xMin = x(stats.min);
  const xQ1 = x(stats.q1);
  const xMed = x(stats.median);
  const xQ3 = x(stats.q3);
  const xMax = x(stats.max);
  const xMean = x(stats.mean);
  const boxLeft = Math.min(xQ1, xQ3);
  const boxRight = Math.max(xQ1, xQ3);
  const boxW = Math.max(1, boxRight - boxLeft); // ensure a 1px sliver if Q1==Q3
  const tooltip =
    `${label}: n=${stats.n.toLocaleString()} alive cells · ` +
    `min ${stats.min} · Q1 ${Math.round(stats.q1)} · ` +
    `median ${Math.round(stats.median)} · Q3 ${Math.round(stats.q3)} · ` +
    `max ${stats.max} · mean ${Math.round(stats.mean)}`;
  return (
    <div className="flex items-center gap-2" title={tooltip}>
      <span
        className="text-[10px] font-mono w-9 shrink-0"
        style={{ color }}
      >
        {label}
      </span>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        width="100%"
        height={H}
        className="block"
      >
        {/* whisker baseline */}
        <line
          x1={xMin}
          x2={xMax}
          y1={H / 2}
          y2={H / 2}
          stroke={color}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
        {/* whisker caps */}
        <line
          x1={xMin}
          x2={xMin}
          y1={H * 0.3}
          y2={H * 0.7}
          stroke={color}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
        <line
          x1={xMax}
          x2={xMax}
          y1={H * 0.3}
          y2={H * 0.7}
          stroke={color}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
        {/* IQR box */}
        <rect
          x={boxLeft}
          y={H * 0.15}
          width={boxW}
          height={H * 0.7}
          fill={color}
          fillOpacity={0.25}
          stroke={color}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
        {/* median */}
        <line
          x1={xMed}
          x2={xMed}
          y1={H * 0.1}
          y2={H * 0.9}
          stroke={color}
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
        {/* mean (faint dashed) */}
        <line
          x1={xMean}
          x2={xMean}
          y1={H * 0.2}
          y2={H * 0.8}
          stroke="currentColor"
          strokeWidth={1}
          strokeDasharray="2,2"
          vectorEffect="non-scaling-stroke"
          className="text-gray-500 dark:text-gray-400"
        />
      </svg>
      <span className="text-[10px] font-mono tabular-nums w-16 shrink-0 text-right text-gray-600 dark:text-gray-300">
        {Math.round(stats.median)}
      </span>
    </div>
  );
}
