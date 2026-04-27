import { useMemo, useRef } from "react";
import { ReplayData } from "../lib/replayLoader";

// Per-species, per-axis population mean across the recorded window.
//
// PopulationStrip shows population *size* over time — this is its companion
// for population *phenotype* over time. Four axes (w_eat, w_act, w_prey,
// w_pred); each axis gets a stacked sparkline showing prey + pred mean.
// The replay window only covers ~1000 sim steps so the trajectory is short
// — but it's enough to spot directional drift within the window and to
// orient the user before they dig into AgentInspector.

interface Props {
  data: ReplayData;
  frameIdx: number;
  onFrameChange: (frame: number) => void;
}

const COLOR_PREY = "#4ade80";
const COLOR_PRED = "#f87171";

const AXIS_RANGE = 2.0;
const LABELS: [string, string][] = [
  ["w_eat", "food drive"],
  ["w_act", "action cost"],
  ["w_prey", "social / chase"],
  ["w_pred", "fear / hunt"],
];

interface Series {
  prey: Float32Array; // (n_frames,) — NaN where the species had zero alive
  pred: Float32Array;
}

function computeSeries(data: ReplayData): Series[] {
  // Returns one Series per weight axis (4 total). Mean is computed over alive
  // agents at each frame; if a species had zero alive, the value is NaN so
  // the line breaks rather than collapsing to zero.
  const n = data.meta.n_frames;
  const N = data.meta.max_agents;
  const w = data.rewardWeights!;
  const out: Series[] = [];
  for (let axis = 0; axis < 4; axis++) {
    out.push({ prey: new Float32Array(n), pred: new Float32Array(n) });
  }
  for (let f = 0; f < n; f++) {
    const sums = [0, 0, 0, 0, 0, 0, 0, 0]; // [prey_eat, prey_act, prey_prey, prey_pred, pred_eat, ...]
    let preyN = 0;
    let predN = 0;
    for (let s = 0; s < N; s++) {
      if (data.alive[f * N + s] !== 1) continue;
      const isPred = data.species[s] === 1;
      if (isPred) predN++;
      else preyN++;
      const base = (f * N + s) * 4;
      const off = isPred ? 4 : 0;
      sums[off + 0] += w[base + 0];
      sums[off + 1] += w[base + 1];
      sums[off + 2] += w[base + 2];
      sums[off + 3] += w[base + 3];
    }
    for (let axis = 0; axis < 4; axis++) {
      out[axis].prey[f] = preyN > 0 ? sums[axis] / preyN : NaN;
      out[axis].pred[f] = predN > 0 ? sums[4 + axis] / predN : NaN;
    }
  }
  return out;
}

function polylinePts(
  series: Float32Array,
  W: number,
  H: number,
  axis: number,
): string[] {
  // Returns one or more `M x,y L x,y …` substrings — broken on NaN gaps so
  // dead-species frames render as blanks rather than diagonal jumps.
  const n = series.length;
  if (n === 0) return [];
  const segs: string[] = [];
  let cur: string[] = [];
  for (let f = 0; f < n; f++) {
    const v = series[f];
    if (Number.isNaN(v)) {
      if (cur.length > 1) segs.push(`M ${cur.join(" L ")}`);
      cur = [];
      continue;
    }
    const clamped = Math.max(-axis, Math.min(axis, v));
    const x = (f / (n - 1)) * W;
    const y = H - ((clamped + axis) / (2 * axis)) * H;
    cur.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  if (cur.length > 1) segs.push(`M ${cur.join(" L ")}`);
  return segs;
}

export default function WeightTrajectoryStrip({
  data,
  frameIdx,
  onFrameChange,
}: Props) {
  const W = 1000;
  const H = 36;

  const series = useMemo(() => computeSeries(data), [data]);

  const nFrames = data.meta.n_frames;
  const playX = nFrames > 1 ? (frameIdx / (nFrames - 1)) * W : 0;

  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const frameAtEvent = (clientX: number, target: SVGSVGElement): number => {
    const rect = target.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return Math.round(frac * (nFrames - 1));
  };

  if (!data.rewardWeights) {
    return (
      <div className="border border-gray-200 dark:border-gray-800 rounded p-3 text-xs text-gray-500">
        Reward-weight trajectory unavailable — phenotype not recorded in this
        replay. Re-run with the v2 recorder to populate.
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className="w-full">
      <div className="text-[10px] font-mono text-gray-500 dark:text-gray-400 mb-1 px-1 flex justify-between">
        <span>population mean reward weight (alive only)</span>
        <span>axis ±{AXIS_RANGE.toFixed(1)}</span>
      </div>
      <div className="grid grid-cols-1 gap-1">
        {LABELS.map(([key, hint], axis) => {
          const preySegs = polylinePts(series[axis].prey, W, H, AXIS_RANGE);
          const predSegs = polylinePts(series[axis].pred, W, H, AXIS_RANGE);
          const preyNow = series[axis].prey[frameIdx];
          const predNow = series[axis].pred[frameIdx];
          return (
            <div key={key} className="flex items-center gap-2">
              <div className="w-20 shrink-0 text-[10px] font-mono text-gray-700 dark:text-gray-300">
                <div>{key}</div>
                <div className="text-gray-500 text-[9px]">{hint}</div>
              </div>
              <svg
                viewBox={`0 0 ${W} ${H}`}
                preserveAspectRatio="none"
                width="100%"
                height={H}
                className="block border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-900/50 cursor-crosshair flex-1"
                onClick={(e) => onFrameChange(frameAtEvent(e.clientX, e.currentTarget))}
                onPointerMove={(e) => {
                  if (e.buttons === 1) {
                    onFrameChange(frameAtEvent(e.clientX, e.currentTarget));
                  }
                }}
              >
                {/* zero line */}
                <line
                  x1={0}
                  x2={W}
                  y1={H / 2}
                  y2={H / 2}
                  stroke="currentColor"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  vectorEffect="non-scaling-stroke"
                  className="text-gray-300 dark:text-gray-700"
                />
                {preySegs.map((d, i) => (
                  <path
                    key={`prey-${i}`}
                    d={d}
                    fill="none"
                    stroke={COLOR_PREY}
                    strokeWidth={1.4}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                {predSegs.map((d, i) => (
                  <path
                    key={`pred-${i}`}
                    d={d}
                    fill="none"
                    stroke={COLOR_PRED}
                    strokeWidth={1.4}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                <line
                  x1={playX}
                  x2={playX}
                  y1={0}
                  y2={H}
                  stroke="currentColor"
                  strokeWidth={1}
                  vectorEffect="non-scaling-stroke"
                  className="text-gray-400 dark:text-gray-500"
                />
              </svg>
              <div className="w-24 shrink-0 text-[10px] font-mono text-right tabular-nums">
                <span style={{ color: COLOR_PREY }}>
                  {Number.isNaN(preyNow) ? "—" : preyNow.toFixed(2)}
                </span>
                <span className="mx-1 text-gray-400">/</span>
                <span style={{ color: COLOR_PRED }}>
                  {Number.isNaN(predNow) ? "—" : predNow.toFixed(2)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
