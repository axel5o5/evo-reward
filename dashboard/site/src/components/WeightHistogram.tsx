import { useMemo, useState } from "react";
import { ReplayData } from "../lib/replayLoader";

interface Props {
  data: ReplayData;
  frameIdx: number;
  // When set + alive at frameIdx, render a marker at the agent's current
  // weight on the active axis. Lets the user read at a glance whether the
  // selected individual is typical-of-species or a tail outlier.
  selectedSlot?: number | null;
}

type Axis = 0 | 1 | 2 | 3;
const AXIS_LABELS: Record<Axis, string> = {
  0: "w_eat",
  1: "w_act",
  2: "w_prey",
  3: "w_pred",
};

// Histogram spans ±HIST_RANGE with HIST_BINS bins. Matches the reward-weight
// axis used by AgentInspector so visual comparison is apples-to-apples.
const HIST_RANGE = 2.0;
const HIST_BINS = 24;
const COLOR_PREY = "#4ade80";
const COLOR_PRED = "#f87171";

function bucket(v: number): number {
  // Map [-R, R] → [0, BINS). Clip extremes into edge bins.
  const t = (v + HIST_RANGE) / (2 * HIST_RANGE);
  const b = Math.floor(t * HIST_BINS);
  if (b < 0) return 0;
  if (b >= HIST_BINS) return HIST_BINS - 1;
  return b;
}

function computeHistograms(
  data: ReplayData,
  frameIdx: number,
  axis: Axis,
): { prey: Int32Array; pred: Int32Array; preyN: number; predN: number; preyMean: number; predMean: number } {
  const N = data.meta.max_agents;
  const prey = new Int32Array(HIST_BINS);
  const pred = new Int32Array(HIST_BINS);
  let preyN = 0;
  let predN = 0;
  let preySum = 0;
  let predSum = 0;
  const weights = data.rewardWeights!;
  for (let slot = 0; slot < N; slot++) {
    if (data.alive[frameIdx * N + slot] !== 1) continue;
    const v = weights[(frameIdx * N + slot) * 4 + axis];
    const b = bucket(v);
    if (data.species[slot] === 1) {
      pred[b] += 1;
      predN += 1;
      predSum += v;
    } else {
      prey[b] += 1;
      preyN += 1;
      preySum += v;
    }
  }
  return {
    prey,
    pred,
    preyN,
    predN,
    preyMean: preyN > 0 ? preySum / preyN : 0,
    predMean: predN > 0 ? predSum / predN : 0,
  };
}

export default function WeightHistogram({ data, frameIdx, selectedSlot = null }: Props) {
  const [axis, setAxis] = useState<Axis>(3);

  const hist = useMemo(() => {
    if (!data.rewardWeights) return null;
    return computeHistograms(data, frameIdx, axis);
  }, [data, frameIdx, axis]);

  // Selected agent's weight value + same-species percentile. We use mid-rank
  // percentile (count below + half the ties) because tied phenotypes are
  // common — early in training many agents share an inherited weight vector,
  // and naive `count_below / total` would slam those onto 0% or 100%.
  const selectedInfo = useMemo(() => {
    if (selectedSlot === null || !data.rewardWeights) return null;
    const N = data.meta.max_agents;
    if (data.alive[frameIdx * N + selectedSlot] !== 1) return null;
    const v = data.rewardWeights[(frameIdx * N + selectedSlot) * 4 + axis];
    const isPred = data.species[selectedSlot] === 1;
    let lower = 0;
    let tied = 0;
    let total = 0;
    const w = data.rewardWeights;
    for (let s = 0; s < N; s++) {
      if (data.alive[frameIdx * N + s] !== 1) continue;
      if ((data.species[s] === 1) !== isPred) continue;
      const wv = w[(frameIdx * N + s) * 4 + axis];
      total++;
      if (wv < v) lower++;
      else if (wv === v) tied++;
    }
    const percentile =
      total > 0 ? ((lower + tied / 2) / total) * 100 : null;
    return { value: v, isPred, percentile, total };
  }, [data, frameIdx, axis, selectedSlot]);

  if (!data.rewardWeights) {
    return (
      <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-white dark:bg-gray-950 text-xs text-gray-500">
        Phenotype not recorded — re-run with the v2 replay recorder to see the
        reward-weight distribution.
      </div>
    );
  }

  const { prey, pred, preyN, predN, preyMean, predMean } = hist!;
  const peak = Math.max(1, ...prey, ...pred);

  // SVG canvas: one bar pair per bin, stacked horizontally, with separate
  // prey/pred heights since species sizes differ wildly.
  const W = 480;
  const H = 80;
  const binW = W / HIST_BINS;

  return (
    <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-white dark:bg-gray-950">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
          Reward-weight distribution
        </div>
        <div className="flex gap-1">
          {(Object.keys(AXIS_LABELS) as unknown as Axis[])
            .map((k) => Number(k) as Axis)
            .map((k) => (
              <button
                key={k}
                onClick={() => setAxis(k)}
                className={`px-2 py-0.5 text-xs font-mono rounded border transition ${
                  axis === k
                    ? "bg-blue-600 border-blue-600 text-white"
                    : "border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-blue-400"
                }`}
              >
                {AXIS_LABELS[k]}
              </button>
            ))}
        </div>
      </div>

      <div className="text-[10px] font-mono text-gray-500 mb-1 flex justify-between">
        <span>
          prey n={preyN} μ={preyMean.toFixed(2)}
        </span>
        <span className="opacity-60">axis ±{HIST_RANGE.toFixed(1)}</span>
        <span>
          pred n={predN} μ={predMean.toFixed(2)}
        </span>
      </div>
      {selectedInfo && (
        <div
          className="text-[10px] font-mono mb-1 flex items-center gap-1.5"
          style={{ color: selectedInfo.isPred ? COLOR_PRED : COLOR_PREY }}
          title="The selected/pinned agent's value on this axis, and where it ranks within its species at this frame (mid-rank percentile)."
        >
          <span aria-hidden>▾</span>
          <span>
            selected: {AXIS_LABELS[axis]}={selectedInfo.value.toFixed(2)}
            {selectedInfo.percentile !== null && (
              <>
                {" · "}
                {Math.round(selectedInfo.percentile)}
                <span className="opacity-70">
                  th pct of {selectedInfo.isPred ? "pred" : "prey"}
                </span>
              </>
            )}
          </span>
        </div>
      )}

      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        width="100%"
        height={H}
        className="block border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-900/50"
      >
        {/* center baseline */}
        <line
          x1={W / 2}
          x2={W / 2}
          y1={0}
          y2={H}
          stroke="currentColor"
          strokeWidth={1}
          strokeDasharray="2 2"
          vectorEffect="non-scaling-stroke"
          className="text-gray-300 dark:text-gray-700"
        />
        {/* Prey = filled green; Predator = outline red on top. Overlaid so
            you can see modes diverge at a glance. */}
        {Array.from({ length: HIST_BINS }, (_, i) => {
          const py = (prey[i] / peak) * H;
          const pd = (pred[i] / peak) * H;
          return (
            <g key={i}>
              {py > 0 && (
                <rect
                  x={i * binW + 0.5}
                  y={H - py}
                  width={Math.max(0, binW - 1)}
                  height={py}
                  fill={COLOR_PREY}
                  fillOpacity={0.45}
                />
              )}
              {pd > 0 && (
                <rect
                  x={i * binW + 0.5}
                  y={H - pd}
                  width={Math.max(0, binW - 1)}
                  height={pd}
                  fill="none"
                  stroke={COLOR_PRED}
                  strokeWidth={1.25}
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </g>
          );
        })}
        {/* Mean markers */}
        <MeanTick value={preyMean} color={COLOR_PREY} H={H} />
        <MeanTick value={predMean} color={COLOR_PRED} H={H} />
        {/* Selected-agent marker — drawn last so it sits on top of bars
            and mean ticks. */}
        {selectedInfo && (
          <SelectedTick
            value={selectedInfo.value}
            color={selectedInfo.isPred ? COLOR_PRED : COLOR_PREY}
            H={H}
          />
        )}
      </svg>

      <div className="mt-1 text-[10px] font-mono text-gray-500 flex justify-between">
        <span>−{HIST_RANGE.toFixed(1)}</span>
        <span>0</span>
        <span>+{HIST_RANGE.toFixed(1)}</span>
      </div>
    </div>
  );
}

function MeanTick({ value, color, H }: { value: number; color: string; H: number }) {
  const v = Math.max(-HIST_RANGE, Math.min(HIST_RANGE, value));
  const x = ((v + HIST_RANGE) / (2 * HIST_RANGE)) * 480;
  return (
    <line
      x1={x}
      x2={x}
      y1={0}
      y2={H}
      stroke={color}
      strokeWidth={1.5}
      vectorEffect="non-scaling-stroke"
      opacity={0.9}
    />
  );
}

// Selected-agent tick. Distinguished from MeanTick by a downward triangle
// pointer at the top edge — easier to read at a glance than a thicker line
// against the same colored bars.
function SelectedTick({ value, color, H }: { value: number; color: string; H: number }) {
  const v = Math.max(-HIST_RANGE, Math.min(HIST_RANGE, value));
  const x = ((v + HIST_RANGE) / (2 * HIST_RANGE)) * 480;
  const HEAD = 6;
  return (
    <g>
      <line
        x1={x}
        x2={x}
        y1={HEAD}
        y2={H}
        stroke={color}
        strokeWidth={2}
        vectorEffect="non-scaling-stroke"
      />
      <polygon
        points={`${x - HEAD},0 ${x + HEAD},0 ${x},${HEAD}`}
        fill={color}
      />
    </g>
  );
}
