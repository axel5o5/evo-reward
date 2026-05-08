import { useMemo, useState } from "react";
import { ReplayData } from "../lib/replayLoader";
import RewardLandscape from "./RewardLandscape";
import RewardMlpDiagram from "./RewardMlpDiagram";

interface Props {
  data: ReplayData;
  frameIdx: number;
  slot: number;
  pinned: boolean;
  onTogglePin: () => void;
  onClose: () => void;
  // Emitted when user clicks "jump to parent". Parent lookup + state wiring
  // lives in Replay.tsx since it needs to mutate frameIdx + pinnedSlot.
  onJumpToParent?: (parentAgentId: number) => void;
}

const COLOR_PREY = "#4ade80";
const COLOR_PRED = "#f87171";
const COLOR_DEAD = "#9ca3af"; // gray-400

// Maximum energy axis for the sparkline. Matches the alpha-fade scale used on
// the canvas so the two visual cues share a reference.
const ENERGY_CAP = 200;

// Reward-weight bar axis. reward_weights_init_std defaults to ~0.5; weights
// drift over generations but rarely exceed |2| in the runs we've inspected.
const WEIGHT_AXIS = 2.0;

// Action components are tanh-ish in [-1, 1]; small headroom for clarity.
const ACTION_AXIS = 1.1;

const WEIGHT_LABELS = ["w_eat", "w_act", "w_prey", "w_pred"] as const;

function computeAgentSeries(
  data: ReplayData,
  slot: number,
): {
  energy: Float32Array;
  alive: Uint8Array;
  actionFwd: Float32Array | null;
  actionTurn: Float32Array | null;
  distance: number;
  aliveFirst: number;
  aliveLast: number;
} {
  const n = data.meta.n_frames;
  const N = data.meta.max_agents;
  const energy = new Float32Array(n);
  const alive = new Uint8Array(n);
  const hasAction = data.action !== null;
  const actionFwd = hasAction ? new Float32Array(n) : null;
  const actionTurn = hasAction ? new Float32Array(n) : null;
  let distance = 0;
  let prevX = 0;
  let prevY = 0;
  let havePrev = false;
  let first = -1;
  let last = -1;
  for (let f = 0; f < n; f++) {
    const a = data.alive[f * N + slot];
    alive[f] = a;
    energy[f] = data.energy[f * N + slot];
    if (hasAction) {
      actionFwd![f] = data.action![(f * N + slot) * 2];
      actionTurn![f] = data.action![(f * N + slot) * 2 + 1];
    }
    if (a) {
      const x = data.pos[(f * N + slot) * 2];
      const y = data.pos[(f * N + slot) * 2 + 1];
      if (havePrev) {
        const dx = x - prevX;
        const dy = y - prevY;
        distance += Math.sqrt(dx * dx + dy * dy);
      }
      prevX = x;
      prevY = y;
      havePrev = true;
      if (first < 0) first = f;
      last = f;
    } else {
      // Gap resets the trajectory so slot-reuse across death doesn't inflate
      // distance with a giant teleport segment.
      havePrev = false;
    }
  }
  return { energy, alive, actionFwd, actionTurn, distance, aliveFirst: first, aliveLast: last };
}

// Build two SVG path `d` strings — one for alive segments (species color),
// one for dead segments (grey) — driven by a value series on [0, maxV] mapped
// to an H-tall strip. Segments break on alive-state transitions so line style
// tracks the on/off state.
function splitSparkline(
  series: Float32Array,
  alive: Uint8Array,
  W: number,
  H: number,
  minV: number,
  maxV: number,
): { alive: string; dead: string } {
  const n = series.length;
  if (n === 0) return { alive: "", dead: "" };
  const range = maxV - minV || 1;
  const aliveSegs: string[] = [];
  const deadSegs: string[] = [];
  let curAlive: string[] = [];
  let curDead: string[] = [];
  const flush = () => {
    if (curAlive.length > 1) aliveSegs.push(`M ${curAlive.join(" L ")}`);
    if (curDead.length > 1) deadSegs.push(`M ${curDead.join(" L ")}`);
    curAlive = [];
    curDead = [];
  };
  let lastAlive: number | null = null;
  for (let f = 0; f < n; f++) {
    const x = (f / (n - 1)) * W;
    const v = series[f];
    const y = H - ((v - minV) / range) * H;
    const pt = `${x.toFixed(2)},${y.toFixed(2)}`;
    const isAlive = alive[f] === 1;
    if (lastAlive !== null && isAlive !== (lastAlive === 1)) flush();
    if (isAlive) curAlive.push(pt);
    else curDead.push(pt);
    lastAlive = isAlive ? 1 : 0;
  }
  flush();
  return { alive: aliveSegs.join(" "), dead: deadSegs.join(" ") };
}

function energySparkline(
  energy: Float32Array,
  alive: Uint8Array,
  W: number,
  H: number,
): { alive: string; dead: string; maxE: number } {
  let maxE = ENERGY_CAP;
  for (let i = 0; i < energy.length; i++) if (energy[i] > maxE) maxE = energy[i];
  const s = splitSparkline(energy, alive, W, H, 0, maxE);
  return { ...s, maxE };
}

export default function AgentInspector({
  data,
  frameIdx,
  slot,
  pinned,
  onTogglePin,
  onClose,
  onJumpToParent,
}: Props) {
  const series = useMemo(() => computeAgentSeries(data, slot), [data, slot]);
  const W = 240;
  const H = 40;
  const HA = 24;
  const spark = useMemo(
    () => energySparkline(series.energy, series.alive, W, H),
    [series],
  );
  const actFwd = useMemo(
    () =>
      series.actionFwd
        ? splitSparkline(series.actionFwd, series.alive, W, HA, -ACTION_AXIS, ACTION_AXIS)
        : null,
    [series],
  );
  const actTurn = useMemo(
    () =>
      series.actionTurn
        ? splitSparkline(series.actionTurn, series.alive, W, HA, -ACTION_AXIS, ACTION_AXIS)
        : null,
    [series],
  );

  const N = data.meta.max_agents;
  const aliveNow = data.alive[frameIdx * N + slot] === 1;
  const species = data.species[slot];
  const radius = data.radii[slot];
  const x = data.pos[(frameIdx * N + slot) * 2];
  const y = data.pos[(frameIdx * N + slot) * 2 + 1];
  const energy = data.energy[frameIdx * N + slot];
  const angleRad = data.angle[frameIdx * N + slot];
  const angleDeg = ((angleRad * 180) / Math.PI + 360) % 360;

  // v2 fields — null on v1 replays, in which case the phenotype/lineage
  // sections render a "not recorded" fallback.
  const agentId = data.agentIds ? data.agentIds[frameIdx * N + slot] : null;
  const parentId = data.parentIds ? data.parentIds[frameIdx * N + slot] : null;
  const age = data.ages ? data.ages[frameIdx * N + slot] : null;
  const birthStep =
    age !== null && age >= 0 ? data.stepNums[frameIdx] - age : null;
  const weights = data.rewardWeights
    ? Array.from(
        data.rewardWeights.subarray((frameIdx * N + slot) * 4, (frameIdx * N + slot + 1) * 4),
      )
    : null;

  const playX =
    series.energy.length > 1 ? (frameIdx / (series.energy.length - 1)) * W : 0;

  const speciesLabel = species === 1 ? "predator" : "prey";
  const speciesColor = species === 1 ? COLOR_PRED : COLOR_PREY;

  const canJumpParent =
    onJumpToParent && parentId !== null && parentId >= 0;

  return (
    <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-white dark:bg-gray-950">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ background: speciesColor }}
          />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            slot {slot} · {speciesLabel}
            {agentId !== null && (
              <span className="text-gray-500 font-mono"> · id {agentId}</span>
            )}
          </span>
          {!aliveNow && (
            <span className="text-[10px] uppercase tracking-wide text-gray-500">
              dead
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onTogglePin}
            className={`px-2 py-0.5 text-xs rounded border transition ${
              pinned
                ? "bg-amber-500 border-amber-500 text-white"
                : "border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-amber-400"
            }`}
            title="Pin this agent — draws trail + ring on canvas"
          >
            {pinned ? "📌 pinned" : "pin"}
          </button>
          <button
            onClick={onClose}
            className="px-2 py-0.5 text-xs rounded border border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-red-400 hover:text-red-500"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono text-gray-700 dark:text-gray-300 mb-2">
        <div>energy <span className="text-gray-500">{energy.toFixed(1)}</span></div>
        <div>radius <span className="text-gray-500">{radius.toFixed(1)}</span></div>
        <div>heading <span className="text-gray-500">{angleDeg.toFixed(0)}°</span></div>
        <div>distance <span className="text-gray-500">{series.distance.toFixed(0)}</span></div>
        <div className="col-span-2">
          pos <span className="text-gray-500">({x.toFixed(1)}, {y.toFixed(1)})</span>
        </div>
        {age !== null && age >= 0 && (
          <div className="col-span-2">
            age{" "}
            <span className="text-gray-700 dark:text-gray-300 tabular-nums">
              {age.toLocaleString()}
            </span>
            <span className="text-gray-500"> steps</span>
            {birthStep !== null && (
              <span className="text-gray-500">
                {" "}· born @ step {birthStep.toLocaleString()}
              </span>
            )}
          </div>
        )}
        <div className="col-span-2 flex items-center gap-2">
          <span>parent</span>
          {parentId === null ? (
            <span className="text-gray-500 italic">not recorded</span>
          ) : parentId < 0 ? (
            <span className="text-gray-500">founder</span>
          ) : (
            <button
              onClick={() => canJumpParent && onJumpToParent!(parentId)}
              disabled={!canJumpParent}
              className="text-blue-600 dark:text-blue-400 hover:underline disabled:no-underline disabled:text-gray-500"
              title="Jump to parent's last-alive frame and pin"
            >
              id {parentId} →
            </button>
          )}
        </div>
        {series.aliveFirst >= 0 && (
          <div className="col-span-2 text-[10px] text-gray-500">
            alive frames {series.aliveFirst}–{series.aliveLast} of {data.meta.n_frames}
          </div>
        )}
      </div>

      {weights && (
        <div className="mb-2">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
            reward weights{" "}
            <span className="normal-case tracking-normal">(±{WEIGHT_AXIS.toFixed(1)})</span>
          </div>
          <div className="grid grid-cols-[auto_1fr_auto] gap-x-2 gap-y-0.5 items-center text-[11px] font-mono">
            {WEIGHT_LABELS.map((label, i) => {
              const v = weights[i];
              const pct = Math.min(1, Math.abs(v) / WEIGHT_AXIS);
              const isNeg = v < 0;
              return (
                <WeightBarRow
                  key={label}
                  label={label}
                  value={v}
                  pct={pct}
                  isNeg={isNeg}
                />
              );
            })}
          </div>
        </div>
      )}

      {(data.meta.genome_arch === "mlp" ||
        data.meta.genome_arch === "temporal") && (() => {
        const genome =
          agentId !== null && agentId >= 0
            ? data.genomesById?.get(agentId) ?? null
            : null;
        const sourceLabel =
          agentId !== null && agentId >= 0 ? `id ${agentId}` : undefined;
        return (
          <div className="mb-2 flex flex-col gap-2">
            <MlpStimuliPanel
              genome={genome}
              sourceLabel={sourceLabel}
              arch={data.meta.genome_arch}
              contextWindow={data.meta.genome_shape?.context_window ?? 1}
            />
          </div>
        );
      })()}

      <div>
        <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-0.5">
          energy <span className="normal-case tracking-normal">(0–{spark.maxE.toFixed(0)})</span>
        </div>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          width="100%"
          height={H}
          className="block border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-900/50"
        >
          {spark.dead && (
            <path d={spark.dead} fill="none" stroke={COLOR_DEAD} strokeWidth={1} vectorEffect="non-scaling-stroke" />
          )}
          {spark.alive && (
            <path d={spark.alive} fill="none" stroke={speciesColor} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
          )}
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
      </div>

      {actFwd && actTurn && (
        <div className="mt-2 grid grid-cols-1 gap-1">
          <ActionTrack
            label="forward"
            spark={actFwd}
            W={W}
            H={HA}
            color={speciesColor}
            playX={playX}
          />
          <ActionTrack
            label="turn"
            spark={actTurn}
            W={W}
            H={HA}
            color={speciesColor}
            playX={playX}
          />
        </div>
      )}
    </div>
  );
}

function WeightBarRow({
  label,
  value,
  pct,
  isNeg,
}: {
  label: string;
  value: number;
  pct: number;
  isNeg: boolean;
}) {
  return (
    <>
      <span className="text-gray-600 dark:text-gray-400">{label}</span>
      <div className="relative h-3 rounded bg-gray-100 dark:bg-gray-800 overflow-hidden">
        {/* center baseline */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300 dark:bg-gray-700" />
        <div
          className={isNeg ? "absolute top-0 bottom-0 bg-red-400/70" : "absolute top-0 bottom-0 bg-blue-500/70"}
          style={
            isNeg
              ? { right: "50%", width: `${pct * 50}%` }
              : { left: "50%", width: `${pct * 50}%` }
          }
        />
      </div>
      <span className="text-gray-700 dark:text-gray-300 tabular-nums text-right">
        {value >= 0 ? "+" : ""}
        {value.toFixed(2)}
      </span>
    </>
  );
}

function ActionTrack({
  label,
  spark,
  W,
  H,
  color,
  playX,
}: {
  label: string;
  spark: { alive: string; dead: string };
  W: number;
  H: number;
  color: string;
  playX: number;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-0.5">
        {label}{" "}
        <span className="normal-case tracking-normal">(±{ACTION_AXIS.toFixed(1)})</span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        width="100%"
        height={H}
        className="block border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-900/50"
      >
        {/* zero baseline */}
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
        {spark.dead && (
          <path d={spark.dead} fill="none" stroke={COLOR_DEAD} strokeWidth={1} vectorEffect="non-scaling-stroke" />
        )}
        {spark.alive && (
          <path d={spark.alive} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        )}
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
    </div>
  );
}

// Wraps RewardLandscape + RewardMlpDiagram and owns a shared 4-d stimulus
// state. Lifted out so dragging a slider feeds both views in lockstep —
// the heatmap re-samples its slice, the network diagram re-runs the
// forward pass, and the user sees how a single stimulus change ripples
// through the evolved reward function.
//
// For MLP genomes (Axis 1, 4-d input) the stimulus vector is fed directly.
// For temporal genomes (Axis 3, k*4-d input) we tile the same 4-d vector
// across all k time steps — interpretable as "the agent has been seeing
// this stimulus pattern for the last k steps." This is a one-axis-of-many
// projection of the temporal reward surface but it's the only one users
// can drive with 4 sliders; richer temporal viz is a follow-up.
const MLP_STIMULUS_LABELS = ["n_eaten", "motor_norm", "s_prey", "s_pred"] as const;
const MLP_STIMULUS_RANGES: [number, number][] = [
  [0, 3],
  [0, 1],
  [0, 1],
  [0, 1],
];

function MlpStimuliPanel({
  genome,
  sourceLabel,
  arch,
  contextWindow,
}: {
  genome: import("../lib/rewardMlp").MlpGenome | null;
  sourceLabel?: string;
  arch: "mlp" | "temporal";
  contextWindow: number;
}) {
  const [stimuli, setStimuli] = useState<Float32Array>(
    () => new Float32Array([1.0, 0.5, 0.5, 0.5]),
  );

  const updateStimulus = (i: number, v: number) => {
    setStimuli((prev) => {
      const out = new Float32Array(prev);
      out[i] = v;
      return out;
    });
  };

  // Build the input vector the network actually consumes. MLP: 4-d as-is.
  // Temporal: tile to length k*4 by repeating the 4-d slider state across
  // all k time windows.
  const networkInput = useMemo(() => {
    if (arch === "mlp") return stimuli;
    const k = Math.max(1, contextWindow);
    const out = new Float32Array(k * 4);
    for (let t = 0; t < k; t++) {
      out[t * 4 + 0] = stimuli[0];
      out[t * 4 + 1] = stimuli[1];
      out[t * 4 + 2] = stimuli[2];
      out[t * 4 + 3] = stimuli[3];
    }
    return out;
  }, [arch, contextWindow, stimuli]);

  return (
    <div className="flex flex-col gap-2">
      <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-white dark:bg-gray-950">
        <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">
          stimuli{" "}
          <span className="normal-case tracking-normal">
            {arch === "temporal"
              ? `(tiled across ${contextWindow}-step window)`
              : "(drives both views)"}
          </span>
        </div>
        <div className="flex flex-col gap-1">
          {MLP_STIMULUS_LABELS.map((label, i) => (
            <div key={label} className="flex items-center gap-2 text-[10px] font-mono">
              <span className="w-20 text-gray-600 dark:text-gray-400">{label}</span>
              <input
                type="range"
                min={MLP_STIMULUS_RANGES[i][0]}
                max={MLP_STIMULUS_RANGES[i][1]}
                step={(MLP_STIMULUS_RANGES[i][1] - MLP_STIMULUS_RANGES[i][0]) / 100}
                value={stimuli[i]}
                onChange={(e) => updateStimulus(i, Number(e.target.value))}
                className="flex-1"
              />
              <span className="w-10 text-right tabular-nums text-gray-700 dark:text-gray-300">
                {stimuli[i].toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {genome && (
        <RewardMlpDiagram
          genome={genome}
          heldValues={networkInput}
          inputLabels={arch === "mlp" ? MLP_STIMULUS_LABELS : undefined}
        />
      )}

      {/* Landscape only renders for the 4-d MLP input — its 2-axis slice
          assumes 4 inputs so it doesn't generalize to the temporal MLP. */}
      {arch === "mlp" && (
        <RewardLandscape
          genome={genome}
          sourceLabel={sourceLabel}
          heldValues={stimuli}
        />
      )}
    </div>
  );
}
