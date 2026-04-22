import { useMemo } from "react";
import { ReplayData } from "../lib/replayLoader";

interface Props {
  data: ReplayData;
  frameIdx: number;
  slot: number;
  pinned: boolean;
  onTogglePin: () => void;
  onClose: () => void;
}

const COLOR_PREY = "#4ade80";
const COLOR_PRED = "#f87171";
const COLOR_DEAD = "#9ca3af"; // gray-400

// Maximum energy axis for the sparkline. Matches the alpha-fade scale used on
// the canvas so the two visual cues share a reference.
const ENERGY_CAP = 200;

function computeAgentSeries(
  data: ReplayData,
  slot: number,
): {
  energy: Float32Array;
  alive: Uint8Array;
  distance: number;
  aliveFirst: number;
  aliveLast: number;
} {
  const n = data.meta.n_frames;
  const N = data.meta.max_agents;
  const energy = new Float32Array(n);
  const alive = new Uint8Array(n);
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
  return { energy, alive, distance, aliveFirst: first, aliveLast: last };
}

function energySparkline(
  energy: Float32Array,
  alive: Uint8Array,
  W: number,
  H: number,
): { alive: string; dead: string; maxE: number } {
  const n = energy.length;
  if (n === 0) return { alive: "", dead: "", maxE: ENERGY_CAP };
  let maxE = ENERGY_CAP;
  for (let i = 0; i < n; i++) if (energy[i] > maxE) maxE = energy[i];
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
    const y = H - (energy[f] / maxE) * H;
    const pt = `${x.toFixed(2)},${y.toFixed(2)}`;
    const isAlive = alive[f] === 1;
    if (lastAlive !== null && isAlive !== (lastAlive === 1)) flush();
    if (isAlive) curAlive.push(pt);
    else curDead.push(pt);
    lastAlive = isAlive ? 1 : 0;
  }
  flush();
  return { alive: aliveSegs.join(" "), dead: deadSegs.join(" "), maxE };
}

export default function AgentInspector({
  data,
  frameIdx,
  slot,
  pinned,
  onTogglePin,
  onClose,
}: Props) {
  const series = useMemo(() => computeAgentSeries(data, slot), [data, slot]);
  const W = 240;
  const H = 40;
  const spark = useMemo(
    () => energySparkline(series.energy, series.alive, W, H),
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

  const playX =
    series.energy.length > 1 ? (frameIdx / (series.energy.length - 1)) * W : 0;

  const speciesLabel = species === 1 ? "predator" : "prey";
  const speciesColor = species === 1 ? COLOR_PRED : COLOR_PREY;

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
        {series.aliveFirst >= 0 && (
          <div className="col-span-2 text-[10px] text-gray-500">
            alive frames {series.aliveFirst}–{series.aliveLast} of {data.meta.n_frames}
          </div>
        )}
      </div>

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
    </div>
  );
}
