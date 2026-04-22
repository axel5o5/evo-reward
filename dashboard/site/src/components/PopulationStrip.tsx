import { useMemo, useRef } from "react";
import { ReplayData } from "../lib/replayLoader";
import { ReplayStats } from "../lib/replayStats";

interface Props {
  data: ReplayData;
  stats: ReplayStats;
  frameIdx: number;
  onFrameChange: (frame: number) => void;
  // Optional: vertical height in px. Width tracks the parent.
  heightPx?: number;
}

const COLOR_PREY = "#4ade80"; // green-400 (matches canvas)
const COLOR_PRED = "#f87171"; // red-400

// SVG viewBox Y is inverted (0 at top). We plot counts with a flat y-flip:
// y = H - count/maxCount * H.

function polylinePoints(counts: Uint16Array, W: number, H: number, maxCount: number): string {
  const n = counts.length;
  if (n <= 1) return "";
  // Downsample when n > W * 2 — a dashboard frame is ~720px, replays are ~1024
  // frames; we still render one point per frame on the default size, but cap
  // points to avoid pathological cases.
  const maxPoints = 2048;
  const stride = Math.max(1, Math.ceil(n / maxPoints));
  const parts: string[] = [];
  for (let f = 0; f < n; f += stride) {
    const x = (f / (n - 1)) * W;
    const y = H - (counts[f] / maxCount) * H;
    parts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  // Ensure final frame is included so the line ends flush right.
  const lastF = n - 1;
  const lastX = W;
  const lastY = H - (counts[lastF] / maxCount) * H;
  parts.push(`${lastX.toFixed(2)},${lastY.toFixed(2)}`);
  return parts.join(" ");
}

export default function PopulationStrip({
  data,
  stats,
  frameIdx,
  onFrameChange,
  heightPx = 48,
}: Props) {
  const { prey, pred, maxCount } = stats;

  const W = 1000; // viewBox width; preserveAspectRatio='none' stretches it
  const H = 100;

  const preyPts = useMemo(() => polylinePoints(prey, W, H, maxCount), [prey, maxCount]);
  const predPts = useMemo(() => polylinePoints(pred, W, H, maxCount), [pred, maxCount]);

  const nFrames = data.meta.n_frames;
  const playX = nFrames > 1 ? (frameIdx / (nFrames - 1)) * W : 0;

  const svgRef = useRef<SVGSVGElement | null>(null);
  const frameAtEvent = (clientX: number): number => {
    const svg = svgRef.current;
    if (!svg) return frameIdx;
    const rect = svg.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return Math.round(frac * (nFrames - 1));
  };

  const preyNow = prey[frameIdx] ?? 0;
  const predNow = pred[frameIdx] ?? 0;

  return (
    <div className="w-full">
      <div className="flex items-center justify-between text-[10px] font-mono text-gray-500 dark:text-gray-400 mb-1 px-1">
        <span>
          <span style={{ color: COLOR_PREY }}>●</span> prey {preyNow}
          <span className="mx-2" />
          <span style={{ color: COLOR_PRED }}>●</span> pred {predNow}
        </span>
        <span>peak {maxCount}</span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        width="100%"
        height={heightPx}
        className="block border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-900/50 cursor-crosshair"
        onClick={(e) => onFrameChange(frameAtEvent(e.clientX))}
        onPointerMove={(e) => {
          if (e.buttons === 1) onFrameChange(frameAtEvent(e.clientX));
        }}
      >
        <polyline
          points={preyPts}
          fill="none"
          stroke={COLOR_PREY}
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
        <polyline
          points={predPts}
          fill="none"
          stroke={COLOR_PRED}
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
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
