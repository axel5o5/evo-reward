import { useEffect, useMemo, useRef, useState } from "react";
import {
  MlpGenome,
  fitLinearEquivalent,
  sampleRewardGrid,
} from "../lib/rewardMlp";
import { weightColorRgb } from "../lib/weightColor";

// 2D heatmap of an MLP reward genome's response over a chosen pair of stimulus
// axes. The genome comes in via the `genome` prop — typically the agent
// pinned in AgentInspector, decoded from a v3 replay's `genomesById`. The
// synthetic fixtures in public/fixtures/mlp_reward_examples.json are now a
// debug-only path (see scripts/bake_mlp_reward_fixture.py); production
// renders use real evolved networks.

const GRID_SIZE = 64;
// Stimulus ranges roughly cover the values reward.py applies coefs to
// (n_eaten in 0–3, motor in 0–1, sensors in 0–1). The MLP genome takes RAW
// values (no fixed coefs), so we sample the same range the simulator would.
const AXIS_RANGES: [number, number][] = [
  [0, 3], // n_eaten
  [0, 1], // motor_norm
  [0, 1], // s_prey
  [0, 1], // s_pred
];

const INPUT_LABELS = ["n_eaten", "motor_norm", "s_prey", "s_pred"];

interface Props {
  // The MLP genome to inspect. Null/undefined = render an empty placeholder
  // (e.g. on linear replays, or when no agent is pinned).
  genome: MlpGenome | null;
  // Caller-supplied label for the source — e.g. "agent 47" — shown in the
  // header so the user knows whose landscape they're looking at.
  sourceLabel?: string;
  // Length-4 stimulus state owned by the parent (AgentInspector). The two
  // non-axis dimensions read from this; the two axis dimensions are swept
  // across their full ranges by the heatmap and ignore the held values.
  // Lifted out of this component so RewardMlpDiagram can share the same
  // forward-pass inputs.
  heldValues: Float32Array;
  // Optional override: if a parent wants to lock the user to a specific
  // axis pair, pass it here.
  defaultAxisX?: number;
  defaultAxisY?: number;
}

export default function RewardLandscape({
  genome,
  sourceLabel,
  heldValues,
  defaultAxisX = 3, // s_pred (fear)
  defaultAxisY = 2, // s_prey (social)
}: Props) {
  const [axisX, setAxisX] = useState(defaultAxisX);
  const [axisY, setAxisY] = useState(defaultAxisY);

  const labels = INPUT_LABELS;

  const { values, min, max, absMax } = useMemo(() => {
    if (!genome) return { values: null, min: 0, max: 0, absMax: 0 };
    const grid = sampleRewardGrid(
      genome,
      axisX,
      axisY,
      AXIS_RANGES[axisX],
      AXIS_RANGES[axisY],
      GRID_SIZE,
      heldValues,
    );
    return {
      values: grid.values,
      min: grid.min,
      max: grid.max,
      absMax: Math.max(Math.abs(grid.min), Math.abs(grid.max), 1e-6),
    };
  }, [genome, axisX, axisY, heldValues]);

  const linFit = useMemo(() => {
    if (!genome) return null;
    return fitLinearEquivalent(genome, AXIS_RANGES, 5);
  }, [genome]);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !values) return;
    const isDark =
      typeof document !== "undefined" &&
      document.documentElement.classList.contains("dark");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = GRID_SIZE;
    canvas.height = GRID_SIZE;
    const img = ctx.createImageData(GRID_SIZE, GRID_SIZE);
    for (let yi = 0; yi < GRID_SIZE; yi++) {
      // Y-flip so positive Y reads upward in the rendered image (matches
      // the rest of the dashboard's "world-up" convention).
      const srcY = GRID_SIZE - 1 - yi;
      for (let xi = 0; xi < GRID_SIZE; xi++) {
        const v = values[srcY * GRID_SIZE + xi];
        const [r, g, b] = weightColorRgb(v, absMax, isDark);
        const o = (yi * GRID_SIZE + xi) * 4;
        img.data[o + 0] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = b;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [values, absMax]);

  if (!genome) {
    return (
      <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 text-xs text-gray-500">
        Reward landscape unavailable — no MLP genome for the selected agent.
      </div>
    );
  }

  return (
    <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-white dark:bg-gray-950">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
          Reward landscape
          {sourceLabel && (
            <span className="text-[10px] font-normal text-gray-500 ml-1">
              · {sourceLabel}
            </span>
          )}
        </div>
      </div>

      <div className="flex gap-3 items-start">
        <div className="flex-shrink-0">
          <canvas
            ref={canvasRef}
            className="border border-gray-200 dark:border-gray-800 rounded"
            style={{
              width: 192,
              height: 192,
              imageRendering: "pixelated",
            }}
          />
          <div className="flex justify-between text-[9px] font-mono text-gray-500 mt-0.5 w-48">
            <span>
              {labels[axisX]} {AXIS_RANGES[axisX][0].toFixed(1)}
            </span>
            <span>{AXIS_RANGES[axisX][1].toFixed(1)}</span>
          </div>
          <div className="flex justify-between text-[9px] font-mono text-gray-500 w-48">
            <span>r ∈ [{min.toFixed(2)}, {max.toFixed(2)}]</span>
          </div>
        </div>

        <div className="flex-1 flex flex-col gap-2">
          <div className="grid grid-cols-2 gap-2">
            <AxisPicker
              label="X"
              value={axisX}
              onChange={(v) => {
                if (v === axisY) setAxisY(axisX);
                setAxisX(v);
              }}
              labels={labels}
            />
            <AxisPicker
              label="Y"
              value={axisY}
              onChange={(v) => {
                if (v === axisX) setAxisX(axisY);
                setAxisY(v);
              }}
              labels={labels}
            />
          </div>

          <div className="text-[10px] text-gray-500 leading-tight">
            X / Y axes sweep their full range across the heatmap. Other two
            inputs are held at the values set by the sliders above (shared
            with the network diagram).
          </div>

          {linFit && (
            <LinearEquivalent
              labels={labels}
              weights={linFit.weights}
              bias={linFit.bias}
              residualRms={linFit.residualRms}
              targetRms={linFit.targetRms}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function AxisPicker({
  label,
  value,
  onChange,
  labels,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  labels: string[];
}) {
  return (
    <label className="flex flex-col text-[10px] uppercase tracking-wider text-gray-500">
      {label}
      <select
        className="mt-0.5 text-xs border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-900 px-1 py-0.5 normal-case tracking-normal"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {labels.map((l, i) => (
          <option key={i} value={i}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );
}

function LinearEquivalent({
  labels,
  weights,
  bias,
  residualRms,
  targetRms,
}: {
  labels: string[];
  weights: number[];
  bias: number;
  residualRms: number;
  targetRms: number;
}) {
  const nonlinearity = targetRms > 1e-6 ? residualRms / targetRms : 0;
  const maxAbs = Math.max(...weights.map(Math.abs), 1e-6);
  return (
    <div className="border-t border-gray-200 dark:border-gray-800 pt-2 mt-1">
      <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">
        linear-equivalent fit
      </div>
      <div className="grid grid-cols-[auto_1fr_auto] gap-x-2 gap-y-0.5 items-center text-[10px] font-mono">
        {labels.map((label, i) => {
          const v = weights[i];
          const pct = Math.min(1, Math.abs(v) / maxAbs);
          const isNeg = v < 0;
          return (
            <Bar
              key={label}
              label={label}
              value={v}
              pct={pct}
              isNeg={isNeg}
            />
          );
        })}
      </div>
      <div className="mt-1 text-[10px] text-gray-500">
        bias <span className="font-mono">{bias.toFixed(2)}</span> · nonlinearity{" "}
        <span className="font-mono">
          {(nonlinearity * 100).toFixed(0)}%
        </span>{" "}
        <span className="text-[9px]">
          (residual / target RMS — 0 = pure linear, →1 = highly nonlinear)
        </span>
      </div>
    </div>
  );
}

function Bar({
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
      <div className="relative h-2.5 rounded bg-gray-100 dark:bg-gray-800 overflow-hidden">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300 dark:bg-gray-700" />
        <div
          className={
            isNeg
              ? "absolute top-0 bottom-0 bg-red-400/70"
              : "absolute top-0 bottom-0 bg-blue-500/70"
          }
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
