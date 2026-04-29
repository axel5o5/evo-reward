import { useMemo } from "react";
import { MlpGenome, MlpLayer } from "../lib/rewardMlp";
import { weightColorRgb } from "../lib/weightColor";

// Live network diagram for an MLP reward genome — 4 → 8 → 8 → 1 with edge
// color/opacity encoding evolved weights and node fill encoding activation
// for the current stimulus state. Driven by the same heldValues array that
// feeds RewardLandscape, so the two views share a forward pass and a user
// dragging a slider sees both update together.
//
// Edges: blue = positive weight, red = negative (matches weightColor's
// reward-weight scale). Opacity + width scale with |w|/maxWeight, so the
// dominant pathways stand out and small near-zero weights fade. With 104
// edges packed into a 360-px SVG this is what keeps the picture readable.
//
// Nodes: input nodes always reflect the current heldValues. Hidden and
// output nodes show post-activation values from the forward pass. The
// diverging weightColor scale is reused for activation fill so positive
// (blue) / negative (red) read consistently across the dashboard.

interface Props {
  genome: MlpGenome;
  // Length 4: [n_eaten, motor_norm, s_prey, s_pred]. Same array used by
  // RewardLandscape to fill held axes — owned by the parent.
  heldValues: Float32Array;
  isDark?: boolean;
}

const NODE_R = 8;
const SVG_W = 360;
const SVG_H = 240;
const PAD_Y = 24;
const LAYER_X = [40, 140, 240, SVG_W - 30] as const;
const LAYER_LABELS = ["input", "hidden 1", "hidden 2", "reward"] as const;
const INPUT_LABELS = ["n_eaten", "motor", "s_prey", "s_pred"] as const;

function denseTanh(x: Float32Array, layer: MlpLayer): Float32Array {
  const inDim = layer.kernel.length;
  const outDim = layer.bias.length;
  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let s = layer.bias[j];
    for (let i = 0; i < inDim; i++) s += x[i] * layer.kernel[i][j];
    out[j] = Math.tanh(s);
  }
  return out;
}

function denseLinear(x: Float32Array, layer: MlpLayer): Float32Array {
  const inDim = layer.kernel.length;
  const outDim = layer.bias.length;
  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let s = layer.bias[j];
    for (let i = 0; i < inDim; i++) s += x[i] * layer.kernel[i][j];
    out[j] = s;
  }
  return out;
}

function nodeY(layerSize: number, idx: number): number {
  if (layerSize <= 1) return SVG_H / 2;
  const usableH = SVG_H - 2 * PAD_Y;
  return PAD_Y + (idx / (layerSize - 1)) * usableH;
}

export default function RewardMlpDiagram({ genome, heldValues, isDark = false }: Props) {
  const { activations, weightAbsMax, actAbsMax, layers } = useMemo(() => {
    const a0 = heldValues;
    const a1 = denseTanh(a0, genome.Dense_0);
    const a2 = denseTanh(a1, genome.Dense_1);
    const a3 = denseLinear(a2, genome.Dense_2);

    const ls = [genome.Dense_0, genome.Dense_1, genome.Dense_2];
    let wMax = 1e-6;
    for (const layer of ls) {
      for (const row of layer.kernel) {
        for (const w of row) {
          const a = Math.abs(w);
          if (a > wMax) wMax = a;
        }
      }
    }
    // Activation scale: max over hidden+output. Inputs get their own scale
    // because n_eaten ranges 0–3 and the rest 0–1 — including them would
    // wash the hidden layers grey by comparison.
    let aMax = 1e-6;
    for (const arr of [a1, a2, a3]) {
      for (const v of arr) {
        const a = Math.abs(v);
        if (a > aMax) aMax = a;
      }
    }
    return { activations: [a0, a1, a2, a3], weightAbsMax: wMax, actAbsMax: aMax, layers: ls };
  }, [genome, heldValues]);

  // Sort edges by |weight| ascending so dominant ones render on top.
  const edges = useMemo(() => {
    type Edge = { x1: number; y1: number; x2: number; y2: number; w: number; key: string };
    const out: Edge[] = [];
    for (let li = 0; li < layers.length; li++) {
      const layer = layers[li];
      const inDim = layer.kernel.length;
      const outDim = layer.bias.length;
      for (let i = 0; i < inDim; i++) {
        for (let j = 0; j < outDim; j++) {
          out.push({
            x1: LAYER_X[li],
            y1: nodeY(inDim, i),
            x2: LAYER_X[li + 1],
            y2: nodeY(outDim, j),
            w: layer.kernel[i][j],
            key: `${li}-${i}-${j}`,
          });
        }
      }
    }
    out.sort((a, b) => Math.abs(a.w) - Math.abs(b.w));
    return out;
  }, [layers]);

  const reward = activations[3][0];
  // Input node activation scale (covers 0..3 for n_eaten, 0..1 for others).
  const INPUT_AXIS = 2.0;

  return (
    <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-white dark:bg-gray-950">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
          Network <span className="text-[10px] font-normal text-gray-500">live activations · weights</span>
        </div>
        <div className="text-[10px] font-mono text-gray-500">
          reward ={" "}
          <span className="text-gray-800 dark:text-gray-200 tabular-nums">
            {reward >= 0 ? "+" : ""}
            {reward.toFixed(2)}
          </span>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        width="100%"
        height={SVG_H}
        preserveAspectRatio="xMidYMid meet"
        className="block"
      >
        {/* Layer labels */}
        {LAYER_LABELS.map((label, i) => (
          <text
            key={`lbl-${i}`}
            x={LAYER_X[i]}
            y={12}
            textAnchor="middle"
            className="fill-gray-500 text-[9px] font-mono"
          >
            {label}
          </text>
        ))}

        {/* Edges */}
        {edges.map(({ x1, y1, x2, y2, w, key }) => {
          const [r, g, b] = weightColorRgb(w, weightAbsMax, isDark);
          const norm = Math.abs(w) / weightAbsMax;
          return (
            <line
              key={key}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={`rgb(${r},${g},${b})`}
              strokeOpacity={Math.max(0.05, norm)}
              strokeWidth={Math.max(0.5, Math.min(2.5, norm * 2.5))}
            >
              <title>{`weight ${w >= 0 ? "+" : ""}${w.toFixed(3)}`}</title>
            </line>
          );
        })}

        {/* Nodes */}
        {activations.map((arr, layerIdx) =>
          Array.from(arr).map((v, i) => {
            const cx = LAYER_X[layerIdx];
            const cy = nodeY(arr.length, i);
            // Input layer: fixed scale so 1.0 reads as moderately blue.
            // Other layers: relative to actAbsMax so structure is visible
            // even when the network outputs small magnitudes.
            const axis = layerIdx === 0 ? INPUT_AXIS : actAbsMax;
            const [r, g, b] = weightColorRgb(v, axis, isDark);
            const labelText =
              layerIdx === 0
                ? `${INPUT_LABELS[i]} = ${v.toFixed(2)}`
                : layerIdx === 3
                  ? `reward = ${v.toFixed(3)}`
                  : `${LAYER_LABELS[layerIdx]}[${i}] = ${v.toFixed(3)}`;
            return (
              <circle
                key={`n-${layerIdx}-${i}`}
                cx={cx}
                cy={cy}
                r={NODE_R}
                fill={`rgb(${r},${g},${b})`}
                stroke="currentColor"
                strokeWidth={1}
                className="text-gray-400 dark:text-gray-600"
              >
                <title>{labelText}</title>
              </circle>
            );
          })
        )}

        {/* Input labels (left of column 0) */}
        {INPUT_LABELS.map((label, i) => (
          <text
            key={`il-${i}`}
            x={LAYER_X[0] - NODE_R - 4}
            y={nodeY(4, i)}
            textAnchor="end"
            dominantBaseline="middle"
            className="fill-gray-600 dark:fill-gray-400 text-[9px] font-mono"
          >
            {label}
          </text>
        ))}
      </svg>
    </div>
  );
}
