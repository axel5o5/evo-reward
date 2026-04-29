import { useMemo } from "react";
import { MlpGenome, MlpLayer } from "../lib/rewardMlp";
import { weightColorRgb } from "../lib/weightColor";

// Live network diagram for any-depth MLP genome (Axis 1: 4 → 8 → 8 → 1,
// Axis 3 temporal: 40 → 16 → 16 → 1, future variants of arbitrary depth).
// Edges are colored by weight sign (blue +, red -) with opacity + width
// scaling by |w| / maxWeight, so dominant pathways stand out and small
// weights fade. Nodes fill with the same diverging palette to show layer
// activations from a live forward pass through the genome.
//
// All structural parameters — column count, column positions, per-column
// node spacing — derive from genome.layers, so swapping architectures
// (re-running with a different mlp_hidden_size, or moving to temporal)
// requires no change here. The synthetic-fixture path is gone; the only
// caller is AgentInspector, which gets genomes from the v3 replay loader.

interface Props {
  genome: MlpGenome;
  // Per-input stimulus values, length = genome.layers[0].kernel.length.
  // Owned by the parent (AgentInspector) so dragging a slider updates
  // both this component and the reward-landscape heatmap in lockstep.
  heldValues: Float32Array;
  // Optional labels for input axes — falls back to "x_<i>" if absent.
  inputLabels?: readonly string[];
  isDark?: boolean;
}

const SVG_W = 360;
const SVG_H = 240;
const PAD_X_LEFT = 40;
const PAD_X_RIGHT = 30;
const PAD_Y_TOP = 24;
const PAD_Y_BOTTOM = 12;

// Input axis used to color the input-layer nodes (separate from the per-run
// activation scale because n_eaten ranges 0..3 while sensors are 0..1 — a
// shared scale would wash the smaller inputs grey).
const INPUT_COLOR_AXIS = 2.0;

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

// Vertical position of the i-th node in a layer of `layerSize` nodes.
// Centers the column when the layer is small, fills available height
// when it's wide. Caps node radius for very wide layers so they don't
// overlap (Axis 3 temporal has 40 input nodes).
function nodeY(layerSize: number, idx: number): number {
  if (layerSize <= 1) return SVG_H / 2;
  const usableH = SVG_H - PAD_Y_TOP - PAD_Y_BOTTOM;
  return PAD_Y_TOP + (idx / (layerSize - 1)) * usableH;
}

function nodeRadius(maxLayerSize: number): number {
  // Pack with a bit of breathing room. At 8 nodes → radius 8, at 40 → ~3.
  const usableH = SVG_H - PAD_Y_TOP - PAD_Y_BOTTOM;
  const spacing = usableH / Math.max(1, maxLayerSize - 1);
  return Math.max(2.5, Math.min(8, spacing * 0.45));
}

export default function RewardMlpDiagram({
  genome,
  heldValues,
  inputLabels,
  isDark = false,
}: Props) {
  const {
    activations,
    layers,
    columnX,
    nodeR,
    weightAbsMax,
    actAbsMax,
  } = useMemo(() => {
    // Forward pass — collect activations at every column. Column 0 is the
    // raw input; columns 1..L are the post-activation outputs of each layer.
    const ls = genome.layers;
    if (ls.length === 0) {
      return {
        activations: [heldValues],
        layers: [] as MlpLayer[],
        columnX: [PAD_X_LEFT],
        nodeR: 6,
        weightAbsMax: 1,
        actAbsMax: 1,
      };
    }
    const acts: Float32Array[] = [heldValues];
    for (let i = 0; i < ls.length - 1; i++) {
      acts.push(denseTanh(acts[acts.length - 1], ls[i]));
    }
    acts.push(denseLinear(acts[acts.length - 1], ls[ls.length - 1]));

    let wMax = 1e-6;
    for (const layer of ls) {
      for (const row of layer.kernel) {
        for (const w of row) {
          const a = Math.abs(w);
          if (a > wMax) wMax = a;
        }
      }
    }
    // Activation scale spans hidden + output but skips the input layer
    // (its values can be on a different scale than the post-tanh outputs).
    let aMax = 1e-6;
    for (let i = 1; i < acts.length; i++) {
      for (const v of acts[i]) {
        const a = Math.abs(v);
        if (a > aMax) aMax = a;
      }
    }

    // Equally space columns across the SVG width.
    const nCols = acts.length;
    const usableW = SVG_W - PAD_X_LEFT - PAD_X_RIGHT;
    const cols: number[] = [];
    for (let i = 0; i < nCols; i++) {
      cols.push(
        nCols === 1
          ? PAD_X_LEFT + usableW / 2
          : PAD_X_LEFT + (i / (nCols - 1)) * usableW,
      );
    }

    let widest = 0;
    for (const a of acts) if (a.length > widest) widest = a.length;
    const r = nodeRadius(widest);

    return {
      activations: acts,
      layers: ls,
      columnX: cols,
      nodeR: r,
      weightAbsMax: wMax,
      actAbsMax: aMax,
    };
  }, [genome, heldValues]);

  // Edge geometry, sorted by |w| ascending so dominant ones render on top.
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
            x1: columnX[li],
            y1: nodeY(inDim, i),
            x2: columnX[li + 1],
            y2: nodeY(outDim, j),
            w: layer.kernel[i][j],
            key: `${li}-${i}-${j}`,
          });
        }
      }
    }
    out.sort((a, b) => Math.abs(a.w) - Math.abs(b.w));
    return out;
  }, [layers, columnX]);

  const reward = activations[activations.length - 1][0];

  // Column labels: input, hidden 1, hidden 2, ..., hidden N-1, reward.
  const columnLabels: string[] = [];
  if (activations.length > 0) {
    columnLabels.push("input");
    for (let i = 1; i < activations.length - 1; i++) {
      columnLabels.push(`hidden ${i}`);
    }
    if (activations.length > 1) columnLabels.push("reward");
  }

  const inputDim = activations[0]?.length ?? 0;
  // Only show input labels when they're meaningful and the layer is small
  // enough to not overlap (40-d temporal input would be too dense).
  const showInputLabels = inputLabels && inputDim <= 8;

  return (
    <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-3 bg-white dark:bg-gray-950">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
          Network{" "}
          <span className="text-[10px] font-normal text-gray-500">
            live activations · weights · {layers.length} layer
            {layers.length === 1 ? "" : "s"}
          </span>
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
        {/* Column labels (top of each layer column) */}
        {columnLabels.map((label, i) => (
          <text
            key={`col-${i}`}
            x={columnX[i]}
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
              strokeWidth={Math.max(0.3, Math.min(2.5, norm * 2.5))}
            >
              <title>{`weight ${w >= 0 ? "+" : ""}${w.toFixed(3)}`}</title>
            </line>
          );
        })}

        {/* Nodes — input layer uses a fixed colormap axis so 1.0 reads as
            moderately blue; other layers use the per-forward-pass max so
            structure is visible even when the network outputs small
            magnitudes. */}
        {activations.map((arr, layerIdx) =>
          Array.from(arr).map((v, i) => {
            const cx = columnX[layerIdx];
            const cy = nodeY(arr.length, i);
            const axis = layerIdx === 0 ? INPUT_COLOR_AXIS : actAbsMax;
            const [r, g, b] = weightColorRgb(v, axis, isDark);
            const tipName =
              layerIdx === 0
                ? inputLabels?.[i] ?? `x_${i}`
                : layerIdx === activations.length - 1
                  ? "reward"
                  : `${columnLabels[layerIdx]}[${i}]`;
            return (
              <circle
                key={`n-${layerIdx}-${i}`}
                cx={cx}
                cy={cy}
                r={nodeR}
                fill={`rgb(${r},${g},${b})`}
                stroke="currentColor"
                strokeWidth={1}
                className="text-gray-400 dark:text-gray-600"
              >
                <title>{`${tipName} = ${v.toFixed(3)}`}</title>
              </circle>
            );
          })
        )}

        {/* Input labels — only when meaningful (named) and layer fits */}
        {showInputLabels &&
          inputLabels.slice(0, inputDim).map((label, i) => (
            <text
              key={`il-${i}`}
              x={columnX[0] - nodeR - 4}
              y={nodeY(inputDim, i)}
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
