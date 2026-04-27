// Diverging colormap for reward-weight values.
//
// Negative → red, zero → neutral grey, positive → blue. RGB stops are
// interpolated in linear space so the midpoint reads as actually neutral.
// Intuition: a prey's w_pred is the "fear" axis; values around -2 mean strong
// fear, so dyeing the canvas in red makes evolved fear visually obvious.
//
// `axis` is the absolute value at which the colormap saturates. Values outside
// ±axis are clamped. Default 2.0 matches the AgentInspector / WeightHistogram
// axis so all three views read against the same scale.

const NEG = [0xef, 0x44, 0x44]; // red-500
const ZERO_LIGHT = [0xe5, 0xe7, 0xeb]; // gray-200
const ZERO_DARK = [0x37, 0x41, 0x51]; // gray-700
const POS = [0x3b, 0x82, 0xf6]; // blue-500

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function weightColorRgb(
  value: number,
  axis = 2.0,
  isDark = false,
): [number, number, number] {
  const zero = isDark ? ZERO_DARK : ZERO_LIGHT;
  const t = Math.max(-1, Math.min(1, value / axis));
  if (t < 0) {
    const m = -t;
    return [
      Math.round(lerp(zero[0], NEG[0], m)),
      Math.round(lerp(zero[1], NEG[1], m)),
      Math.round(lerp(zero[2], NEG[2], m)),
    ];
  }
  return [
    Math.round(lerp(zero[0], POS[0], t)),
    Math.round(lerp(zero[1], POS[1], t)),
    Math.round(lerp(zero[2], POS[2], t)),
  ];
}

export function weightColor(value: number, axis = 2.0, isDark = false): string {
  const [r, g, b] = weightColorRgb(value, axis, isDark);
  return `rgb(${r},${g},${b})`;
}

// Axis label → index into the 4-vector reward genome stored in replays.
// Kept here (rather than ReplayCanvas) so other consumers (legend, future
// weight-trajectory strip) share one source of truth.
export const WEIGHT_AXIS_INDEX: Record<WeightAxisKey, number> = {
  w_eat: 0,
  w_act: 1,
  w_prey: 2,
  w_pred: 3,
};

export type WeightAxisKey = "w_eat" | "w_act" | "w_prey" | "w_pred";
export type ColorByKey = "species" | WeightAxisKey;

export const WEIGHT_AXIS_LABELS: Record<WeightAxisKey, string> = {
  w_eat: "food drive",
  w_act: "action cost",
  w_prey: "social / chase",
  w_pred: "fear / hunt",
};
