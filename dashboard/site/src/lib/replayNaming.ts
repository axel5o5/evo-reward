import { ReplayIndexEntry } from "./replayLoader";

const TITLE_OVERRIDES: Record<string, string> = {
  baseline_faithful: "Baseline Replication — Faithful",
  baseline_simplified: "Baseline Replication — Simplified",
  axis2_social_obs: "Axis 2 — Social Observation",
  axis3_temporal_reward: "Axis 3 — Temporal Reward",
  axis4_lstm_policy: "Axis 4 — LSTM Policy",
  demo_random_init: "Initial Conditions Demo",
};

const VARIANT_OVERRIDES: Record<string, string> = {
  baseline_faithful: "Baseline",
  baseline_simplified: "Baseline",
  axis2_social_obs: "Axis 2",
  axis3_temporal_reward: "Axis 3",
  axis4_lstm_policy: "Axis 4",
  demo_random_init: "Demo",
};

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function prettyToken(token: string): string {
  const low = token.toLowerCase();
  if (low === "lstm") return "LSTM";
  if (low === "ppo") return "PPO";
  if (low === "jax") return "JAX";
  if (low === "kd") return "K&D";
  if (low.length === 0) return token;
  return low[0].toUpperCase() + low.slice(1);
}

export function prettyId(id: string): string {
  return id
    .split(/[_-]+/)
    .filter((part) => part.length > 0)
    .map(prettyToken)
    .join(" ");
}

export function displayExperimentName(exp: string): string {
  return TITLE_OVERRIDES[exp] ?? prettyId(exp);
}

// Mirrors configs/ vs configs/archive/ — keep in sync when an experiment
// graduates or is shelved (see configs/README.md and configs/archive/README.md).
const ACTIVE_EXPS = new Set<string>([
  "axis1_residual",
  "axis2_aligned",
  "baseline_faithful",
]);

export type ExpCategory = "active" | "archive";

export function categoryForExp(exp: string): ExpCategory {
  return ACTIVE_EXPS.has(exp) ? "active" : "archive";
}

export function variantForExp(exp: string): string {
  if (VARIANT_OVERRIDES[exp]) return VARIANT_OVERRIDES[exp];
  const axis = exp.match(/^axis(\d+)/i);
  if (axis) return `Axis ${axis[1]}`;
  if (exp.startsWith("baseline")) return "Baseline";
  if (exp.startsWith("demo")) return "Demo";
  return "Custom";
}

// Matches both legacy ISO-with-time tags (`2026-04-21T1447Z_*`) and the
// post-rename date-only / date-prefixed form (`2026-04-21`, `2026-04-21_*`).
const DATE_PREFIX_RE =
  /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2})(\d{2})Z)?(?:[_-](.+))?$/;

function parseRunTagTimestamp(runTag: string): string | null {
  const m = runTag.match(DATE_PREFIX_RE);
  if (!m) return null;
  const month = Number(m[2]);
  const day = Number(m[3]);
  if (!Number.isFinite(month) || !Number.isFinite(day) || month < 1 || month > 12) {
    return null;
  }
  return `${MONTHS[month - 1]} ${day}`;
}

// Returns a Date for the run's date prefix, or null for tags that don't
// start with YYYY-MM-DD. Used for date-banded grouping in the run picker.
export function parseRunTagDate(runTag?: string): Date | null {
  if (!runTag) return null;
  const m = runTag.match(DATE_PREFIX_RE);
  if (!m) return null;
  const year = Number(m[1]);
  const month = Number(m[2]);
  const day = Number(m[3]);
  if (
    !Number.isFinite(year) ||
    !Number.isFinite(month) ||
    !Number.isFinite(day) ||
    month < 1 || month > 12 || day < 1 || day > 31
  ) {
    return null;
  }
  return new Date(year, month - 1, day);
}

export function displayRunTag(runTag?: string): string {
  if (!runTag || runTag.length === 0) return "Current";
  const m = runTag.match(DATE_PREFIX_RE);
  if (m) {
    const ts = parseRunTagTimestamp(runTag);
    const suffix = m[6] ?? "";
    if (ts && suffix.length > 0) return `${ts} · ${prettyId(suffix)}`;
    if (ts) return ts;
  }
  return prettyId(runTag);
}

export interface ReplayDisplay {
  title: string;
  variant: string;
  runLabel: string;
  rawId: string;
  chips: string[];
}

export function describeReplay(entry: ReplayIndexEntry): ReplayDisplay {
  const variant = variantForExp(entry.exp);
  const runLabel = displayRunTag(entry.run_tag);
  const chips = [
    variant,
    `Seed ${entry.seed}`,
    `Step ${entry.start_step.toLocaleString()}`,
    `${entry.n_frames.toLocaleString()} frames`,
  ];
  const dateLabel = entry.run_tag ? parseRunTagTimestamp(entry.run_tag) : null;
  if (dateLabel) chips.push(dateLabel);
  return {
    title: displayExperimentName(entry.exp),
    variant,
    runLabel,
    rawId: `${entry.run_tag || "current"}:${entry.exp}:seed_${entry.seed}:step_${entry.start_step}`,
    chips,
  };
}
