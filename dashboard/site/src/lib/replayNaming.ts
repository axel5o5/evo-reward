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

export function variantForExp(exp: string): string {
  if (VARIANT_OVERRIDES[exp]) return VARIANT_OVERRIDES[exp];
  const axis = exp.match(/^axis(\d+)/i);
  if (axis) return `Axis ${axis[1]}`;
  if (exp.startsWith("baseline")) return "Baseline";
  if (exp.startsWith("demo")) return "Demo";
  return "Custom";
}

function parseRunTagTimestamp(runTag: string): string | null {
  // 2026-04-21T1447Z_post-d19 -> Apr 21
  const m = runTag.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})Z(?:[_-].+)?$/);
  if (!m) return null;
  const month = Number(m[2]);
  const day = Number(m[3]);
  if (!Number.isFinite(month) || !Number.isFinite(day) || month < 1 || month > 12) {
    return null;
  }
  return `${MONTHS[month - 1]} ${day}`;
}

export function displayRunTag(runTag?: string): string {
  if (!runTag || runTag.length === 0) return "Current";
  const ts = parseRunTagTimestamp(runTag);
  if (ts) {
    const suffix = runTag.replace(/^\d{4}-\d{2}-\d{2}T\d{4}Z[_-]?/, "");
    return suffix.length > 0 ? `${ts} · ${prettyId(suffix)}` : ts;
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
