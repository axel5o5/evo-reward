import { ArchiveSummary, ReplayIndexEntry } from "./replayLoader";

// experiment_name parsing.
//
// Two schemas exist in the wild:
//   1. NEW (post §15.23-§15.24 reorg): <prefix>_<mech>_<tier> where
//        prefix ∈ {axis1, axis2, axis12, baseline}
//        tier   ∈ {tiny, small, med, full}
//        mech   = the descriptive middle (e.g. residual_reward_mlp,
//                 social_heading, kd_linear)
//      Plus the special standalone `baseline_faithful` (paper-pure, no tier).
//   2. LEGACY: free-form names produced before the reorg. Listed in
//      LEGACY_ALIASES so old replays in the index still render readably.
//
// `parseExp` picks the new schema first, falls back to the legacy alias
// table, then heuristics. Adding a new tier file requires no code change.

export type ExpVariant =
  | "Baseline (paper)"
  | "Baseline"
  | "Axis 1"
  | "Axis 2"
  | "Axis 1+2"
  | "Demo"
  | "Custom";

export type ExpCategory = "active" | "archive";
export type ExpTier = "tiny" | "small" | "med" | "full";

export interface ParsedExp {
  variant: ExpVariant;
  mechanism: string | null;
  tier: ExpTier | null;
  category: ExpCategory;
  isLegacy: boolean;
  // Pretty title; tier rendered as a separate chip, not in the title.
  title: string;
}

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

// Order matters for the prefix alternation: `axis12` must come before `axis1`
// so it isn't shadowed. (Backtracking would still rescue it, but explicit
// ordering avoids the costly retry.)
const NEW_SCHEMA_RE =
  /^(axis12|axis1|axis2|baseline)(?:_(.+))?_(tiny|small|med|full)$/;

// Pretty rendering for known mechanism stems. Multi-mechanism combinations
// (e.g. axis12) are matched as full keys here so the joiner is explicit.
const MECHANISM_PRETTY: Record<string, string> = {
  residual_reward_mlp: "Residual Reward MLP",
  social_heading: "Social Heading",
  kd_linear: "K&D Linear (control)",
  residual_reward_mlp_social_heading: "Residual Reward MLP + Social Heading",
};

// Replays from before the reorg. Keys are exact `experiment_name` strings;
// these will never reappear in new configs, so this list only grows when an
// old run shows up in index.json that we haven't seen rendered yet.
const LEGACY_ALIASES: Record<
  string,
  { title: string; variant: ExpVariant; mechanism: string | null }
> = {
  axis1_residual: {
    title: "Axis 1 — Residual Reward MLP",
    variant: "Axis 1",
    mechanism: "Residual Reward MLP",
  },
  axis1_mlp_reward: {
    title: "Axis 1 — MLP Reward (replaced)",
    variant: "Axis 1",
    mechanism: "MLP Reward",
  },
  axis2_aligned: {
    title: "Axis 2 — Bin-aligned Heading",
    variant: "Axis 2",
    mechanism: "Bin-aligned Heading",
  },
  axis2_social_obs: {
    title: "Axis 2 — Social Observation",
    variant: "Axis 2",
    mechanism: "Social Observation",
  },
  axis2_both_1: {
    title: "Axis 2 — Both (v1)",
    variant: "Axis 2",
    mechanism: "Both",
  },
  axis2_cross_1: {
    title: "Axis 2 — Cross (v1)",
    variant: "Axis 2",
    mechanism: "Cross",
  },
  axis3_temporal_reward: {
    title: "Axis 3 — Temporal Reward",
    variant: "Custom",
    mechanism: "Temporal Reward",
  },
  axis4_lstm_policy: {
    title: "Axis 4 — LSTM Policy",
    variant: "Custom",
    mechanism: "LSTM Policy",
  },
  baseline_endpoint: {
    title: "Baseline (endpoint)",
    variant: "Baseline",
    mechanism: null,
  },
  baseline_med_ddb: {
    title: "Baseline + DDB (med)",
    variant: "Baseline",
    mechanism: "DDB",
  },
  baseline_med_ddb_ddm: {
    title: "Baseline + DDB+DDM (med)",
    variant: "Baseline",
    mechanism: "DDB+DDM",
  },
  baseline_smol_ddb: {
    title: "Baseline + DDB (small)",
    variant: "Baseline",
    mechanism: "DDB",
  },
  demo_random_init: {
    title: "Initial Conditions Demo",
    variant: "Demo",
    mechanism: null,
  },
};

function prettyToken(tok: string): string {
  const low = tok.toLowerCase();
  if (low === "lstm") return "LSTM";
  if (low === "ppo") return "PPO";
  if (low === "jax") return "JAX";
  if (low === "kd") return "K&D";
  if (low === "mlp") return "MLP";
  if (low === "ddb") return "DDB";
  if (low === "ddm") return "DDM";
  if (low.length === 0) return tok;
  return low[0].toUpperCase() + low.slice(1);
}

export function prettyId(id: string): string {
  return id
    .split(/[_-]+/)
    .filter((p) => p.length > 0)
    .map(prettyToken)
    .join(" ");
}

function prettyMechanism(stem: string): string {
  return MECHANISM_PRETTY[stem] ?? prettyId(stem);
}

export function parseExp(exp: string): ParsedExp {
  // 1. baseline_faithful is the K&D paper-pure reference; stands apart from
  //    the tiered baseline/ controls because it has no scaffolds.
  if (exp === "baseline_faithful") {
    return {
      variant: "Baseline (paper)",
      mechanism: null,
      tier: null,
      category: "active",
      isLegacy: false,
      title: "Baseline — K&D Paper-faithful",
    };
  }

  // 2. New schema (axis1/axis2/axis12/baseline + tier).
  const m = exp.match(NEW_SCHEMA_RE);
  if (m) {
    const [, prefix, mechStem, tierRaw] = m;
    const tier = tierRaw as ExpTier;
    const variant: ExpVariant =
      prefix === "axis1" ? "Axis 1" :
      prefix === "axis2" ? "Axis 2" :
      prefix === "axis12" ? "Axis 1+2" :
      "Baseline";
    const mechanism = mechStem ? prettyMechanism(mechStem) : null;
    let title: string;
    if (variant === "Baseline") {
      title = mechanism ? `Baseline — ${mechanism}` : "Baseline";
    } else {
      title = mechanism ? `${variant} — ${mechanism}` : variant;
    }
    return {
      variant,
      mechanism,
      tier,
      category: "active",
      isLegacy: false,
      title,
    };
  }

  // 3. Legacy alias table.
  const legacy = LEGACY_ALIASES[exp];
  if (legacy) {
    return {
      variant: legacy.variant,
      mechanism: legacy.mechanism,
      tier: null,
      category: "archive",
      isLegacy: true,
      title: legacy.title,
    };
  }

  // 4. Unknown — best-effort heuristics so something renders.
  const axisM = exp.match(/^axis(\d+)/i);
  if (axisM) {
    const n = axisM[1];
    const variant: ExpVariant =
      n === "1" ? "Axis 1" :
      n === "2" ? "Axis 2" :
      n === "12" ? "Axis 1+2" :
      "Custom";
    return {
      variant,
      mechanism: null,
      tier: null,
      category: "archive",
      isLegacy: true,
      title: prettyId(exp),
    };
  }
  if (exp.startsWith("baseline")) {
    return {
      variant: "Baseline",
      mechanism: null,
      tier: null,
      category: "archive",
      isLegacy: true,
      title: prettyId(exp),
    };
  }
  if (exp.startsWith("demo")) {
    return {
      variant: "Demo",
      mechanism: null,
      tier: null,
      category: "archive",
      isLegacy: true,
      title: prettyId(exp),
    };
  }
  return {
    variant: "Custom",
    mechanism: null,
    tier: null,
    category: "archive",
    isLegacy: true,
    title: prettyId(exp),
  };
}

// -- Public helpers (existing call sites) ------------------------------------

export function displayExperimentName(exp: string): string {
  return parseExp(exp).title;
}

export function variantForExp(exp: string): ExpVariant {
  return parseExp(exp).variant;
}

export function categoryForExp(exp: string): ExpCategory {
  return parseExp(exp).category;
}

export function tierForExp(exp: string): ExpTier | null {
  return parseExp(exp).tier;
}

// -- Run-tag formatting ------------------------------------------------------

// Matches both legacy ISO-with-time tags (`2026-04-21T1447Z_*`) and the
// post-rename date-only / date-prefixed form (`2026-04-21`, `2026-04-21_*`).
const DATE_PREFIX_RE =
  /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2})(\d{2})Z)?(?:[_-](.+))?$/;

function parseRunTagTimestamp(runTag: string): string | null {
  const m = runTag.match(DATE_PREFIX_RE);
  if (!m) return null;
  const month = Number(m[2]);
  const day = Number(m[3]);
  if (
    !Number.isFinite(month) || !Number.isFinite(day) ||
    month < 1 || month > 12
  ) {
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
    !Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day) ||
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

// -- Run-level stats from the archive summary --------------------------------

export interface RunStats {
  finalStep: number;
  extinct: boolean;
  // "mixed" only when multiple seeds under the same tag extincted in
  // different species; rare but possible.
  extinctSpecies: "prey" | "pred" | "none" | "mixed";
  extinctionStep: number | null;
  // Number of (exp, seed) tuples this aggregates. >1 is rare in practice.
  count: number;
}

// Aggregate per-tag stats. A run_tag identifies a launch; multiple (exp, seed)
// pairs may share a tag in principle, so we aggregate: take the longest run
// for finalStep, mark extinct if any did, surface "mixed" if seeds disagree.
export function tagStats(
  summary: ArchiveSummary | null,
  tag: string,
): RunStats | null {
  if (!summary || tag === "current") return null;
  const matches = summary.runs.filter((r) => r.run_tag === tag);
  if (matches.length === 0) return null;
  const finalStep = matches.reduce(
    (m, r) => Math.max(m, r.final_step ?? 0),
    0,
  );
  const extinctRuns = matches.filter((r) => r.extinct);
  const extinct = extinctRuns.length > 0;
  let extinctSpecies: RunStats["extinctSpecies"] = "none";
  let extinctionStep: number | null = null;
  if (extinct) {
    const speciesSet = new Set(
      extinctRuns
        .map((r) => r.extinct_species)
        .filter((s) => s === "prey" || s === "pred"),
    );
    if (speciesSet.size === 1) {
      extinctSpecies = (speciesSet.values().next().value ?? "none") as
        "prey" | "pred";
    } else if (speciesSet.size > 1) {
      extinctSpecies = "mixed";
    }
    extinctionStep = extinctRuns.reduce<number | null>((m, r) => {
      if (r.extinction_step === null) return m;
      if (m === null) return r.extinction_step;
      return Math.min(m, r.extinction_step);
    }, null);
  }
  return {
    finalStep,
    extinct,
    extinctSpecies,
    extinctionStep,
    count: matches.length,
  };
}

// Run-stats keyed on (exp, seed, tag), for the "Selected replay" card where
// we know the exact run, not just the tag.
export function runStatsFor(
  summary: ArchiveSummary | null,
  exp: string,
  seed: number,
  tag: string,
): RunStats | null {
  if (!summary || tag === "current") return null;
  const r = summary.runs.find(
    (s) => s.exp === exp && s.seed === seed && s.run_tag === tag,
  );
  if (!r) return null;
  return {
    finalStep: r.final_step ?? 0,
    extinct: r.extinct,
    extinctSpecies:
      r.extinct_species === "none" || !r.extinct
        ? "none"
        : (r.extinct_species as "prey" | "pred"),
    extinctionStep: r.extinction_step,
    count: 1,
  };
}

// "1.0M", "250K", "12.5K", "8000". One decimal once the number reaches
// the next prefix; integer for the first decade past it ("10M", not "10.0M").
export function formatSteps(n: number): string {
  if (n >= 10_000_000) return `${Math.round(n / 1_000_000)}M`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 100_000) return `${Math.round(n / 1_000)}K`;
  if (n >= 10_000) return `${(n / 1_000).toFixed(1)}K`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

// -- describeReplay ----------------------------------------------------------

export interface ReplayDisplay {
  title: string;
  variant: ExpVariant;
  tier: ExpTier | null;
  runLabel: string;
  rawId: string;
  chips: string[];
}

export function describeReplay(entry: ReplayIndexEntry): ReplayDisplay {
  const parsed = parseExp(entry.exp);
  const runLabel = displayRunTag(entry.run_tag);
  const chips: string[] = [parsed.variant];
  if (parsed.tier) chips.push(parsed.tier);
  chips.push(`Seed ${entry.seed}`);
  chips.push(`Step ${entry.start_step.toLocaleString()}`);
  chips.push(`${entry.n_frames.toLocaleString()} frames`);
  const dateLabel = entry.run_tag ? parseRunTagTimestamp(entry.run_tag) : null;
  if (dateLabel) chips.push(dateLabel);
  return {
    title: parsed.title,
    variant: parsed.variant,
    tier: parsed.tier,
    runLabel,
    rawId: `${entry.run_tag || "current"}:${entry.exp}:seed_${entry.seed}:step_${entry.start_step}`,
    chips,
  };
}
