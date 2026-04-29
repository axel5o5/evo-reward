// Replay data loader.
//
// Source is controlled by VITE_REPLAYS_BASE_URL (falls back to "/replays/").
// Today that resolves to same-origin static files shipped with the Vercel
// build; swap it for a GCS public prefix or a serverless endpoint later
// without touching this file's consumers.
//
// Binary format evolves — meta.json.version gates any breaking changes.
// v1 sections (in file order) and their dtypes when quantize=true:
//   pos           uint16   (scale = meta.scales.pos)
//   angle         float32
//   energy        uint8    (scale = meta.scales.energy)
//   alive         uint8    (boolean)
//   food_pos      uint16   (scale = meta.scales.food_pos)
//   food_active   uint8    (boolean)
//   step_nums     int32
//   species       int32    (static, length = max_agents)
//   radii         float32  (static, length = max_agents)
// v2 adds per-frame identity/lineage/phenotype tracks. Dtypes below are the
// quantized form (what ships to GCS); quantize=false falls back to raw:
//   agent_ids      uint16  (scale via meta.id_base, 0 = -1 sentinel)
//   parent_ids     uint16  (same encoding as agent_ids)
//   ages           int32                                — birth_step = step - age
//   reward_weights int8    (scale = meta.scales.reward_weights)
//   action         int8    (scale = meta.scales.action)
// v3 tags genome architecture in meta and, for non-linear runs, ships
// per-genome rows instead of the per-frame reward_weights field:
//   reward_genomes_byid   float32 (n_unique, genome_dim)   — MLP / temporal
//   reward_genomes_idmap  int32   (n_unique,)              — agent_id per row
// Linear v3 replays keep the v2 reward_weights pipeline. The loader exposes
// `genomesById: Map<number, MlpGenome>` for v3-MLP replays; consumers fall
// back to `null` for v1/v2/linear.
// Sizes at defaults (length=1000, max_agents=500): v2 ≈ 14 MB vs v1 ≈ 8 MB.
// v3 MLP adds ~240 KB (h=8) on top of the v2 baseline, after dropping the
// 4-vector reward_weights section.
// Consumers always see Float32Array / Int32Array — dequantization happens
// here, eagerly. v1 replays still load; the v2 fields are null.

export interface ReplayIndexEntry {
  exp: string;
  seed: number;
  start_step: number;
  n_frames: number;
  path: string;
  size_bytes: number;
  // Optional: present on replays uploaded via the tagged layout (see
  // docs/emevo-diff.md D18). Empty/undefined = legacy untagged layout
  // (we treat that as "current" post-D18 by convention).
  run_tag?: string;
  // Optional: downsampled prey/pred counts for a thumbnail next to each
  // selector dot. Populated by scripts/gen_replay_thumbnails.py. Same length
  // on both arrays (default 60 points); missing = render no thumbnail.
  sparkline?: {
    prey: number[];
    pred: number[];
  };
}

export interface ReplayIndex {
  replays: ReplayIndexEntry[];
}

type Dtype = "float32" | "uint8" | "int8" | "int32" | "uint16";

interface SectionMeta {
  offset: number;
  length: number;
  dtype: Dtype;
  shape: number[];
}

export interface GenomeLayoutEntry {
  // Path into the nested genome dict from the dashboard's perspective —
  // recorder strips Flax's leading "params" wrapper so consumers see a
  // bare {Dense_0: {...}, Dense_1: {...}, ...} shape.
  path: string[];
  shape: number[];
  offset: number;
}

export interface ReplayMeta {
  version: number;
  start_step: number;
  n_frames: number;
  max_agents: number;
  food_max: number;
  world_size: number;
  frames_bin: string;
  frames_bin_size: number;
  sections: Record<string, SectionMeta>;
  // Present on quantized replays. scales maps section name → multiplicative
  // scale for dequantization (e.g. pos uint16 * scale → world units).
  quantize?: boolean;
  scales?: Record<string, number>;
  // v2 only: additive offset for agent_ids / parent_ids when stored as
  // uint16. Absolute id = (stored - 1) + id_base; stored 0 → -1 sentinel.
  id_base?: number;
  // v3 only: genome-architecture metadata. genome_arch is "linear" |
  // "mlp" | "temporal". For non-linear runs, genome_layout describes how
  // to re-nest a flat row from `reward_genomes_byid` into the per-layer
  // kernel/bias structure used by the dashboard's forward pass.
  genome_arch?: "linear" | "mlp" | "temporal";
  genome_shape?: Record<string, number>;
  genome_layout?: GenomeLayoutEntry[];
  genome_dim?: number;
}

// MlpGenome is shared with rewardMlp.ts (the forward-pass + landscape
// sampler) — keep one source of truth so the loader output drops directly
// into the existing TS code path.
import type { MlpGenome, MlpLayer } from "./rewardMlp";

export interface ReplayData {
  meta: ReplayMeta;
  // Per-frame arrays — sliced on demand via frameView().
  pos: Float32Array;        // length = n_frames * max_agents * 2
  angle: Float32Array;      // length = n_frames * max_agents
  energy: Float32Array;     // length = n_frames * max_agents
  alive: Uint8Array;        // length = n_frames * max_agents
  foodPos: Float32Array;    // length = n_frames * food_max * 2
  foodActive: Uint8Array;   // length = n_frames * food_max
  stepNums: Int32Array;     // length = n_frames
  species: Int32Array;      // length = max_agents (static)
  radii: Float32Array;      // length = max_agents (static)
  // v2 per-frame fields. Null for v1 replays — consumers must handle both.
  agentIds: Int32Array | null;        // length = n_frames * max_agents
  parentIds: Int32Array | null;       // length = n_frames * max_agents
  ages: Int32Array | null;            // length = n_frames * max_agents
  rewardWeights: Float32Array | null; // length = n_frames * max_agents * 4
  action: Float32Array | null;        // length = n_frames * max_agents * 2
  // v3 — populated only for MLP genomes. One MLP per agent_id seen during
  // the recording window. Null on v1/v2 and on linear v3 replays. The
  // recorder snapshots the genome the first frame an agent is observed,
  // which is exact (not a sample) because genomes are mutated only at
  // birth and frozen for the agent's lifetime.
  genomesById: Map<number, MlpGenome> | null;
}

export function replaysBaseUrl(): string {
  const env = (import.meta as unknown as { env?: Record<string, string> }).env;
  const fromEnv = env?.VITE_REPLAYS_BASE_URL;
  const raw = fromEnv && fromEnv.length > 0 ? fromEnv : "/replays/";
  return raw.endsWith("/") ? raw : raw + "/";
}

export async function fetchIndex(): Promise<ReplayIndex> {
  const base = replaysBaseUrl();
  const res = await fetch(base + "index.json", { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`replays index not found at ${base}index.json (${res.status})`);
  }
  return (await res.json()) as ReplayIndex;
}

export async function fetchReplay(entry: ReplayIndexEntry): Promise<ReplayData> {
  const base = replaysBaseUrl();
  const dir = base + entry.path.replace(/\\/g, "/") + "/";
  const [metaRes, binRes] = await Promise.all([
    fetch(dir + "meta.json", { cache: "no-cache" }),
    fetch(dir + "frames.bin", { cache: "no-cache" }),
  ]);
  if (!metaRes.ok) throw new Error(`meta.json not found at ${dir}`);
  if (!binRes.ok) throw new Error(`frames.bin not found at ${dir}`);
  const meta = (await metaRes.json()) as ReplayMeta;
  const buf = await binRes.arrayBuffer();
  return decodeReplay(meta, buf);
}

// -- section helpers ---------------------------------------------------------

function elementSize(dtype: Dtype): number {
  switch (dtype) {
    case "float32":
    case "int32":
      return 4;
    case "uint16":
      return 2;
    case "uint8":
    case "int8":
      return 1;
  }
}

function sectionOrThrow(meta: ReplayMeta, name: string): SectionMeta {
  const s = meta.sections[name];
  if (!s) throw new Error(`replay meta missing section: ${name}`);
  return s;
}

/** Zero-copy typed-array view over a section. Use when no dtype conversion is needed. */
function rawView<T>(
  buf: ArrayBuffer,
  s: SectionMeta,
  ctor: new (b: ArrayBuffer, o: number, n: number) => T,
): T {
  const count = s.length / elementSize(s.dtype);
  return new ctor(buf, s.offset, count);
}

/** Load a section as Float32Array, dequantizing from uint16/uint8 if needed. */
function loadFloat32(buf: ArrayBuffer, meta: ReplayMeta, name: string): Float32Array {
  const s = sectionOrThrow(meta, name);
  if (s.dtype === "float32") {
    return rawView(buf, s, Float32Array);
  }
  const scale = meta.scales?.[name];
  if (scale === undefined) {
    throw new Error(`section ${name} is ${s.dtype} but no scale provided in meta.scales`);
  }
  if (s.dtype === "uint16") {
    const src = rawView(buf, s, Uint16Array);
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = src[i] * scale;
    return dst;
  }
  if (s.dtype === "uint8") {
    const src = rawView(buf, s, Uint8Array);
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = src[i] * scale;
    return dst;
  }
  if (s.dtype === "int8") {
    const src = rawView(buf, s, Int8Array);
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = src[i] * scale;
    return dst;
  }
  throw new Error(`unexpected dtype ${s.dtype} for float section ${name}`);
}

// Load an agent-id section (agent_ids / parent_ids). When quantized (uint16)
// the encoding is: stored 0 → -1 sentinel, stored k≥1 → k - 1 + id_base.
// Unquantized sections (int32) pass through unchanged.
function loadIds(buf: ArrayBuffer, meta: ReplayMeta, name: string): Int32Array | null {
  const s = meta.sections[name];
  if (!s) return null;
  if (s.dtype === "int32") return rawView(buf, s, Int32Array);
  if (s.dtype === "uint16") {
    const base = meta.id_base ?? 0;
    const src = rawView(buf, s, Uint16Array);
    const dst = new Int32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      dst[i] = src[i] === 0 ? -1 : src[i] - 1 + base;
    }
    return dst;
  }
  throw new Error(`unexpected dtype ${s.dtype} for id section ${name}`);
}

function optionalRaw<T>(
  buf: ArrayBuffer,
  meta: ReplayMeta,
  name: string,
  ctor: new (b: ArrayBuffer, o: number, n: number) => T,
): T | null {
  const s = meta.sections[name];
  if (!s) return null;
  return rawView(buf, s, ctor);
}

function decodeGenomes(buf: ArrayBuffer, meta: ReplayMeta): Map<number, MlpGenome> | null {
  if (meta.genome_arch !== "mlp") return null;
  const rowsSec = meta.sections["reward_genomes_byid"];
  const idsSec = meta.sections["reward_genomes_idmap"];
  const layout = meta.genome_layout;
  if (!rowsSec || !idsSec || !layout || layout.length === 0) return null;

  const rows = rawView(buf, rowsSec, Float32Array);
  const ids = rawView(buf, idsSec, Int32Array);
  const dim = meta.genome_dim ?? rowsSec.shape[1];
  const out = new Map<number, MlpGenome>();
  for (let r = 0; r < ids.length; r++) {
    const flat = rows.subarray(r * dim, (r + 1) * dim);
    out.set(ids[r], unflattenMlp(flat, layout));
  }
  return out;
}

// Re-nest a flat genome row into an ordered list of MlpLayer objects.
// Layer order tracks first-seen-in-layout order, which equals the
// recorder's `tree_flatten_with_path` order — input → first hidden → ...
// → output. Depth and per-layer widths are read from the layout itself,
// so any architecture (4→8→8→1, 40→16→16→1, future deeper variants) works
// with no code change.
//
// Each entry's `path` is `[<layer-name>, "kernel"|"bias"]` after the
// recorder strips Flax's "params" prefix.
function unflattenMlp(flat: Float32Array, layout: GenomeLayoutEntry[]): MlpGenome {
  // Group entries by layer name (path[0]). Use a Map to preserve insertion
  // order. Each layer collects its kernel + bias as they're encountered.
  const byLayer = new Map<string, { kernel?: number[][]; bias?: number[] }>();

  for (const entry of layout) {
    if (entry.path.length !== 2) {
      throw new Error(
        `unsupported genome layout path depth ${entry.path.length}: ` +
          `${entry.path.join("/")}`,
      );
    }
    const [layerName, leafName] = entry.path;
    if (!byLayer.has(layerName)) byLayer.set(layerName, {});
    const layer = byLayer.get(layerName)!;
    const size = entry.shape.reduce((a, b) => a * b, 1);
    const slice = flat.subarray(entry.offset, entry.offset + size);

    if (leafName === "kernel") {
      if (entry.shape.length !== 2) {
        throw new Error(`kernel must be 2-d, got shape ${entry.shape}`);
      }
      const [rows, cols] = entry.shape;
      const matrix: number[][] = [];
      for (let r = 0; r < rows; r++) {
        const row: number[] = new Array(cols);
        for (let c = 0; c < cols; c++) row[c] = slice[r * cols + c];
        matrix.push(row);
      }
      layer.kernel = matrix;
    } else if (leafName === "bias") {
      if (entry.shape.length !== 1) {
        throw new Error(`bias must be 1-d, got shape ${entry.shape}`);
      }
      layer.bias = Array.from(slice);
    } else {
      throw new Error(`unexpected genome leaf "${leafName}" in ${layerName}`);
    }
  }

  const layers: MlpLayer[] = [];
  for (const [layerName, parts] of byLayer) {
    if (!parts.kernel || !parts.bias) {
      throw new Error(`layer ${layerName} missing kernel or bias`);
    }
    layers.push({ kernel: parts.kernel, bias: parts.bias });
  }
  return { layers };
}

function decodeReplay(meta: ReplayMeta, buf: ArrayBuffer): ReplayData {
  return {
    meta,
    pos: loadFloat32(buf, meta, "pos"),
    angle: loadFloat32(buf, meta, "angle"),
    energy: loadFloat32(buf, meta, "energy"),
    alive: rawView(buf, sectionOrThrow(meta, "alive"), Uint8Array),
    foodPos: loadFloat32(buf, meta, "food_pos"),
    foodActive: rawView(buf, sectionOrThrow(meta, "food_active"), Uint8Array),
    stepNums: rawView(buf, sectionOrThrow(meta, "step_nums"), Int32Array),
    species: rawView(buf, sectionOrThrow(meta, "species"), Int32Array),
    radii: rawView(buf, sectionOrThrow(meta, "radii"), Float32Array),
    agentIds: loadIds(buf, meta, "agent_ids"),
    parentIds: loadIds(buf, meta, "parent_ids"),
    ages: optionalRaw(buf, meta, "ages", Int32Array),
    rewardWeights: meta.sections["reward_weights"]
      ? loadFloat32(buf, meta, "reward_weights")
      : null,
    action: meta.sections["action"] ? loadFloat32(buf, meta, "action") : null,
    genomesById: decodeGenomes(buf, meta),
  };
}

// -- per-frame views ---------------------------------------------------------

// Zero-copy per-frame slices. Returned typed arrays share memory with `data`,
// so do not mutate them.
export interface FrameView {
  pos: Float32Array;        // (max_agents, 2) flat
  angle: Float32Array;      // (max_agents,)
  energy: Float32Array;     // (max_agents,)
  alive: Uint8Array;        // (max_agents,)
  foodPos: Float32Array;    // (food_max, 2) flat
  foodActive: Uint8Array;   // (food_max,)
  step: number;
  // v2 — null on v1 replays
  agentIds: Int32Array | null;        // (max_agents,)
  parentIds: Int32Array | null;       // (max_agents,)
  ages: Int32Array | null;            // (max_agents,)
  rewardWeights: Float32Array | null; // (max_agents, 4) flat
  action: Float32Array | null;        // (max_agents, 2) flat
}

export function frameView(data: ReplayData, frameIdx: number): FrameView {
  const n = data.meta.max_agents;
  const f = data.meta.food_max;
  const i = Math.max(0, Math.min(frameIdx, data.meta.n_frames - 1));
  const slice = <T extends Int32Array | Float32Array>(
    arr: T | null,
    stride: number,
  ): T | null =>
    arr ? (arr.subarray(i * n * stride, (i + 1) * n * stride) as T) : null;
  return {
    pos: data.pos.subarray(i * n * 2, (i + 1) * n * 2),
    angle: data.angle.subarray(i * n, (i + 1) * n),
    energy: data.energy.subarray(i * n, (i + 1) * n),
    alive: data.alive.subarray(i * n, (i + 1) * n),
    foodPos: data.foodPos.subarray(i * f * 2, (i + 1) * f * 2),
    foodActive: data.foodActive.subarray(i * f, (i + 1) * f),
    step: data.stepNums[i],
    agentIds: slice(data.agentIds, 1),
    parentIds: slice(data.parentIds, 1),
    ages: slice(data.ages, 1),
    rewardWeights: slice(data.rewardWeights, 4),
    action: slice(data.action, 2),
  };
}
