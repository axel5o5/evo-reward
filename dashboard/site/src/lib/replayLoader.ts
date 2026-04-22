// Replay data loader.
//
// Source is controlled by VITE_REPLAYS_BASE_URL (falls back to "/replays/").
// Today that resolves to same-origin static files shipped with the Vercel
// build; swap it for a GCS public prefix or a serverless endpoint later
// without touching this file's consumers.
//
// Binary format evolves — meta.json.version gates any future breaking changes.
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
// When quantize=false (older replays), pos/food_pos/energy are float32 directly.
// Consumers always see Float32Array — dequantization happens here, eagerly.

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

type Dtype = "float32" | "uint8" | "int32" | "uint16";

interface SectionMeta {
  offset: number;
  length: number;
  dtype: Dtype;
  shape: number[];
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
  // Present on v1 quantized replays only.
  quantize?: boolean;
  scales?: Record<string, number>;
}

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
  throw new Error(`unexpected dtype ${s.dtype} for float section ${name}`);
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
}

export function frameView(data: ReplayData, frameIdx: number): FrameView {
  const n = data.meta.max_agents;
  const f = data.meta.food_max;
  const i = Math.max(0, Math.min(frameIdx, data.meta.n_frames - 1));
  return {
    pos: data.pos.subarray(i * n * 2, (i + 1) * n * 2),
    angle: data.angle.subarray(i * n, (i + 1) * n),
    energy: data.energy.subarray(i * n, (i + 1) * n),
    alive: data.alive.subarray(i * n, (i + 1) * n),
    foodPos: data.foodPos.subarray(i * f * 2, (i + 1) * f * 2),
    foodActive: data.foodActive.subarray(i * f, (i + 1) * f),
    step: data.stepNums[i],
  };
}
