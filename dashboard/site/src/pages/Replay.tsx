import { useCallback, useEffect, useRef, useState } from "react";
import ReplayCanvas from "../components/ReplayCanvas";
import ReplaySelector from "../components/ReplaySelector";
import PopulationStrip from "../components/PopulationStrip";
import {
  ReplayData,
  ReplayIndex,
  ReplayIndexEntry,
  fetchIndex,
  fetchReplay,
  replaysBaseUrl,
} from "../lib/replayLoader";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

// URL <-> replay selection.
// We encode the selection as ?tag=&exp=&seed=&step=, and the playhead as &frame=.
// `tag` is omitted when the replay is untagged (treated as "current").
function urlParamsFor(selected: ReplayIndexEntry | null, frameIdx: number): URLSearchParams {
  const q = new URLSearchParams();
  if (!selected) return q;
  if (selected.run_tag) q.set("tag", selected.run_tag);
  q.set("exp", selected.exp);
  q.set("seed", String(selected.seed));
  q.set("step", String(selected.start_step));
  if (frameIdx > 0) q.set("frame", String(frameIdx));
  return q;
}

function matchFromUrl(
  replays: ReplayIndexEntry[],
  params: URLSearchParams,
): ReplayIndexEntry | null {
  const tag = params.get("tag") ?? "";
  const exp = params.get("exp");
  const seed = params.get("seed");
  const step = params.get("step");
  if (!exp || seed === null) return null;
  const seedN = Number(seed);
  const stepN = step !== null ? Number(step) : null;
  const matches = replays.filter(
    (r) => (r.run_tag || "") === tag && r.exp === exp && r.seed === seedN,
  );
  if (matches.length === 0) return null;
  if (stepN !== null) {
    const exact = matches.find((r) => r.start_step === stepN);
    if (exact) return exact;
  }
  // Fall back to earliest start_step for the (tag, exp, seed) tuple.
  return matches.sort((a, b) => a.start_step - b.start_step)[0];
}

export default function Replay() {
  const [index, setIndex] = useState<ReplayIndex | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ReplayIndexEntry | null>(null);
  const [data, setData] = useState<ReplayData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showHeading, setShowHeading] = useState(true);
  const [energyAlpha, setEnergyAlpha] = useState(true);

  // --- load index on mount; honor URL params if present ---
  // URL schema: ?tag=&exp=&seed=&step=&frame= — see urlParamsFor/matchFromUrl.
  // The initial ?frame= is applied once the replay finishes loading (below).
  const initialFrameFromUrl = useRef<number | null>(null);
  useEffect(() => {
    fetchIndex()
      .then((idx) => {
        setIndex(idx);
        if (idx.replays.length === 0) return;
        const params = new URLSearchParams(window.location.search);
        const fromUrl = matchFromUrl(idx.replays, params);
        const frameParam = params.get("frame");
        if (frameParam !== null) {
          const n = Number(frameParam);
          if (Number.isFinite(n) && n >= 0) initialFrameFromUrl.current = n;
        }
        setSelected(fromUrl ?? idx.replays[0]);
      })
      .catch((e) => setIndexError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- load selected replay ---
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    setData(null);
    setFrameIdx(0);
    setPlaying(false);
    fetchReplay(selected)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setLoading(false);
        // Apply URL-provided frame only once, on the first replay load.
        if (initialFrameFromUrl.current !== null) {
          const clamped = Math.max(
            0,
            Math.min(d.meta.n_frames - 1, initialFrameFromUrl.current),
          );
          setFrameIdx(clamped);
          initialFrameFromUrl.current = null;
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setLoadError(String(e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  // --- URL sync: replay identity (immediate) and frame (debounced) ---
  // We replaceState so Back doesn't accumulate per-frame entries; scrubbing
  // the slider would otherwise spam history.
  useEffect(() => {
    if (!selected) return;
    const q = urlParamsFor(selected, frameIdx);
    const search = q.toString();
    const target = search ? `?${search}` : window.location.pathname;
    if (window.location.search !== (search ? `?${search}` : "")) {
      window.history.replaceState(null, "", target + window.location.hash);
    }
  }, [selected]);

  useEffect(() => {
    if (!selected) return;
    const t = window.setTimeout(() => {
      const q = urlParamsFor(selected, frameIdx);
      const search = q.toString();
      const target = search ? `?${search}` : window.location.pathname;
      if (window.location.search !== (search ? `?${search}` : "")) {
        window.history.replaceState(null, "", target + window.location.hash);
      }
    }, 200);
    return () => window.clearTimeout(t);
  }, [frameIdx, selected]);

  // --- playback loop ---
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);
  const accumRef = useRef<number>(0);

  useEffect(() => {
    if (!playing || !data) return;

    const BASE_FPS = 60;
    const step = (now: number) => {
      if (lastTickRef.current === 0) lastTickRef.current = now;
      const dt = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      accumRef.current += dt * BASE_FPS * speed;
      const advance = Math.floor(accumRef.current);
      if (advance > 0) {
        accumRef.current -= advance;
        setFrameIdx((idx) => {
          const next = idx + advance;
          if (next >= data.meta.n_frames - 1) {
            setPlaying(false);
            return data.meta.n_frames - 1;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTickRef.current = 0;
      accumRef.current = 0;
    };
  }, [playing, speed, data]);

  const togglePlay = useCallback(() => {
    if (!data) return;
    if (frameIdx >= data.meta.n_frames - 1) setFrameIdx(0);
    setPlaying((p) => !p);
  }, [data, frameIdx]);

  return (
    <div className="max-w-6xl mx-auto py-8 px-6">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Replay</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Deterministic re-simulation of a saved checkpoint — watch the predator/prey
          dynamics unfold step by step. Source:{" "}
          <code className="text-xs">{replaysBaseUrl()}</code>
        </p>
      </header>

      {indexError && (
        <div className="mb-4 p-3 rounded bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 text-sm">
          Could not load replay index: {indexError}
          <div className="mt-1 text-xs opacity-80">
            Generate one with{" "}
            <code>python scripts/replay.py all --exp &lt;name&gt; --seed &lt;n&gt; --steps 1000</code>
          </div>
        </div>
      )}

      {index && index.replays.length === 0 && (
        <div className="p-6 border border-dashed border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400">
          No replays yet. Render one with{" "}
          <code>python scripts/replay.py all --exp baseline_faithful --seed 0 --steps 1000</code>.
        </div>
      )}

      {index && index.replays.length > 0 && (
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left: controls */}
          <aside className="lg:w-72 flex flex-col gap-4">
            <ReplaySelector
              replays={index.replays}
              selected={selected}
              onSelect={setSelected}
            />
            {selected?.run_tag && (
              <p className="text-xs text-amber-700 dark:text-amber-400">
                ⚠ Archived run: <code>{selected.run_tag}</code> — see
                <a
                  href="https://github.com/axel5o5/evo-reward/blob/main/docs/emevo-diff.md#d18"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline ml-1"
                >
                  emevo-diff.md D18
                </a>{" "}
                for context.
              </p>
            )}

            <div className="flex items-center gap-2">
              <button
                onClick={togglePlay}
                disabled={!data}
                className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium disabled:opacity-50"
              >
                {playing ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => {
                  setPlaying(false);
                  setFrameIdx(0);
                }}
                disabled={!data}
                className="px-2 py-1.5 rounded border border-gray-300 dark:border-gray-700 text-sm disabled:opacity-50"
              >
                ⏮
              </button>
              <button
                onClick={() => {
                  setPlaying(false);
                  setFrameIdx((i) => Math.max(0, i - 1));
                }}
                disabled={!data}
                className="px-2 py-1.5 rounded border border-gray-300 dark:border-gray-700 text-sm disabled:opacity-50"
              >
                −1
              </button>
              <button
                onClick={() => {
                  setPlaying(false);
                  setFrameIdx((i) =>
                    data ? Math.min(data.meta.n_frames - 1, i + 1) : i
                  );
                }}
                disabled={!data}
                className="px-2 py-1.5 rounded border border-gray-300 dark:border-gray-700 text-sm disabled:opacity-50"
              >
                +1
              </button>
            </div>

            <div>
              <label className="block text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
                Speed
              </label>
              <div className="flex gap-1">
                {SPEEDS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setSpeed(s)}
                    className={`px-2 py-1 rounded text-xs font-mono border transition ${
                      speed === s
                        ? "bg-blue-100 dark:bg-blue-900/40 border-blue-400 dark:border-blue-600 text-blue-800 dark:text-blue-200"
                        : "border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                    }`}
                  >
                    {s}×
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-2 text-sm text-gray-700 dark:text-gray-300">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showHeading}
                  onChange={(e) => setShowHeading(e.target.checked)}
                />
                Show heading
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={energyAlpha}
                  onChange={(e) => setEnergyAlpha(e.target.checked)}
                />
                Fade low-energy
              </label>
            </div>

            <div className="text-xs text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-800 pt-3 space-y-1">
              <div>
                Prey <span className="inline-block w-2 h-2 rounded-full bg-green-400 align-middle" />{" "}
                Predator{" "}
                <span className="inline-block w-2 h-2 rounded-full bg-red-400 align-middle" /> Food{" "}
                <span className="inline-block w-2 h-2 rounded-full bg-blue-400 align-middle" />
              </div>
              <div>
                Determinism: replay is bit-for-bit faithful for ≤ one rollout (1024 steps)
                from the saved state; policies then diverge from training since no PPO
                update fires.
              </div>
            </div>
          </aside>

          {/* Right: canvas + scrub */}
          <section className="flex-1 flex flex-col gap-3">
            <div className="aspect-square w-full max-w-[720px] mx-auto border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden bg-slate-50 dark:bg-slate-950">
              {loading && (
                <div className="w-full h-full flex items-center justify-center text-gray-500 dark:text-gray-400 text-sm">
                  Loading replay…
                </div>
              )}
              {loadError && (
                <div className="w-full h-full flex items-center justify-center text-red-600 dark:text-red-400 text-sm p-4 text-center">
                  {loadError}
                </div>
              )}
              {data && !loading && (
                <ReplayCanvas
                  data={data}
                  frameIdx={frameIdx}
                  showHeading={showHeading}
                  energyAlpha={energyAlpha}
                  className="w-full h-full"
                />
              )}
            </div>

            {data && (
              <div className="max-w-[720px] w-full mx-auto flex flex-col gap-2">
                <PopulationStrip
                  data={data}
                  frameIdx={frameIdx}
                  onFrameChange={(f) => {
                    setPlaying(false);
                    setFrameIdx(f);
                  }}
                />
                <input
                  type="range"
                  min={0}
                  max={data.meta.n_frames - 1}
                  value={frameIdx}
                  onChange={(e) => {
                    setPlaying(false);
                    setFrameIdx(parseInt(e.target.value, 10));
                  }}
                  className="w-full"
                />
                <div className="flex justify-between text-xs font-mono text-gray-500 dark:text-gray-400">
                  <span>frame {frameIdx + 1} / {data.meta.n_frames}</span>
                  <span>sim step {data.stepNums[frameIdx].toLocaleString()}</span>
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
