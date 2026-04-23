import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import ReplayCanvas from "../components/ReplayCanvas";
import PopulationStrip from "../components/PopulationStrip";
import {
  ReplayData,
  ReplayIndex,
  ReplayIndexEntry,
  fetchIndex,
  fetchReplay,
} from "../lib/replayLoader";
import { computeReplayStats } from "../lib/replayStats";

// Compact replay key: "tag:exp:seed:step". `tag` is empty for untagged
// replays (rendered as just ":exp:seed:step"). Kept deliberately human-
// readable so users can hand-edit the URL when experimenting with pairs.
type Key = { tag: string; exp: string; seed: number; step: number };

function parseKey(s: string | null): Key | null {
  if (!s) return null;
  const parts = s.split(":");
  if (parts.length !== 4) return null;
  const [tag, exp, seed, step] = parts;
  const sN = Number(seed);
  const stN = Number(step);
  if (!exp || !Number.isFinite(sN) || !Number.isFinite(stN)) return null;
  return { tag, exp, seed: sN, step: stN };
}

function matchEntry(replays: ReplayIndexEntry[], key: Key): ReplayIndexEntry | null {
  return (
    replays.find(
      (r) =>
        (r.run_tag || "") === key.tag &&
        r.exp === key.exp &&
        r.seed === key.seed &&
        r.start_step === key.step,
    ) ?? null
  );
}

function labelFor(e: ReplayIndexEntry): string {
  const tag = e.run_tag || "current";
  return `${tag} / ${e.exp} / seed ${e.seed} @ step ${e.start_step.toLocaleString()}`;
}

export default function ReplayCompare() {
  const [index, setIndex] = useState<ReplayIndex | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [dataA, setDataA] = useState<ReplayData | null>(null);
  const [dataB, setDataB] = useState<ReplayData | null>(null);
  const [entryA, setEntryA] = useState<ReplayIndexEntry | null>(null);
  const [entryB, setEntryB] = useState<ReplayIndexEntry | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetchIndex()
      .then(setIndex)
      .catch((e) => setIndexError(String(e)));
  }, []);

  // Load both replays from URL keys. Re-resolves on index load or URL change.
  useEffect(() => {
    if (!index) return;
    const q = new URLSearchParams(window.location.search);
    const keyA = parseKey(q.get("a"));
    const keyB = parseKey(q.get("b"));
    if (!keyA || !keyB) {
      setErr("Missing ?a= or ?b= query parameter. Format: tag:exp:seed:step");
      return;
    }
    const eA = matchEntry(index.replays, keyA);
    const eB = matchEntry(index.replays, keyB);
    if (!eA) {
      setErr(`No replay matches a=${q.get("a")}`);
      return;
    }
    if (!eB) {
      setErr(`No replay matches b=${q.get("b")}`);
      return;
    }
    setErr(null);
    setEntryA(eA);
    setEntryB(eB);
    setDataA(null);
    setDataB(null);
    setFrameIdx(0);
    setPlaying(false);
    Promise.all([fetchReplay(eA), fetchReplay(eB)])
      .then(([a, b]) => {
        setDataA(a);
        setDataB(b);
      })
      .catch((e) => setErr(String(e)));
  }, [index]);

  // Shared max frame — run the slider against the shorter of the two.
  const nFrames = useMemo(() => {
    if (!dataA || !dataB) return 0;
    return Math.min(dataA.meta.n_frames, dataB.meta.n_frames);
  }, [dataA, dataB]);

  const statsA = useMemo(() => (dataA ? computeReplayStats(dataA) : null), [dataA]);
  const statsB = useMemo(() => (dataB ? computeReplayStats(dataB) : null), [dataB]);

  const seedMismatch =
    entryA && entryB && entryA.seed !== entryB.seed;

  // --- playback loop (shared scrubber) ---
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);
  const accumRef = useRef<number>(0);
  useEffect(() => {
    if (!playing || nFrames === 0) return;
    const BASE_FPS = 60;
    const step = (now: number) => {
      if (lastTickRef.current === 0) lastTickRef.current = now;
      const dt = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      accumRef.current += dt * BASE_FPS;
      const adv = Math.floor(accumRef.current);
      if (adv > 0) {
        accumRef.current -= adv;
        setFrameIdx((i) => {
          const nxt = i + adv;
          if (nxt >= nFrames - 1) {
            setPlaying(false);
            return nFrames - 1;
          }
          return nxt;
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
  }, [playing, nFrames]);

  const togglePlay = useCallback(() => {
    if (nFrames === 0) return;
    if (frameIdx >= nFrames - 1) setFrameIdx(0);
    setPlaying((p) => !p);
  }, [nFrames, frameIdx]);

  return (
    <div className="max-w-7xl mx-auto py-8 px-6">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Replay — Compare
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Two replays side-by-side on a shared scrubber.{" "}
          <Link to="/replay" className="text-blue-600 dark:text-blue-400 underline">
            ← back to single replay
          </Link>
        </p>
        <p className="text-xs text-gray-500 mt-1">
          URL format:{" "}
          <code>/replay/compare?a=tag:exp:seed:step&amp;b=tag:exp:seed:step</code>
          . Leave <code>tag</code> empty for untagged runs (keep the colon).
        </p>
      </header>

      {indexError && (
        <div className="mb-4 p-3 rounded bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 text-sm">
          Could not load replay index: {indexError}
        </div>
      )}

      {err && (
        <div className="mb-4 p-3 rounded bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 text-sm">
          {err}
        </div>
      )}

      {seedMismatch && (
        <div className="mb-4 p-3 rounded bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 text-xs">
          Seeds differ ({entryA?.seed} vs {entryB?.seed}) — initial conditions
          are not matched, so differences here conflate policy and noise.
        </div>
      )}

      {dataA && dataB && entryA && entryB && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
            {[
              { data: dataA, stats: statsA, entry: entryA },
              { data: dataB, stats: statsB, entry: entryB },
            ].map((p, i) => (
              <div key={i} className="flex flex-col gap-2">
                <div className="text-sm font-mono text-gray-700 dark:text-gray-300">
                  {String.fromCharCode(65 + i)}. {labelFor(p.entry)}
                </div>
                <div className="aspect-square border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden bg-slate-50 dark:bg-slate-950">
                  <ReplayCanvas
                    data={p.data}
                    frameIdx={Math.min(frameIdx, p.data.meta.n_frames - 1)}
                    className="w-full h-full"
                  />
                </div>
                {p.stats && (
                  <PopulationStrip
                    data={p.data}
                    stats={p.stats}
                    frameIdx={Math.min(frameIdx, p.data.meta.n_frames - 1)}
                    onFrameChange={(f) => {
                      setPlaying(false);
                      setFrameIdx(f);
                    }}
                  />
                )}
              </div>
            ))}
          </div>

          <div className="flex flex-col gap-2 max-w-3xl mx-auto">
            <div className="flex items-center gap-2">
              <button
                onClick={togglePlay}
                className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium"
              >
                {playing ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => {
                  setPlaying(false);
                  setFrameIdx(0);
                }}
                className="px-2 py-1.5 rounded border border-gray-300 dark:border-gray-700 text-sm"
              >
                ⏮
              </button>
            </div>
            <input
              type="range"
              min={0}
              max={nFrames - 1}
              value={frameIdx}
              onChange={(e) => {
                setPlaying(false);
                setFrameIdx(parseInt(e.target.value, 10));
              }}
              className="w-full"
            />
            <div className="flex justify-between text-xs font-mono text-gray-500 dark:text-gray-400">
              <span>frame {frameIdx + 1} / {nFrames}</span>
              <span>
                sim A {dataA.stepNums[Math.min(frameIdx, dataA.meta.n_frames - 1)].toLocaleString()}{" "}
                · B {dataB.stepNums[Math.min(frameIdx, dataB.meta.n_frames - 1)].toLocaleString()}
              </span>
            </div>
          </div>
        </>
      )}

      {!dataA && !err && (
        <div className="p-6 border border-dashed border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400 text-sm">
          Loading replays…
        </div>
      )}
    </div>
  );
}
