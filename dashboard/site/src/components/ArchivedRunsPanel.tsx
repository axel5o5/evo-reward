import { useMemo, useState } from "react";
import {
  ArchiveRunSummary,
  ArchiveSummary,
  ReplayIndexEntry,
} from "../lib/replayLoader";
import { displayExperimentName, displayRunTag } from "../lib/replayNaming";

type SortKey = "exp" | "final_step" | "extinction_step" | "peak_prey" | "peak_pred";
type SortDir = "asc" | "desc";

interface Props {
  // Used to grey-out runs whose replays still exist in index.json (so the
  // panel reads as "what's been archived" vs "what's still live").
  liveReplays: ReplayIndexEntry[];
  // Hoisted from Replay.tsx so the page only fetches summary.json once.
  // Null while loading or when the bucket has no summary yet.
  summary: ArchiveSummary | null;
}

export default function ArchivedRunsPanel({ liveReplays, summary }: Props) {
  const [open, setOpen] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("final_step");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // Per-run live checkpoint counts. Used to classify each archived row as
  // live (every archived ckpt is still on disk), thinned (some pruned), or
  // pruned (none on disk). Just `Set.has` would call every row "live" after
  // a thin-only prune, which hides the actual state.
  const liveCounts = useMemo(() => {
    const m = new Map<string, number>();
    for (const r of liveReplays) {
      const key = `${r.exp}::${r.seed}::${r.run_tag || ""}`;
      m.set(key, (m.get(key) ?? 0) + 1);
    }
    return m;
  }, [liveReplays]);

  const rows = useMemo(() => {
    if (!summary) return [] as ArchiveRunSummary[];
    const sorted = [...summary.runs].sort((a, b) => cmp(a, b, sortKey));
    return sortDir === "asc" ? sorted : sorted.reverse();
  }, [summary, sortKey, sortDir]);

  // All hooks must run on every render — keep this above the early returns
  // below or React throws "Rendered more hooks than during previous render".
  const stateCounts = useMemo(() => {
    let live = 0, thinned = 0, pruned = 0;
    if (!summary) return { live, thinned, pruned };
    for (const r of summary.runs) {
      const key = `${r.exp}::${r.seed}::${r.run_tag || ""}`;
      const n = liveCounts.get(key) ?? 0;
      if (n === 0) pruned++;
      else if (n < r.n_checkpoints) thinned++;
      else live++;
    }
    return { live, thinned, pruned };
  }, [summary, liveCounts]);

  if (!summary) {
    // Quietly absent until the archive script has been uploaded — no need to
    // burden every page with a "no summary yet" notice.
    return null;
  }

  const total = summary.runs.length;
  const extinctCount = summary.runs.filter((r) => r.extinct).length;

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-2 px-4 py-3 text-left"
      >
        <div>
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Archived runs ({total})
          </div>
          <div className="text-[11px] text-gray-500 dark:text-gray-400">
            {stateCounts.live} live · {stateCounts.thinned} thinned ·{" "}
            {stateCounts.pruned} pruned · {extinctCount} went extinct
          </div>
        </div>
        <span className="text-gray-400 text-sm">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="border-t border-gray-200 dark:border-gray-800 overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 dark:bg-gray-900/60 text-gray-600 dark:text-gray-400">
              <tr>
                <Th onClick={() => toggleSort("exp", sortKey, sortDir, setSortKey, setSortDir)}
                    active={sortKey === "exp"} dir={sortDir}>exp / run_tag</Th>
                <Th>seed</Th>
                <Th onClick={() => toggleSort("final_step", sortKey, sortDir, setSortKey, setSortDir)}
                    active={sortKey === "final_step"} dir={sortDir} align="right">final step</Th>
                <Th align="center">ckpts</Th>
                <Th onClick={() => toggleSort("extinction_step", sortKey, sortDir, setSortKey, setSortDir)}
                    active={sortKey === "extinction_step"} dir={sortDir} align="right">extinct@</Th>
                <Th align="center">species</Th>
                <Th onClick={() => toggleSort("peak_prey", sortKey, sortDir, setSortKey, setSortDir)}
                    active={sortKey === "peak_prey"} dir={sortDir} align="right">peak prey</Th>
                <Th onClick={() => toggleSort("peak_pred", sortKey, sortDir, setSortKey, setSortDir)}
                    active={sortKey === "peak_pred"} dir={sortDir} align="right">peak pred</Th>
                <Th align="center">trajectory</Th>
                <Th align="center">status</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const key = `${r.exp}::${r.seed}::${r.run_tag || ""}`;
                const liveN = liveCounts.get(key) ?? 0;
                const status: "live" | "thinned" | "pruned" =
                  liveN === 0
                    ? "pruned"
                    : liveN < r.n_checkpoints
                      ? "thinned"
                      : "live";
                return (
                  <tr
                    key={key}
                    className={`border-t border-gray-100 dark:border-gray-800 ${
                      status === "pruned" ? "text-gray-500 dark:text-gray-400" : ""
                    }`}
                  >
                    <td className="px-3 py-2">
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {displayExperimentName(r.exp)}
                      </div>
                      <div className="text-[10px] font-mono opacity-70">
                        {displayRunTag(r.run_tag || undefined)}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-center">{r.seed}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {r.final_step !== null ? r.final_step.toLocaleString() : "—"}
                    </td>
                    <td className="px-3 py-2 text-center">{r.n_checkpoints}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {r.extinction_step !== null
                        ? r.extinction_step.toLocaleString()
                        : "—"}
                    </td>
                    <td className="px-3 py-2 text-center">
                      {r.extinct ? (
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            r.extinct_species === "pred"
                              ? "bg-red-100 dark:bg-red-950/40 text-red-700 dark:text-red-300"
                              : "bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300"
                          }`}
                        >
                          {r.extinct_species}
                        </span>
                      ) : (
                        <span className="text-[10px]">—</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">{r.peak_prey}</td>
                    <td className="px-3 py-2 text-right font-mono">{r.peak_pred}</td>
                    <td className="px-3 py-2">
                      <TrajectorySpark traj={r.trajectory} />
                    </td>
                    <td
                      className="px-3 py-2 text-center"
                      title={
                        status === "thinned"
                          ? `${liveN} of ${r.n_checkpoints} checkpoints still on disk`
                          : status === "live"
                            ? `all ${r.n_checkpoints} checkpoints on disk`
                            : `binaries pruned, summary preserved`
                      }
                    >
                      <span
                        className={`text-[10px] ${
                          status === "live"
                            ? "text-emerald-600 dark:text-emerald-400"
                            : status === "thinned"
                              ? "text-amber-600 dark:text-amber-400"
                              : "text-gray-400"
                        }`}
                      >
                        {status === "thinned"
                          ? `thinned ${liveN}/${r.n_checkpoints}`
                          : status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function cmp(a: ArchiveRunSummary, b: ArchiveRunSummary, key: SortKey): number {
  switch (key) {
    case "exp": {
      const ae = a.exp.localeCompare(b.exp);
      if (ae !== 0) return ae;
      const at = (a.run_tag || "").localeCompare(b.run_tag || "");
      if (at !== 0) return at;
      return a.seed - b.seed;
    }
    case "final_step":
      return (a.final_step ?? 0) - (b.final_step ?? 0);
    case "extinction_step":
      // Nulls (alive) sort below all extinction steps.
      return (a.extinction_step ?? Infinity) - (b.extinction_step ?? Infinity);
    case "peak_prey":
      return a.peak_prey - b.peak_prey;
    case "peak_pred":
      return a.peak_pred - b.peak_pred;
  }
}

function toggleSort(
  key: SortKey,
  current: SortKey,
  dir: SortDir,
  setKey: (k: SortKey) => void,
  setDir: (d: SortDir) => void,
) {
  if (key === current) {
    setDir(dir === "asc" ? "desc" : "asc");
  } else {
    setKey(key);
    setDir("desc");
  }
}

function Th({
  children,
  onClick,
  active,
  dir,
  align,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  active?: boolean;
  dir?: SortDir;
  align?: "right" | "center";
}) {
  const alignClass =
    align === "right" ? "text-right" : align === "center" ? "text-center" : "text-left";
  return (
    <th
      onClick={onClick}
      className={`px-3 py-2 font-medium uppercase tracking-wider text-[10px] ${alignClass} ${
        onClick ? "cursor-pointer select-none hover:text-gray-900 dark:hover:text-gray-200" : ""
      } ${active ? "text-gray-900 dark:text-gray-100" : ""}`}
    >
      {children}
      {active && <span className="ml-1">{dir === "asc" ? "▲" : "▼"}</span>}
    </th>
  );
}

const SPARK_W = 80;
const SPARK_H = 18;

function TrajectorySpark({ traj }: { traj: { step: number; prey: number; pred: number }[] }) {
  if (traj.length < 2) {
    return <span className="text-[10px] text-gray-400">—</span>;
  }
  const maxC = Math.max(1, ...traj.map((p) => p.prey), ...traj.map((p) => p.pred));
  const stepMin = traj[0].step;
  const stepMax = traj[traj.length - 1].step;
  const xRange = Math.max(1, stepMax - stepMin);

  const pts = (key: "prey" | "pred") =>
    traj
      .map((p) => {
        const x = ((p.step - stepMin) / xRange) * SPARK_W;
        const y = SPARK_H - (p[key] / maxC) * SPARK_H;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <svg
      width={SPARK_W}
      height={SPARK_H}
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      preserveAspectRatio="none"
      className="block rounded border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60"
    >
      <polyline
        points={pts("prey")}
        fill="none"
        stroke="#4ade80"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <polyline
        points={pts("pred")}
        fill="none"
        stroke="#f87171"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
