import { useState, useMemo } from "react";
import gitData from "../data/git-timeline.json";

const PHASE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  "0": { bg: "bg-blue-100 dark:bg-blue-900/40", border: "border-blue-400 dark:border-blue-600", text: "text-blue-700 dark:text-blue-300" },
  "1": { bg: "bg-green-100 dark:bg-green-900/40", border: "border-green-400 dark:border-green-600", text: "text-green-700 dark:text-green-300" },
  "2": { bg: "bg-purple-100 dark:bg-purple-900/40", border: "border-purple-400 dark:border-purple-600", text: "text-purple-700 dark:text-purple-300" },
  "3": { bg: "bg-amber-100 dark:bg-amber-900/40", border: "border-amber-400 dark:border-amber-600", text: "text-amber-700 dark:text-amber-300" },
};

export default function SessionTimeline() {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [filterPhase, setFilterPhase] = useState<string | "all">("all");

  const commits = useMemo(() => {
    let list = gitData.commits;
    if (filterPhase !== "all") list = list.filter((c) => c.phase === filterPhase);
    return list;
  }, [filterPhase]);

  const phases = useMemo(() => {
    const set = new Set<string>();
    gitData.commits.forEach((c) => { if (c.phase) set.add(c.phase); });
    return Array.from(set).sort();
  }, []);

  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Session Timeline</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Git commit history visualized as a project timeline. {gitData.commits.length} total commits. Phase-tagged commits are highlighted.
      </p>

      <div className="flex gap-2 mb-6 items-center">
        <span className="text-sm text-gray-500 dark:text-gray-400">Filter:</span>
        <button onClick={() => setFilterPhase("all")}
          className={`px-3 py-1 text-xs rounded-full ${filterPhase === "all" ? "bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-900" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"}`}>
          All ({gitData.commits.length})
        </button>
        {phases.map((p) => {
          const count = gitData.commits.filter((c) => c.phase === p).length;
          return (
            <button key={p} onClick={() => setFilterPhase(p === filterPhase ? "all" : p)}
              className={`px-3 py-1 text-xs rounded-full ${filterPhase === p ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"}`}>
              Phase {p} ({count})
            </button>
          );
        })}
      </div>

      {/* Horizontal timeline */}
      <div className="overflow-x-auto mb-8 pb-4">
        <div className="relative flex items-end gap-1 min-w-max" style={{ height: 180 }}>
          <div className="absolute bottom-8 left-0 right-0 h-0.5 bg-gray-300 dark:bg-gray-700" />
          {commits.map((commit, idx) => {
            const isPhase = !!commit.phase;
            const colors = commit.phase ? PHASE_COLORS[commit.phase] || { bg: "bg-gray-200 dark:bg-gray-700", border: "border-gray-400 dark:border-gray-600", text: "text-gray-700 dark:text-gray-300" }
              : { bg: "bg-gray-100 dark:bg-gray-700", border: "border-gray-300 dark:border-gray-600", text: "text-gray-500 dark:text-gray-400" };
            return (
              <div key={commit.hash} className="relative flex flex-col items-center" style={{ width: isPhase ? 20 : 12 }}
                onMouseEnter={() => setHoveredIdx(idx)} onMouseLeave={() => setHoveredIdx(null)}>
                <div className={`rounded-full border-2 ${colors.border} ${colors.bg} z-10 cursor-pointer ${isPhase ? "w-4 h-4" : "w-2.5 h-2.5"}`}
                  style={{ marginBottom: isPhase ? 14 : 17 }} />
                {(idx === 0 || idx === commits.length - 1 || (isPhase && idx % 2 === 0)) && (
                  <div className="text-[8px] text-gray-400 dark:text-gray-500 absolute bottom-0 whitespace-nowrap transform -rotate-45 origin-top-left" style={{ left: 0 }}>
                    {commit.date.split(" ")[0]}
                  </div>
                )}
                {hoveredIdx === idx && (
                  <div className="absolute bottom-20 left-1/2 -translate-x-1/2 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-3 z-20">
                    <div className="font-mono text-xs text-gray-400 dark:text-gray-500 mb-1">{commit.hash}</div>
                    <div className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-1">{commit.message}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">{commit.date}</div>
                    <div className="text-xs text-gray-400 dark:text-gray-500">{commit.author}</div>
                    {commit.phase && <div className={`mt-1 inline-block px-1.5 py-0.5 text-[10px] rounded ${colors.bg} ${colors.text}`}>Phase {commit.phase}</div>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Commit list */}
      <h2 className="font-semibold mb-3">Commit Log</h2>
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        {commits.map((commit, idx) => {
          const isPhase = !!commit.phase;
          const isExpanded = expandedIdx === idx;
          const hasBody = !!(commit as Record<string, unknown>).body;
          const colors = commit.phase ? PHASE_COLORS[commit.phase] || { bg: "bg-gray-50 dark:bg-gray-800", text: "text-gray-600 dark:text-gray-400" } : { bg: "", text: "" };
          return (
            <div key={commit.hash} className={`border-b border-gray-100 dark:border-gray-800 text-sm ${
              isPhase ? colors.bg : idx % 2 === 0 ? "bg-white dark:bg-gray-950" : "bg-gray-50/50 dark:bg-gray-900/50"
            }`}>
              <div
                className={`flex items-center px-3 py-2 ${hasBody ? "cursor-pointer hover:bg-gray-100/50 dark:hover:bg-gray-800/50" : ""}`}
                onClick={() => hasBody && setExpandedIdx(isExpanded ? null : idx)}
              >
                <div className="font-mono text-xs text-gray-400 dark:text-gray-500 w-20 shrink-0">{commit.hash}</div>
                <div className="flex-1 text-gray-800 dark:text-gray-200">
                  {isPhase && commit.phase && (
                    <span className={`inline-block px-1.5 py-0.5 text-[10px] rounded mr-2 ${colors.bg} ${colors.text} border ${PHASE_COLORS[commit.phase]?.border || "border-gray-200 dark:border-gray-600"}`}>
                      Phase {commit.phase}
                    </span>
                  )}
                  {commit.message}
                </div>
                <div className="flex items-center gap-2 shrink-0 ml-4">
                  <div className="text-xs text-gray-400 dark:text-gray-500">{commit.date.split(" ")[0]}</div>
                  {hasBody && (
                    <span className={`text-gray-400 dark:text-gray-500 text-xs transition-transform ${isExpanded ? "rotate-90" : ""}`}>▶</span>
                  )}
                </div>
              </div>
              {isExpanded && hasBody && (
                <div className="px-3 pb-3 pl-24">
                  <pre className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap font-mono leading-relaxed bg-gray-50 dark:bg-gray-800/50 rounded p-3">
                    {(commit as Record<string, unknown>).body as string}
                  </pre>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
