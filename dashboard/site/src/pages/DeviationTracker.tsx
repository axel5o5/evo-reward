import { useState } from "react";
import deviationsData from "../data/deviations.json";

const RISK_LEVELS: Record<string, { bg: string; text: string }> = {
  high: { bg: "bg-red-100 dark:bg-red-900/30", text: "text-red-700 dark:text-red-400" },
  medium: { bg: "bg-amber-100 dark:bg-amber-900/30", text: "text-amber-700 dark:text-amber-400" },
  low: { bg: "bg-green-100 dark:bg-green-900/30", text: "text-green-700 dark:text-green-400" },
};

function inferRisk(dev: typeof deviationsData.deviations[number]): string {
  const text = (dev.title + " " + dev.body_preview).toLowerCase();
  if (text.includes("critical") || text.includes("wrong") || text.includes("different")) return "high";
  if (text.includes("minor") || text.includes("cosmetic") || text.includes("resolved")) return "low";
  return "medium";
}

export default function DeviationTracker() {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [showResolved, setShowResolved] = useState(true);

  const deviations = deviationsData.deviations;
  const filtered = showResolved ? deviations : deviations.filter((d) => !d.resolved);
  const resolvedCount = deviations.filter((d) => d.resolved).length;
  const activeCount = deviations.length - resolvedCount;

  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Deviation Tracker</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Differences between our implementation and the emevo source / K&D paper.
        {deviations.length} deviations tracked ({resolvedCount} resolved, {activeCount} active).
      </p>

      <div className="flex gap-4 mb-6 items-center">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer">
          <input type="checkbox" checked={showResolved} onChange={(e) => setShowResolved(e.target.checked)} className="rounded" />
          Show resolved ({resolvedCount})
        </label>
      </div>

      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 dark:bg-gray-800 text-left text-gray-600 dark:text-gray-400 border-b dark:border-gray-700">
              <th className="px-3 py-2.5 font-medium w-16">ID</th>
              <th className="px-3 py-2.5 font-medium">Title</th>
              <th className="px-3 py-2.5 font-medium w-24">Status</th>
              <th className="px-3 py-2.5 font-medium w-20">Risk</th>
              <th className="px-3 py-2.5 font-medium w-12"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((dev) => {
              const isExpanded = expandedId === dev.id;
              const risk = inferRisk(dev);
              const riskColors = RISK_LEVELS[risk] || RISK_LEVELS.medium;
              return (
                <tr key={dev.id} className="group">
                  <td colSpan={5} className="p-0">
                    <div>
                      <button onClick={() => setExpandedId(isExpanded ? null : dev.id)}
                        className={`w-full text-left flex items-center border-b border-gray-100 dark:border-gray-800 px-3 py-2.5
                          hover:bg-gray-50 dark:hover:bg-gray-800 transition ${dev.resolved ? "opacity-60" : ""}`}>
                        <span className="w-16 shrink-0 font-mono text-xs text-gray-500 dark:text-gray-400">{dev.id}</span>
                        <span className={`flex-1 ${dev.resolved ? "line-through text-gray-400 dark:text-gray-500" : "text-gray-800 dark:text-gray-200"}`}>{dev.title}</span>
                        <span className={`w-24 shrink-0 text-xs px-2 py-0.5 rounded ${dev.resolved ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400" : "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400"}`}>
                          {dev.resolved ? "Resolved" : "Active"}
                        </span>
                        <span className={`w-20 shrink-0 text-xs px-2 py-0.5 rounded ml-2 ${riskColors.bg} ${riskColors.text}`}>{risk}</span>
                        <span className="w-8 shrink-0 text-gray-400 dark:text-gray-500 text-xs text-right">{isExpanded ? "−" : "+"}</span>
                      </button>
                      {isExpanded && (
                        <div className="px-3 py-3 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
                          <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">{dev.body_preview}</div>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {filtered.length === 0 && <div className="text-center py-8 text-gray-400 dark:text-gray-500">No deviations match your filters</div>}
    </div>
  );
}
