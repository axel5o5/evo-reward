import { useState, useMemo } from "react";
import configData from "../data/config-schema.json";

type ConfigParam = (typeof configData.parameters)[number];

export default function ConfigReference() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSection, setSelectedSection] = useState<string | "all">("all");
  const [showDiscrepanciesOnly, setShowDiscrepanciesOnly] = useState(false);

  // Group by section
  const sections = useMemo(() => {
    const sectionSet = new Set<string>();
    configData.parameters.forEach((p) => {
      if (p.section) sectionSet.add(p.section);
    });
    return Array.from(sectionSet);
  }, []);

  // Detect discrepancies (params where inline_comment mentions "differs" or "code differs")
  const hasDiscrepancy = (p: ConfigParam) => {
    const text = (p.inline_comment + " " + p.comment).toLowerCase();
    return text.includes("differs") || text.includes("code differs") || text.includes("paper says");
  };

  // Filter parameters
  const filteredParams = useMemo(() => {
    let params = configData.parameters;

    if (selectedSection !== "all") {
      params = params.filter((p) => p.section === selectedSection);
    }

    if (showDiscrepanciesOnly) {
      params = params.filter(hasDiscrepancy);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      params = params.filter((p) =>
        p.key.toLowerCase().includes(q) ||
        p.section.toLowerCase().includes(q) ||
        (p.comment || "").toLowerCase().includes(q) ||
        (p.inline_comment || "").toLowerCase().includes(q) ||
        String(p.value).toLowerCase().includes(q)
      );
    }

    return params;
  }, [searchQuery, selectedSection, showDiscrepanciesOnly]);

  const discrepancyCount = configData.parameters.filter(hasDiscrepancy).length;

  return (
    <div className="max-w-6xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Config Reference</h1>
      <p className="text-gray-600 mb-6">
        All {configData.parameters.length} parameters from{" "}
        <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">{configData.source}</code>.
        {discrepancyCount > 0 && (
          <span className="text-amber-600 ml-2">
            {discrepancyCount} parameter(s) where code differs from paper.
          </span>
        )}
      </p>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6 items-center">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search parameters..."
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-64 focus:outline-none focus:border-blue-400"
        />

        <select
          value={selectedSection}
          onChange={(e) => setSelectedSection(e.target.value)}
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400"
        >
          <option value="all">All sections</option>
          {sections.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>

        <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
          <input
            type="checkbox"
            checked={showDiscrepanciesOnly}
            onChange={(e) => setShowDiscrepanciesOnly(e.target.checked)}
            className="rounded"
          />
          Show discrepancies only ({discrepancyCount})
        </label>

        <span className="text-sm text-gray-400">
          Showing {filteredParams.length} of {configData.parameters.length}
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 text-left text-gray-600 border-b">
              <th className="px-3 py-2.5 font-medium">Section</th>
              <th className="px-3 py-2.5 font-medium">Parameter</th>
              <th className="px-3 py-2.5 font-medium">Value</th>
              <th className="px-3 py-2.5 font-medium">Source / Notes</th>
            </tr>
          </thead>
          <tbody>
            {filteredParams.map((p, i) => {
              const isDiscrepancy = hasDiscrepancy(p);
              return (
                <tr
                  key={p.key}
                  className={`border-b border-gray-100 ${
                    isDiscrepancy ? "bg-amber-50" : i % 2 === 0 ? "bg-white" : "bg-gray-50/50"
                  }`}
                >
                  <td className="px-3 py-2 text-xs text-gray-400">{p.section}</td>
                  <td className="px-3 py-2">
                    <code className="text-xs font-medium text-gray-800">{p.key}</code>
                  </td>
                  <td className="px-3 py-2">
                    <span className="font-mono text-xs font-semibold text-blue-700">
                      {typeof p.value === "boolean" ? String(p.value) : String(p.value)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-500 max-w-md">
                    {isDiscrepancy && (
                      <span className="inline-block bg-amber-200 text-amber-800 text-[10px] px-1.5 py-0.5 rounded font-medium mr-1">
                        DIFFERS
                      </span>
                    )}
                    {p.inline_comment && <span>{p.inline_comment}</span>}
                    {p.comment && p.inline_comment && <br />}
                    {p.comment && <span className="text-gray-400">{p.comment}</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {filteredParams.length === 0 && (
        <div className="text-center py-8 text-gray-400">No parameters match your filters</div>
      )}
    </div>
  );
}
