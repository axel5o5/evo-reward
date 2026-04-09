import { useState } from "react";
import obsData from "../data/obs-layout.json";

const SEGMENT_COLORS: Record<string, { bg: string; border: string; text: string; label: string }> = {
  proximity_sensors: { bg: "bg-blue-100 dark:bg-blue-900/40", border: "border-blue-400 dark:border-blue-600", text: "text-blue-800 dark:text-blue-300", label: "Proximity" },
  tactile_collision: { bg: "bg-green-100 dark:bg-green-900/40", border: "border-green-400 dark:border-green-600", text: "text-green-800 dark:text-green-300", label: "Tactile" },
  velocity: { bg: "bg-orange-100 dark:bg-orange-900/40", border: "border-orange-400 dark:border-orange-600", text: "text-orange-800 dark:text-orange-300", label: "Velocity" },
  angle: { bg: "bg-purple-100 dark:bg-purple-900/40", border: "border-purple-400 dark:border-purple-600", text: "text-purple-800 dark:text-purple-300", label: "Angle" },
  angular_velocity: { bg: "bg-purple-100 dark:bg-purple-900/40", border: "border-purple-300 dark:border-purple-600", text: "text-purple-700 dark:text-purple-300", label: "Ang. Vel." },
  energy: { bg: "bg-red-100 dark:bg-red-900/40", border: "border-red-400 dark:border-red-600", text: "text-red-800 dark:text-red-300", label: "Energy" },
};

type Segment = (typeof obsData.segments)[number];

export default function ObservationLayout() {
  const [selectedSegment, setSelectedSegment] = useState<Segment | null>(null);

  const totalDim = obsData.total_dim;

  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Observation Vector Layout</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Each agent receives a <strong>{totalDim}-dimensional</strong> observation vector every step.
        Click a segment to explore its internal structure.
      </p>

      {/* Stacked bar */}
      <div className="mb-8">
        <div className="flex rounded-lg overflow-hidden border border-gray-300 dark:border-gray-600 h-16">
          {obsData.segments.map((seg) => {
            const dims = seg.end - seg.start + 1;
            const widthPct = (dims / totalDim) * 100;
            const colors = SEGMENT_COLORS[seg.name] || { bg: "bg-gray-100 dark:bg-gray-800", border: "border-gray-300", text: "text-gray-700 dark:text-gray-300", label: seg.name };
            const isSelected = selectedSegment?.name === seg.name;

            return (
              <button
                key={seg.name}
                onClick={() => setSelectedSegment(isSelected ? null : seg)}
                className={`${colors.bg} ${colors.text} flex items-center justify-center text-xs font-medium
                  hover:brightness-95 dark:hover:brightness-125 transition cursor-pointer relative
                  ${isSelected ? `ring-2 ring-offset-1 dark:ring-offset-gray-950 ring-blue-500` : ""}`}
                style={{ width: `${widthPct}%`, minWidth: dims > 3 ? "40px" : "20px" }}
                title={`${seg.name}: indices ${seg.start}-${seg.end} (${dims} dims)`}
              >
                {dims > 10 && (
                  <div className="text-center leading-tight">
                    <div className="font-semibold">{colors.label}</div>
                    <div className="text-[10px] opacity-70">{dims}</div>
                  </div>
                )}
                {dims <= 10 && dims > 3 && <span className="text-[10px]">{dims}</span>}
              </button>
            );
          })}
        </div>
        <div className="flex justify-between text-xs text-gray-400 dark:text-gray-500 mt-1 px-1">
          <span>0</span>
          <span>128</span>
          <span>200</span>
          <span>{totalDim - 1}</span>
        </div>
      </div>

      {/* Segment legend */}
      <div className="flex flex-wrap gap-3 mb-8">
        {obsData.segments.map((seg) => {
          const dims = seg.end - seg.start + 1;
          const colors = SEGMENT_COLORS[seg.name] || { bg: "bg-gray-100", border: "border-gray-300", text: "text-gray-700", label: seg.name };
          return (
            <div key={seg.name} className={`flex items-center gap-2 text-sm ${colors.text}`}>
              <div className={`w-3 h-3 rounded ${colors.bg} border ${colors.border}`} />
              <span>{colors.label} ({dims})</span>
            </div>
          );
        })}
      </div>

      {/* Detail panel */}
      {selectedSegment && (
        <SegmentDetail segment={selectedSegment} onClose={() => setSelectedSegment(null)} />
      )}

      {/* Summary table */}
      {!selectedSegment && (
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
          <h2 className="font-semibold mb-3">Observation Vector Summary</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 dark:text-gray-400 border-b dark:border-gray-700">
                <th className="pb-2 pr-4">Segment</th>
                <th className="pb-2 pr-4">Indices</th>
                <th className="pb-2 pr-4">Shape</th>
                <th className="pb-2 pr-4">Range</th>
                <th className="pb-2">Description</th>
              </tr>
            </thead>
            <tbody>
              {obsData.segments.map((seg) => (
                <tr key={seg.name} className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2 pr-4 font-medium">{seg.name.replace(/_/g, " ")}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{seg.start}–{seg.end}</td>
                  <td className="py-2 pr-4 font-mono text-xs">[{seg.shape.join(", ")}]</td>
                  <td className="py-2 pr-4 font-mono text-xs">{seg.range}</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">{seg.description}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="font-semibold">
                <td className="pt-2">Total</td>
                <td className="pt-2 font-mono text-xs">0–{totalDim - 1}</td>
                <td className="pt-2 font-mono text-xs">[{totalDim}]</td>
                <td colSpan={2}></td>
              </tr>
            </tfoot>
          </table>
        </div>
      )}
    </div>
  );
}

function SegmentDetail({ segment, onClose }: { segment: Segment; onClose: () => void }) {
  const dims = segment.end - segment.start + 1;
  const colors = SEGMENT_COLORS[segment.name] || { bg: "bg-gray-100", border: "border-gray-300", text: "text-gray-700", label: segment.name };

  return (
    <div className={`rounded-lg p-6 border-2 ${colors.border} ${colors.bg} mb-6`}>
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className={`text-xl font-bold ${colors.text}`}>
            {segment.name.replace(/_/g, " ")}
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Indices {segment.start}–{segment.end} ({dims} dimensions) — Shape: [{segment.shape.join(", ")}]
          </p>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-xl">&times;</button>
      </div>

      <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">{segment.description}</p>
      {"details" in segment && segment.details && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">{segment.details}</p>
      )}

      {segment.name === "proximity_sensors" && <ProximityDetail />}
      {segment.name === "tactile_collision" && <TactileDetail />}
    </div>
  );
}

function ProximityDetail() {
  const channels = ["prey", "predator", "food", "wall"];
  const channelColors = ["text-blue-700 dark:text-blue-400", "text-red-700 dark:text-red-400", "text-green-700 dark:text-green-400", "text-gray-700 dark:text-gray-400"];
  const nChannels = 4;

  return (
    <div>
      <h3 className="font-semibold mb-2">32 Sensors x 4 Channels</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
        FOV: 120 degrees forward arc. Range: 200 world units. Winner-take-all per sensor:
        only the closest detected type gets a positive value; other channels = -1.0.
      </p>

      <div className="overflow-x-auto mb-4">
        <div className="inline-block">
          <div className="flex gap-0.5 mb-1 ml-12">
            {channels.map((ch, i) => (
              <div key={ch} className={`w-7 text-[9px] text-center font-medium ${channelColors[i]}`}>
                {ch.slice(0, 4)}
              </div>
            ))}
          </div>
          {Array.from({ length: 10 }).map((_, rowIdx) => {
            const sensorIdx = rowIdx < 8 ? rowIdx : 32 - (10 - rowIdx);
            const showEllipsis = rowIdx === 8;
            if (showEllipsis) {
              return (
                <div key="ellipsis" className="flex items-center gap-0.5 mb-0.5">
                  <div className="w-10 text-right text-[9px] text-gray-400 pr-1">...</div>
                  <div className="text-gray-400 text-xs">...</div>
                </div>
              );
            }
            return (
              <div key={sensorIdx} className="flex items-center gap-0.5 mb-0.5">
                <div className="w-10 text-right text-[9px] text-gray-500 dark:text-gray-400 pr-1 font-mono">
                  s{sensorIdx}
                </div>
                {channels.map((ch, chIdx) => {
                  const idx = sensorIdx * nChannels + chIdx;
                  return (
                    <div
                      key={ch}
                      className="w-7 h-5 rounded-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 text-[8px] text-center leading-5 text-gray-400"
                      title={`Index ${idx}: sensor ${sensorIdx}, channel ${ch}`}
                    >
                      {idx}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      <div className="text-xs bg-white/50 dark:bg-gray-800/50 rounded p-3 border border-blue-200 dark:border-blue-800">
        <strong>Winner-take-all:</strong> For each sensor, only the closest object type gets a positive
        value (inverse distance, 0 to 1). All other channels are set to -1.0. This means if a prey is
        closest to sensor 5, then s<sub>5</sub>[prey] &gt; 0 and s<sub>5</sub>[predator] = s<sub>5</sub>[food] = s<sub>5</sub>[wall] = -1.
        <br /><br />
        <strong>For reward computation:</strong> s<sub>prey</sub><sup>k</sup> = channel 0 (prey) of sensor k.
        s<sub>pred</sub><sup>k</sup> = channel 1 (predator). These are aggregated (mean over 32 sensors)
        and multiplied by the genome weights w<sub>prey</sub> and w<sub>pred</sub>.
      </div>
    </div>
  );
}

function TactileDetail() {
  const types = ["conspecific", "other_species", "food", "wall"];
  const typeColors = ["bg-blue-200 dark:bg-blue-800", "bg-red-200 dark:bg-red-800", "bg-green-200 dark:bg-green-800", "bg-gray-200 dark:bg-gray-700"];
  const nBins = 18;

  return (
    <div>
      <h3 className="font-semibold mb-2">4 Type Channels x 18 Bins</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
        Binary contact sensors arranged around the agent body at 20-degree intervals (360/18 = 20).
        Each bin reports whether contact of each type is occurring (0 or 1).
      </p>

      <div className="overflow-x-auto mb-4">
        <div className="inline-block">
          <div className="flex gap-0.5 mb-1 ml-20">
            {Array.from({ length: nBins }).map((_, i) => (
              <div key={i} className="w-5 text-[7px] text-center text-gray-400">{i * 20}°</div>
            ))}
          </div>
          {types.map((type, typeIdx) => (
            <div key={type} className="flex items-center gap-0.5 mb-0.5">
              <div className="text-right text-[9px] text-gray-500 dark:text-gray-400 pr-1 truncate" style={{ width: "75px" }}>
                {type}
              </div>
              {Array.from({ length: nBins }).map((_, binIdx) => {
                const idx = 128 + typeIdx * nBins + binIdx;
                return (
                  <div
                    key={binIdx}
                    className={`w-5 h-5 rounded-sm ${typeColors[typeIdx]} border border-gray-200 dark:border-gray-600 text-[7px] text-center leading-5 text-gray-500 dark:text-gray-400`}
                    title={`Index ${idx}: ${type}, bin ${binIdx} (${binIdx * 20}°)`}
                  >
                    {idx}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
