import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import depsData from "../data/module-deps.json";

const LAYER_COLORS: Record<string, { fill: string; fillDark: string; stroke: string; label: string }> = {
  environment: { fill: "#dcfce7", fillDark: "#14532d", stroke: "#16a34a", label: "Environment" },
  lifecycle: { fill: "#dcfce7", fillDark: "#14532d", stroke: "#16a34a", label: "Environment" },
  agents: { fill: "#dbeafe", fillDark: "#1e3a5f", stroke: "#2563eb", label: "Logic" },
  reward: { fill: "#dbeafe", fillDark: "#1e3a5f", stroke: "#2563eb", label: "Logic" },
  policy: { fill: "#ede9fe", fillDark: "#2e1065", stroke: "#7c3aed", label: "Learning" },
  ppo: { fill: "#ede9fe", fillDark: "#2e1065", stroke: "#7c3aed", label: "Learning" },
  evolution: { fill: "#dbeafe", fillDark: "#1e3a5f", stroke: "#2563eb", label: "Logic" },
  metrics: { fill: "#f3f4f6", fillDark: "#1f2937", stroke: "#6b7280", label: "Infra" },
};

const KNOWN_EDGES = [
  { source: "agents", target: "environment" },
  { source: "agents", target: "reward" },
  { source: "lifecycle", target: "environment" },
  { source: "evolution", target: "lifecycle" },
  { source: "ppo", target: "policy" },
];

interface ModuleNode extends d3.SimulationNodeDatum {
  name: string;
  lines: number;
  file: string;
}

interface ModuleEdge {
  source: string | ModuleNode;
  target: string | ModuleNode;
}

const MODULE_API: Record<string, string[]> = {
  environment: ["create_world(config) -> WorldState", "step_physics(state, actions) -> state", "check_eating(state) -> events", "reset_world(config) -> WorldState"],
  lifecycle: ["update_energies(agents, events) -> agents", "process_births_and_deaths(agents, config) -> (alive, newborns)", "regenerate_food(state) -> state", "hazard_prob(age, energy, config) -> float", "birth_prob(energy, config) -> float"],
  agents: ["get_observation(state, agent) -> ndarray[205]", "extract_stimuli(obs) -> dict", "compute_reward(genome, stimuli, config) -> float"],
  reward: ["linear_reward(weights, stimuli, config) -> float", "extract_sensor_aggregates(obs, config) -> dict"],
  evolution: ["mutate_genome(parent_weights, config, rng) -> child_weights", "spawn_offspring(parent, config, rng) -> child"],
  policy: ["create_policy(config) -> params", "sample_action(params, obs) -> (action, logprob, value)", "evaluate_actions(params, obs, actions) -> (logprobs, values, entropy)"],
  ppo: ["ppo_update(buffer, params, config) -> new_params", "compute_gae(rewards, values, dones, config) -> advantages", "RolloutBuffer — stores transitions"],
  metrics: ["MetricsLog — accumulates step data", "log_step(step, agents, log) -> None", "save_checkpoint(step, agents, path) -> None", "save_metrics(log, path) -> None"],
};

export default function CodebaseMap() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<ModuleNode | null>(null);

  const nodes: ModuleNode[] = depsData.nodes.map((n) => ({ ...n }));
  const edges: ModuleEdge[] = (depsData.edges.length > 0 ? depsData.edges : KNOWN_EDGES).map((e) => ({ ...e }));

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const width = 700, height = 500;
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    const g = svg.append("g");

    svg.call(d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.5, 3]).on("zoom", (event) => g.attr("transform", event.transform)) as any);

    const isDark = document.documentElement.classList.contains("dark");

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id((d: any) => d.name).distance(120))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d: any) => Math.sqrt(d.lines) * 1.5 + 20));

    const link = g.append("g").selectAll("line").data(edges).join("line")
      .attr("stroke", isDark ? "#475569" : "#cbd5e1").attr("stroke-width", 1.5).attr("marker-end", "url(#arrow)");

    svg.append("defs").append("marker").attr("id", "arrow").attr("viewBox", "0 0 10 10")
      .attr("refX", 25).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
      .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", isDark ? "#64748b" : "#94a3b8");

    const node = g.append("g").selectAll("g").data(nodes).join("g")
      .style("cursor", "pointer").on("click", (_, d) => setSelectedNode(d))
      .call(d3.drag<SVGGElement, ModuleNode>()
        .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }) as any);

    node.append("circle")
      .attr("r", (d) => Math.sqrt(d.lines) * 1.2 + 10)
      .attr("fill", (d) => isDark ? (LAYER_COLORS[d.name]?.fillDark || "#1f2937") : (LAYER_COLORS[d.name]?.fill || "#f3f4f6"))
      .attr("stroke", (d) => LAYER_COLORS[d.name]?.stroke || "#6b7280").attr("stroke-width", 2);

    node.append("text").text((d) => d.name).attr("text-anchor", "middle").attr("dy", "0.35em")
      .attr("font-size", "11px").attr("font-weight", "600").attr("fill", (d) => LAYER_COLORS[d.name]?.stroke || "#374151");

    node.append("text").text((d) => `${d.lines}L`).attr("text-anchor", "middle").attr("dy", "1.5em")
      .attr("font-size", "9px").attr("fill", isDark ? "#6b7280" : "#9ca3af");

    simulation.on("tick", () => {
      link.attr("x1", (d: any) => d.source.x).attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x).attr("y2", (d: any) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, []);

  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Codebase Map</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-4">
        Force-directed graph of the 8 source modules. Nodes sized by line count, edges show import dependencies. Click a node to see its public API.
      </p>
      <div className="flex flex-wrap gap-4 mb-4">
        {[
          { label: "Environment", color: "#16a34a", bg: "#dcfce7", bgDark: "#14532d" },
          { label: "Logic", color: "#2563eb", bg: "#dbeafe", bgDark: "#1e3a5f" },
          { label: "Learning", color: "#7c3aed", bg: "#ede9fe", bgDark: "#2e1065" },
          { label: "Infra", color: "#6b7280", bg: "#f3f4f6", bgDark: "#1f2937" },
        ].map((l) => (
          <div key={l.label} className="flex items-center gap-1.5 text-sm">
            <div className="w-3 h-3 rounded-full border-2" style={{ borderColor: l.color }} />
            <span style={{ color: l.color }}>{l.label}</span>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <svg ref={svgRef} className="w-full border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900" style={{ height: 500 }} />
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          {selectedNode ? (
            <>
              <h3 className="font-semibold text-lg mb-1" style={{ color: LAYER_COLORS[selectedNode.name]?.stroke }}>
                {selectedNode.name}.py
              </h3>
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                {selectedNode.lines} lines — {LAYER_COLORS[selectedNode.name]?.label || "Infra"} layer
              </div>
              <div className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">Public API</div>
              <div className="space-y-1.5">
                {(MODULE_API[selectedNode.name] || []).map((fn) => (
                  <div key={fn} className="text-xs font-mono bg-white dark:bg-gray-800 rounded px-2 py-1.5 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300">
                    {fn}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-sm text-gray-400 dark:text-gray-500 mt-4">Click a module node to see its API</div>
          )}
        </div>
      </div>
    </div>
  );
}
