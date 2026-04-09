import { useState } from "react";
import simData from "../data/sim-loop-steps.json";

const MODULE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  "agents.py": { bg: "bg-blue-50 dark:bg-blue-900/30", border: "border-blue-300 dark:border-blue-700", text: "text-blue-700 dark:text-blue-300" },
  "policy.py": { bg: "bg-purple-50 dark:bg-purple-900/30", border: "border-purple-300 dark:border-purple-700", text: "text-purple-700 dark:text-purple-300" },
  "environment.py": { bg: "bg-green-50 dark:bg-green-900/30", border: "border-green-300 dark:border-green-700", text: "text-green-700 dark:text-green-300" },
  "lifecycle.py": { bg: "bg-amber-50 dark:bg-amber-900/30", border: "border-amber-300 dark:border-amber-700", text: "text-amber-700 dark:text-amber-300" },
  "ppo.py": { bg: "bg-indigo-50 dark:bg-indigo-900/30", border: "border-indigo-300 dark:border-indigo-700", text: "text-indigo-700 dark:text-indigo-300" },
  "metrics.py": { bg: "bg-gray-50 dark:bg-gray-800", border: "border-gray-300 dark:border-gray-600", text: "text-gray-700 dark:text-gray-300" },
};

const STEP_DETAILS: Record<number, { signature: string; params: string[]; dataFlow: string }> = {
  1: { signature: "get_observation(world_state, agent) -> ndarray[205]", params: ["32 proximity sensors x 4 channels", "18 tactile bins x 4 types", "velocity (2D), angle, angular velocity, energy"], dataFlow: "WorldState + Agent position -> obs vector for each alive agent" },
  2: { signature: "sample_action(policy_params, obs) -> (action, log_prob, value)", params: ["MLP: obs(205) -> tanh(64) -> tanh(64) -> mean(2) + log_std(2)", "Action sampled from Normal(mean, exp(log_std))"], dataFlow: "obs -> policy network -> action + log_prob + value estimate -> rollout buffer" },
  3: { signature: "step_physics(world_state, actions) -> new_world_state", params: ["phyjax2d 2D rigid body simulation", "Actions mapped via sigmoid: [-20, 80]", "Collision detection between all entities"], dataFlow: "current positions + actions -> new positions, velocities, contacts" },
  4: { signature: "check_eating(world_state) -> eating_events", params: ["Prey-food overlap detection", "Predator-prey overlap + mouth cone check", "Mouth: 60 deg, range 40-80 units"], dataFlow: "positions + radii -> list of (eater_id, eaten_id) pairs" },
  5: { signature: "compute_reward(agent, stimuli) -> float", params: ["r = w_eat*n_eaten + 0.01*w_act*(||f||/F_max) + 0.1*w_prey*agg(s_prey) + 0.1*w_pred*agg(s_pred)", "Sensor aggregation: mean over 32 sensors"], dataFlow: "genome weights + current stimuli -> scalar reward -> rollout buffer" },
  6: { signature: "update_energies(agents, eating_events) -> updated_agents", params: ["Prey: +1.0 per food, -c_b per step, -c_a*||action|| per step", "Predator: +eta*prey_energy per kill, -d_b, -d_a*||action||", "Capped at energy_capacity (1000)"], dataFlow: "eating events + actions -> new energy values for all agents" },
  7: { signature: "process_births_and_deaths(agents, config) -> (alive, newborns)", params: ["Death: h(t,e) = kappa_h * energy_term * age_term", "Birth: b(e) = kappa_b / (1 + exp(zeta - beta_b*e))", "Offspring: 40% parent energy, mutated genome, random policy"], dataFlow: "energy + age -> death probability, birth probability -> population changes" },
  8: { signature: "regenerate_food(world_state) -> updated_food", params: ["Linear growth: +0.5 food per step", "Cap: 600 total, max 10 new per step", "Random placement in 960x960 world"], dataFlow: "current food count -> new food items placed randomly" },
  9: { signature: "ppo_update(rollout_buffer, policy_params) -> new_params", params: ["Triggered when buffer reaches 1024 steps", "10 epochs, minibatch size 256", "Clipped surrogate (eps=0.2), entropy bonus (0.001)"], dataFlow: "full rollout buffer -> gradient updates -> new policy parameters" },
  10: { signature: "log_step(step, agents, metrics_log) -> None", params: ["Log interval: every 10,000 steps", "Checkpoint interval: every 25,000 steps", "Metrics: population, weights, energy, rates"], dataFlow: "aggregate stats -> metrics.npz (append), checkpoint .pkl (snapshot)" },
};

export default function SimLoopDiagram() {
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  return (
    <div className="max-w-4xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Simulation Loop</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        The 10-step per-tick loop that drives the simulation. Each step executes once per
        timestep (10.24M total steps). Click a step to see function details.
      </p>

      {/* Module legend */}
      <div className="flex flex-wrap gap-3 mb-8">
        {Object.entries(MODULE_COLORS).map(([mod, colors]) => (
          <div key={mod} className="flex items-center gap-1.5 text-sm">
            <div className={`w-3 h-3 rounded border ${colors.border} ${colors.bg}`} />
            <span className={colors.text}>{mod}</span>
          </div>
        ))}
      </div>

      {/* Vertical stepper */}
      <div className="relative">
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-700" />

        {simData.steps.map((step, idx) => {
          const colors = MODULE_COLORS[step.module] || { bg: "bg-gray-50 dark:bg-gray-800", border: "border-gray-300 dark:border-gray-600", text: "text-gray-700 dark:text-gray-300" };
          const isExpanded = expandedStep === step.number;
          const details = STEP_DETAILS[step.number];

          return (
            <div key={step.number} className="relative mb-2">
              <div className={`absolute left-5 w-7 h-7 rounded-full border-2 ${colors.border} ${colors.bg}
                flex items-center justify-center text-xs font-bold ${colors.text} z-10`}>
                {step.number}
              </div>

              <div className="ml-16">
                <button
                  onClick={() => setExpandedStep(isExpanded ? null : step.number)}
                  className={`w-full text-left rounded-lg border p-3 transition
                    ${isExpanded
                      ? `${colors.bg} ${colors.border} border-2`
                      : "bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800"
                    }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-medium text-gray-800 dark:text-gray-200">{step.action}</span>
                      <span className={`ml-2 text-xs ${colors.text}`}>
                        [{step.module} → {step.function}]
                      </span>
                    </div>
                    <span className="text-gray-400 text-xs">{isExpanded ? "−" : "+"}</span>
                  </div>
                </button>

                {isExpanded && details && (
                  <div className={`mt-1 rounded-b-lg border border-t-0 ${colors.border} p-4 ${colors.bg}`}>
                    <div className="font-mono text-xs text-gray-700 dark:text-gray-300 bg-white/60 dark:bg-gray-800/60 rounded p-2 mb-3">
                      {details.signature}
                    </div>
                    <div className="mb-3">
                      <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">KEY PARAMETERS</div>
                      <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                        {details.params.map((p, i) => (
                          <li key={i} className="flex items-start">
                            <span className="text-gray-400 mr-2">•</span>{p}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">DATA FLOW</div>
                      <p className="text-sm text-gray-700 dark:text-gray-300">{details.dataFlow}</p>
                    </div>
                    <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
                      Module: <span className="underline">src/{step.module}</span>
                    </div>
                  </div>
                )}
              </div>

              {idx < simData.steps.length - 1 && (
                <div className="absolute left-[33px] bottom-[-4px] text-gray-300 dark:text-gray-600 text-xs">↓</div>
              )}
            </div>
          );
        })}

        <div className="ml-16 mt-4 rounded-lg border border-dashed border-gray-300 dark:border-gray-600 p-3 bg-gray-50 dark:bg-gray-900">
          <div className="text-sm text-gray-500 dark:text-gray-400 text-center">
            ↺ Repeat from step 1 — next timestep
            <span className="block text-xs text-gray-400 dark:text-gray-500 mt-1">
              10,240,000 total steps (~10-12 hours on A100)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
