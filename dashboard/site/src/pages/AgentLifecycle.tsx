import { useState, useMemo } from "react";

export default function AgentLifecycle() {
  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Agent Lifecycle</h1>
      <p className="text-gray-600 mb-8">
        Each agent follows a continuous birth-death lifecycle. Reproduction and death
        are probabilistic, driven by energy — no explicit fitness function.
      </p>

      <LifecycleFlowchart />
      <div className="grid grid-cols-2 gap-8 mt-10">
        <HazardCalculator />
        <BirthCalculator />
      </div>
      <SpeciesParameters />
    </div>
  );
}

function LifecycleFlowchart() {
  const steps = [
    { label: "Birth", desc: "Inherit mutated genome from parent. Random policy init. Get 40% of parent's energy.", color: "bg-green-100 border-green-400 text-green-800" },
    { label: "Observe → Act → Physics → Eat → Reward → Energy", desc: "The inner loop: get obs, run policy, step physics, compute reward, update energy. Repeats every timestep.", color: "bg-blue-100 border-blue-400 text-blue-800", loop: true },
    { label: "PPO Update", desc: "Every 1024 steps, run 10 epochs of clipped PPO on the agent's rollout buffer. Updates the policy to maximize the agent's personal reward signal.", color: "bg-purple-100 border-purple-400 text-purple-800", loop: true },
    { label: "Death Check", desc: "Each step: sample from hazard function h(t,e). Probability increases with age and decreases with energy. Also die if energy < 0.", color: "bg-red-100 border-red-400 text-red-800" },
    { label: "Birth Check", desc: "Each step: sample from birth function b(e). Probability increases with energy above threshold. If triggered, spawn offspring.", color: "bg-amber-100 border-amber-400 text-amber-800" },
  ];

  return (
    <div className="flex flex-col items-center">
      {steps.map((step, i) => (
        <div key={step.label} className="flex flex-col items-center">
          <div className={`rounded-lg border-2 px-6 py-3 max-w-md text-center ${step.color}`}>
            <div className="font-semibold">{step.label}</div>
            <div className="text-xs mt-1 opacity-80">{step.desc}</div>
          </div>
          {i < steps.length - 1 && (
            <div className="flex flex-col items-center my-1">
              <div className="w-0.5 h-4 bg-gray-300" />
              {step.loop && (
                <div className="text-[10px] text-gray-400">↺ repeat</div>
              )}
              <div className="text-gray-400">↓</div>
            </div>
          )}
        </div>
      ))}

      {/* Branch for death vs reproduce */}
      <div className="flex gap-16 mt-3">
        <div className="flex flex-col items-center">
          <div className="text-gray-400 mb-1">↙ death</div>
          <div className="rounded-lg border-2 px-4 py-2 bg-red-50 border-red-300 text-red-700 text-sm text-center">
            Removed from world
          </div>
        </div>
        <div className="flex flex-col items-center">
          <div className="text-gray-400 mb-1">↘ reproduce</div>
          <div className="rounded-lg border-2 px-4 py-2 bg-green-50 border-green-300 text-green-700 text-sm text-center">
            <div>Child: mutated genome</div>
            <div className="text-xs opacity-70">+ fresh random policy</div>
          </div>
          <div className="text-gray-400 text-xs mt-1">↺ child starts lifecycle</div>
        </div>
      </div>
    </div>
  );
}

function HazardCalculator() {
  const [age, setAge] = useState(5000);
  const [energy, setEnergy] = useState(50);
  const [species, setSpecies] = useState<"prey" | "predator">("prey");

  const params = species === "prey"
    ? { alpha_t: 4e-7, beta_t: 2e-6 }
    : { alpha_t: 2e-7, beta_t: 4e-6 };

  const kappa_h = 0.01;
  const alpha_e = 0.02;
  const beta_h = 0.2;

  const hazard = useMemo(() => {
    const energy_term = 1 - 1 / (1 + alpha_e * Math.exp(-beta_h * energy));
    const age_term = params.alpha_t * Math.exp(params.beta_t * age);
    return kappa_h * energy_term * age_term;
  }, [age, energy, species]);

  return (
    <div className="bg-red-50 rounded-lg p-5 border border-red-200">
      <h2 className="font-semibold text-red-800 mb-3">Hazard Function h(t, e)</h2>
      <div className="font-mono text-xs text-red-700 bg-white/60 rounded p-2 mb-3">
        h(t,e) = κ_h · (1 - 1/(1 + α_e·exp(-β_h·e))) · α_t·exp(β_t·t)
      </div>

      <div className="space-y-3 mb-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>Species</span>
            <div className="flex gap-2">
              <button className={`px-2 py-0.5 text-xs rounded ${species === "prey" ? "bg-blue-500 text-white" : "bg-gray-100"}`}
                onClick={() => setSpecies("prey")}>Prey</button>
              <button className={`px-2 py-0.5 text-xs rounded ${species === "predator" ? "bg-red-500 text-white" : "bg-gray-100"}`}
                onClick={() => setSpecies("predator")}>Predator</button>
            </div>
          </div>
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>Age (steps)</span>
            <span className="font-mono text-xs">{age.toLocaleString()}</span>
          </div>
          <input type="range" min={0} max={50000} step={100} value={age}
            onChange={(e) => setAge(parseInt(e.target.value))}
            className="w-full" style={{ accentColor: "#dc2626" }} />
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>Energy</span>
            <span className="font-mono text-xs">{energy}</span>
          </div>
          <input type="range" min={0} max={500} step={1} value={energy}
            onChange={(e) => setEnergy(parseInt(e.target.value))}
            className="w-full" style={{ accentColor: "#dc2626" }} />
        </div>
      </div>

      <div className="text-center">
        <div className="text-xs text-red-600 uppercase tracking-wide">Death probability per step</div>
        <div className="text-3xl font-bold text-red-700">{hazard.toExponential(3)}</div>
        <div className="text-xs text-gray-500 mt-1">
          Expected lifetime: ~{hazard > 0 ? Math.round(1 / hazard).toLocaleString() : "∞"} steps
        </div>
      </div>
    </div>
  );
}

function BirthCalculator() {
  const [energy, setEnergy] = useState(80);
  const [species, setSpecies] = useState<"prey" | "predator">("prey");

  const zeta = species === "prey" ? 10.0 : 100.0;
  const kappa_b = 1e-3;
  const beta_b = 0.1;

  const birthProb = useMemo(() => {
    return kappa_b / (1 + Math.exp(zeta - beta_b * energy));
  }, [energy, species]);

  return (
    <div className="bg-green-50 rounded-lg p-5 border border-green-200">
      <h2 className="font-semibold text-green-800 mb-3">Birth Function b(e)</h2>
      <div className="font-mono text-xs text-green-700 bg-white/60 rounded p-2 mb-3">
        b(e) = κ_b / (1 + exp(ζ - β_b·e))
      </div>

      <div className="space-y-3 mb-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>Species</span>
            <div className="flex gap-2">
              <button className={`px-2 py-0.5 text-xs rounded ${species === "prey" ? "bg-blue-500 text-white" : "bg-gray-100"}`}
                onClick={() => setSpecies("prey")}>Prey (ζ=10)</button>
              <button className={`px-2 py-0.5 text-xs rounded ${species === "predator" ? "bg-red-500 text-white" : "bg-gray-100"}`}
                onClick={() => setSpecies("predator")}>Predator (ζ=100)</button>
            </div>
          </div>
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>Energy</span>
            <span className="font-mono text-xs">{energy}</span>
          </div>
          <input type="range" min={0} max={500} step={1} value={energy}
            onChange={(e) => setEnergy(parseInt(e.target.value))}
            className="w-full" style={{ accentColor: "#16a34a" }} />
        </div>
      </div>

      <div className="text-center">
        <div className="text-xs text-green-600 uppercase tracking-wide">Birth probability per step</div>
        <div className="text-3xl font-bold text-green-700">{birthProb.toExponential(3)}</div>
        <div className="text-xs text-gray-500 mt-1">
          {species === "prey"
            ? "Prey need ~20-30 energy for meaningful birth probability"
            : "Predators need ~240-260 energy (much higher threshold)"}
        </div>
      </div>

      <div className="mt-3 text-xs text-gray-500">
        On birth: parent loses 40% energy, child receives 40% of parent's energy.
        Child genome = parent genome + t(df=2, scale=0.4) noise. Child policy: random init.
      </div>
    </div>
  );
}

function SpeciesParameters() {
  return (
    <div className="mt-8 bg-gray-50 rounded-lg p-5 border border-gray-200">
      <h2 className="font-semibold mb-3">Species-Specific Parameters</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 border-b">
            <th className="pb-2 pr-4">Parameter</th>
            <th className="pb-2 pr-4">Prey</th>
            <th className="pb-2 pr-4">Predator</th>
            <th className="pb-2">Notes</th>
          </tr>
        </thead>
        <tbody className="text-gray-700">
          <tr className="border-b border-gray-100">
            <td className="py-2 pr-4 font-mono text-xs">ζ (birth threshold)</td>
            <td className="py-2 pr-4">10</td>
            <td className="py-2 pr-4">100</td>
            <td className="py-2 text-xs text-gray-500">Predators need much more energy to reproduce</td>
          </tr>
          <tr className="border-b border-gray-100">
            <td className="py-2 pr-4 font-mono text-xs">α_t (age hazard)</td>
            <td className="py-2 pr-4">4e-7</td>
            <td className="py-2 pr-4">2e-7</td>
            <td className="py-2 text-xs text-gray-500">Prey age faster (shorter natural lifespan)</td>
          </tr>
          <tr className="border-b border-gray-100">
            <td className="py-2 pr-4 font-mono text-xs">β_t (age scaling)</td>
            <td className="py-2 pr-4">2e-6</td>
            <td className="py-2 pr-4">4e-6</td>
            <td className="py-2 text-xs text-gray-500">Predator aging accelerates faster</td>
          </tr>
          <tr className="border-b border-gray-100">
            <td className="py-2 pr-4 font-mono text-xs">c_b / d_b (basal cost)</td>
            <td className="py-2 pr-4">1e-4</td>
            <td className="py-2 pr-4">4e-3</td>
            <td className="py-2 text-xs text-gray-500">Predators 40x more expensive to maintain</td>
          </tr>
          <tr className="border-b border-gray-100">
            <td className="py-2 pr-4 font-mono text-xs">radius</td>
            <td className="py-2 pr-4">10</td>
            <td className="py-2 pr-4">14</td>
            <td className="py-2 text-xs text-gray-500">Predators are physically larger</td>
          </tr>
          <tr>
            <td className="py-2 pr-4 font-mono text-xs">initial population</td>
            <td className="py-2 pr-4">150</td>
            <td className="py-2 pr-4">10</td>
            <td className="py-2 text-xs text-gray-500">15:1 prey-predator ratio</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
