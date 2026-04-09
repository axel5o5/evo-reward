import { useState, useMemo } from "react";

const WEIGHT_RANGE = { min: -10, max: 10, step: 0.1 };

export default function RewardEquation() {
  // Genome weights
  const [wEat, setWEat] = useState(2.0);
  const [wAct, setWAct] = useState(-0.5);
  const [wPrey, setWPrey] = useState(1.5);
  const [wPred, setWPred] = useState(-3.0);

  // Stimulus inputs
  const [nEaten, setNEaten] = useState(0);
  const [motorNorm, setMotorNorm] = useState(0.3);
  const [preyProximity, setPreyProximity] = useState(0.2);
  const [predProximity, setPredProximity] = useState(0.6);

  const F_MAX = Math.sqrt(80 * 80 + 80 * 80); // ~113.14

  const reward = useMemo(() => {
    const term_eat = wEat * nEaten;
    const term_act = 0.01 * wAct * motorNorm;
    const term_prey = 0.1 * wPrey * preyProximity;
    const term_pred = 0.1 * wPred * predProximity;
    return { total: term_eat + term_act + term_prey + term_pred, term_eat, term_act, term_prey, term_pred };
  }, [wEat, wAct, wPrey, wPred, nEaten, motorNorm, preyProximity, predProximity]);

  const rewardColor = reward.total >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400";

  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Reward Equation</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Interactive explorer for the K&D reward function. Drag sliders to see how
        genome weights and stimulus inputs combine to produce reward.
      </p>

      {/* Equation display */}
      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-5 mb-8 font-mono text-sm leading-relaxed border border-gray-200 dark:border-gray-700">
        <div className="text-center text-base mb-3 font-semibold text-gray-700 dark:text-gray-300">
          r = w<sub>eat</sub> · n<sub>eaten</sub> + 0.01 · w<sub>act</sub> · (‖f‖ / F<sub>max</sub>) + 0.1 · w<sub>prey</sub> · agg(s<sub>prey</sub>) + 0.1 · w<sub>pred</sub> · agg(s<sub>pred</sub>)
        </div>
        <div className="grid grid-cols-4 gap-3 text-xs text-gray-500 dark:text-gray-400">
          <div className="text-center">
            <span className="font-semibold text-gray-700 dark:text-gray-300">{reward.term_eat.toFixed(4)}</span>
            <br />food term
          </div>
          <div className="text-center">
            <span className="font-semibold text-gray-700 dark:text-gray-300">{reward.term_act.toFixed(4)}</span>
            <br />motor term
          </div>
          <div className="text-center">
            <span className="font-semibold text-gray-700 dark:text-gray-300">{reward.term_prey.toFixed(4)}</span>
            <br />prey proximity
          </div>
          <div className="text-center">
            <span className="font-semibold text-gray-700 dark:text-gray-300">{reward.term_pred.toFixed(4)}</span>
            <br />predator proximity
          </div>
        </div>
      </div>

      {/* Reward output */}
      <div className="text-center mb-8">
        <div className="text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1">Computed Reward</div>
        <div className={`text-5xl font-bold ${rewardColor}`}>
          {reward.total >= 0 ? "+" : ""}{reward.total.toFixed(4)}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-8">
        {/* Genome weights */}
        <div>
          <h2 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">Genome Weights (evolved)</h2>
          <WeightSlider label="w_eat" sublabel="food reward" value={wEat} onChange={setWEat} />
          <WeightSlider label="w_act" sublabel="motor cost" value={wAct} onChange={setWAct} />
          <WeightSlider label="w_prey" sublabel="conspecific" value={wPrey} onChange={setWPrey} />
          <WeightSlider label="w_pred" sublabel="predator (fear)" value={wPred} onChange={setWPred} />
        </div>

        {/* Stimulus inputs */}
        <div>
          <h2 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">Stimulus Inputs (from environment)</h2>

          <div className="mb-5">
            <div className="flex justify-between text-sm mb-1">
              <span className="font-medium">n_eaten</span>
              <span className="text-gray-500 dark:text-gray-400">{nEaten}</span>
            </div>
            <div className="flex gap-3">
              <button
                className={`px-4 py-1.5 rounded text-sm font-medium transition ${nEaten === 0 ? "bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-900" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"}`}
                onClick={() => setNEaten(0)}
              >0 (no food)</button>
              <button
                className={`px-4 py-1.5 rounded text-sm font-medium transition ${nEaten === 1 ? "bg-green-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"}`}
                onClick={() => setNEaten(1)}
              >1 (ate food)</button>
            </div>
          </div>

          <StimulusSlider label="‖f‖ / F_max" sublabel="motor output norm" value={motorNorm}
            onChange={setMotorNorm} min={0} max={1} />
          <StimulusSlider label="agg(s_prey)" sublabel="prey proximity" value={preyProximity}
            onChange={setPreyProximity} min={0} max={1} />
          <StimulusSlider label="agg(s_pred)" sublabel="predator proximity" value={predProximity}
            onChange={setPredProximity} min={0} max={1} />
        </div>
      </div>

      {/* Key insight callout */}
      <div className={`mt-8 p-5 rounded-lg border-2 ${
        wPred < 0 && predProximity > 0.3
          ? "border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-950/30"
          : "border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900"
      }`}>
        <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">Key Insight: The Emergence of Fear</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
          When <strong>w<sub>pred</sub> is negative</strong> (currently {wPred.toFixed(1)}) and a predator
          is nearby (proximity = {predProximity.toFixed(2)}), the predator term contributes{" "}
          <span className={reward.term_pred < 0 ? "text-red-600 dark:text-red-400 font-semibold" : "text-gray-700 dark:text-gray-300"}>
            {reward.term_pred.toFixed(4)}
          </span>{" "}
          to the reward. This negative reward signal teaches the policy to <em>avoid</em> predators —
          fear emerges from evolution without being hand-designed.
        </p>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
          The 0.01 and 0.1 coefficients are fixed architecture, not part of the genome.
          F<sub>max</sub> = √(80² + 80²) ≈ {F_MAX.toFixed(2)}.
          Genome order: [w_eat, w_act, w_prey, w_pred] — canonical everywhere.
        </p>
      </div>

      {/* Preset scenarios */}
      <div className="mt-8">
        <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">Preset Scenarios</h3>
        <div className="flex gap-3 flex-wrap">
          <PresetButton label="Fearful prey" onClick={() => {
            setWEat(2.0); setWAct(-0.5); setWPrey(1.5); setWPred(-3.0);
            setPredProximity(0.8); setPreyProximity(0.2); setNEaten(0); setMotorNorm(0.7);
          }} />
          <PresetButton label="Hungry prey (no fear)" onClick={() => {
            setWEat(4.0); setWAct(-0.2); setWPrey(0.1); setWPred(0.0);
            setPredProximity(0.5); setPreyProximity(0.1); setNEaten(1); setMotorNorm(0.3);
          }} />
          <PresetButton label="Social prey" onClick={() => {
            setWEat(1.0); setWAct(-0.1); setWPrey(5.0); setWPred(-1.0);
            setPredProximity(0.1); setPreyProximity(0.9); setNEaten(0); setMotorNorm(0.2);
          }} />
          <PresetButton label="Aggressive predator" onClick={() => {
            setWEat(5.0); setWAct(-1.0); setWPrey(3.0); setWPred(0.5);
            setPredProximity(0.0); setPreyProximity(0.8); setNEaten(1); setMotorNorm(0.9);
          }} />
          <PresetButton label="Initial (random)" onClick={() => {
            setWEat(0.0); setWAct(0.0); setWPrey(0.0); setWPred(0.0);
            setPredProximity(0.0); setPreyProximity(0.0); setNEaten(0); setMotorNorm(0.0);
          }} />
        </div>
      </div>
    </div>
  );
}

function WeightSlider({ label, sublabel, value, onChange }: {
  label: string; sublabel: string; value: number;
  onChange: (v: number) => void; color?: string;
}) {
  return (
    <div className="mb-5">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium">{label} <span className="text-gray-400 dark:text-gray-500 text-xs">({sublabel})</span></span>
        <span className={`font-mono font-semibold ${value >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
          {value >= 0 ? "+" : ""}{value.toFixed(1)}
        </span>
      </div>
      <input
        type="range"
        min={WEIGHT_RANGE.min}
        max={WEIGHT_RANGE.max}
        step={WEIGHT_RANGE.step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 rounded-lg appearance-none cursor-pointer"
        style={{ accentColor: value >= 0 ? "#16a34a" : "#dc2626" }}
      />
      <div className="flex justify-between text-xs text-gray-400 dark:text-gray-500">
        <span>{WEIGHT_RANGE.min}</span>
        <span>0</span>
        <span>{WEIGHT_RANGE.max}</span>
      </div>
    </div>
  );
}

function StimulusSlider({ label, sublabel, value, onChange, min, max }: {
  label: string; sublabel: string; value: number;
  onChange: (v: number) => void; min: number; max: number;
}) {
  return (
    <div className="mb-5">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium">{label} <span className="text-gray-400 dark:text-gray-500 text-xs">({sublabel})</span></span>
        <span className="font-mono text-gray-700 dark:text-gray-300">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 rounded-lg appearance-none cursor-pointer"
        style={{ accentColor: "#6b7280" }}
      />
      <div className="flex justify-between text-xs text-gray-400 dark:text-gray-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function PresetButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded border border-gray-300 dark:border-gray-600 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
    >
      {label}
    </button>
  );
}
