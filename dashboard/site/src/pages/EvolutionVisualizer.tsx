import { useState, useMemo, useCallback } from "react";

function gaussianSample(rng: () => number): number {
  let u1 = 0, u2 = 0;
  while (u1 === 0) u1 = rng();
  while (u2 === 0) u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function tSample(df: number, scale: number, rng: () => number): number {
  let chi2 = 0;
  for (let i = 0; i < df; i++) {
    const z = gaussianSample(rng);
    chi2 += z * z;
  }
  const z = gaussianSample(rng);
  return scale * z / Math.sqrt(chi2 / df);
}

function createRng(seed: number) {
  let t = seed + 0x6D2B79F5;
  return () => {
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateHistogram(samples: number[], bins: number, range: [number, number]): { centers: number[]; counts: number[] } {
  const [lo, hi] = range;
  const binWidth = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  const centers = Array.from({ length: bins }, (_, i) => lo + (i + 0.5) * binWidth);
  for (const s of samples) {
    if (s >= lo && s < hi) {
      const idx = Math.floor((s - lo) / binWidth);
      counts[Math.min(idx, bins - 1)]++;
    }
  }
  const total = samples.length * binWidth;
  return { centers, counts: counts.map((c) => c / total) };
}

export default function EvolutionVisualizer() {
  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Evolution Visualizer</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-8">
        Understanding how mutation distributions shape evolutionary dynamics.
        K&D (2025) uses Student's t(df=2) mutation — heavier tails than Gaussian
        allow occasional large jumps in reward weight space.
      </p>
      <div className="space-y-10">
        <DistributionComparison />
        <MutationDemo />
        <WhyHeavyTails />
      </div>
    </div>
  );
}

function DistributionComparison() {
  const data = useMemo(() => {
    const n = 10000;
    const rng = createRng(42);
    const gaussian = Array.from({ length: n }, () => 0.4 * gaussianSample(rng));
    const cauchy = Array.from({ length: n }, () => tSample(1, 0.4, rng));
    const t2 = Array.from({ length: n }, () => tSample(2, 0.4, rng));
    const range: [number, number] = [-4, 4];
    const bins = 80;
    return { gaussian: generateHistogram(gaussian, bins, range), cauchy: generateHistogram(cauchy, bins, range), t2: generateHistogram(t2, bins, range), range };
  }, []);

  const maxDensity = Math.max(...data.gaussian.counts, ...data.cauchy.counts, ...data.t2.counts);

  const W = 700, H = 300;
  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  function xScale(v: number) { return margin.left + ((v - data.range[0]) / (data.range[1] - data.range[0])) * plotW; }
  function yScale(v: number) { return margin.top + plotH - (v / (maxDensity * 1.1)) * plotH; }

  function makePath(hist: { centers: number[]; counts: number[] }) {
    return hist.centers.map((c, i) => `${i === 0 ? "M" : "L"} ${xScale(c).toFixed(1)} ${yScale(hist.counts[i]).toFixed(1)}`).join(" ");
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-3">Panel 1: Mutation Distribution Comparison</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        10,000 samples from each distribution (scale=0.4). Notice the heavy tails of t(df=2) compared to Gaussian.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-3xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg">
        <line x1={margin.left} y1={margin.top + plotH} x2={margin.left + plotW} y2={margin.top + plotH} stroke="currentColor" className="text-gray-300 dark:text-gray-600" strokeWidth={1} />
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + plotH} stroke="currentColor" className="text-gray-300 dark:text-gray-600" strokeWidth={1} />
        {[-4, -2, 0, 2, 4].map((v) => (
          <text key={v} x={xScale(v)} y={margin.top + plotH + 20} textAnchor="middle" className="text-[10px] fill-gray-500 dark:fill-gray-400">{v}</text>
        ))}
        <text x={margin.left + plotW / 2} y={H - 5} textAnchor="middle" className="text-[11px] fill-gray-600 dark:fill-gray-400">Mutation magnitude</text>
        <path d={makePath(data.gaussian)} fill="none" stroke="#4e79a7" strokeWidth={2} opacity={0.8} />
        <path d={makePath(data.t2)} fill="none" stroke="#e15759" strokeWidth={2.5} />
        <path d={makePath(data.cauchy)} fill="none" stroke="#76b7b2" strokeWidth={1.5} strokeDasharray="4 2" opacity={0.7} />
        <line x1={xScale(0)} y1={margin.top} x2={xScale(0)} y2={margin.top + plotH} stroke="currentColor" className="text-gray-300 dark:text-gray-600" strokeWidth={1} strokeDasharray="3 3" />
        <rect x={W - 170} y={margin.top + 5} width={150} height={70} className="fill-white dark:fill-gray-800" stroke="currentColor" rx={4} strokeWidth={0.5} />
        <line x1={W - 160} y1={margin.top + 22} x2={W - 140} y2={margin.top + 22} stroke="#4e79a7" strokeWidth={2} />
        <text x={W - 135} y={margin.top + 26} className="text-[10px] fill-gray-700 dark:fill-gray-300">Gaussian (s=0.4)</text>
        <line x1={W - 160} y1={margin.top + 42} x2={W - 140} y2={margin.top + 42} stroke="#e15759" strokeWidth={2.5} />
        <text x={W - 135} y={margin.top + 46} className="text-[10px] fill-gray-700 dark:fill-gray-300">t(df=2, s=0.4) — K&D 2025</text>
        <line x1={W - 160} y1={margin.top + 62} x2={W - 140} y2={margin.top + 62} stroke="#76b7b2" strokeWidth={1.5} strokeDasharray="4 2" />
        <text x={W - 135} y={margin.top + 66} className="text-[10px] fill-gray-700 dark:fill-gray-300">Cauchy (t df=1, s=0.4)</text>
      </svg>
    </div>
  );
}

function MutationDemo() {
  const [parentGenome, setParentGenome] = useState([2.0, -0.5, 1.5, -3.0]);
  const [childGenome, setChildGenome] = useState<number[] | null>(null);
  const [mutationNoise, setMutationNoise] = useState<number[] | null>(null);
  const [mutationCount, setMutationCount] = useState(0);

  const WEIGHT_NAMES = ["w_eat", "w_act", "w_prey", "w_pred"];
  const CLIP = 100;

  const mutate = useCallback(() => {
    const rng = createRng(Date.now() + mutationCount);
    const noise = WEIGHT_NAMES.map(() => tSample(2, 0.4, rng));
    const child = parentGenome.map((w, i) => Math.max(-CLIP, Math.min(CLIP, w + noise[i])));
    setMutationNoise(noise);
    setChildGenome(child);
    setMutationCount((c) => c + 1);
  }, [parentGenome, mutationCount]);

  const maxVal = 6;
  const barW = 280;

  function renderBar(value: number, color: string) {
    const w = (Math.abs(value) / maxVal) * (barW / 2);
    const x = value >= 0 ? barW / 2 : barW / 2 - w;
    return (
      <div className="relative h-7 bg-gray-100 dark:bg-gray-800 rounded" style={{ width: barW }}>
        <div className="absolute top-0 bottom-0 w-px bg-gray-300 dark:bg-gray-600" style={{ left: barW / 2 }} />
        <div className={`absolute top-0.5 bottom-0.5 rounded ${color}`} style={{ left: x, width: Math.max(w, 2) }} />
        <span className="absolute right-1 top-0.5 text-[10px] text-gray-500 dark:text-gray-400 font-mono">{value.toFixed(2)}</span>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-3">Panel 2: Mutate a Genome</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Click "Mutate" to add t(df=2, scale=0.4) noise to the parent genome. Repeat to see how the child genome drifts.
      </p>
      <div className="grid grid-cols-3 gap-6">
        <div>
          <div className="text-sm font-semibold mb-2 text-gray-600 dark:text-gray-300">Parent Genome</div>
          {WEIGHT_NAMES.map((wn, i) => (
            <div key={wn} className="mb-2">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-0.5">{wn}</div>
              {renderBar(parentGenome[i], "bg-blue-400 dark:bg-blue-500")}
            </div>
          ))}
        </div>
        <div className="flex flex-col items-center justify-center">
          <button onClick={mutate} className="px-5 py-2.5 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 transition mb-3">Mutate</button>
          {mutationNoise && (
            <div>
              <div className="text-xs text-gray-400 dark:text-gray-500 text-center mb-1">noise added:</div>
              {WEIGHT_NAMES.map((wn, i) => (
                <div key={wn} className="text-xs font-mono text-gray-500 dark:text-gray-400 text-center">
                  {wn}: {mutationNoise[i] >= 0 ? "+" : ""}{mutationNoise[i].toFixed(3)}
                </div>
              ))}
            </div>
          )}
        </div>
        <div>
          <div className="text-sm font-semibold mb-2 text-gray-600 dark:text-gray-300">Child Genome</div>
          {childGenome ? (
            WEIGHT_NAMES.map((wn, i) => (
              <div key={wn} className="mb-2">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-0.5">{wn}</div>
                {renderBar(childGenome[i], "bg-green-400 dark:bg-green-500")}
              </div>
            ))
          ) : (
            <div className="text-sm text-gray-400 dark:text-gray-500 mt-8 text-center">Click Mutate</div>
          )}
          {childGenome && (
            <button onClick={() => { setParentGenome(childGenome); setChildGenome(null); setMutationNoise(null); }}
              className="mt-3 text-xs text-blue-500 dark:text-blue-400 hover:underline">
              Use child as new parent →
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function WhyHeavyTails() {
  return (
    <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg p-5">
      <h2 className="text-xl font-semibold mb-2">Panel 3: Why Heavy Tails Matter</h2>
      <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
        <p><strong>Gaussian mutation</strong> (the standard approach) produces small, incremental changes. Most mutations move the reward weights by &lt;1 standard deviation. The population explores reward space slowly and can get stuck in local optima.</p>
        <p><strong>Student's t(df=2)</strong> mutation (K&D 2025) has much heavier tails. While most mutations are still small, there's a meaningful probability of large jumps — mutations 5-10x larger than typical. These occasional large jumps:</p>
        <ul className="list-disc pl-5 space-y-1">
          <li>Enable faster exploration of reward weight space</li>
          <li>Allow escape from local optima in the reward-fitness landscape</li>
          <li>Create more phenotypic diversity in the population</li>
          <li>Better model biological mutation which has occasional large-effect mutations</li>
        </ul>
        <p><strong>Cauchy (t df=1)</strong> has even heavier tails, but K&D found t(df=2) works better in practice. The 2024 paper used Cauchy with scale=0.02; the 2025 paper switched to t(df=2) with scale=0.4.</p>
      </div>
    </div>
  );
}
