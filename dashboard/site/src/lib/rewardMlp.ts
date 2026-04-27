// MLP reward forward pass + fixture loader.
//
// Mirrors src/reward.py::RewardMLP exactly:
//   input(4) -> Dense(8, tanh) -> Dense(8, tanh) -> Dense(1, linear)
//
// Lives entirely in the browser — fixtures are tiny (~17 KB JSON for 4
// archetypes), forward pass is ~120 multiplies, can be evaluated thousands
// of times per second to drive a real-time landscape heatmap. When recorder
// v3 lands and ships per-agent genomes inside replays, this module is what
// AgentInspector calls; the fixture path is an interim demonstration.

export interface MlpLayer {
  kernel: number[][]; // shape (in, out)
  bias: number[]; // shape (out,)
}

export interface MlpGenome {
  Dense_0: MlpLayer;
  Dense_1: MlpLayer;
  Dense_2: MlpLayer;
}

export interface MlpFixture {
  name: string;
  description: string;
  params: MlpGenome;
}

export interface MlpFixtureFile {
  version: number;
  arch: string;
  input_dim: number;
  input_labels: string[];
  hidden_size: number;
  synthetic: boolean;
  note?: string;
  fixtures: MlpFixture[];
}

function denseTanh(x: Float32Array, layer: MlpLayer): Float32Array {
  const inDim = layer.kernel.length;
  const outDim = layer.bias.length;
  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let s = layer.bias[j];
    for (let i = 0; i < inDim; i++) {
      s += x[i] * layer.kernel[i][j];
    }
    out[j] = Math.tanh(s);
  }
  return out;
}

function denseLinear(x: Float32Array, layer: MlpLayer): Float32Array {
  const inDim = layer.kernel.length;
  const outDim = layer.bias.length;
  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let s = layer.bias[j];
    for (let i = 0; i < inDim; i++) {
      s += x[i] * layer.kernel[i][j];
    }
    out[j] = s;
  }
  return out;
}

export function evalRewardMlp(genome: MlpGenome, stimuli: Float32Array): number {
  const h1 = denseTanh(stimuli, genome.Dense_0);
  const h2 = denseTanh(h1, genome.Dense_1);
  const out = denseLinear(h2, genome.Dense_2);
  return out[0];
}

// Sample the reward function across a 2D stimulus grid for the heatmap.
// `axisX`, `axisY` index into the 4-d stimulus vector. Other axes are held
// at the values supplied in `held`. Returns flat row-major (gridY × gridX)
// reward values + min/max for colormap normalization.
export function sampleRewardGrid(
  genome: MlpGenome,
  axisX: number,
  axisY: number,
  rangeX: [number, number],
  rangeY: [number, number],
  gridSize: number,
  held: Float32Array,
): { values: Float32Array; min: number; max: number } {
  const values = new Float32Array(gridSize * gridSize);
  const stim = new Float32Array(held);
  let lo = Infinity;
  let hi = -Infinity;
  for (let yi = 0; yi < gridSize; yi++) {
    const ty = yi / (gridSize - 1);
    const yv = rangeY[0] + (rangeY[1] - rangeY[0]) * ty;
    stim[axisY] = yv;
    for (let xi = 0; xi < gridSize; xi++) {
      const tx = xi / (gridSize - 1);
      const xv = rangeX[0] + (rangeX[1] - rangeX[0]) * tx;
      stim[axisX] = xv;
      const r = evalRewardMlp(genome, stim);
      values[yi * gridSize + xi] = r;
      if (r < lo) lo = r;
      if (r > hi) hi = r;
    }
  }
  return { values, min: lo, max: hi };
}

// Best-linear-approximation: fits r ≈ w·s + b across a sampled grid (here we
// use the same grid as the heatmap so the linear-equivalent is comparable to
// what the user is looking at). Returns (4,) weights + bias + residual norm.
//
// Solves least-squares via the normal equations on a 4-d feature space — at
// this size (a few hundred samples) it's cheap and avoids pulling in an
// SVD library.
export function fitLinearEquivalent(
  genome: MlpGenome,
  rangePerAxis: [number, number][] = [
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
  ],
  samplesPerAxis = 5,
): { weights: number[]; bias: number; residualRms: number; targetRms: number } {
  // 4-d Latin-hypercube-ish grid (full Cartesian product = 5^4 = 625 samples).
  const M = samplesPerAxis;
  const total = M ** 4;
  const X = new Float32Array(total * 5); // [1, s0, s1, s2, s3]
  const Y = new Float32Array(total);
  const stim = new Float32Array(4);
  let row = 0;
  for (let i0 = 0; i0 < M; i0++) {
    const v0 = rangePerAxis[0][0] + ((rangePerAxis[0][1] - rangePerAxis[0][0]) * i0) / (M - 1);
    stim[0] = v0;
    for (let i1 = 0; i1 < M; i1++) {
      const v1 = rangePerAxis[1][0] + ((rangePerAxis[1][1] - rangePerAxis[1][0]) * i1) / (M - 1);
      stim[1] = v1;
      for (let i2 = 0; i2 < M; i2++) {
        const v2 = rangePerAxis[2][0] + ((rangePerAxis[2][1] - rangePerAxis[2][0]) * i2) / (M - 1);
        stim[2] = v2;
        for (let i3 = 0; i3 < M; i3++) {
          const v3 = rangePerAxis[3][0] + ((rangePerAxis[3][1] - rangePerAxis[3][0]) * i3) / (M - 1);
          stim[3] = v3;
          X[row * 5 + 0] = 1;
          X[row * 5 + 1] = v0;
          X[row * 5 + 2] = v1;
          X[row * 5 + 3] = v2;
          X[row * 5 + 4] = v3;
          Y[row] = evalRewardMlp(genome, stim);
          row++;
        }
      }
    }
  }
  // Solve (XᵀX) β = Xᵀy via 5×5 Gauss-Jordan.
  const beta = solveNormalEq(X, Y, 5, total);
  const bias = beta[0];
  const weights = [beta[1], beta[2], beta[3], beta[4]];
  // Residual + target RMS for "nonlinearity utilization" metric.
  let resSq = 0;
  let targetSq = 0;
  let yMean = 0;
  for (let i = 0; i < total; i++) yMean += Y[i];
  yMean /= total;
  for (let i = 0; i < total; i++) {
    const yhat =
      bias +
      weights[0] * X[i * 5 + 1] +
      weights[1] * X[i * 5 + 2] +
      weights[2] * X[i * 5 + 3] +
      weights[3] * X[i * 5 + 4];
    const e = Y[i] - yhat;
    resSq += e * e;
    const dy = Y[i] - yMean;
    targetSq += dy * dy;
  }
  return {
    weights,
    bias,
    residualRms: Math.sqrt(resSq / total),
    targetRms: Math.sqrt(targetSq / total),
  };
}

function solveNormalEq(
  X: Float32Array,
  Y: Float32Array,
  d: number,
  n: number,
): Float32Array {
  // A = XᵀX (d×d), b = XᵀY (d). Then solve A β = b.
  const A = new Float32Array(d * d);
  const b = new Float32Array(d);
  for (let r = 0; r < n; r++) {
    for (let i = 0; i < d; i++) {
      const xi = X[r * d + i];
      b[i] += xi * Y[r];
      for (let j = i; j < d; j++) {
        A[i * d + j] += xi * X[r * d + j];
      }
    }
  }
  // Symmetrize.
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < i; j++) A[i * d + j] = A[j * d + i];
  }
  // Gauss-Jordan with partial pivoting.
  const aug = new Float32Array(d * (d + 1));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) aug[i * (d + 1) + j] = A[i * d + j];
    aug[i * (d + 1) + d] = b[i];
  }
  for (let i = 0; i < d; i++) {
    // pivot
    let maxRow = i;
    let maxAbs = Math.abs(aug[i * (d + 1) + i]);
    for (let r = i + 1; r < d; r++) {
      const v = Math.abs(aug[r * (d + 1) + i]);
      if (v > maxAbs) {
        maxAbs = v;
        maxRow = r;
      }
    }
    if (maxRow !== i) {
      for (let c = 0; c <= d; c++) {
        const tmp = aug[i * (d + 1) + c];
        aug[i * (d + 1) + c] = aug[maxRow * (d + 1) + c];
        aug[maxRow * (d + 1) + c] = tmp;
      }
    }
    const piv = aug[i * (d + 1) + i] || 1e-9;
    for (let c = i; c <= d; c++) aug[i * (d + 1) + c] /= piv;
    for (let r = 0; r < d; r++) {
      if (r === i) continue;
      const f = aug[r * (d + 1) + i];
      if (f === 0) continue;
      for (let c = i; c <= d; c++) {
        aug[r * (d + 1) + c] -= f * aug[i * (d + 1) + c];
      }
    }
  }
  const beta = new Float32Array(d);
  for (let i = 0; i < d; i++) beta[i] = aug[i * (d + 1) + d];
  return beta;
}

export async function fetchMlpFixtures(): Promise<MlpFixtureFile> {
  const res = await fetch("/fixtures/mlp_reward_examples.json", {
    cache: "no-cache",
  });
  if (!res.ok) {
    throw new Error(`MLP fixture not found (${res.status})`);
  }
  return (await res.json()) as MlpFixtureFile;
}
