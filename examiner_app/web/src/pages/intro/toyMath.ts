// Pure maths for the browser-side delta-rule toy (4 x 4 memory). No API calls.
// Conventions: Mat[i][j] has row i (value index) and column j (key index), so S k is a read over columns.

export type Vec = number[];
export type Mat = number[][];

export const SIZE = 4;

/** Small deterministic PRNG so that a given seed always gives the same key and value. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function norm(v: Vec): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

/** Starting memory: entries in [-0.8, 0.8]. */
export function makeMemory(seed: number): Mat {
  const rng = mulberry32(seed);
  return Array.from({ length: SIZE }, () => Array.from({ length: SIZE }, () => (rng() * 2 - 1) * 0.8));
}

/** A key of unit length and a value with entries in [-1, 1]. */
export function makeKeyValue(seed: number): { key: Vec; value: Vec } {
  const rng = mulberry32(seed * 7919 + 17);
  const uni = () => rng() * 2 - 1;
  const raw = Array.from({ length: SIZE }, () => uni());
  const n = norm(raw) || 1;
  const key = raw.map((x) => x / n);
  const value = Array.from({ length: SIZE }, () => uni());
  return { key, value };
}

function matVec(M: Mat, x: Vec): Vec {
  return M.map((row) => row.reduce((s, m, j) => s + m * x[j], 0));
}

export interface StepResult {
  /** S~ = S diag(alpha): after the fade. */
  faded: Mat;
  /** S~ k: what the faded memory returns for this key. */
  read: Vec;
  /** r = v - S~ k: what was missing. */
  residual: Vec;
  /** S' = S~ + beta r k^T. */
  next: Mat;
  /** S' k: what the memory returns after the write. */
  readAfter: Vec;
  /** ||v - S~ k||: read error before the write. */
  errBefore: number;
  /** ||v - S' k||: read error after the write. */
  errAfter: number;
}

export function runStep(memory: Mat, alpha: Vec, key: Vec, value: Vec, beta: number): StepResult {
  const faded = memory.map((row) => row.map((x, j) => x * alpha[j]));
  const read = matVec(faded, key);
  const residual = value.map((v, i) => v - read[i]);
  const next = faded.map((row, i) => row.map((x, j) => x + beta * residual[i] * key[j]));
  const readAfter = matVec(next, key);
  const afterGap = value.map((v, i) => v - readAfter[i]);
  return { faded, read, residual, next, readAfter, errBefore: norm(residual), errAfter: norm(afterGap) };
}
