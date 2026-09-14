import { decodeFloats, decodePositions } from "./client";
import type { SampleMeta } from "./types";

export const PRERENDER_DEFAULT = {
  n: 6,
  l: 2,
  m: 0,
  system: "c",
  basis: "complex",
  count: 100000,
} as const;

export interface PrerenderedSample {
  meta: SampleMeta;
  positions: Float32Array;
  density: Float32Array;
  phase: Float32Array | null;
}

export function matchesPrerenderDefault(s: {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: string;
  count: number;
}): boolean {
  return (
    s.n === PRERENDER_DEFAULT.n &&
    s.l === PRERENDER_DEFAULT.l &&
    s.m === PRERENDER_DEFAULT.m &&
    s.system === PRERENDER_DEFAULT.system &&
    s.basis === PRERENDER_DEFAULT.basis &&
    s.count === PRERENDER_DEFAULT.count
  );
}

const BASE = "/prerender/default";

export async function loadPrerenderedSample(): Promise<PrerenderedSample | null> {
  try {
    const [metaRes, posRes, denRes, phaseRes] = await Promise.all([
      fetch(`${BASE}/meta.json`),
      fetch(`${BASE}/positions.bin`),
      fetch(`${BASE}/density.bin`),
      fetch(`${BASE}/phase.bin`),
    ]);
    if (!metaRes.ok || !posRes.ok || !denRes.ok) return null;
    const meta = (await metaRes.json()) as SampleMeta;
    if (meta.kind !== "sample") return null;
    const positions = decodePositions(await posRes.arrayBuffer());
    const density = decodeFloats(await denRes.arrayBuffer());
    if (positions.length / 3 !== meta.count) return null;
    if (density.length !== meta.count) return null;
    const phase = phaseRes.ok ? decodeFloats(await phaseRes.arrayBuffer()) : null;
    if (phase && phase.length !== meta.count) return null;
    return { meta, positions, density, phase };
  } catch {
    return null;
  }
}
