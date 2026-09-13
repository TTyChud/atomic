import type { IsoMeta } from "../api/types";
import { phaseColor, srgbToLinear } from "./colormap";

export function buildSurfaceColors(phase: Float32Array): Float32Array {
  const out = new Float32Array(phase.length * 3);
  for (let i = 0; i < phase.length; i++) {
    const [r, g, b] = phaseColor(phase[i]);
    out[3 * i] = srgbToLinear(r / 255);
    out[3 * i + 1] = srgbToLinear(g / 255);
    out[3 * i + 2] = srgbToLinear(b / 255);
  }
  return out;
}

export function enclosedCaption(meta: IsoMeta): string {
  const inside = (meta.enclosed_fraction.value * 100).toFixed(1);
  const outside = (meta.outside_fraction * 100).toFixed(1);
  return `this surface encloses ${inside}% of the electron, which is outside it ${outside}% of the time`;
}

export function surfaceExtent(vertices: Float32Array): number {
  let max = 0;
  for (let i = 0; i < vertices.length; i += 3) {
    const r = Math.hypot(vertices[i], vertices[i + 1], vertices[i + 2]);
    if (r > max) max = r;
  }
  return max;
}

export function componentsCaption(meta: IsoMeta): string {
  const pieces = `${meta.components} piece${meta.components === 1 ? "" : "s"}`;
  if (meta.l === 0) return pieces;
  return `${pieces} at this resolution (lobes closer than one cell come out joined)`;
}
