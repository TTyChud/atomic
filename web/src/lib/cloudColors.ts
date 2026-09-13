import type { ColorMode } from "./urlState";
import { densityT, lutColor, maxOf, phaseColor, srgbToLinear } from "./colormap";
import { INFERNO } from "./luts";

export function buildCloudColors(
  mode: ColorMode,
  density: Float32Array | null,
  phase: Float32Array | null,
): Float32Array | null {
  if (mode === "density" && density) {
    const vmax = maxOf(density);
    const out = new Float32Array(density.length * 3);
    for (let i = 0; i < density.length; i++) {
      const [r, g, b] = lutColor(INFERNO, densityT(density[i], vmax));
      out[3 * i] = srgbToLinear(r / 255);
      out[3 * i + 1] = srgbToLinear(g / 255);
      out[3 * i + 2] = srgbToLinear(b / 255);
    }
    return out;
  }
  if (mode === "phase" && phase) {
    const out = new Float32Array(phase.length * 3);
    for (let i = 0; i < phase.length; i++) {
      const [r, g, b] = phaseColor(phase[i]);
      out[3 * i] = srgbToLinear(r / 255);
      out[3 * i + 1] = srgbToLinear(g / 255);
      out[3 * i + 2] = srgbToLinear(b / 255);
    }
    return out;
  }
  return null;
}
