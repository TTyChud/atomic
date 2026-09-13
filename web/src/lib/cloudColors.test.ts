import { describe, expect, it } from "vitest";
import { buildCloudColors } from "./cloudColors";
import { srgbToLinear } from "./colormap";
import { INFERNO } from "./luts";

describe("buildCloudColors", () => {
  const density = new Float32Array([0, 2, 1]);
  const phase = new Float32Array([0, Math.PI / 2]);

  it("solid mode returns null (material colour handles it)", () => {
    expect(buildCloudColors("solid", density, phase)).toBeNull();
  });

  it("density mode maps extremes through the inferno LUT in linear space", () => {
    const colors = buildCloudColors("density", density, null);
    expect(colors).not.toBeNull();
    expect(colors).toHaveLength(9);
    const top = INFERNO[255];
    expect(colors![3]).toBeCloseTo(srgbToLinear(top[0] / 255), 6);
    expect(colors![4]).toBeCloseTo(srgbToLinear(top[1] / 255), 6);
    expect(colors![5]).toBeCloseTo(srgbToLinear(top[2] / 255), 6);
    const bottom = INFERNO[0];
    expect(colors![0]).toBeCloseTo(srgbToLinear(bottom[0] / 255), 6);
    expect(colors![2]).toBeCloseTo(srgbToLinear(bottom[2] / 255), 6);
  });

  it("phase mode needs the phase channel", () => {
    expect(buildCloudColors("phase", density, null)).toBeNull();
    expect(buildCloudColors("phase", null, phase)).toHaveLength(6);
  });

  it("density mode without data returns null", () => {
    expect(buildCloudColors("density", null, phase)).toBeNull();
  });
});
