import { describe, expect, it } from "vitest";
import { URL_DEFAULTS } from "../lib/urlState";
import { tourReset } from "./apply";

const systems = [
  { key: "h", name: "hydrogen", has_gsz: true },
  { key: "s", name: "sulfur", has_gsz: false },
] as never[];

describe("tourReset", () => {
  it("carries every input the step asked for", () => {
    const out = tourReset({ ...URL_DEFAULTS, n: 3, l: 2, view: "levels" }, systems);
    expect(out.n).toBe(3);
    expect(out.l).toBe(2);
    expect(out.view).toBe("levels");
  });

  it("clears every derived field, not just the level payloads", () => {
    const out = tourReset({ ...URL_DEFAULTS, n: 2 }, systems);
    for (const k of [
      "stateInfo",
      "positions",
      "density",
      "phase",
      "meta",
      "plane",
      "iso",
      "radial",
      "levels",
      "spectrum",
      "absorptionData",
    ]) {
      expect(out[k as keyof typeof out], `${k} not cleared`).toBeNull();
    }
    expect(out.status).toBe("idle");
    expect(out.planeStatus).toBe("idle");
    expect(out.isoStatus).toBe("idle");
  });

  it("clears the payloads that live outside INVALIDATED", () => {
    const out = tourReset({ ...URL_DEFAULTS, system: "he" }, systems);
    expect(out.ghost).toBeNull();
    expect(out.ghostStatus).toBe("idle");
    expect(out.hfLevels).toBeNull();
    expect(out.forceLaw).toBeNull();
    expect(out.forceStatus).toBe("idle");
    expect(out.whatif).toBeNull();
    expect(out.whatifStatus).toBe("idle");
  });

  it("moves an atom with no GSZ parameters onto Hartree-Fock", () => {
    const out = tourReset({ ...URL_DEFAULTS, system: "s", model: "gsz" }, systems);
    expect(out.model).toBe("hf");
  });

  it("leaves a resolvable model alone", () => {
    const out = tourReset({ ...URL_DEFAULTS, system: "h", model: "gsz" }, systems);
    expect(out.model).toBe("gsz");
  });

  it("moves the ghost toggle without clobbering the orbit data", () => {
    const out = tourReset({ ...URL_DEFAULTS, ghost: true }, systems);
    expect(out.ghostOn).toBe(true);
    expect(out.ghost).toBeNull();
    expect(out.ghostStatus).toBe("idle");
  });
});
