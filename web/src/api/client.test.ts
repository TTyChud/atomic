import { describe, expect, it } from "vitest";
import { num, thumbnailUrl } from "./client";

describe("thumbnailUrl", () => {
  it("leaves the real-physics defaults off the query", () => {
    expect(thumbnailUrl(2, 1, 0, "he+", "complex", 96)).toBe(
      "/api/thumbnail/2/1/0?system=he%2B&basis=complex&size=96",
    );
  });

  it("carries the model, config and counterfactual flags for Hartree-Fock", () => {
    const url = thumbnailUrl(2, 1, 0, "c", "complex", 96, {
      model: "hf", config: "1s2 2s2 2p2", exchange: false, pauli: true,
    });
    expect(url).toContain("model=hf");
    expect(url).toContain("config=1s2%202s2%202p2");
    expect(url).toContain("exchange=false");
    expect(url).not.toContain("pauli=false");
  });
});

describe("num", () => {
  it("leaves ordinary numbers alone", () => {
    expect(num(10000)).toBe("10000");
    expect(num(1e13)).toBe("10000000000000");
    expect(num(656.28)).toBe("656.28");
    expect(num(-1.5)).toBe("-1.5");
  });

  it("escapes the plus in exponential form", () => {
    expect(String(1e22)).toBe("1e+22");
    expect(num(1e22)).toBe("1e%2B22");
    expect(num(1e21)).toBe("1e%2B21");
    expect(decodeURIComponent(num(1e26))).toBe("1e+26");
  });

  it("survives a round trip through URL decoding for every slider extreme", () => {
    for (const v of [1e4, 1e13, 1e22, 1e14, 1e20, 1e26, 1e2, 1e7]) {
      const parsed = Number(
        new URLSearchParams(`x=${num(v)}`).get("x"),
      );
      expect(parsed).toBe(v);
    }
  });

  it("would have caught the raw interpolation it replaces", () => {
    const raw = new URLSearchParams(`x=${String(1e22)}`).get("x");
    expect(raw).toBe("1e 22");
    expect(Number(raw)).toBeNaN();
  });
});
