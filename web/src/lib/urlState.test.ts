import { describe, expect, it } from "vitest";
import {
  currentUrlState,
  isNewPlace,
  parseAppUrl,
  serializeAppUrl,
  URL_DEFAULTS,
  type UrlState,
} from "./urlState";

describe("parseAppUrl", () => {
  it("round-trips a full orbital address", () => {
    const parsed = parseAppUrl("?n=3&l=1&m=-1&system=mu-h");
    expect(parsed).toMatchObject({ n: 3, l: 1, m: -1, system: "mu-h" });
    const back = serializeAppUrl({ ...URL_DEFAULTS, ...parsed });
    expect(parseAppUrl(back)).toMatchObject({ n: 3, l: 1, m: -1, system: "mu-h" });
  });

  it("drops junk instead of poisoning the store", () => {
    expect(parseAppUrl("?n=banana&view=warp&system=!!!")).toEqual({});
    expect(parseAppUrl("?l=9&m=9")).toMatchObject({ n: 6, l: 5, m: 5 });
  });

  it("demotes phase to density on the real basis", () => {
    expect(parseAppUrl("?basis=real&color=phase")).toMatchObject({
      basis: "real",
      colorMode: "density",
    });
    expect(parseAppUrl("?color=phase")).toMatchObject({ colorMode: "phase" });
  });

  it("reads view, basis and plane quantity", () => {
    expect(parseAppUrl("?view=radial&basis=real&plane=psi")).toMatchObject({
      view: "radial",
      basis: "real",
      planeQuantity: "psi",
    });
  });
});

describe("serializeAppUrl", () => {
  it("omits defaults entirely", () => {
    expect(serializeAppUrl(URL_DEFAULTS)).toBe("");
  });

  it("writes only what moved", () => {
    expect(serializeAppUrl({ ...URL_DEFAULTS, n: 2, view: "levels" })).toBe("?n=2&view=levels");
  });

  it("round-trips lab multipliers and force-law settings", () => {
    const state: UrlState = {
      ...URL_DEFAULTS,
      labConst: { ...URL_DEFAULTS.labConst, e: 2 },
      forcePreset: "yukawa",
      forceParams: { lambda: 6 },
      forceL: 1,
    };
    const back = parseAppUrl(serializeAppUrl({ ...state }));
    expect(back).toMatchObject({
      labConst: { hbar: 1, e: 2, m_e: 1, eps0: 1, c: 1 },
      forcePreset: "yukawa",
      forceParams: { lambda: 6 },
      forceL: 1,
    });
  });

  it("reads custom expressions only for the custom preset", () => {
    expect(parseAppUrl("?preset=custom&expr=-1%2Fr")).toMatchObject({
      forcePreset: "custom",
      forceExpr: "-1/r",
    });
    expect(parseAppUrl("?expr=-1%2Fr").forceExpr).toBeUndefined();
  });

  it("clamps lab multipliers into range", () => {
    expect(parseAppUrl("?e=99").labConst).toMatchObject({ e: 4 });
  });
});

describe("currentUrlState", () => {
  it("picks the addressable slice", () => {
    expect(currentUrlState({ ...URL_DEFAULTS, system: "he+" })).toMatchObject({
      system: "he+",
      n: 6,
    });
  });
});

describe("the default place", () => {
  it("opens on carbon", () => {
    expect(URL_DEFAULTS.system).toBe("c");
    expect(URL_DEFAULTS.n).toBe(6);
  });
});

describe("model selection", () => {
  it("round-trips the model key", () => {
    const state = parseAppUrl("?system=ar&model=hf");
    expect(state.model).toBe("hf");
    const back = serializeAppUrl({ ...URL_DEFAULTS, system: "ar", model: "hf" });
    expect(back).toContain("model=hf");
    expect(parseAppUrl(back)).toMatchObject({ system: "ar", model: "hf" });
  });

  it("defaults to gsz so existing deep links keep resolving as before", () => {
    expect(parseAppUrl("?system=he+").model).toBeUndefined();
    expect(URL_DEFAULTS.model).toBe("gsz");
  });

  it("omits the default from the serialized URL", () => {
    expect(serializeAppUrl(URL_DEFAULTS)).toBe("");
    expect(serializeAppUrl({ ...URL_DEFAULTS, model: "hf" })).toContain("model=hf");
  });

  it("ignores an unknown model rather than passing it to the store", () => {
    expect(parseAppUrl("?model=dft").model).toBeUndefined();
  });
});

describe("many-electron controls", () => {
  it("round-trips config, nox, nopauli and compare", () => {
    const parsed = parseAppUrl("?model=hf&config=1s2%202s1&nox=1&compare=1");
    expect(parsed).toMatchObject({
      model: "hf",
      config: "1s2 2s1",
      exchange: false,
      compare: true,
    });
    const back = serializeAppUrl({ ...URL_DEFAULTS, ...parsed });
    expect(back).toContain("config=1s2+2s1");
    expect(back).toContain("nox=1");
    expect(back).toContain("compare=1");
    expect(parseAppUrl(back)).toMatchObject({
      config: "1s2 2s1",
      exchange: false,
      compare: true,
    });
  });

  it("reads nopauli as the collapse it means, exchange off included", () => {
    expect(parseAppUrl("?nopauli=1")).toMatchObject({ pauli: false, exchange: false });
    const back = serializeAppUrl({ ...URL_DEFAULTS, pauli: false, exchange: false });
    expect(back).toContain("nopauli=1");
  });

  it("drops a malformed config rather than sending it", () => {
    expect(parseAppUrl("?config=banana").config).toBeUndefined();
  });

  it("omits the real-physics defaults from the serialized URL", () => {
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("nox");
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("nopauli");
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("compare");
  });

  it("a link cannot land anyone in altered physics by omission", () => {
    expect(URL_DEFAULTS.exchange).toBe(true);
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("nox");
    expect(parseAppUrl("?system=ne&model=hf").exchange).toBeUndefined();
  });

  it("the stronger counterfactual cannot arrive by omission either", () => {
    expect(URL_DEFAULTS.pauli).toBe(true);
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("nopauli");
    expect(parseAppUrl("?system=ne&model=hf").pauli).toBeUndefined();
  });

  it("writes both keys for the collapse, since it means both things", () => {
    const off = serializeAppUrl({ ...URL_DEFAULTS, pauli: false, exchange: false });
    expect(off).toContain("nopauli=1");
    expect(off).toContain("nox=1");
  });

  it("does not confuse model=hf with the hyperfine hf=1 flag", () => {
    const q = serializeAppUrl({ ...URL_DEFAULTS, model: "hf", hyperfine: true });
    const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
    expect(back.model).toBe("hf");
    expect(back.hyperfine).toBe(true);

    const onlyHyperfine = { ...URL_DEFAULTS, ...parseAppUrl("?hf=1") };
    expect(onlyHyperfine.hyperfine).toBe(true);
    expect(onlyHyperfine.model).toBe("gsz");
  });
});

describe("lab charge", () => {
  it("round-trips the What-If Z stepper", () => {
    expect(parseAppUrl("?z=3")).toMatchObject({ labZ: 3 });
    const back = serializeAppUrl({ ...URL_DEFAULTS, labZ: 3 });
    expect(back).toContain("z=3");
    expect(parseAppUrl(back)).toMatchObject({ labZ: 3 });
  });

  it("clamps junk charge into the hydrogenic range", () => {
    expect(parseAppUrl("?z=99")).toMatchObject({ labZ: 10 });
    expect(parseAppUrl("?z=banana").labZ).toBeUndefined();
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("z=");
  });
});

describe("what earns a history entry", () => {
  it("counts a different state, atom, model, view or tour step as a place", () => {
    for (const move of [
      { n: 2, l: 1, m: 0 },
      { system: "he+" },
      { view: "spectrum" as const },
      { model: "hf" as const },
      { config: "1s2 2s1" },
      { tour: "hydrogen-honestly" },
      { step: 2 },
    ]) {
      expect(isNewPlace(URL_DEFAULTS, { ...URL_DEFAULTS, ...move })).toBe(true);
    }
  });

  it("counts a setting on the place you are already in as no move at all", () => {
    for (const tweak of [
      { bField: 3 },
      { eField: 2 },
      { temperatureK: 12000 },
      { logNe: 15 },
      { logColumn: 22 },
      { isoFraction: 0.5 },
      { colorMode: "density" as const },
      { nucleusMode: "hidden" as const },
      { fineStructure: true },
    ]) {
      expect(isNewPlace(URL_DEFAULTS, { ...URL_DEFAULTS, ...tweak })).toBe(false);
    }
  });

  it("is not a move when nothing moved", () => {
    expect(isNewPlace(URL_DEFAULTS, { ...URL_DEFAULTS })).toBe(false);
  });
});

describe("ghost deep links", () => {
  it("round-trips the overlay toggle", () => {
    expect(parseAppUrl("?ghost=1")).toMatchObject({ ghost: true });
    expect(parseAppUrl("?ghost=0")).toMatchObject({ ghost: false });
    expect(parseAppUrl("?ghost=banana").ghost).toBeUndefined();
    const back = serializeAppUrl({ ...URL_DEFAULTS, ghost: true });
    expect(back).toContain("ghost=1");
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("ghost");
  });
});
