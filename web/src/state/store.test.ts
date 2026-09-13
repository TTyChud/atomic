import { beforeEach, describe, expect, it, vi } from "vitest";
import { isScreenedLevels } from "../api/client";
import { FLAGSHIP_TOUR_ID } from "../tours/registry";
import { useAppStore } from "./store";

const SYSTEM = {
  key: "h",
  name: "Hydrogen",
  z: 1,
  mu_ratio: { value: 1, unit: "m_e", label: "m", provenance: null },
  m_over_m_nucleus: 0,
  description: "h",
  nuclear_radius: null,
  nuclear_radius_fm: null,
  kind: "hydrogenic",
  n_electrons: null,
  has_gsz: true,
};

const SAMPLE_META = {
  kind: "sample",
  count: 2,
  dtype: "float32",
  layout: "xyz-interleaved",
  unit: "bohr",
  n: 1,
  l: 0,
  m: 0,
  basis: "complex",
  system: "h",
  model: "hydrogenic",
  provenance: null,
  channels: [
    { name: "positions", dtype: "float32", unit: "bohr", provenance: null },
    { name: "density", dtype: "float32", unit: "bohr^-3", provenance: null },
    { name: "phase", dtype: "float32", unit: "rad", provenance: null },
  ],
};

const PLANE_META = {
  kind: "plane",
  resolution: 4,
  dtype: "float32",
  layout: "row-major",
  quantity: "density",
  unit: "bohr^-3",
  label: "density",
  half_extent: 10,
  axis_unit: "bohr",
  n: 1,
  l: 0,
  m: 0,
  basis: "complex",
  system: "h",
  model: "hydrogenic",
  provenance: null,
};

function stubFetch() {
  vi.stubGlobal(
    "fetch",
    async (url: string, init?: { method?: string }) => {
      const method = init?.method ?? "GET";
      if (url === "/api/systems") return json({ systems: [SYSTEM] });
      if (url.startsWith("/api/state/")) {
        return json({ n: 1, l: 0, m: 0, system: SYSTEM, energy: { value: -0.5 } });
      }
      if (url.startsWith("/api/levels")) {
        return json({
          system: SYSTEM,
          n_max: 6,
          fine_structure: false,
          alpha: 1 / 137,
          dirac: false,
          b_field: 0,
          e_field: 0,
          hyperfine: false,
          fine: null,
          gross: [],
        });
      }
      if (url.startsWith("/api/spectrum")) {
        return json({
          system: SYSTEM, n_max: 6, fine_structure: false,
          lines: [], comparison: null, reference_citation: null,
          tolerance_relative: null, intensity_note: null,
        });
      }
      if (url.startsWith("/api/radial/")) {
        return json({ n: 1, l: 0, system: SYSTEM });
      }
      if (url.startsWith("/api/constants")) {
        const obs = (value: number) => ({
          quantity: { value, unit: "u", label: "o", provenance: null },
          ratio: 1,
          changed: false,
        });
        return json({ alpha: obs(0.0073), bohr_radius_pm: obs(52.9), hartree_ev: obs(27.2), altered: false });
      }
      if (url.startsWith("/api/classical")) {
        const q = (value: number) => ({ value, unit: "u", label: "g", provenance: null });
        return json({
          n: 1, system_key: "h", z: 1,
          orbits: [{ n: 1, radius_bohr: q(1), radius_pm: q(52.9) }],
          r0_bohr: q(1), collapse_time_s: q(1e-11),
          orbital_period_s: q(1e-16), orbit_count: q(2e5),
        });
      }
      if (url.startsWith("/api/forcelaw")) {
        return json({
          preset: "powerlaw", params: { p: 1 }, l: 0, z: 1, system: SYSTEM,
          counterfactual: [], bound_count: 0, requested_count: 4,
          reference: { kind: "levels", items: [] },
          potential_curve: { r: [], v_ev: [], provenance: null },
          expression: null,
        });
      }
      if (url === "/api/jobs/sample" && method === "POST") {
        return json({ id: "j1", status: "pending", progress: 0, error: null });
      }
      if (url === "/api/jobs/plane" && method === "POST") {
        return json({ id: "p1", status: "pending", progress: 0, error: null });
      }
      if (url === "/api/jobs/j1/meta") return json(SAMPLE_META);
      if (url === "/api/jobs/p1/meta") return json(PLANE_META);
      if (url.startsWith("/api/jobs/j1/data")) {
        if (url.includes("density") || url.includes("phase")) return bytes(8);
        return bytes(24);
      }
      if (url.startsWith("/api/jobs/p1/data")) return bytes(64);
      throw new Error(`unstubbed ${method} ${url}`);
    },
  );
}

function json(body: unknown) {
  return { ok: true, json: async () => body } as Response;
}

function bytes(n: number) {
  return { ok: true, arrayBuffer: async () => new ArrayBuffer(n) } as Response;
}

beforeEach(() => {
  vi.unstubAllGlobals();
  stubFetch();
  useAppStore.getState().setQuantumNumbers(1, 0, 0);
  useAppStore.getState().setSystem("h");
  useAppStore.getState().setView("cloud");
});

describe("loaders", () => {
  it("loads systems, state, radial and levels", async () => {
    const s = useAppStore.getState();
    await s.loadSystems();
    expect(useAppStore.getState().systems.map((x) => x.key)).toEqual(["h"]);
    await s.loadStateInfo();
    expect(useAppStore.getState().stateInfo?.energy.value).toBe(-0.5);
    await s.loadRadial();
    expect(useAppStore.getState().radial?.n).toBe(1);
    await s.loadLevels();
    const lv = useAppStore.getState().levels;
    if (lv === null || isScreenedLevels(lv)) throw new Error("expected hydrogenic levels");
    expect(lv.n_max).toBe(6);
  });

  it("samples a cloud through the job lifecycle", async () => {
    await useAppStore.getState().sample();
    const s = useAppStore.getState();
    expect(s.status).toBe("ready");
    expect(s.positions).toHaveLength(6);
    expect(s.density).toHaveLength(2);
    expect(s.phase).toHaveLength(2);
    expect(s.meta?.count).toBe(2);
  });

  it("loads a plane grid", async () => {
    await useAppStore.getState().loadPlane();
    const s = useAppStore.getState();
    expect(s.planeStatus).toBe("ready");
    expect(s.plane?.values).toHaveLength(16);
    expect(s.plane?.meta.resolution).toBe(4);
  });

  it("quantum changes invalidate derived data", async () => {
    await useAppStore.getState().sample();
    expect(useAppStore.getState().status).toBe("ready");
    useAppStore.getState().setQuantumNumbers(2, 1, 0);
    const s = useAppStore.getState();
    expect(s.positions).toBeNull();
    expect(s.meta).toBeNull();
    expect(s.status).toBe("idle");
    expect(s.n).toBe(2);
  });

  it("records engine failures instead of throwing", async () => {
    vi.stubGlobal("fetch", async () => {
      throw new Error("down");
    });
    await useAppStore.getState().sample();
    const s = useAppStore.getState();
    expect(s.status).toBe("error");
    expect(s.error).toContain("down");
  });

  it("loads the what-if lab and the ghost", async () => {
    const s = useAppStore.getState();
    await s.loadWhatIf();
    expect(useAppStore.getState().whatif?.report.altered).toBe(false);
    expect(useAppStore.getState().whatifStatus).toBe("ready");
    await s.loadGhost();
    expect(useAppStore.getState().ghost?.z).toBe(1);
    useAppStore.getState().setLabConst({ e: 2 });
    expect(useAppStore.getState().labConst.e).toBe(2);
    expect(useAppStore.getState().whatif).toBeNull();
  });

  it("loads force laws and tracks params", async () => {
    const s = useAppStore.getState();
    await s.loadForceLaw();
    expect(useAppStore.getState().forceLaw?.preset).toBe("powerlaw");
    expect(useAppStore.getState().forceStatus).toBe("ready");
    useAppStore.getState().setForcePreset("yukawa");
    expect(useAppStore.getState().forceParams).toEqual({ lambda: 3 });
    expect(useAppStore.getState().forceLaw).toBeNull();
    useAppStore.getState().setForceParam("lambda", 6);
    expect(useAppStore.getState().forceParams.lambda).toBe(6);
    useAppStore.getState().setForceL(1);
    expect(useAppStore.getState().forceL).toBe(1);
    useAppStore.getState().setForceExpr("-1/r");
    expect(useAppStore.getState().forceExpr).toBe("-1/r");
  });

  it("level-detail controls clear cached levels so stale physics never renders", async () => {
    const s = useAppStore.getState();
    await s.loadLevels();
    expect(useAppStore.getState().levels).not.toBeNull();
    useAppStore.getState().setFineStructure(true);
    expect(useAppStore.getState().levels).toBeNull();
    await s.loadLevels();
    useAppStore.getState().setDirac(true);
    expect(useAppStore.getState().levels).toBeNull();
    await s.loadLevels();
    useAppStore.getState().setBField(2);
    expect(useAppStore.getState().levels).toBeNull();
    await s.loadLevels();
    useAppStore.getState().setEField(30);
    expect(useAppStore.getState().levels).toBeNull();
    await s.loadLevels();
    useAppStore.getState().setHyperfine(true);
    expect(useAppStore.getState().levels).toBeNull();
    useAppStore.getState().setFineStructure(false);
    expect(useAppStore.getState().dirac).toBe(false);
  });

  it("loads the spectrum and re-asks when intensity toggles", async () => {
    const s = useAppStore.getState();
    await s.loadSpectrum();
    expect(useAppStore.getState().spectrum).not.toBeNull();
    useAppStore.getState().setIntensities(false);
    expect(useAppStore.getState().spectrum).toBeNull();
    expect(useAppStore.getState().intensities).toBe(false);
  });

  it("threads the level-detail controls into the levels URL", async () => {
    let seen = "";
    vi.stubGlobal("fetch", async (url: string) => {
      seen = url;
      return json({
        system: SYSTEM, n_max: 6, fine_structure: true, alpha: 1 / 137,
        dirac: true, b_field: 2, e_field: 30, hyperfine: true, fine: null, gross: [],
      });
    });
    useAppStore.setState({
      fineStructure: true, dirac: true, bField: 2, eField: 30, hyperfine: true,
    });
    await useAppStore.getState().loadLevels();
    expect(seen).toContain("fine_structure=true");
    expect(seen).toContain("dirac=true");
    expect(seen).toContain("b_field=2");
    expect(seen).toContain("e_field=30");
    expect(seen).toContain("hyperfine=true");
  });

  it("couples the counterfactual switches both ways and clears derived data", async () => {
    const s = useAppStore.getState();
    expect(s.exchange).toBe(true);
    expect(s.pauli).toBe(true);
    useAppStore.getState().setExchange(false);
    expect(useAppStore.getState().exchange).toBe(false);
    expect(useAppStore.getState().pauli).toBe(true);
    useAppStore.getState().setExchange(true);
    expect(useAppStore.getState().pauli).toBe(true);
    useAppStore.getState().setPauli(false);
    expect(useAppStore.getState().pauli).toBe(false);
    expect(useAppStore.getState().exchange).toBe(false);
    useAppStore.getState().setPauli(true);
    expect(useAppStore.getState().exchange).toBe(true);
  });

  it("threads model, config and the switches into the radial URL", async () => {
    let seen = "";
    vi.stubGlobal("fetch", async (url: string) => {
      seen = url;
      return json({ n: 2, l: 1, system: SYSTEM });
    });
    useAppStore.setState({
      model: "hf", config: "1s2 2s2 2p5 3s1", exchange: false, compare: true,
    });
    await useAppStore.getState().loadRadial();
    expect(seen).toContain("model=hf");
    expect(seen).toContain("config=1s2%202s2%202p5%203s1");
    expect(seen).toContain("exchange=false");
    expect(seen).toContain("compare=true");
  });

  it("threads the four fields into sample job bodies", async () => {
    let body = "";
    vi.stubGlobal("fetch", async (url: string, init?: { method?: string; body?: string }) => {
      if (url === "/api/jobs/sample") body = String(init?.body ?? "");
      return json({ id: "j1", status: "pending", progress: 0, error: null });
    });
    useAppStore.setState({ model: "hf", config: "1s2 2s1", exchange: false, pauli: true });
    await useAppStore.getState().sample().catch(() => undefined);
    const sent = JSON.parse(body === "" ? "{}" : body) as Record<string, unknown>;
    expect(sent["model"]).toBe("hf");
    expect(sent["config"]).toBe("1s2 2s1");
    expect(sent["exchange"]).toBe(false);
    expect(sent["pauli"]).toBe(true);
  });
});

describe("the Back button's landing pad", () => {
  function pretendLoaded() {
    useAppStore.setState({
      positions: new Float32Array(3),
      density: new Float32Array(1),
      phase: new Float32Array(1),
      stateInfo: {} as never,
      plane: {} as never,
      radial: {} as never,
      levels: {} as never,
      spectrum: {} as never,
      status: "ready",
    });
  }

  it("applies a URL as a whole state, not as a patch", () => {
    useAppStore.setState({ bField: 4, fineStructure: true, colorMode: "solid" });
    useAppStore.getState().applyUrl({ n: 3, l: 1, m: 0 });
    const s = useAppStore.getState();
    expect([s.n, s.l, s.m]).toEqual([3, 1, 0]);
    expect(s.bField).toBe(0);
    expect(s.fineStructure).toBe(false);
    expect(s.colorMode).toBe("density");
  });

  it("clears everything the previous place derived", () => {
    pretendLoaded();
    useAppStore.getState().applyUrl({ system: "mu-h" });
    const s = useAppStore.getState();
    expect(s.positions).toBeNull();
    expect(s.plane).toBeNull();
    expect(s.levels).toBeNull();
    expect(s.spectrum).toBeNull();
    expect(s.status).toBe("idle");
  });

  it("steps back into a tour, and out of one", () => {
    useAppStore.getState().applyUrl({ tour: FLAGSHIP_TOUR_ID, step: 2 });
    expect(useAppStore.getState().tourId).toBe(FLAGSHIP_TOUR_ID);
    expect(useAppStore.getState().stepIndex).toBe(2);
    expect(useAppStore.getState().savedState).not.toBeNull();
    useAppStore.getState().applyUrl({});
    expect(useAppStore.getState().tourId).toBeNull();
    expect(useAppStore.getState().savedState).toBeNull();
  });
});

describe("the default place", () => {
  it("opens on carbon", () => {
    expect(useAppStore.getInitialState().system).toBe("c");
    expect(useAppStore.getInitialState().n).toBe(6);
  });
});

describe("the counterfactual switches", () => {
  it("default to real physics, so no session lands in the counterfactual", () => {
    expect(useAppStore.getInitialState().exchange).toBe(true);
    expect(useAppStore.getInitialState().pauli).toBe(true);
  });

  it("setExchange drops the solve, which was for the other model", () => {
    useAppStore.setState({ hfLevels: { fake: true } as never, hfStatus: "ready" });
    useAppStore.getState().setExchange(false);
    const s = useAppStore.getState();
    expect(s.exchange).toBe(false);
    expect(s.hfLevels).toBeNull();
    expect(s.hfStatus).toBe("idle");
  });

  it("setPauli drops the solve and resets the configuration to the new ground rule", () => {
    useAppStore.setState({
      hfLevels: { fake: true } as never,
      hfStatus: "ready",
      config: "1s2 2s2 2p6",
    });
    useAppStore.getState().setPauli(false);
    const s = useAppStore.getState();
    expect(s.pauli).toBe(false);
    expect(s.exchange).toBe(false);
    expect(s.config).toBeNull();
    expect(s.hfLevels).toBeNull();
    expect(s.hfStatus).toBe("idle");
  });

  it("turning exchange back on restores the cap", () => {
    useAppStore.setState({ pauli: false, exchange: false });
    useAppStore.getState().setExchange(true);
    expect(useAppStore.getState().pauli).toBe(true);
  });

  it("never holds pauli off with exchange on", () => {
    const store = useAppStore.getState();
    for (const step of [
      () => store.setPauli(false),
      () => store.setExchange(false),
      () => store.setExchange(true),
      () => store.setPauli(false),
      () => store.setPauli(true),
    ]) {
      step();
      const s = useAppStore.getState();
      expect(!s.pauli && s.exchange).toBe(false);
    }
  });

  it("altered physics does not follow the user to the next atom", () => {
    useAppStore.setState({ exchange: false });
    useAppStore.getState().setSystem("ar");
    expect(useAppStore.getState().exchange).toBe(true);

    useAppStore.setState({ pauli: false, exchange: false });
    useAppStore.getState().setSystem("ar");
    expect(useAppStore.getState().pauli).toBe(true);
    expect(useAppStore.getState().exchange).toBe(true);
  });
});
