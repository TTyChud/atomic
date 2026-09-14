import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAppStore } from "./store";

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

const PRE_META = {
  kind: "sample",
  count: 4,
  dtype: "float32",
  layout: "xyz-interleaved",
  unit: "bohr",
  n: 6,
  l: 2,
  m: 0,
  basis: "complex",
  system: "c",
  model: "screened",
  provenance: null,
  channels: [
    { name: "positions", dtype: "float32", unit: "bohr", provenance: null },
    { name: "density", dtype: "float32", unit: "bohr^-3", provenance: null },
    { name: "phase", dtype: "float32", unit: "rad", provenance: null },
  ],
};

function jsonResponse(body: unknown, ok = true) {
  return {
    ok,
    json: async () => body,
    arrayBuffer: async () => new ArrayBuffer(0),
  } as unknown as Response;
}

function binResponse(n: number) {
  return {
    ok: true,
    arrayBuffer: async () => new ArrayBuffer(n),
  } as Response;
}

function stubJobFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: { method?: string }) => {
      const method = init?.method ?? "GET";
      if (url === "/api/systems") return jsonResponse({ systems: [SYSTEM] });
      if (url === "/api/jobs/sample" && method === "POST") {
        return jsonResponse({ id: "j1", status: "pending", progress: 0, error: null });
      }
      if (url === "/api/jobs/j1/meta") return jsonResponse(SAMPLE_META);
      if (url.startsWith("/api/jobs/j1/data")) {
        if (url.includes("density") || url.includes("phase")) return binResponse(8);
        return binResponse(24);
      }
      if (url.startsWith("/prerender/default/")) return jsonResponse(null, false);
      throw new Error(`unstubbed ${method} ${url}`);
    }),
  );
}

describe("prerendered default fast path", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    useAppStore.setState({
      n: 6, l: 2, m: 0, system: "c", basis: "complex", count: 100000,
      positions: null, density: null, phase: null, meta: null,
      status: "idle", progress: 0, error: null,
    });
  });

  it("loads the prerendered default without touching the job API", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("meta.json")) return jsonResponse(PRE_META);
      if (url.endsWith("positions.bin")) return binResponse(48);
      if (url.endsWith("density.bin")) return binResponse(16);
      if (url.endsWith("phase.bin")) return binResponse(16);
      throw new Error(`unstubbed ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    await useAppStore.getState().sample();
    const s = useAppStore.getState();
    expect(s.status).toBe("ready");
    expect(s.meta?.model).toBe("screened");
    expect(s.positions).toHaveLength(12);
    expect(fetchMock).toHaveBeenCalledWith("/prerender/default/meta.json");
    expect(fetchMock).not.toHaveBeenCalledWith("/api/jobs/sample", expect.anything());
  });

  it("falls back to the engine when prerendered assets are missing", async () => {
    stubJobFetch();
    await useAppStore.getState().sample();
    const s = useAppStore.getState();
    expect(s.status).toBe("ready");
    expect(s.meta?.count).toBe(2);
  });

  it("does not use the prerender path for other states", async () => {
    stubJobFetch();
    useAppStore.getState().setQuantumNumbers(3, 1, 0);
    await useAppStore.getState().sample();
    const s = useAppStore.getState();
    expect(s.status).toBe("ready");
    expect(s.meta?.count).toBe(2);
  });
});
