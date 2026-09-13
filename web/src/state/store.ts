import { create } from "zustand";
import {
  createIsoJob,
  createPlaneJob,
  createSampleJob,
  createHFJob,
  getAbsorption,
  getChannel,
  getClassical,
  getConstants,
  getCurveOfGrowth,
  getForceLaw,
  getIndexChannel,
  getJobMeta,
  getLevels,
  getRadial,
  getSpectrum,
  getState,
  getSystems,
  isHFLevels,
  isScreenedLevels,
  type Basis,
  type ConstMultipliers,
  type PlaneQuantity,
} from "../api/client";
import type {
  AbsorptionInfo,
  ClassicalGhost,
  ConstantsReport,
  CurveOfGrowthInfo,
  ForceLawResult,
  HFLevels,
  IsoMeta,
  JobMeta,
  LevelsResponse,
  PlaneMeta,
  RadialResponse,
  SampleMeta,
  ScreenedLevels,
  SpectrumResponse,
  StateResponse,
  SystemInfo,
} from "../api/types";
import type { ForcePreset } from "../lib/forceLaw";
import { DEFAULT_EXPR, defaultParams } from "../lib/forceLaw";
import type { Snap } from "../lib/sheet";
import type { AtomModel, ColorMode, SurfaceMode, UrlState, ViewMode } from "../lib/urlState";
import { URL_DEFAULTS, currentUrlState } from "../lib/urlState";
import type { NucleusMode } from "../lib/nucleus";
import { isAlphaValid } from "../lib/whatif";
import { manyElectronParams, resolveModel } from "../lib/hfModel";
import { tourReset } from "../tours/apply";
import { tourById } from "../tours/registry";
import { readMemory, rememberCompleted, rememberDismissed, shouldInvite } from "../tours/seen";
import { clampStep, stepState } from "../tours/step";

export type LoadStatus = "idle" | "loading" | "ready" | "error";

export interface PlaneData {
  meta: PlaneMeta;
  values: Float32Array;
}

export interface IsoData {
  meta: IsoMeta;
  vertices: Float32Array;
  triangles: Uint32Array;
  phase: Float32Array;
}

export const ISO_FRACTIONS = [0.5, 0.75, 0.9, 0.95, 0.99] as const;

interface AppState {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: Basis;
  view: ViewMode;
  colorMode: ColorMode;
  planeQuantity: PlaneQuantity;
  count: number;
  sheet: Snap;
  systems: SystemInfo[];
  stateInfo: StateResponse | null;
  positions: Float32Array | null;
  density: Float32Array | null;
  phase: Float32Array | null;
  meta: SampleMeta | null;
  status: LoadStatus;
  progress: number;
  error: string | null;
  plane: PlaneData | null;
  planeStatus: LoadStatus;
  radial: RadialResponse | null;
  levels: LevelsResponse | ScreenedLevels | null;
  spectrum: SpectrumResponse | null;
  intensities: boolean;
  thermal: boolean;
  temperatureK: number;
  logNe: number;
  profile: boolean;
  logResolvingPower: number | null;
  profileZoom: [number, number] | null;
  showCurveOfGrowth: boolean;
  curveOfGrowth: CurveOfGrowthInfo | null;
  absorption: boolean;
  logColumn: number;
  absorptionData: AbsorptionInfo | null;
  labConst: ConstMultipliers;
  labZ: number;
  whatif: {
    report: ConstantsReport;
    real: LevelsResponse;
    altered: LevelsResponse | null;
  } | null;
  whatifStatus: LoadStatus;
  ghost: ClassicalGhost | null;
  ghostStatus: LoadStatus;
  ghostOn: boolean;
  setGhostOn: (ghostOn: boolean) => void;
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
  forceL: number;
  forceExpr: string;
  forceLaw: ForceLawResult | null;
  forceStatus: LoadStatus;
  fineStructure: boolean;
  dirac: boolean;
  bField: number;
  eField: number;
  hyperfine: boolean;
  model: AtomModel;
  setModel: (model: AtomModel) => void;
  hfLevels: HFLevels | null;
  hfStatus: LoadStatus;
  loadHF: () => Promise<void>;
  ensureHF: () => Promise<boolean>;
  config: string | null;
  setConfig: (config: string | null) => void;
  exchange: boolean;
  setExchange: (exchange: boolean) => void;
  pauli: boolean;
  setPauli: (pauli: boolean) => void;
  compare: boolean;
  setCompare: (compare: boolean) => void;
  setQuantumNumbers: (n: number, l: number, m: number) => void;
  setSystem: (system: string) => void;
  setBasis: (basis: Basis) => void;
  setView: (view: ViewMode) => void;
  setColorMode: (colorMode: ColorMode) => void;
  setPlaneQuantity: (planeQuantity: PlaneQuantity) => void;
  setCount: (count: number) => void;
  setSheet: (sheet: Snap) => void;
  loadSystems: () => Promise<void>;
  loadStateInfo: () => Promise<void>;
  sample: () => Promise<void>;
  loadPlane: () => Promise<void>;
  loadRadial: () => Promise<void>;
  loadLevels: () => Promise<void>;
  loadSpectrum: () => Promise<void>;
  setIntensities: (intensities: boolean) => void;
  setThermal: (thermal: boolean) => void;
  setTemperatureK: (temperatureK: number) => void;
  setLogNe: (logNe: number) => void;
  setProfile: (profile: boolean) => void;
  setLogResolvingPower: (logResolvingPower: number | null) => void;
  setProfileZoom: (profileZoom: [number, number] | null) => void;
  setShowCurveOfGrowth: (showCurveOfGrowth: boolean) => void;
  loadCurveOfGrowth: (lambdaNm: number) => Promise<void>;
  setAbsorption: (absorption: boolean) => void;
  setLogColumn: (logColumn: number) => void;
  loadAbsorption: () => Promise<void>;
  setLabConst: (partial: Partial<ConstMultipliers>) => void;
  setLabZ: (labZ: number) => void;
  loadWhatIf: () => Promise<void>;
  loadGhost: () => Promise<void>;
  setForcePreset: (preset: ForcePreset) => void;
  setForceParam: (name: string, value: number) => void;
  setForceL: (l: number) => void;
  setForceExpr: (expr: string) => void;
  loadForceLaw: () => Promise<void>;
  setFineStructure: (on: boolean) => void;
  setDirac: (on: boolean) => void;
  setBField: (b: number) => void;
  setEField: (e: number) => void;
  setHyperfine: (hyperfine: boolean) => void;
  nucleusMode: NucleusMode;
  setNucleusMode: (nucleusMode: NucleusMode) => void;
  surfaceMode: SurfaceMode;
  setSurfaceMode: (surfaceMode: SurfaceMode) => void;
  isoFraction: number;
  setIsoFraction: (isoFraction: number) => void;
  iso: IsoData | null;
  isoStatus: LoadStatus;
  isoProgress: number;
  loadIso: () => Promise<void>;
  tourId: string | null;
  stepIndex: number;
  savedState: UrlState | null;
  applyUrl: (patch: Partial<UrlState>) => void;
  startTour: (id: string, step?: number) => void;
  exitTour: () => void;
  finishTour: () => void;
  goToStep: (i: number) => void;
  inviteOpen: boolean;
  completedTours: string[];
  dismissInvite: () => void;
}

export const INVALIDATED = {
  stateInfo: null,
  positions: null,
  density: null,
  phase: null,
  meta: null,
  status: "idle",
  progress: 0,
  error: null,
  plane: null,
  planeStatus: "idle",
  iso: null,
  isoStatus: "idle",
  isoProgress: 0,
  radial: null,
  levels: null,
  spectrum: null,
  hfLevels: null,
  hfStatus: "idle",
  curveOfGrowth: null,
  absorptionData: null,
  profileZoom: null as [number, number] | null,
  ghost: null,
  ghostStatus: "idle",
} as const;

const startingMemory = readMemory();

let seedCounter = 1;

async function waitMeta(jobId: string, timeoutMs = 30000): Promise<JobMeta> {
  const start = Date.now();
  for (;;) {
    try {
      return await getJobMeta(jobId);
    } catch {
      if (Date.now() - start > timeoutMs) throw new Error("job timed out");
      await new Promise((r) => setTimeout(r, 200));
    }
  }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export const useAppStore = create<AppState>()((set, get) => ({
  n: 6,
  l: 0,
  m: 0,
  system: "c",
  basis: "complex",
  view: "cloud",
  colorMode: "solid",
  planeQuantity: "density",
  count: 100000,
  sheet: "collapsed",
  systems: [],
  stateInfo: null,
  positions: null,
  density: null,
  phase: null,
  meta: null,
  status: "idle",
  progress: 0,
  error: null,
  plane: null,
  planeStatus: "idle",
  radial: null,
  levels: null,
  spectrum: null,
  intensities: true,
  model: "gsz",
  hfLevels: null,
  hfStatus: "idle",
  config: null,
  exchange: true,
  pauli: true,
  compare: false,
  curveOfGrowth: null,
  absorptionData: null,
  temperatureK: 10000,
  logNe: 13,
  logResolvingPower: null,
  profileZoom: null,
  thermal: false,
  profile: false,
  showCurveOfGrowth: false,
  absorption: false,
  logColumn: 20,
  labConst: { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 },
  labZ: 1,
  whatif: null,
  whatifStatus: "idle",
  ghost: null,
  ghostStatus: "idle",
  ghostOn: false,
  forcePreset: "powerlaw",
  forceParams: defaultParams("powerlaw"),
  forceL: 0,
  forceExpr: DEFAULT_EXPR,
  forceLaw: null,
  forceStatus: "idle",
  fineStructure: false,
  dirac: false,
  bField: 0,
  eField: 0,
  hyperfine: false,
  nucleusMode: "marker",
  surfaceMode: "cloud",
  isoFraction: 0.9,
  iso: null,
  isoStatus: "idle",
  isoProgress: 0,
  tourId: null,
  stepIndex: 0,
  savedState: null,
  inviteOpen: shouldInvite(startingMemory),
  completedTours: startingMemory.completed,

  setQuantumNumbers: (n, l, m) => set({ n, l, m, ...INVALIDATED }),
  setSystem: (system) =>
    set({
      system,
      ...INVALIDATED,
      model: resolveModel(get().systems, system, get().model),
      config: null,
      exchange: true,
      pauli: true,
      compare: false,
    }),
  setBasis: (basis) => set({ basis, ...INVALIDATED }),
  setView: (view) => set({ view }),
  setColorMode: (colorMode) => set({ colorMode }),
  setPlaneQuantity: (planeQuantity) => set({ planeQuantity, plane: null, planeStatus: "idle" }),
  setCount: (count) => set({ count }),
  setGhostOn: (ghostOn) => set({ ghostOn }),
  setSheet: (sheet) => set({ sheet }),

  loadSystems: async () => {
    const systems = (await getSystems()).systems;
    set({ systems, model: resolveModel(systems, get().system, get().model) });
  },

  loadStateInfo: async () => {
    const s = get();
    try {
      set({ stateInfo: await getState(s.n, s.l, s.m, s.system) });
    } catch {
      set({ stateInfo: null });
    }
  },

  sample: async () => {
    const s = get();
    set({ status: "loading", progress: 0, error: null });
    try {
      const job = await createSampleJob({
        n: s.n,
        l: s.l,
        m: s.m,
        count: s.count,
        seed: seedCounter++ % 1000000,
        basis: s.basis,
        system: s.system,
        ...manyElectronParams(s),
      });
      const meta = await waitMeta(job.id);
      if (meta.kind !== "sample") throw new Error(`unexpected job kind ${(meta as JobMeta).kind}`);
      const positions = await getChannel(job.id);
      const density = await getChannel(job.id, "density");
      const phase = meta.channels.some((c) => c.name === "phase")
        ? await getChannel(job.id, "phase")
        : null;
      set({ positions, density, phase, meta, status: "ready", progress: 1 });
    } catch (e) {
      set({ status: "error", error: message(e) });
    }
  },

  loadPlane: async () => {
    const s = get();
    set({ planeStatus: "loading" });
    try {
      const job = await createPlaneJob({
        n: s.n,
        l: s.l,
        m: s.m,
        quantity: s.planeQuantity,
        basis: s.basis,
        system: s.system,
        resolution: 256,
        ...manyElectronParams(s),
      });
      const meta = await waitMeta(job.id);
      if (meta.kind !== "plane") throw new Error(`unexpected job kind ${(meta as JobMeta).kind}`);
      const values = await getChannel(job.id);
      set({ plane: { meta, values }, planeStatus: "ready" });
    } catch {
      set({ planeStatus: "error" });
    }
  },

  loadRadial: async () => {
    const s = get();
    const p = manyElectronParams(s);
    set({
      radial: await getRadial(
        s.n, s.l, s.system, 400, p.model, p.config, p.exchange, p.pauli, s.compare,
      ),
    });
  },

  loadLevels: async () => {
    const s = get();
    set({
      levels: await getLevels(
        s.system,
        6,
        s.fineStructure,
        undefined,
        s.config,
        s.dirac,
        s.bField,
        s.eField,
        s.hyperfine,
      ),
    });
  },

  loadSpectrum: async () => {
    const s = get();
    set({
      spectrum: await getSpectrum(
        s.system, 6, s.fineStructure, s.intensities,
        s.thermal ? { temperatureK: s.temperatureK, electronDensityCm3: 10 ** s.logNe } : null,
        {
          on: s.profile,
          resolvingPower: s.logResolvingPower === null ? null : 10 ** s.logResolvingPower,
          window: s.profileZoom,
        },
      ),
    });
  },

  setIntensities: (intensities) => set({ intensities, spectrum: null }),
  setThermal: (thermal) => set({ thermal, spectrum: null }),
  setTemperatureK: (temperatureK) => set({ temperatureK, spectrum: null }),
  setLogNe: (logNe) => set({ logNe, spectrum: null }),
  setProfile: (profile) => set({ profile, spectrum: null }),
  setLogResolvingPower: (logResolvingPower) => set({ logResolvingPower, spectrum: null }),
  setProfileZoom: (profileZoom) =>
    set({ profileZoom, spectrum: null, curveOfGrowth: null, absorptionData: null }),
  setShowCurveOfGrowth: (showCurveOfGrowth) => set({ showCurveOfGrowth }),
  setAbsorption: (absorption) => set({ absorption, absorptionData: null }),
  setLogColumn: (logColumn) => set({ logColumn, absorptionData: null }),

  setFineStructure: (fineStructure) => set({ fineStructure, dirac: false, levels: null }),
  setNucleusMode: (nucleusMode) => set({ nucleusMode }),
  setSurfaceMode: (surfaceMode) => set({ surfaceMode, iso: null, isoStatus: "idle", isoProgress: 0 }),
  setIsoFraction: (isoFraction) => set({ isoFraction, iso: null, isoStatus: "idle", isoProgress: 0 }),
  loadIso: async () => {
    if (get().isoStatus === "loading") return;
    if (!(await get().ensureHF())) return;
    const s = get();
    set({ isoStatus: "loading", isoProgress: 0, error: null });
    try {
      const job = await createIsoJob({
        n: s.n,
        l: s.l,
        m: s.m,
        fraction: s.isoFraction,
        basis: s.basis,
        system: s.system,
        ...manyElectronParams(s),
      });
      const meta = await waitMeta(job.id);
      if (meta.kind !== "isosurface") throw new Error(`unexpected job kind ${(meta as JobMeta).kind}`);
      const [vertices, triangles, phase] = await Promise.all([
        getChannel(job.id, "vertices"),
        getIndexChannel(job.id, "triangles"),
        getChannel(job.id, "phase"),
      ]);
      set({ iso: { meta, vertices, triangles, phase }, isoStatus: "ready", isoProgress: 1 });
    } catch (e) {
      set({ isoStatus: "error", error: message(e) });
    }
  },
  applyUrl: (patch) =>
    set((s) => {
      const next: UrlState = { ...URL_DEFAULTS, ...patch };
      return {
        ...tourReset(next, s.systems),
        tourId: next.tour,
        stepIndex: next.step,
        savedState: next.tour
          ? (s.savedState ?? { ...currentUrlState(s), ghost: s.ghostOn })
          : null,
      };
    }),
  startTour: (id, step = 0) => {
    const tour = tourById(id);
    if (!tour) return;
    rememberDismissed();
    set((s) => {
      const i = clampStep(tour, step);
      return {
        ...tourReset(stepState(tour.steps[i]), s.systems),
        tourId: id,
        stepIndex: i,
        inviteOpen: false,
        savedState: s.savedState ?? { ...currentUrlState(s), ghost: s.ghostOn },
      };
    });
  },
  goToStep: (i) =>
    set((s) => {
      const tour = s.tourId ? tourById(s.tourId) : null;
      if (!tour) return {};
      const next = clampStep(tour, i);
      return { ...tourReset(stepState(tour.steps[next]), s.systems), stepIndex: next };
    }),
  exitTour: () =>
    set((s) => ({
      ...(s.savedState ? tourReset(s.savedState, s.systems) : {}),
      tourId: null,
      stepIndex: 0,
      savedState: null,
    })),
  finishTour: () => {
    const id = get().tourId;
    if (id) set({ completedTours: rememberCompleted(id).completed });
    get().exitTour();
  },
  dismissInvite: () => {
    rememberDismissed();
    set({ inviteOpen: false });
  },
  setDirac: (dirac) => set({ dirac, levels: null }),
  setBField: (bField) => set({ bField, levels: null }),
  setEField: (eField) => set({ eField, levels: null }),
  setHyperfine: (hyperfine) => set({ hyperfine, levels: null }),
  setModel: (model) => set({ model, ...INVALIDATED }),
  setConfig: (config) => set({ config, ...INVALIDATED }),
  setExchange: (exchange) =>
    set((s) => ({
      exchange,
      pauli: exchange ? true : s.pauli,
      ...INVALIDATED,
    })),
  setPauli: (pauli) =>
    set({
      pauli,
      exchange: pauli,
      config: null,
      ...INVALIDATED,
    }),
  setCompare: (compare) => set({ compare, radial: null }),

  loadHF: async () => {
    const s = get();
    if (s.hfStatus === "loading") return;
    const info = s.systems.find((x) => x.key === s.system);
    if (info === undefined || info.kind !== "screened") {
      set({ hfLevels: null, hfStatus: "idle" });
      return;
    }
    set({ hfStatus: "loading" });
    try {
      const job = await createHFJob({
        z: info.z,
        config: s.config ?? undefined,
        exchange: s.exchange,
        pauli: s.pauli,
      });
      const meta = await waitMeta(job.id);
      set({ hfLevels: isHFLevels(meta) ? meta : null, hfStatus: "ready" });
    } catch {
      set({ hfLevels: null, hfStatus: "error" });
    }
  },

  ensureHF: async () => {
    if (get().systems.length === 0) await get().loadSystems();
    if (get().model !== "hf") return true;
    if (get().hfLevels !== null) return true;
    await get().loadHF();
    return get().hfLevels !== null;
  },

  setLabConst: (partial) => {
    const labConst = { ...get().labConst, ...partial };
    set({ labConst, whatif: null, whatifStatus: "idle" });
  },

  setLabZ: (labZ) => set({ labZ, whatif: null, whatifStatus: "idle" }),

  loadWhatIf: async () => {
    const { labConst, labZ } = get();
    const sys = `z${labZ}`;
    set({ whatifStatus: "loading", error: null });
    try {
      const report = await getConstants(labConst);
      const alpha = report.alpha.quantity.value;
      const real = await getLevels(sys, 6, true);
      if (isScreenedLevels(real)) throw new Error("what-if expects hydrogenic levels");
      const alteredRaw =
        report.altered && isAlphaValid(alpha)
          ? await getLevels(sys, 6, true, alpha)
          : null;
      if (alteredRaw !== null && isScreenedLevels(alteredRaw)) {
        throw new Error("what-if expects hydrogenic levels");
      }
      set({ whatif: { report, real, altered: alteredRaw }, whatifStatus: "ready" });
    } catch (e) {
      set({ whatifStatus: "error", error: message(e) });
    }
  },

  loadGhost: async () => {
    const s = get();
    set({ ghostStatus: "loading" });
    try {
      set({ ghost: await getClassical(s.system, s.n), ghostStatus: "ready" });
    } catch {
      set({ ghostStatus: "error" });
    }
  },

  setForcePreset: (forcePreset) =>
    set({ forcePreset, forceParams: defaultParams(forcePreset), forceLaw: null, forceStatus: "idle" }),
  setForceParam: (name, value) =>
    set((s) => ({ forceParams: { ...s.forceParams, [name]: value } })),
  setForceL: (forceL) => set({ forceL }),
  setForceExpr: (forceExpr) => set({ forceExpr }),

  loadForceLaw: async () => {
    const s = get();
    set({ forceStatus: "loading" });
    try {
      const forceLaw = await getForceLaw({
        system: s.system,
        preset: s.forcePreset,
        params: s.forceParams,
        l: s.forceL,
        expr: s.forcePreset === "custom" ? s.forceExpr : undefined,
      });
      set({ forceLaw, forceStatus: "ready" });
    } catch {
      set({ forceStatus: "error" });
    }
  },

  loadCurveOfGrowth: async (lambdaNm) => {
    const { system, fineStructure, temperatureK, logNe, logResolvingPower } = get();
    set({
      curveOfGrowth: await getCurveOfGrowth({
        system,
        nMax: 6,
        fineStructure,
        lambdaNm,
        thermal: { temperatureK, electronDensityCm3: 10 ** logNe },
        resolvingPower: logResolvingPower === null ? null : 10 ** logResolvingPower,
      }),
    });
  },

  loadAbsorption: async () => {
    const {
      system, fineStructure, temperatureK, logNe, logResolvingPower,
      logColumn, profileZoom,
    } = get();
    set({
      absorptionData: await getAbsorption({
        system,
        nMax: 6,
        fineStructure,
        columnDensityM2: 10 ** logColumn,
        thermal: { temperatureK, electronDensityCm3: 10 ** logNe },
        resolvingPower: logResolvingPower === null ? null : 10 ** logResolvingPower,
        window: profileZoom,
      }),
    });
  },
}));
