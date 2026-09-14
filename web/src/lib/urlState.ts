import type { Basis, ConstMultipliers, PlaneQuantity } from "../api/client";
import type { NucleusMode } from "./nucleus";
import { N_MAX, clampState } from "./quantum";
import {
  DEFAULT_EXPR,
  PRESET_PARAMS,
  clampParam,
  defaultParams,
  validateExprClient,
  type ForcePreset,
} from "./forceLaw";
import { CONST_MAX, CONST_MIN, CONSTANT_KEYS, type ConstantKey } from "./whatif";

export type ViewMode =
  | "cloud" | "plane" | "radial" | "levels" | "spectrum" | "whatif" | "forcelaw";
export type ColorMode = "solid" | "density" | "phase";
export type AtomModel = "gsz" | "hf";
export type SurfaceMode = "cloud" | "surface" | "both";

export interface UrlState {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: Basis;
  view: ViewMode;
  colorMode: ColorMode;
  planeQuantity: PlaneQuantity;
  labConst: ConstMultipliers;
  labZ: number;
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
  forceL: number;
  forceExpr: string;
  fineStructure: boolean;
  dirac: boolean;
  bField: number;
  eField: number;
  hyperfine: boolean;
  intensities: boolean;
  thermal: boolean;
  temperatureK: number;
  logNe: number;
  profile: boolean;
  logResolvingPower: number | null;
  profileZoom: [number, number] | null;
  absorption: boolean;
  logColumn: number;
  nucleusMode: NucleusMode;
  surfaceMode: SurfaceMode;
  isoFraction: number;
  model: AtomModel;
  config: string | null;
  exchange: boolean;
  pauli: boolean;
  compare: boolean;
  tour: string | null;
  step: number;
  ghost: boolean;
}

export const URL_DEFAULTS: UrlState = {
  n: 6,
  l: 2,
  m: 0,
  system: "c",
  basis: "complex",
  view: "cloud",
  colorMode: "density",
  planeQuantity: "density",
  labConst: { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 },
  labZ: 1,
  forcePreset: "powerlaw",
  forceParams: defaultParams("powerlaw"),
  forceL: 0,
  forceExpr: DEFAULT_EXPR,
  fineStructure: false,
  dirac: false,
  bField: 0,
  eField: 0,
  hyperfine: false,
  intensities: true,
  thermal: false,
  temperatureK: 10000,
  logNe: 13,
  profile: false,
  logResolvingPower: null,
  profileZoom: null,
  absorption: false,
  logColumn: 20,
  nucleusMode: "marker",
  surfaceMode: "cloud",
  isoFraction: 0.9,
  model: "gsz",
  config: null,
  exchange: true,
  pauli: true,
  compare: false,
  tour: null,
  step: 0,
  ghost: false,
};

const VIEWS: ViewMode[] = [
  "cloud", "plane", "radial", "levels", "spectrum", "whatif", "forcelaw",
];
const COLORS: ColorMode[] = ["solid", "density", "phase"];
const BASES: Basis[] = ["complex", "real"];
const PLANES: PlaneQuantity[] = ["density", "psi"];
const SYSTEM_KEY = /^[a-z0-9+-]{1,16}$/;
const FORCE_PRESETS: ForcePreset[] = [
  "powerlaw",
  "yukawa",
  "harmonic",
  "finitewell",
  "coulombcore",
  "custom",
];
const MODELS: AtomModel[] = ["gsz", "hf"];
const NUCLEUS: NucleusMode[] = ["hidden", "true-scale", "marker"];
const SURFACES: SurfaceMode[] = ["cloud", "surface", "both"];
const CONFIG_RE = /^(\d[spdfgh]\d+)( \d[spdfgh]\d+)*$/;

const CONST_PARAMS: Record<ConstantKey, string> = {
  hbar: "hbar",
  e: "e",
  m_e: "me",
  eps0: "eps0",
  c: "c",
};

function pickEnum<T extends string>(raw: string | null, allowed: T[]): T | undefined {
  return allowed.includes(raw as T) ? (raw as T) : undefined;
}

function pickInt(raw: string | null): number | undefined {
  if (raw === null || !/^-?\d+$/.test(raw)) return undefined;
  return Number(raw);
}

function pickFloat(raw: string | null): number | undefined {
  if (raw === null || !/^-?\d*\.?\d+(e-?\d+)?$/i.test(raw)) return undefined;
  const v = Number(raw);
  return Number.isFinite(v) ? v : undefined;
}

export function currentUrlState(s: Omit<UrlState, "tour" | "step" | "ghost">): UrlState {
  return {
    n: s.n,
    l: s.l,
    m: s.m,
    system: s.system,
    basis: s.basis,
    view: s.view,
    colorMode: s.colorMode,
    planeQuantity: s.planeQuantity,
    labConst: s.labConst,
    labZ: s.labZ,
    forcePreset: s.forcePreset,
    forceParams: s.forceParams,
    forceL: s.forceL,
    forceExpr: s.forceExpr,
    fineStructure: s.fineStructure,
    dirac: s.dirac,
    bField: s.bField,
    eField: s.eField,
    hyperfine: s.hyperfine,
    intensities: s.intensities,
    thermal: s.thermal,
    temperatureK: s.temperatureK,
    logNe: s.logNe,
    profile: s.profile,
    logResolvingPower: s.logResolvingPower,
    profileZoom: s.profileZoom,
    absorption: s.absorption,
    logColumn: s.logColumn,
    nucleusMode: s.nucleusMode,
    surfaceMode: s.surfaceMode,
    isoFraction: s.isoFraction,
    model: s.model,
    config: s.config,
    exchange: s.exchange,
    pauli: s.pauli,
    compare: s.compare,
    tour: null,
    step: 0,
    ghost: false,
  };
}

export function parseAppUrl(search: string): Partial<UrlState> {
  const q = new URLSearchParams(search);
  const out: Partial<UrlState> = {};

  const n = pickInt(q.get("n"));
  const l = pickInt(q.get("l"));
  const m = pickInt(q.get("m"));
  if (n !== undefined || l !== undefined || m !== undefined) {
    const clamped = clampState(
      Math.min(n ?? URL_DEFAULTS.n, N_MAX),
      l ?? URL_DEFAULTS.l,
      m ?? URL_DEFAULTS.m,
    );
    out.n = clamped.n;
    out.l = clamped.l;
    out.m = clamped.m;
  }

  const system = q.get("system");
  if (system !== null && SYSTEM_KEY.test(system)) out.system = system;

  const basis = pickEnum(q.get("basis"), BASES);
  if (basis) out.basis = basis;
  const view = pickEnum(q.get("view"), VIEWS);
  if (view) out.view = view;
  let color = pickEnum(q.get("color"), COLORS);
  if (color === "phase" && (basis ?? URL_DEFAULTS.basis) === "real") color = "density";
  if (color) out.colorMode = color;
  const plane = pickEnum(q.get("plane"), PLANES);
  if (plane) out.planeQuantity = plane;

  const lc: Partial<ConstMultipliers> = {};
  for (const k of CONSTANT_KEYS) {
    const v = pickFloat(q.get(CONST_PARAMS[k]));
    if (v !== undefined && v > 0) lc[k] = Math.min(Math.max(v, CONST_MIN), CONST_MAX);
  }
  if (Object.keys(lc).length > 0) out.labConst = { ...URL_DEFAULTS.labConst, ...lc };

  const z = pickInt(q.get("z"));
  if (z !== undefined) out.labZ = Math.min(Math.max(z, 1), 10);

  const presetRaw = q.get("preset");
  const preset = pickEnum(presetRaw, FORCE_PRESETS) ?? "powerlaw";
  const params = defaultParams(preset);
  let sawForceParam = false;
  for (const spec of PRESET_PARAMS[preset]) {
    const v = pickFloat(q.get(spec.name));
    if (v !== undefined) {
      params[spec.name] = clampParam(spec, v);
      sawForceParam = true;
    }
  }
  if ((presetRaw !== null && pickEnum(presetRaw, FORCE_PRESETS) !== undefined) || sawForceParam) {
    out.forcePreset = preset;
    out.forceParams = params;
  }

  const fl = pickInt(q.get("fl"));
  if (fl !== undefined && fl >= 0) out.forceL = fl;

  if (out.forcePreset === "custom") {
    const rawExpr = q.get("expr");
    if (rawExpr !== null && validateExprClient(rawExpr) === null) out.forceExpr = rawExpr;
  }

  const fs = q.get("fs");
  if (fs === "1" || fs === "true") out.fineStructure = true;
  else if (fs === "0" || fs === "false") out.fineStructure = false;
  if (q.get("dirac") === "1") out.dirac = true;
  const b = Number(q.get("b"));
  if (Number.isFinite(b) && b > 0) out.bField = b;
  const ef = Number(q.get("ef"));
  if (Number.isFinite(ef) && ef > 0) out.eField = ef;
  if (q.get("hf") === "1") out.hyperfine = true;
  if (q.get("int") === "0") out.intensities = false;

  if (q.get("lte") === "1") out.thermal = true;
  const tk = pickFloat(q.get("tk"));
  if (tk !== undefined && tk >= 1e2 && tk <= 1e6) out.temperatureK = tk;
  const ne = pickFloat(q.get("ne"));
  if (ne !== undefined && ne >= 4 && ne <= 22) out.logNe = ne;

  if (q.get("prof") === "1") out.profile = true;
  const rp = pickFloat(q.get("rp"));
  if (rp !== undefined && rp >= 2 && rp <= 7) out.logResolvingPower = rp;
  const zoom = q.get("zoom");
  if (zoom) {
    const [lo, hi] = zoom.split(",").map(Number);
    if (Number.isFinite(lo) && Number.isFinite(hi) && lo > 0 && hi > lo) {
      out.profileZoom = [lo, hi];
    }
  }

  if (q.get("abs") === "1") out.absorption = true;
  const col = pickFloat(q.get("col"));
  if (col !== undefined && col >= 14 && col <= 26) out.logColumn = col;

  const nucleus = pickEnum(q.get("nucleus"), NUCLEUS);
  if (nucleus) out.nucleusMode = nucleus;
  const surf = pickEnum(q.get("surf"), SURFACES);
  if (surf) out.surfaceMode = surf;
  const isoFraction = pickFloat(q.get("iso"));
  if (isoFraction !== undefined && isoFraction > 0 && isoFraction < 1) {
    out.isoFraction = isoFraction;
  }

  const tour = q.get("tour");
  if (tour) {
    out.tour = tour;
    const step = pickInt(q.get("step"));
    out.step = step !== undefined && step >= 0 ? step : 0;
  }
  const ghost = q.get("ghost");
  if (ghost === "1" || ghost === "true") out.ghost = true;
  else if (ghost === "0" || ghost === "false") out.ghost = false;
  const model = pickEnum(q.get("model"), MODELS);
  if (model) out.model = model;
  const config = q.get("config");
  if (config !== null && CONFIG_RE.test(config)) out.config = config;
  if (q.get("nox") === "1") out.exchange = false;
  if (q.get("nopauli") === "1") {
    out.pauli = false;
    out.exchange = false;
  }
  if (q.get("compare") === "1") out.compare = true;

  return out;
}

export function serializeAppUrl(state: UrlState): string {
  const q = new URLSearchParams();
  if (state.n !== URL_DEFAULTS.n) q.set("n", String(state.n));
  if (state.l !== URL_DEFAULTS.l) q.set("l", String(state.l));
  if (state.m !== URL_DEFAULTS.m) q.set("m", String(state.m));
  if (state.system !== URL_DEFAULTS.system) q.set("system", state.system);
  if (state.basis !== URL_DEFAULTS.basis) q.set("basis", state.basis);
  if (state.view !== URL_DEFAULTS.view) q.set("view", state.view);
  if (state.colorMode !== URL_DEFAULTS.colorMode) q.set("color", state.colorMode);
  if (state.planeQuantity !== URL_DEFAULTS.planeQuantity) q.set("plane", state.planeQuantity);
  for (const k of CONSTANT_KEYS) {
    if (Math.abs(state.labConst[k] - URL_DEFAULTS.labConst[k]) > 1e-9) {
      q.set(CONST_PARAMS[k], String(state.labConst[k]));
    }
  }
  if (state.labZ !== URL_DEFAULTS.labZ) q.set("z", String(state.labZ));
  if (state.forcePreset !== URL_DEFAULTS.forcePreset) q.set("preset", state.forcePreset);
  for (const spec of PRESET_PARAMS[state.forcePreset]) {
    const v = state.forceParams[spec.name];
    if (v !== undefined && Math.abs(v - spec.default) > 1e-9) q.set(spec.name, String(v));
  }
  if (state.forceL !== URL_DEFAULTS.forceL) q.set("fl", String(state.forceL));
  if (state.forcePreset === "custom" && state.forceExpr !== URL_DEFAULTS.forceExpr) {
    q.set("expr", state.forceExpr);
  }
  if (state.fineStructure !== URL_DEFAULTS.fineStructure) q.set("fs", "1");
  if (state.dirac && state.fineStructure) q.set("dirac", "1");
  if (state.bField > 0 && state.fineStructure) q.set("b", String(state.bField));
  if (state.eField > 0) q.set("ef", String(state.eField));
  if (state.hyperfine) q.set("hf", "1");
  if (!state.intensities) q.set("int", "0");
  if (state.thermal) {
    q.set("lte", "1");
    if (state.temperatureK !== URL_DEFAULTS.temperatureK) {
      q.set("tk", String(state.temperatureK));
    }
    if (state.logNe !== URL_DEFAULTS.logNe) q.set("ne", String(state.logNe));
  }
  if (state.profile) {
    q.set("prof", "1");
    if (state.logResolvingPower !== null) {
      q.set("rp", String(state.logResolvingPower));
    }
    if (state.profileZoom) {
      q.set("zoom", `${state.profileZoom[0]},${state.profileZoom[1]}`);
    }
  }
  if (state.absorption) {
    q.set("abs", "1");
    if (state.logColumn !== URL_DEFAULTS.logColumn) {
      q.set("col", String(state.logColumn));
    }
  }
  if (state.nucleusMode !== URL_DEFAULTS.nucleusMode) q.set("nucleus", state.nucleusMode);
  if (state.surfaceMode !== URL_DEFAULTS.surfaceMode) q.set("surf", state.surfaceMode);
  if (state.surfaceMode !== "cloud" && state.isoFraction !== URL_DEFAULTS.isoFraction) {
    q.set("iso", String(state.isoFraction));
  }
  if (state.tour) {
    q.set("tour", state.tour);
    if (state.step !== URL_DEFAULTS.step) q.set("step", String(state.step));
  }
  if (state.ghost !== URL_DEFAULTS.ghost) q.set("ghost", "1");
  if (state.model !== URL_DEFAULTS.model) q.set("model", state.model);
  if (state.config) q.set("config", state.config);
  if (!state.exchange) q.set("nox", "1");
  if (!state.pauli) q.set("nopauli", "1");
  if (state.compare) q.set("compare", "1");
  const s = q.toString();
  return s ? `?${s}` : "";
}

const NAV_KEYS = [
  "n",
  "l",
  "m",
  "system",
  "view",
  "model",
  "config",
  "tour",
  "step",
] as const satisfies readonly (keyof UrlState)[];

export function isNewPlace(a: UrlState, b: UrlState): boolean {
  return NAV_KEYS.some((k) => a[k] !== b[k]);
}
