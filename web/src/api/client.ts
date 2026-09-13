import { apiUrl } from "../lib/apiBase";
import { currentEngineMode, engine } from "../engine/engine";
import type {
  AbsorptionInfo,
  ClassicalGhost,
  ConstantsReport,
  ForceLawResult,
  HFLevels,
  JobInfo,
  JobMeta,
  LevelsResponse,
  RadialResponse,
  ScreenedLevels,
  SpectrumResponse,
  StateResponse,
  SystemsResponse,
} from "./types";

export function isScreenedLevels(
  body: LevelsResponse | ScreenedLevels,
): body is ScreenedLevels {
  return "orbitals" in body;
}

export type Basis = "complex" | "real";
export type PlaneQuantity = "density" | "psi";

export function num(v: number): string {
  return encodeURIComponent(String(v));
}

export function key(v: string): string {
  return encodeURIComponent(v);
}

/**
 * Transport for every engine call.
 *
 * Same-origin localhost (dev, `atomic serve`, `vite preview`) routes to the
 * in-browser engine — the real Python package running in a Web Worker, no
 * server involved. Any other origin (a deployed UI) uses plain fetch against
 * the network API. Absolute URLs (split deploy) always go over the network.
 */
async function request(url: string, init?: RequestInit): Promise<Response> {
  if (isLocalRelative(url)) return engine.request(url, init);
  return fetch(url, init);
}

function isLocalRelative(url: string): boolean {
  if (/^https?:\/\//i.test(url)) return false;
  // currentEngineMode honors the ?engine=server|device override on top of the
  // origin policy, so the override switches the whole transport, not just the
  // badge. Absolute URLs always go over the network regardless.
  return currentEngineMode() === "local";
}

async function getJson<T>(url: string): Promise<T> {
  const res = await request(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

async function errorFrom(url: string, res: Response): Promise<Error> {
  let detail: string | null = null;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    detail = null;
  }
  return new Error(detail ?? `${url}: HTTP ${res.status}`);
}

/**
 * Fetch a binary payload through the standard transport and decode it.
 * One status-check-and-decode path for every octet-stream endpoint
 * (job channels, thumbnails), device engine and network alike.
 */
async function binary<T>(
  path: string,
  decode: (buffer: ArrayBuffer) => T,
): Promise<T> {
  const res = await request(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return decode(await res.arrayBuffer());
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await errorFrom(url, res);
  return res.json() as Promise<T>;
}

export function getSystems(): Promise<SystemsResponse> {
  return getJson(apiUrl("/api/systems"));
}

export function getState(
  n: number,
  l: number,
  m: number,
  system: string,
): Promise<StateResponse> {
  return getJson(apiUrl(`/api/state/${n}/${l}/${m}?system=${key(system)}`));
}

export function getRadial(
  n: number,
  l: number,
  system: string,
  points?: number,
  model: "gsz" | "hf" = "gsz",
  config?: string | null,
  exchange = true,
  pauli = true,
  compare = false,
): Promise<RadialResponse> {
  const p = points === undefined ? "" : `&points=${points}`;
  const m = model === "hf" ? "&model=hf" : "";
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const x = exchange ? "" : "&exchange=false";
  const pa = pauli ? "" : "&pauli=false";
  const co = compare ? "&compare=true" : "";
  return getJson(
    apiUrl(`/api/radial/${n}/${l}?system=${key(system)}${p}${m}${c}${x}${pa}${co}`),
  );
}

export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
  alpha?: number,
  config?: string | null,
  dirac = false,
  bField = 0,
  eField = 0,
  hyperfine = false,
): Promise<LevelsResponse | ScreenedLevels> {
  const a = alpha === undefined ? "" : `&alpha=${alpha}`;
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const d = dirac ? "&dirac=true" : "";
  const b = bField > 0 ? `&b_field=${bField}` : "";
  const e = eField > 0 ? `&e_field=${eField}` : "";
  const h = hyperfine ? "&hyperfine=true" : "";
  return getJson(
    apiUrl(
      `/api/levels?system=${key(system)}&n_max=${nMax}&fine_structure=${fineStructure}${a}${c}${d}${b}${e}${h}`,
    ),
  );
}

export interface ThermalParams {
  temperatureK: number;
  electronDensityCm3: number;
}

export interface ProfileParams {
  on: boolean;
  resolvingPower?: number | null;
  window?: [number, number] | null;
}

export function getSpectrum(
  system: string,
  nMax: number,
  fineStructure: boolean,
  intensities = false,
  thermal?: ThermalParams | null,
  profile?: ProfileParams | null,
): Promise<SpectrumResponse> {
  const t = thermal
    ? `&temperature_k=${num(thermal.temperatureK)}` +
      `&electron_density_cm3=${num(thermal.electronDensityCm3)}`
    : "";
  let p = "";
  if (profile?.on) {
    p = "&profile=true";
    if (profile.resolvingPower != null) {
      p += `&resolving_power=${num(profile.resolvingPower)}`;
    }
    if (profile.window) {
      p += `&lambda_min=${num(profile.window[0])}&lambda_max=${num(profile.window[1])}`;
    }
  }
  return getJson(
    apiUrl(
      `/api/spectrum?system=${key(system)}&n_max=${nMax}&fine_structure=${fineStructure}` +
        `&intensities=${intensities}${t}${p}`,
    ),
  );
}

export interface AbsorptionParams {
  system: string;
  nMax: number;
  fineStructure: boolean;
  columnDensityM2: number;
  thermal: ThermalParams;
  resolvingPower?: number | null;
  window?: [number, number] | null;
}

export function getAbsorption(p: AbsorptionParams): Promise<AbsorptionInfo> {
  const r = p.resolvingPower != null ? `&resolving_power=${num(p.resolvingPower)}` : "";
  const w = p.window
    ? `&lambda_min=${num(p.window[0])}&lambda_max=${num(p.window[1])}`
    : "";
  return getJson(
    apiUrl(
      `/api/absorption?system=${key(p.system)}&n_max=${p.nMax}` +
        `&fine_structure=${p.fineStructure}` +
        `&column_density_m2=${num(p.columnDensityM2)}` +
        `&temperature_k=${num(p.thermal.temperatureK)}` +
        `&electron_density_cm3=${num(p.thermal.electronDensityCm3)}${r}${w}`,
    ),
  );
}

export interface ConstMultipliers {
  hbar: number;
  e: number;
  m_e: number;
  eps0: number;
  c: number;
}

export function getConstants(m: ConstMultipliers): Promise<ConstantsReport> {
  return getJson(
    apiUrl(
      `/api/constants?hbar=${m.hbar}&e=${m.e}&m_e=${m.m_e}&eps0=${m.eps0}&c=${m.c}`,
    ),
  );
}

export function getClassical(system: string, n: number): Promise<ClassicalGhost> {
  return getJson(apiUrl(`/api/classical?system=${key(system)}&n=${n}`));
}

export interface ForceLawParams {
  system: string;
  preset: string;
  params: Record<string, number>;
  l: number;
  nStates?: number;
  expr?: string;
}

export function getForceLaw(p: ForceLawParams): Promise<ForceLawResult> {
  const q = new URLSearchParams({
    system: p.system,
    preset: p.preset,
    l: String(p.l),
    n_states: String(p.nStates ?? 4),
  });
  for (const [k, v] of Object.entries(p.params)) q.set(k, String(v));
  if (p.expr !== undefined) q.set("expr", p.expr);
  return getJson(apiUrl(`/api/forcelaw?${q.toString()}`));
}

export interface SampleParams {
  n: number;
  l: number;
  m: number;
  count: number;
  seed?: number;
  basis: Basis;
  system: string;
  model?: "gsz" | "hf";
  config?: string | null;
  exchange?: boolean;
  pauli?: boolean;
}

export function createSampleJob(params: SampleParams): Promise<JobInfo> {
  return postJson(apiUrl("/api/jobs/sample"), { seed: 0, ...params });
}

export interface HFParams {
  z: number;
  n_electrons?: number;
  config?: string;
  exchange?: boolean;
  pauli?: boolean;
}

export function createHFJob(params: HFParams): Promise<JobInfo> {
  return postJson(apiUrl("/api/jobs/hf"), params);
}

export function isHFLevels(meta: JobMeta): meta is HFLevels {
  return meta.kind === "hf";
}

export interface PlaneParams {
  n: number;
  l: number;
  m: number;
  quantity: PlaneQuantity;
  basis: Basis;
  system: string;
  resolution?: number;
  model?: "gsz" | "hf";
  config?: string | null;
  exchange?: boolean;
  pauli?: boolean;
}

export function createPlaneJob(params: PlaneParams): Promise<JobInfo> {
  return postJson(apiUrl("/api/jobs/plane"), { resolution: 256, ...params });
}

export interface IsoParams {
  n: number;
  l: number;
  m: number;
  fraction: number;
  basis: Basis;
  system: string;
  resolution?: number;
  model?: "gsz" | "hf";
  config?: string | null;
  exchange?: boolean;
  pauli?: boolean;
}

export function createIsoJob(params: IsoParams): Promise<JobInfo> {
  return postJson(apiUrl("/api/jobs/isosurface"), { resolution: 96, ...params });
}

export async function getIndexChannel(
  jobId: string,
  channel: string,
): Promise<Uint32Array> {
  return binary<Uint32Array>(
    `/api/jobs/${jobId}/data?channel=${channel}`,
    decodeIndices,
  );
}

export function thumbnailUrl(
  n: number,
  l: number,
  m: number,
  system: string,
  basis: Basis,
  size: number,
  params?: {
    model?: "gsz" | "hf";
    config?: string | null;
    exchange?: boolean;
    pauli?: boolean;
  },
): string {
  const model = params?.model === "hf" ? "&model=hf" : "";
  const cfg = params?.config ? `&config=${encodeURIComponent(params.config)}` : "";
  const x = params?.exchange === false ? "&exchange=false" : "";
  const p = params?.pauli === false ? "&pauli=false" : "";
  return apiUrl(
    `/api/thumbnail/${n}/${l}/${m}?system=${key(system)}&basis=${basis}&size=${size}` +
      `${model}${cfg}${x}${p}`,
  );
}

export function decodeIndices(buffer: ArrayBuffer): Uint32Array {
  if (buffer.byteLength % 12 !== 0) {
    throw new Error(
      `triangle byte length ${buffer.byteLength} is not a multiple of 12 (3 x uint32)`,
    );
  }
  return new Uint32Array(buffer);
}

export function getJobMeta(jobId: string): Promise<JobMeta> {
  return getJson(apiUrl(`/api/jobs/${jobId}/meta`));
}

export async function getChannel(jobId: string, channel?: string): Promise<Float32Array> {
  const path = channel
    ? `/api/jobs/${jobId}/data?channel=${channel}`
    : `/api/jobs/${jobId}/data`;
  return binary<Float32Array>(path, decodeFloats);
}

/** Fetch a thumbnail's bytes through the standard transport. */
export function fetchThumbnail(url: string): Promise<Blob> {
  return binary<Blob>(url, (buffer) => new Blob([buffer]));
}

export function decodeFloats(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`byte length ${buffer.byteLength} is not a multiple of 4 (float32)`);
  }
  return new Float32Array(buffer);
}

export function decodePositions(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 12 !== 0) {
    throw new Error(
      `positions byte length ${buffer.byteLength} is not a multiple of 12 (xyz float32)`,
    );
  }
  return new Float32Array(buffer);
}
