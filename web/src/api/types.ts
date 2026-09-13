export type Fidelity =
  | "exact"
  | "numerical"
  | "approximation"
  | "counterfactual"
  | "visual_liberty";

export interface Provenance {
  fidelity: Fidelity;
  method: string;
  assumptions: string[];
  error_estimate: number | null;
  refinement: string | null;
}

export interface Quantity {
  value: number;
  unit: string;
  label: string;
  provenance: Provenance;
}

export interface FieldData {
  values: number[];
  grid: number[];
  unit: string;
  grid_unit: string;
  label: string;
  provenance: Provenance;
}

export interface SystemInfo {
  key: string;
  name: string;
  z: number;
  mu_ratio: Quantity;
  m_over_m_nucleus: number;
  description: string;
  nuclear_radius: Quantity | null;
  nuclear_radius_fm: Quantity | null;
  kind: "hydrogenic" | "screened";
  n_electrons: number | null;
  has_gsz: boolean;
}

export interface StateResponse {
  n: number;
  l: number;
  m: number;
  system: SystemInfo;
  energy: Quantity;
  energy_ev: Quantity;
  mean_radius: Quantity;
  mean_radius_pm: Quantity;
  angular_momentum: Quantity;
  radial_nodes: number;
  angular_nodes: number;
}

export interface SystemsResponse {
  systems: SystemInfo[];
}

export interface StarkSublevel {
  n1: number;
  n2: number;
  m: number;
  k: number;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface GrossLevel {
  n: number;
  degeneracy: number;
  energy: Quantity;
  energy_ev: Quantity;
  sublevels?: StarkSublevel[] | null;
}

export interface ZeemanSublevel {
  m_j: number;
  branch: string;
  j_label: number;
  high_field_label: string;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface FineLevel {
  n: number;
  l: number;
  j: number;
  energy: Quantity;
  energy_ev: Quantity;
  shift: Quantity;
  shift_ev: Quantity;
  sublevels?: ZeemanSublevel[] | null;
}

export interface HyperfineLevel {
  F: number;
  energy: Quantity;
  energy_ev: Quantity;
  shift: Quantity;
  shift_ev: Quantity;
}

export interface HyperfineShell {
  n: number;
  available: boolean;
  nucleus?: string | null;
  I?: number | null;
  A?: Quantity | null;
  A_ev?: Quantity | null;
  levels: HyperfineLevel[];
  note?: string | null;
  reason?: string | null;
}

export interface LevelsResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  alpha: number;
  gross: GrossLevel[];
  fine: FineLevel[] | null;
  dirac: boolean;
  b_field: number;
  e_field: number;
  hyperfine: boolean;
  hyperfine_shells?: HyperfineShell[] | null;
}

export interface SpectralLineInfo {
  n_upper: number;
  l_upper: number;
  j_upper: number | null;
  n_lower: number;
  l_lower: number;
  j_lower: number | null;
  energy_ev: Quantity;
  wavelength_nm: Quantity;
  einstein_a_s: Quantity | null;
  oscillator_strength: Quantity | null;
  emissivity: Quantity | null;
}

export interface ThermalInfo {
  temperature_k: number;
  electron_density_cm3: number;
  ionized_fraction: Quantity;
  partition_function: Quantity;
}

export interface LineWidthInfo {
  label: string;
  wavelength_nm: number;
  n_upper: number;
  n_lower: number;
  sigma_nm: number;
  gamma_nm: number;
  fwhm_nm: number;
  terms: string[];
}

export interface ProfileInfo {
  wavelength_nm: number[];
  intensity: number[];
  unit: string;
  weight_kind: "emissivity" | "rate" | "uniform";
  resolving_power: number | null;
  flux_closure: number;
  widths: LineWidthInfo[];
  stark_span_nm: Quantity | null;
  stark_note: string | null;
  provenance: Provenance;
}

export type GrowthRegime = "linear" | "saturated" | "damping";

export interface AbsorbingLineInfo {
  wavelength_nm: number;
  label: string;
  oscillator_strength: number;
  lower_column_m2: number;
  tau_centre: number;
  regime: GrowthRegime;
  thin_width_nm: number;
  fwhm_nm: number;
}

export interface AbsorptionInfo {
  wavelength_nm: number[];
  transmission: number[];
  optical_depth: number[];
  lines: AbsorbingLineInfo[];
  thermal: ThermalInfo | null;
  column_density_m2: number;
  equivalent_width_nm: number;
  thin_limit_width_nm: number;
  saturation: number;
  blends: [string, string][];
  flux_closure: number;
  provenance: Provenance;
  column_provenance: Provenance;
}

export interface ComparisonInfo {
  wavelength_nm: number;
  reference_nm: number;
  reference_uncertainty_nm: number | null;
  delta_nm: number;
  relative_error: number;
  within_tolerance: boolean;
}

export interface SpectrumResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  lines: SpectralLineInfo[];
  comparison: ComparisonInfo[] | null;
  reference_citation: string | null;
  tolerance_relative: number | null;
  intensity_note: string | null;
  thermal: ThermalInfo | null;
  profile: ProfileInfo | null;
  profile_note: string | null;
}

export interface ScreenedOrbital {
  n: number;
  l: number;
  label: string;
  occupancy: number;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface ScreenedLevels {
  system: SystemInfo;
  config: string;
  is_ground: boolean;
  orbitals: ScreenedOrbital[];
  total_energy: Quantity;
  total_energy_ev: Quantity;
}

export interface RadialResponse {
  n: number;
  l: number;
  system: SystemInfo;
  r_wavefunction: FieldData;
  radial_probability: FieldData;
  total_density?: FieldData | null;
  density_comparison?: DensityComparison | null;
}

export interface ShellPeak {
  label: string;
  gsz_radius: number | null;
  hf_radius: number | null;
  gsz_depth: number | null;
  hf_depth: number | null;
}

export interface DensityComparison {
  gsz: FieldData;
  hf: FieldData;
  displaced_charge: Quantity;
  shells: ShellPeak[];
  provenance: Provenance;
}

export type JobStatus = "pending" | "running" | "done" | "error";

export interface JobInfo {
  id: string;
  status: JobStatus;
  progress: number;
  error: string | null;
}

export interface ChannelInfo {
  name: string;
  dtype: string;
  unit: string;
  provenance: Provenance;
}

export interface SampleMeta {
  kind: "sample";
  count: number;
  dtype: string;
  layout: string;
  unit: string;
  n: number;
  l: number;
  m: number;
  basis: string;
  system: string;
  model: string;
  provenance: Provenance;
  channels: ChannelInfo[];
}

export interface PlaneMeta {
  kind: "plane";
  resolution: number;
  dtype: string;
  layout: string;
  quantity: "density" | "psi";
  unit: string;
  label: string;
  half_extent: number;
  axis_unit: string;
  n: number;
  l: number;
  m: number;
  basis: string;
  system: string;
  model: string;
  provenance: Provenance;
}

export interface IsoMeta {
  kind: "isosurface";
  vertex_count: number;
  triangle_count: number;
  channels: ChannelInfo[];
  target_fraction: number;
  enclosed_fraction: Quantity;
  outside_fraction: number;
  level: Quantity;
  escaped_fraction: Quantity;
  mesh_volume: Quantity;
  voxel_volume: Quantity;
  area: Quantity;
  components: number;
  half_width: number;
  resolution: number;
  axis_unit: string;
  n: number;
  l: number;
  m: number;
  basis: string;
  system: string;
  model: string;
  label: string;
  provenance: Provenance;
}

export interface HFOrbital {
  n: number;
  l: number;
  label: string;
  occupancy: number;
  energy: Quantity;
  energy_ev: Quantity;
  channel: string;
}

export interface PauliCollapse {
  binding_change: Quantity;
  binding_change_ev: Quantity;
  real_total_energy: Quantity;
  real_total_energy_ev: Quantity;
  real_config: string;
  real_radius: Quantity;
  collapsed_radius: Quantity;
  radius_ratio: Quantity;
  variational_zeta: Quantity;
  variational_energy: Quantity;
  variational_energy_ev: Quantity;
}

export interface ManyElectronParams {
  model: "gsz" | "hf";
  config: string | null;
  exchange: boolean;
  pauli: boolean;
}

export interface HFLevels {
  kind: "hf";
  z: number;
  n_electrons: number;
  symbol: string | null;
  config: string;
  is_ground: boolean;
  exchange: boolean;
  exchange_energy: Quantity | null;
  exchange_energy_ev: Quantity | null;
  pauli: boolean;
  collapse: PauliCollapse | null;
  orbitals: HFOrbital[];
  total_energy: Quantity;
  total_energy_ev: Quantity;
  kinetic: Quantity;
  potential: Quantity;
  virial_ratio: Quantity;
  iterations: number;
  coarse_iterations: number;
  converged: boolean;
  provenance: Provenance;
  grid_channel: string;
  grid_points: number;
  channels: ChannelInfo[];
}

export type JobMeta = SampleMeta | PlaneMeta | IsoMeta | HFLevels;

export interface DerivedObservable {
  quantity: Quantity;
  ratio: number;
  changed: boolean;
}

export interface ConstantsReport {
  alpha: DerivedObservable;
  bohr_radius_pm: DerivedObservable;
  hartree_ev: DerivedObservable;
  altered: boolean;
}

export interface BohrOrbit {
  n: number;
  radius_bohr: Quantity;
  radius_pm: Quantity;
}

export interface ClassicalGhost {
  n: number;
  system_key: string;
  z: number;
  orbits: BohrOrbit[];
  r0_bohr: Quantity;
  collapse_time_s: Quantity;
  orbital_period_s: Quantity;
  orbit_count: Quantity;
}

export interface ForceLawLevel {
  radial_index: number;
  energy: Quantity;
  energy_ev: Quantity;
  trusted: boolean;
}

export interface ReferenceItem {
  label: string;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface Reference {
  kind: "levels" | "markers";
  items: ReferenceItem[];
}

export interface PotentialCurve {
  r: number[];
  v_ev: number[];
  provenance: Provenance;
}

export interface ForceLawResult {
  preset: string;
  params: Record<string, number>;
  l: number;
  z: number;
  system: SystemInfo;
  counterfactual: ForceLawLevel[];
  bound_count: number;
  requested_count: number;
  reference: Reference;
  potential_curve: PotentialCurve;
  expression: string | null;
}
