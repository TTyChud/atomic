
export interface ViewLead {
  title: string;
  lead: string;
  notice?: string;
}

export const VIEW_LEADS: Record<string, ViewLead> = {
  radial: {
    title: "Radial functions",
    lead:
      "Two ways of asking the same thing: how far from the nucleus is the " +
      "electron? One curve is the wavefunction's radial part. The other is the " +
      "chance of finding the electron at each distance.",
    notice:
      "The two curves disagree at r = 0, and that is not a mistake. Hover " +
      "either one to read its value. Count the wiggles while you are there: " +
      "you always get n − ℓ − 1 of them.",
  },
  levels: {
    title: "Energy levels",
    lead:
      "Every energy this atom is allowed to have, drawn as a ladder. The rungs " +
      "are the levels. The gaps between them are the colours of light it gives " +
      "off.",
    notice:
      "Pick a detail below the ladder to zoom in on what splits a single rung: " +
      "relativity, a magnetic field, an electric field, or the nucleus itself.",
  },
  spectrum: {
    title: "Spectrum, checked against NIST",
    lead:
      "Every colour this atom can emit. The wavelengths come out of the level " +
      "ladder, and they are drawn right next to what NIST actually measured.",
    notice:
      "The dots on the axis are the measurements, not the computation. The " +
      "agreement readout counts how many computed lines land inside the stated " +
      "tolerance.",
  },
  whatif: {
    title: "What if the constants were different?",
    lead:
      "Change ℏ, e, mₑ, ε₀ or c, then watch what moves. Usually nothing does, " +
      "because nothing anyone could actually measure has changed.",
    notice:
      "Try a scenario below. The lesson is in what refuses to budge: only " +
      "dimensionless ratios and fixed-ruler scales are observable at all.",
  },
  forcelaw: {
    title: "What if the force law were different?",
    lead:
      "Swap the −Z/r Coulomb pull for something else and the same Schrödinger " +
      "equation gets solved against it. What changes is the ladder.",
    notice:
      "Each potential sits next to the reference it should be judged against, " +
      "so you can see what the swap cost.",
  },
};

export function describeField(tesla: number): string {
  if (tesla <= 0) return "no field";
  if (tesla < 0.5) return "weaker than a lab electromagnet";
  if (tesla < 1.5) return "a strong lab electromagnet";
  if (tesla <= 3.5) return "about an MRI scanner";
  if (tesla <= 12) return "a research superconducting magnet";
  return "among the strongest steady fields built";
}

export function describeEField(mvPerM: number): string {
  if (mvPerM <= 0) return "no field";
  if (mvPerM < 3) return "below air's breakdown strength";
  if (mvPerM < 25) return "past air's breakdown; a vacuum or solid-state field";
  return "strong enough to strip hydrogen's outer shells";
}

export function describeTemperature(kelvin: number): string {
  if (kelvin <= 400) return "around room temperature";
  if (kelvin <= 2000) return "a flame";
  if (kelvin <= 7500) return "a Sun-like photosphere";
  if (kelvin <= 30000) return "a hot blue star";
  return "a stellar corona";
}

export function describeDensity(logNe: number): string {
  if (logNe <= 7) return "a nebula";
  if (logNe <= 11) return "a thin stellar atmosphere";
  if (logNe <= 14) return "a stellar photosphere";
  return "a dense laboratory plasma";
}

export function describeResolvingPower(logR: number | null): string {
  if (logR === null) return "no instrument: the line's own shape";
  if (logR < 3) return "a teaching spectrometer";
  if (logR < 4.5) return "a survey spectrograph";
  if (logR < 6) return "a high-resolution echelle";
  return "beyond what most spectrographs reach";
}

export interface ToleranceRow {
  within_tolerance: boolean;
}

export interface NistSummary {
  within: number;
  total: number;
  allWithin: boolean;
  headline: string;
}

export function nistSummary(
  rows: readonly ToleranceRow[] | null | undefined,
  tolerance: number | null | undefined,
): NistSummary | null {
  if (!rows || rows.length === 0) return null;
  const total = rows.length;
  const within = rows.filter((r) => r.within_tolerance).length;
  const allWithin = within === total;
  const tol =
    typeof tolerance === "number" ? ` (tolerance ${tolerance.toExponential(0)})` : "";
  const headline = allWithin
    ? `All ${total} computed line${total === 1 ? "" : "s"} agree with NIST${tol}.`
    : `${within} of ${total} computed lines agree with NIST${tol}; ` +
      `${total - within} sit${total - within === 1 ? "s" : ""} outside it.`;
  return { within, total, allWithin, headline };
}

export interface ScenarioMultipliers {
  hbar: number;
  e: number;
  m_e: number;
  eps0: number;
  c: number;
}

export interface Scenario {
  key: string;
  label: string;
  blurb: string;
  multipliers: ScenarioMultipliers;
}

const REAL: ScenarioMultipliers = { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 };

export const SCENARIOS: Scenario[] = [
  {
    key: "real",
    label: "Our universe",
    blurb: "Every constant at its measured value. This is the baseline the rest get compared to.",
    multipliers: REAL,
  },
  {
    key: "disguise",
    label: "The same universe in disguise",
    blurb:
      "Two constants scaled so that every exponent cancels out. Nothing " +
      "observable moves, and that is the whole lesson.",
    multipliers: { ...REAL, e: 2, eps0: 4 },
  },
  {
    key: "strong-em",
    label: "Stronger electromagnetism",
    blurb:
      "Charge doubled, so α goes up by four. Atoms shrink, binding jumps, and " +
      "the fine structure finally gets big enough to see.",
    multipliers: { ...REAL, e: 2 },
  },
  {
    key: "heavy-electron",
    label: "A heavier electron",
    blurb:
      "Mass doubled. α does not notice, but the atom halves in size and doubles " +
      "its binding. A muon does exactly this.",
    multipliers: { ...REAL, m_e: 2 },
  },
  {
    key: "broken",
    label: "Past the model's edge",
    blurb:
      "α pushed past 0.5, where perturbative fine structure stops meaning " +
      "anything. Rather than draw it anyway, the app says so.",
    multipliers: { hbar: 0.25, e: 2, m_e: 1, eps0: 0.5, c: 0.25 },
  },
];

export function activeScenario(current: ScenarioMultipliers): Scenario | null {
  return (
    SCENARIOS.find((s) =>
      (Object.keys(s.multipliers) as (keyof ScenarioMultipliers)[]).every(
        (k) => Math.abs(s.multipliers[k] - current[k]) < 1e-9,
      ),
    ) ?? null
  );
}

export const PRESET_BLURBS: Record<string, string> = {
  powerlaw:
    "Coulomb, with the exponent turned into a dial. At p = 1 you get hydrogen, " +
    "solved exactly. Move it and the accidental degeneracy that lets 2s and 2p " +
    "share an energy breaks straight away.",
  yukawa:
    "Coulomb, cut off past a range λ. That is the shape a screened charge " +
    "really has, and a short-ranged well holds only finitely many states, so " +
    "the top of the ladder stops existing.",
  harmonic:
    "A spring instead of a charge. Levels come out evenly spaced instead of " +
    "crowding toward the top, which is the clearest way to see that the 1/n² " +
    "pattern belongs to 1/r and to nothing else.",
  finitewell:
    "A flat-bottomed box, depth V₀ and width a. Make it shallow enough and it " +
    "binds nothing at all. A Coulomb well can never do that.",
  coulombcore:
    "Coulomb plus a repulsive core, the usual stand-in for inner electrons " +
    "holding an outer one away from the nucleus. It lifts the ℓ-degeneracy the " +
    "same way real atoms do.",
  custom:
    "Your own V(r), through the same finite-difference eigensolver every preset " +
    "uses. Every level it produces comes out labelled counterfactual.",
};

export const CONSTANT_BLURBS: Record<string, string> = {
  hbar: "the quantum of action: how grainy the world is",
  e: "the electron's charge: how hard the nucleus pulls",
  m_e: "the electron's mass: how hard it is to shift",
  eps0: "the vacuum's permittivity: how much empty space weakens the pull between charges",
  c: "the speed of light: where relativity starts to matter",
};
