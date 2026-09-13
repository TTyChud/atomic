import { scaleLinear } from "d3-scale";

const UNITS = [
  { unit: "nm" as const, perNm: 1, centreDecimals: 2 },
  { unit: "pm" as const, perNm: 1e3, centreDecimals: 4 },
  { unit: "fm" as const, perNm: 1e6, centreDecimals: 6 },
];

export interface OffsetAxis {
  centreNm: number;
  perNm: number;
  unit: "nm" | "pm" | "fm";
  centreDecimals: number;
}

export function offsetAxis(lo: number, hi: number): OffsetAxis {
  const centreNm = (lo + hi) / 2;
  const span = Math.abs(hi - lo);
  const chosen = UNITS.find((u) => span * u.perNm >= 10) ?? UNITS[UNITS.length - 1];
  return { centreNm, ...chosen };
}

export function formatOffset(axis: OffsetAxis, wavelengthNm: number): string {
  const d = (wavelengthNm - axis.centreNm) * axis.perNm;
  const r = Number(d.toFixed(1));
  if (r === 0) return "0";
  return `${r > 0 ? "+" : ""}${r.toFixed(1)}`;
}

export function offsetTicks(
  axis: OffsetAxis,
  lo: number,
  hi: number,
  count = 6,
): number[] {
  const loOff = (lo - axis.centreNm) * axis.perNm;
  const hiOff = (hi - axis.centreNm) * axis.perNm;
  return scaleLinear([loOff, hiOff], [0, 1])
    .ticks(count)
    .map((o) => axis.centreNm + o / axis.perNm);
}

export function thinTicks(
  ticks: readonly number[],
  x: (v: number) => number,
  minGapPx: number,
): number[] {
  if (ticks.length <= 2) return [...ticks];
  const last = ticks[ticks.length - 1];
  const kept: number[] = [ticks[0]];
  let lastX = x(ticks[0]);
  for (let i = 1; i < ticks.length - 1; i++) {
    const px = x(ticks[i]);
    if (Math.abs(px - lastX) < minGapPx) continue;
    if (Math.abs(x(last) - px) < minGapPx) continue;
    kept.push(ticks[i]);
    lastX = px;
  }
  kept.push(last);
  return kept;
}
