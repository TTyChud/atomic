import { scaleLinear, scaleLog } from "d3-scale";
import type { AbsorptionInfo, AbsorbingLineInfo, GrowthRegime } from "../api/types";
import { formatOffset, offsetAxis, offsetTicks, thinTicks } from "../lib/axis";
import { Notation } from "../lib/mathText";
import { plotHeight } from "../lib/plotSize";
import { Badge } from "./Badge";

export const REGIME_COLOR: Record<GrowthRegime, string> = {
  linear: "#4ade80",
  saturated: "#fbbf24",
  damping: "#7dd3fc",
};

export const REGIME_LABEL: Record<GrowthRegime, string> = {
  linear: "linear (slope 1)",
  saturated: "saturated (slope ≈ 0)",
  damping: "damping (slope ½)",
};

const SHAPE = { ratio: 0.368, min: 220, max: 300 };
const BAND_H = 50;
const M = { left: 62, right: 16, top: 18, bottom: 34 };
const BAND_BAR_H = 22;

export function transmissionPath(
  wavelengthNm: number[],
  transmission: number[],
  x: (v: number) => number,
  y: (v: number) => number,
): string {
  const parts: string[] = [];
  for (let i = 0; i < wavelengthNm.length; i++) {
    if (!(wavelengthNm[i] > 0)) continue;
    const cmd = parts.length === 0 ? "M" : "L";
    parts.push(
      `${cmd}${x(Math.log10(wavelengthNm[i])).toFixed(2)} ${y(transmission[i]).toFixed(2)}`,
    );
  }
  return parts.join(" ");
}

export function transmissionGrey(t: number): string {
  const v = Math.round(255 * Math.min(1, Math.max(0, t)));
  return `rgb(${v} ${v} ${v})`;
}

export function absorptionAxisMode(loNm: number, hiNm: number): "log" | "offset" {
  const centre = (loNm + hiNm) / 2;
  if (!(centre > 0)) return "offset";
  return (hiNm - loNm) / centre >= 0.05 ? "log" : "offset";
}

export interface BandColumn {
  x: number;
  deepest: number;
  mean: number;
}

export function bandColumns(
  logLambda: number[],
  transmission: number[],
  x: (logLambda: number) => number,
  x0: number,
  x1: number,
): BandColumn[] {
  const span = x1 - x0;
  if (!(span > 0) || logLambda.length === 0) return [];
  const n = Math.max(1, Math.round(span));
  const width = span / n;
  const sum = new Float64Array(n);
  const weight = new Float64Array(n);
  const deepest = new Float64Array(n).fill(Infinity);

  for (let i = 0; i < logLambda.length; i++) {
    const left = i === 0 ? logLambda[i] : (logLambda[i - 1] + logLambda[i]) / 2;
    const right =
      i === logLambda.length - 1
        ? logLambda[i]
        : (logLambda[i] + logLambda[i + 1]) / 2;
    const xa = Math.max(x0, Math.min(x(left), x(right)));
    const xb = Math.min(x1, Math.max(x(left), x(right)));
    if (!(xb > xa)) continue;
    const first = Math.min(n - 1, Math.max(0, Math.floor((xa - x0) / width)));
    const last = Math.min(n - 1, Math.max(0, Math.floor((xb - x0) / width)));
    for (let c = first; c <= last; c++) {
      const overlap =
        Math.min(xb, x0 + (c + 1) * width) - Math.max(xa, x0 + c * width);
      if (overlap <= 0) continue;
      sum[c] += transmission[i] * overlap;
      weight[c] += overlap;
      if (transmission[i] < deepest[c]) deepest[c] = transmission[i];
    }
  }

  const out: BandColumn[] = [];
  let carriedDeep = transmission[0];
  let carriedMean = transmission[0];
  for (let c = 0; c < n; c++) {
    if (weight[c] > 0) {
      carriedMean = sum[c] / weight[c];
      carriedDeep = deepest[c];
    }
    out.push({ x: x0 + c * width, deepest: carriedDeep, mean: carriedMean });
  }
  return out;
}

export function bandResolutionNote(drawn: number, mean: number): string {
  if (mean - drawn < 0.05) {
    return (
      "Here the columns are finer than the lines are wide, so a detector pixel"
      + " of the same width would record the same depth: the strip and the"
      + " curve agree."
    );
  }
  return (
    `A detector pixel that wide integrates the flux across it and would reach`
    + ` only ${(100 * mean).toFixed(1)}% of the continuum, against the`
    + ` ${(100 * drawn).toFixed(1)}% the curve reaches at full resolution. That`
    + ` gap is not a drawing error. It is why a low-resolution spectrum of this`
    + ` same gas looks almost blank, and it is exactly the dilution the`
    + ` equivalent width above is immune to.`
  );
}

export function saturationVerdict(saturation: number): string {
  if (saturation > 0.97) {
    return "every line is optically thin, so this spectrum is a faithful census: doubling the gas would double every depth";
  }
  if (saturation > 0.5) {
    return "the strongest lines are starting to saturate, so this spectrum already understates how much gas there is";
  }
  return "this spectrum is badly saturated: the black cores cannot absorb any more, so most of the gas is invisible to these lines";
}

export function AbsorptionView({
  abs,
  width: W,
  zoomed = false,
}: {
  abs: AbsorptionInfo;
  width: number;
  zoomed?: boolean;
}) {
  const H = plotHeight(W, SHAPE.ratio, SHAPE.min, SHAPE.max);
  const BAND_LABEL_Y = H + 14;
  const BAND_Y = H + 20;
  const logLambda = abs.wavelength_nm.map((v) => Math.log10(v));
  const lo = Math.min(...logLambda);
  const hi = Math.max(...logLambda);
  const x = scaleLinear([lo, hi], [M.left, W - M.right]);
  const y = scaleLinear([0, 1], [H - M.bottom, M.top]);
  const path = transmissionPath(abs.wavelength_nm, abs.transmission, x, y);

  const loNm = 10 ** lo;
  const hiNm = 10 ** hi;
  const mode = absorptionAxisMode(loNm, hiNm);
  const offset = offsetAxis(loNm, hiNm);
  const xTicks =
    mode === "log"
      ? thinTicks(
          scaleLog([loNm, hiNm], [M.left, W - M.right]).ticks(8),
          (v) => x(Math.log10(v)),
          34,
        )
      : offsetTicks(offset, loNm, hiNm);
  const xLabel = (v: number) =>
    mode === "log" ? `${v}` : formatOffset(offset, v);

  const band = bandColumns(logLambda, abs.transmission, x, M.left, W - M.right);
  const bandWidth = band.length > 1 ? band[1].x - band[0].x : 1;
  const deepestDrawn = band.reduce((m, c) => Math.min(m, c.deepest), 1);
  const deepestMean = band.reduce((m, c) => Math.min(m, c.mean), 1);

  const strongest = [...abs.lines].sort((a, b) => b.tau_centre - a.tau_centre);
  const present = [...new Set(abs.lines.map((d) => d.regime))];

  return (
    <>
      <div className="view-header">
        <span className="plot-title">
          Absorption: {abs.lines.length}{" "}
          {abs.lines.length === 1 ? "line" : "lines"} against a flat continuum{" "}
          <Badge provenance={abs.provenance} />
        </span>
        <span className="legend-inline">
          {present.map((r) => (
            <span key={r} style={{ color: REGIME_COLOR[r] }}>
              ▎{REGIME_LABEL[r]}
            </span>
          ))}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H + BAND_H}`} style={{ minWidth: W }} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom}
          className="axis"
        />
        <line x1={M.left} x2={M.left} y1={M.top} y2={H - M.bottom} className="axis" />
        {xTicks.map((v) => (
          <g key={v} transform={`translate(${x(Math.log10(v))},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {xLabel(v)}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t} transform={`translate(${M.left},${y(t)})`}>
            <line x2="-5" className="axis" />
            <text x="-9" dy="3" textAnchor="end" className="tick">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <line
          x1={M.left} x2={W - M.right} y1={y(1)} y2={y(1)} className="zero"
        />
        <path d={path} className="transmission-curve" />

        {band.map((c, i) => (
          <rect
            key={i}
            x={c.x}
            y={BAND_Y}
            width={bandWidth}
            height={BAND_BAR_H}
            fill={transmissionGrey(c.deepest)}
            shapeRendering="crispEdges"
          />
        ))}
        <text x={M.left} y={BAND_LABEL_Y} className="tick">
          the deepest absorption in each column, drawn as brightness
        </text>
        {mode === "log" ? (
          <text x={W - M.right} y={H - 4} textAnchor="end" className="tick">
            wavelength [nm, log]
          </text>
        ) : (
          <text
            x={(M.left + W - M.right) / 2}
            y={H - 4}
            textAnchor="middle"
            className="tick"
          >
            λ − {offset.centreNm.toFixed(offset.centreDecimals)} nm [{offset.unit}]
          </text>
        )}
        <text x={4} y={12} className="tick">
          I / I₀
        </text>
      </svg>

      <p className="caption">
        <strong>
          Equivalent width <Notation>{abs.equivalent_width_nm.toExponential(3)}</Notation> nm
        </strong>{" "}
        against <Notation>{abs.thin_limit_width_nm.toExponential(3)}</Notation> nm if nothing
        saturated or overlapped. That is{" "}
        <strong>{(100 * abs.saturation).toFixed(1)}%</strong> of what a naive
        sum predicts. Put plainly: {saturationVerdict(abs.saturation)}.
      </p>

      <p className="caption">
        The strip is {band.length} columns wide, and each one is drawn at the{" "}
        <em>deepest</em> transmission anywhere inside it. That is what makes a
        line narrower than a column findable at all.{" "}
        {bandResolutionNote(deepestDrawn, deepestMean)}
      </p>

      {!zoomed && (
        <p className="caption">
          Across the whole {abs.lines.length}-line range a single line is
          narrower than one pixel, so the plot above shows <em>where</em> the
          gas absorbs, not what any line looks like. The numbers below are the
          full-resolution answer either way. For a line's shape, turn on line
          profiles and <strong>click a line</strong> to zoom both panels to it.
        </p>
      )}

      <p className="caption">
        One column density covers the whole element,{" "}
        <Notation>{abs.column_density_m2.toExponential(2)}</Notation> m⁻², but each line absorbs
        using only the atoms in <em>its own</em> lower level. That is the whole
        reason the Lyman lines come out black while the Balmer lines are
        invisible in the same gas, and it is the one thing an emission spectrum
        cannot show.
      </p>

      <table className="line-table">
        <thead>
          <tr>
            <th>line</th>
            <th>λ [nm]</th>
            <th>f</th>
            <th>lower column [m⁻²]</th>
            <th>τ centre</th>
            <th>branch</th>
          </tr>
        </thead>
        <tbody>
          {strongest.slice(0, 12).map((d: AbsorbingLineInfo, i) => (
            <tr key={`${d.label}-${d.wavelength_nm}-${i}`}>
              <td>
                <Notation>{d.label}</Notation>
              </td>
              <td>{d.wavelength_nm.toFixed(3)}</td>
              <td>
                <Notation>{d.oscillator_strength.toExponential(2)}</Notation>
              </td>
              <td>
                <Notation>{d.lower_column_m2.toExponential(2)}</Notation>
              </td>
              <td>
                <Notation>{d.tau_centre.toExponential(2)}</Notation>
              </td>
              <td style={{ color: REGIME_COLOR[d.regime] }}>{d.regime}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {abs.lines.length > 12 && (
        <p className="caption">
          The 12 deepest of {abs.lines.length} lines, ordered by optical depth
          at line centre.
        </p>
      )}

      {abs.blends.length > 0 && (
        <p className="caption">
          <strong>
            {abs.blends.length} blended{" "}
            {abs.blends.length === 1 ? "pair" : "pairs"}
          </strong>
          :{" "}
          {abs.blends.map((b) => `${b[0]}/${b[1]}`).join(", ")}. Where lines
          overlap, their transmissions multiply rather than their absorptions
          adding, so two lines each removing 60% of the light remove 84%
          together and not 120%. The whole comes out less than the sum of the
          parts by construction, not by approximation.
        </p>
      )}

      <p className="caption">
        Grid closure {abs.flux_closure.toFixed(4)}: the summed optical depth
        integrates to this times the analytic total, on the grid actually used,
        measured rather than assumed. This models absorption only, so the lines
        darken the continuum and never re-emit into it. No core ever reverses.
      </p>
    </>
  );
}
