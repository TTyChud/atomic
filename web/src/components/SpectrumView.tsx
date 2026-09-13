import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect, useState } from "react";
import type { LineWidthInfo, ProfileInfo, SpectralLineInfo } from "../api/types";
import { formatOffset, offsetAxis, offsetTicks, thinTicks } from "../lib/axis";
import {
  describeDensity,
  describeResolvingPower,
  describeTemperature,
  nistSummary,
  VIEW_LEADS,
} from "../lib/explain";
import { withinPlot } from "../lib/hover";
import {
  PROFILE_DECADES,
  SPECTRUM_EMISSIVITY_LIBERTY,
  SPECTRUM_INTENSITY_LIBERTY,
  SPECTRUM_PROFILE_LIBERTY,
} from "../lib/liberties";
import { seriesColor, seriesName } from "../lib/spectrum";
import { plotHeight, usePlotWidth } from "../lib/plotSize";
import { useAppStore } from "../state/store";
import { Notation, mathTspans } from "../lib/mathText";
import { Badge } from "./Badge";
import { AbsorptionView } from "./AbsorptionView";
import { Disclosure } from "./Disclosure";
import { ControlGroup, Slider, Toggle } from "./Field";
import { HoverReadout, usePlotHover } from "./PlotHover";
import { usePlotZoom, ZoomControls } from "./PlotZoom";
import { ViewIntro } from "./ViewIntro";

const SHAPE = {
  floor: 420,
  lines: { ratio: 0.303, min: 190, max: 260 },
  residual: { ratio: 0.247, min: 150, max: 210 },
  zoom: { ratio: 0.309, min: 190, max: 260 },
};
const M = { left: 56, right: 16 };

const TOP = 28;

export type BarQuantity = "rate" | "emissivity";

const PICK: Record<BarQuantity, (ln: SpectralLineInfo) => number | undefined> = {
  rate: (ln) => ln.einstein_a_s?.value,
  emissivity: (ln) => ln.emissivity?.value,
};

export function intensityScale(
  lines: SpectralLineInfo[],
  quantity: BarQuantity = "rate",
) {
  const pick = PICK[quantity];
  const values = lines
    .map(pick)
    .filter((a): a is number => typeof a === "number" && a > 0);
  if (values.length === 0) return null;
  const lo = Math.log10(Math.min(...values));
  const hi = Math.log10(Math.max(...values));
  const span = hi - lo;
  return {
    lo,
    hi,
    quantity,
    value: pick,
    t: (a: number | undefined) =>
      typeof a !== "number" || a <= 0 || span <= 0
        ? span <= 0
          ? 1
          : 0
        : (Math.log10(a) - lo) / span,
  };
}

export function profileScale(intensity: number[], decades = PROFILE_DECADES) {
  let max = 0;
  for (const v of intensity) if (v > max) max = v;
  if (!(max > 0)) return null;
  const hi = Math.log10(max);
  const lo = hi - decades;
  return {
    lo,
    hi,
    max,
    decades,
    t: (v: number) =>
      v <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(v) - lo) / decades)),
  };
}

export function profilePath(
  wavelength: number[],
  intensity: number[],
  x: (lambda: number) => number,
  y: (t: number) => number,
  t: (v: number) => number,
): string {
  const parts: string[] = [];
  for (let i = 0; i < wavelength.length; i++) {
    parts.push(`${i === 0 ? "M" : "L"}${x(wavelength[i]).toFixed(2)} ${y(t(intensity[i])).toFixed(2)}`);
  }
  return parts.join(" ");
}

export function zoomWindow(
  wavelengthNm: number,
  fwhmNm: number,
  halfWidths = 8,
): [number, number] {
  const half = Math.max(fwhmNm * halfWidths, wavelengthNm * 1e-7);
  return [wavelengthNm - half, wavelengthNm + half];
}

export function widthAt(
  widths: LineWidthInfo[],
  wavelengthNm: number,
): LineWidthInfo | null {
  let best: LineWidthInfo | null = null;
  let bestGap = Infinity;
  for (const w of widths) {
    const gap = Math.abs(w.wavelength_nm - wavelengthNm);
    if (gap < bestGap) {
      bestGap = gap;
      best = w;
    }
  }
  return best;
}

export function dominantTerm(w: LineWidthInfo): string {
  const gaussFwhm = 2.3548 * w.sigma_nm;
  const lorentzFwhm = 2 * w.gamma_nm;
  if (gaussFwhm === 0 && lorentzFwhm === 0) return "nothing";
  if (gaussFwhm >= lorentzFwhm) {
    return w.terms.includes("instrumental") && !w.terms.includes("Doppler")
      ? "the spectrograph"
      : "thermal motion";
  }
  return "the upper level's lifetime";
}

export function wavelengthWindow(lines: SpectralLineInfo[], full: boolean) {
  const all = lines.map((ln) => ln.wavelength_nm.value);
  if (full || lines.length === 0) {
    return { lo: Math.min(...all), hi: Math.max(...all), hidden: 0, splittable: false };
  }
  const across = lines
    .filter((ln) => ln.n_upper !== ln.n_lower)
    .map((ln) => ln.wavelength_nm.value);
  if (across.length === 0 || across.length === lines.length) {
    return { lo: Math.min(...all), hi: Math.max(...all), hidden: 0, splittable: false };
  }
  const lo = Math.min(...across);
  const hi = Math.max(...across);
  return {
    lo,
    hi,
    hidden: lines.filter(
      (ln) => ln.wavelength_nm.value < lo || ln.wavelength_nm.value > hi,
    ).length,
    splittable: true,
  };
}



export function residualUnit(tol: number): { scale: number; label: string } {
  if (tol < 1e-3) return { scale: 1e6, label: "parts per million" };
  return { scale: 100, label: "percent" };
}

function ZoomPanel({
  prof,
  width: W,
  window_,
  onClear,
}: {
  prof: ProfileInfo;
  width: number;
  window_: [number, number];
  onClear: () => void;
}) {
  const ZOOM_H = plotHeight(W, SHAPE.zoom.ratio, SHAPE.zoom.min, SHAPE.zoom.max);
  const ZOOM_BASE = ZOOM_H - 40;
  const w = prof.widths.length > 0 ? prof.widths[0] : null;
  const max = Math.max(...prof.intensity, 0);
  const x = scaleLinear(window_, [M.left, W - M.right]);
  const y = scaleLinear([0, max > 0 ? max : 1], [ZOOM_BASE - 6, 16]);
  const path = profilePath(
    prof.wavelength_nm, prof.intensity, x, (t) => t, (v) => y(v),
  );
  const half = w ? w.fwhm_nm / 2 : 0;
  const axis = offsetAxis(window_[0], window_[1]);
  return (
    <div className="zoom-panel">
      <div className="view-header">
        <span className="plot-lead">
          <Notation>{w ? `One line, up close: ${w.label}` : "One line, up close"}</Notation>
        </span>
        <button className="link-button" onClick={onClear} type="button">
          ← back to the full range
        </button>
      </div>
      <p className="plot-blurb">
        Both axes are linear here, so this is the line's real shape. Everywhere
        else it gets drawn as a spike, because a line is far narrower than a
        pixel on an axis covering hundreds of nanometres.{" "}
        <Badge provenance={prof.provenance} />
      </p>
      <svg viewBox={`0 0 ${W} ${ZOOM_H}`} style={{ minWidth: W }} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={ZOOM_BASE} y2={ZOOM_BASE}
          className="axis"
        />
        {offsetTicks(axis, window_[0], window_[1]).map((t) => (
          <g key={t} transform={`translate(${x(t)},${ZOOM_BASE})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {formatOffset(axis, t)}
            </text>
          </g>
        ))}
        <text
          x={(M.left + W - M.right) / 2}
          y={ZOOM_H - 4}
          textAnchor="middle"
          className="tick"
        >
          λ − {axis.centreNm.toFixed(axis.centreDecimals)} nm [{axis.unit}]
        </text>
        {w && max > 0 && (
          <>
            <line
              x1={x(w.wavelength_nm - half)} x2={x(w.wavelength_nm + half)}
              y1={y(max / 2)} y2={y(max / 2)} className="fwhm-bar"
            />
            <text
              x={x(w.wavelength_nm)} y={y(max / 2) - 6}
              textAnchor="middle" className="tick"
            >
              {mathTspans(`FWHM ${w.fwhm_nm.toExponential(2)} nm`)}
            </text>
          </>
        )}
        <path d={path} className="profile-curve" />
      </svg>
      {w && (
        <>
          <p className="caption">
            Mostly <strong>{dominantTerm(w)}</strong> sets this width.
          </p>
          <Disclosure summary="The two widths, and why they do not add">
            <p className="caption">
              Gaussian σ = <Notation>{w.sigma_nm.toExponential(2)}</Notation> nm (
              {w.terms.filter((t) => t !== "natural").join(" + ") || "none"}
              ), Lorentzian γ = <Notation>{w.gamma_nm.toExponential(2)}</Notation> nm (natural).
              These do not add. The shape is their convolution, a Voigt, with a
              Gaussian core and Lorentzian wings, and that is why the far wings
              sit above where a Gaussian would put them.
            </p>
          </Disclosure>
        </>
      )}
    </div>
  );
}

export function SpectrumView() {
  const {
    system, fineStructure, intensities, spectrum, loadSpectrum, setIntensities,
    thermal, temperatureK, logNe, setThermal, setTemperatureK, setLogNe,
    profile, logResolvingPower, profileZoom,
    setProfile, setLogResolvingPower, setProfileZoom,
    absorption, logColumn, absorptionData, setAbsorption, setLogColumn,
    loadAbsorption,
  } = useAppStore();
  const [fullRange, setFullRange] = useState(false);
  const [keepFull, setKeepFull] = useState(false);
  useEffect(() => {
    void loadSpectrum();
  }, [
    system, fineStructure, intensities, thermal, temperatureK, logNe,
    profile, logResolvingPower, profileZoom, loadSpectrum,
  ]);
  const prof0 = spectrum?.profile ?? null;
  const lines0 = spectrum?.lines;
  useEffect(() => {
    if (
      !profile || profileZoom || keepFull || !prof0
      || prof0.widths.length === 0 || !lines0
    ) {
      return;
    }
    let best = prof0.widths[0];
    let bestWeight = -Infinity;
    for (const w of prof0.widths) {
      const ln = lines0.find(
        (l) => Math.abs(l.wavelength_nm.value - w.wavelength_nm) < 1e-9,
      );
      const weight = ln?.emissivity?.value ?? ln?.einstein_a_s?.value ?? 0;
      if (weight > bestWeight) {
        bestWeight = weight;
        best = w;
      }
    }
    setProfileZoom(zoomWindow(best.wavelength_nm, best.fwhm_nm));
  }, [profile, profileZoom, keepFull, prof0, lines0, setProfileZoom]);
  useEffect(() => {
    if (!absorption || !thermal) return;
    void loadAbsorption();
  }, [absorption, thermal, logColumn, system, fineStructure, temperatureK,
      logNe, logResolvingPower, profileZoom, loadAbsorption]);

  const { width: W, compact, ref: wrapRef } = usePlotWidth(SHAPE.floor);
  const LINES_H = plotHeight(W, SHAPE.lines.ratio, SHAPE.lines.min, SHAPE.lines.max);
  const BOTTOM = LINES_H - 46;
  const AXIS_Y = LINES_H - 40;
  const DOT_Y = LINES_H - 43;
  const RES_H = plotHeight(W, SHAPE.residual.ratio, SHAPE.residual.min, SHAPE.residual.max);
  const window_ = spectrum ? wavelengthWindow(spectrum.lines, fullRange) : null;
  const xRange: [number, number] = [M.left, W - M.right];
  const xFull: [number, number] = window_
    ? [window_.lo * 0.9, window_.hi * 1.1]
    : [1, 1000];
  const zoom = usePlotZoom({
    width: W,
    height: LINES_H,
    x: { domain: xFull, range: xRange, log: true },
  });
  const hover = usePlotHover(W, zoom.element);

  if (!spectrum || !window_) {
    return (
      <div className="view-wrap" ref={wrapRef}>
        <ViewIntro lead={VIEW_LEADS.spectrum} />
        <p className="hint-block">Loading the spectrum…</p>
      </div>
    );
  }

  const shown = spectrum.lines.filter(
    (ln) =>
      ln.wavelength_nm.value >= window_.lo && ln.wavelength_nm.value <= window_.hi,
  );
  const x = scaleLog(zoom.x, xRange);
  const nLowers = [...new Set(shown.map((ln) => ln.n_lower))].sort((a, b) => a - b);
  const tol = spectrum.tolerance_relative;
  const comp = spectrum.comparison;
  const yRes = tol ? scaleLinear([-3 * tol, 3 * tol], [RES_H - 48, 14]) : null;
  const clampY = (v: number) => Math.min(Math.max(v, 14), RES_H - 48);
  const nist = nistSummary(comp, tol);

  const isThermal = spectrum.thermal !== null;
  const strength = intensities
    ? intensityScale(shown, isThermal ? "emissivity" : "rate")
    : null;
  const barTop = (ln: SpectralLineInfo) =>
    strength ? BOTTOM - (0.18 + 0.82 * strength.t(strength.value(ln))) * (BOTTOM - TOP)
             : TOP;
  const barOpacity = (ln: SpectralLineInfo) =>
    strength ? 0.3 + 0.7 * strength.t(strength.value(ln)) : 0.9;
  const ionized = spectrum.thermal?.ionized_fraction.value ?? 0;

  const prof = spectrum.profile;
  const trace = prof && !profileZoom ? profileScale(prof.intensity) : null;
  const tracePath =
    prof && trace
      ? profilePath(
          prof.wavelength_nm, prof.intensity, x,
          (t) => BOTTOM - t * (BOTTOM - TOP), trace.t,
        )
      : null;
  const zoomLine = (ln: SpectralLineInfo) => {
    if (!prof) return;
    const w = widthAt(prof.widths, ln.wavelength_nm.value);
    if (!w) return;
    setKeepFull(false);
    setProfileZoom(zoomWindow(ln.wavelength_nm.value, w.fwhm_nm));
  };

  let hoverLine: SpectralLineInfo | null = null;
  if (hover.x !== null && !zoom.dragging && withinPlot(hover.x, M.left, W - M.right)) {
    let bestGap = 7;
    for (const ln of shown) {
      const gap = Math.abs(x(ln.wavelength_nm.value) - hover.x);
      if (gap < bestGap) {
        bestGap = gap;
        hoverLine = ln;
      }
    }
  }
  const hoverLines = hoverLine
    ? [
        `${seriesName(hoverLine.n_lower)}  ${hoverLine.n_upper}→${hoverLine.n_lower}`,
        `λ = ${hoverLine.wavelength_nm.value.toFixed(2)} nm`,
        ...(hoverLine.einstein_a_s
          ? [`A = ${hoverLine.einstein_a_s.value.toExponential(2)} s⁻¹`]
          : []),
        ...(hoverLine.oscillator_strength
          ? [`f = ${hoverLine.oscillator_strength.value.toExponential(2)}`]
          : []),
        ...(hoverLine.emissivity
          ? [`ε = ${hoverLine.emissivity.value.toExponential(2)} eV/s per atom`]
          : []),
        ...(prof ? ["click to draw its shape"] : []),
      ]
    : [];

  return (
    <div className="view-wrap" ref={wrapRef}>
      <ViewIntro
        lead={VIEW_LEADS.spectrum}
        badge={<Badge provenance={spectrum.lines[0].wavelength_nm.provenance} />}
      >
        <Disclosure summary="Where a spectral line comes from">
          <p className="caption">
            An electron dropping from one rung of the ladder to a lower one has
            to put the difference somewhere, so it emits a single photon
            carrying exactly that much energy. Energy fixes colour, which means
            each pair of rungs gives one precise wavelength. This plot is the
            ladder's shape, turned sideways.
          </p>
          <p className="caption">
            The colours say where a line lands. Everything ending on n=1 is the
            Lyman series, in the ultraviolet. Everything ending on n=2 is
            Balmer, the visible one, and Balmer-α at 656 nm is the red in every
            photograph of a nebula.
          </p>
        </Disclosure>
      </ViewIntro>

      <figure className="plot">
        <figcaption>
          <span className="plot-lead">Every line this atom can emit</span>
          <span className="plot-blurb">
            Each bar is one transition, standing at its wavelength.
            {strength
              ? " Bar height comes from line strength, compressed logarithmically."
              : " Every bar is the same height, because strength is not modelled yet."}
            {prof ? " Click any bar to plot its shape." : ""}
          </span>
          <span className="plot-provenance">
            {strength && (
              <Badge
                provenance={
                  isThermal ? SPECTRUM_EMISSIVITY_LIBERTY : SPECTRUM_INTENSITY_LIBERTY
                }
              />
            )}
            {spectrum.thermal && (
              <Badge provenance={spectrum.thermal.ionized_fraction.provenance} />
            )}
            <span className="legend-inline">
              {nLowers.map((nl) => (
                <span key={nl} style={{ color: seriesColor(nl) }}>
                  ▎{seriesName(nl)}
                </span>
              ))}
            </span>
          </span>
        </figcaption>
        <svg
          viewBox={`0 0 ${W} ${LINES_H}`} style={{ minWidth: W }}
          role="img"
          className={
            `levels-svg plot-zoomable${prof ? " plot-hoverable" : ""}` +
            `${zoom.dragging ? " plot-panning" : ""}`
          }
          ref={zoom.ref}
          onPointerDown={zoom.handlers.onPointerDown}
          onPointerMove={(e) => {
            zoom.handlers.onPointerMove(e);
            hover.onPointerMove(e);
          }}
          onPointerUp={zoom.handlers.onPointerUp}
          onPointerCancel={zoom.handlers.onPointerCancel}
          onDoubleClick={zoom.handlers.onDoubleClick}
          onPointerLeave={hover.onPointerLeave}
        >
          <defs>
            <clipPath id="spectrum-clip">
              <rect x={M.left} y={0} width={W - M.right - M.left} height={AXIS_Y} />
            </clipPath>
          </defs>
          {thinTicks(x.ticks(8), x, 34).map((t) => (
            <line
              key={`g-${t}`} x1={x(t)} x2={x(t)} y1={TOP - 12} y2={AXIS_Y}
              className="grid-line"
            />
          ))}
          <line
            x1={M.left} x2={W - M.right} y1={AXIS_Y} y2={AXIS_Y}
            className="axis"
          />
          {thinTicks(x.ticks(8), x, 34).map((t) => (
            <g key={t} transform={`translate(${x(t)},${AXIS_Y})`}>
              <line y2="5" className="axis" />
              <text y="17" textAnchor="middle" className="tick">
                {t}
              </text>
            </g>
          ))}
          <g clipPath="url(#spectrum-clip)">
          {shown.map((ln, i) => (
            <line
              key={i}
              x1={x(ln.wavelength_nm.value)} x2={x(ln.wavelength_nm.value)}
              y1={barTop(ln)} y2={BOTTOM}
              stroke={seriesColor(ln.n_lower)}
              strokeWidth={hoverLine === ln ? 3 : 1.5}
              opacity={barOpacity(ln)}
              className={prof ? "line-clickable" : undefined}
              onClick={prof ? () => zoomLine(ln) : undefined}
            />
          ))}
          {tracePath && <path d={tracePath} className="profile-curve" />}
          {comp?.map((c, i) => (
            <circle
              key={i} cx={x(c.reference_nm)} cy={DOT_Y} r={2.5}
              className={c.within_tolerance ? "ref-ok" : "ref-bad"}
            />
          ))}
          </g>
          <text
            x={(M.left + W - M.right) / 2} y={LINES_H - 3}
            textAnchor="middle" className="axis-title"
          >
            wavelength [nm] (log)
          </text>
          <text x={W - M.right} y={16} textAnchor="end" className="tick">
            {compact
              ? "bars: computed · dots: NIST"
              : "bars: computed · dots on the axis: measured by NIST"}
          </text>
          {hoverLine && (
            <HoverReadout
              px={x(hoverLine.wavelength_nm.value)}
              py={barTop(hoverLine)}
              top={TOP}
              bottom={BOTTOM}
              lines={hoverLines}
              width={W}
              rightMargin={M.right}
            />
          )}
        </svg>
        <ZoomControls zoom={zoom} what="wavelength" />
      </figure>

      {nist ? (
        <section className={`nist-panel${nist.allWithin ? " nist-ok" : " nist-off"}`}>
          <p className="nist-headline">
            <span className="nist-mark" aria-hidden="true">
              {nist.allWithin ? "✓" : "!"}
            </span>
            <Notation>{nist.headline}</Notation>
          </p>
          {yRes && tol && (
            <>
            <svg
              viewBox={`0 0 ${W} ${RES_H}`} style={{ minWidth: W }}
              role="img"
              className={`levels-svg plot-zoomable${zoom.dragging ? " plot-panning" : ""}`}
              ref={zoom.follower}
              onPointerDown={zoom.handlers.onPointerDown}
              onPointerMove={zoom.handlers.onPointerMove}
              onPointerUp={zoom.handlers.onPointerUp}
              onPointerCancel={zoom.handlers.onPointerCancel}
              onDoubleClick={zoom.handlers.onDoubleClick}
            >
              <defs>
                <clipPath id="residual-clip">
                  <rect x={M.left} y={0} width={W - M.right - M.left} height={RES_H} />
                </clipPath>
              </defs>
              {thinTicks(x.ticks(8), x, 34).map((t) => (
                <g key={`rg-${t}`}>
                  <line
                    x1={x(t)} x2={x(t)} y1={14} y2={RES_H - 42}
                    className="grid-line"
                  />
                  <text
                    x={x(t)} y={RES_H - 28} textAnchor="middle" className="tick"
                  >
                    {t}
                  </text>
                </g>
              ))}
              <line
                x1={M.left} x2={W - M.right} y1={RES_H - 42} y2={RES_H - 42}
                className="axis"
              />
              <rect
                x={M.left} width={W - M.left - M.right}
                y={yRes(tol)} height={yRes(-tol) - yRes(tol)} className="tol-band"
              />
              <line x1={M.left} x2={W - M.right} y1={yRes(0)} y2={yRes(0)} className="zero" />
              {yRes.ticks(5).map((t) => (
                <g key={`ry-${t}`} transform={`translate(${M.left},${yRes(t)})`}>
                  <line x2="-4" className="axis" />
                  <text x="-7" dy="0.32em" textAnchor="end" className="tick">
                    {(t * residualUnit(tol).scale).toFixed(0)}
                  </text>
                </g>
              ))}
              <text
                x={11} y={RES_H / 2} textAnchor="middle" className="axis-title"
                transform={`rotate(-90 11 ${RES_H / 2})`}
              >
                {residualUnit(tol).label}
              </text>
              <g clipPath="url(#residual-clip)">
              {comp!.map((c, i) => (
                <circle
                  key={i} cx={x(c.reference_nm)} cy={clampY(yRes(c.relative_error))} r={3}
                  className={c.within_tolerance ? "ref-ok" : "ref-bad"}
                />
              ))}
              </g>
              <text x={M.left} y={12} className="tick">
                {mathTspans(
                  compact
                    ? `(λ_computed − λ_NIST)/λ_NIST · band ±${tol.toExponential(0)}`
                    : `(λ_computed − λ_NIST)/λ_NIST, shaded band = the stated tolerance, ±${tol.toExponential(0)}`,
                )}
              </text>
              <text
                x={(M.left + W - M.right) / 2} y={RES_H - 6}
                textAnchor="middle" className="axis-title"
              >
                wavelength [nm] (log)
              </text>
            </svg>
            <ZoomControls zoom={zoom} what="wavelength" />
            </>
          )}
          <p className="caption">
            Each dot is one measured line. Inside the shaded band, the computed
            wavelength matches the measurement to within the stated tolerance.
            Above or below, it does not.
          </p>
          <p className="caption">
            {spectrum.reference_citation
              ? `Reference: ${spectrum.reference_citation}`
              : "No vendored NIST reference exists for this system."}
          </p>
        </section>
      ) : (
        <p className="hint-block">
          {spectrum.reference_citation
            ? `Reference: ${spectrum.reference_citation}`
            : "No vendored NIST reference exists for this system, so these are computed lines only, honestly unchecked."}
        </p>
      )}

      <div data-tour="spectrum-options">
      <ControlGroup
        title="How bright is each line"
        hint="By default every bar is the same height, because brightness is not modelled at all. These controls add it, one layer at a time."
        tone={intensities ? "active" : "plain"}
      >
        <Toggle
          label="Scale the bars by line strength"
          checked={intensities}
          onChange={setIntensities}
          why={
            intensities && !isThermal
              ? "The bars now show the Einstein A coefficient: how fast an atom in the upper level falls, not how many atoms are up there."
              : "This uses the spontaneous emission rate, which says which transitions an atom prefers before any gas is involved."
          } 
        />
        {intensities && (
          <Toggle
            label="Put it in a real gas (LTE populations)"
            checked={thermal}
            onChange={setThermal}
            why={
              thermal
                ? "Boltzmann sets which level atoms sit in, Saha sets how many are ionized, so the bars are now an emissivity."
                : "A rate is not a brightness until you know how many atoms are in the upper level. Turn this on and a temperature and a density work that out."
            }
          />
        )}
        {intensities && thermal && (
          <>
            <Slider
              label="temperature T"
              readout={
                temperatureK >= 1e4
                  ? `${(temperatureK / 1e3).toFixed(1)}k K`
                  : `${temperatureK.toFixed(0)} K`
              }
              anchor={describeTemperature(temperatureK)}
              min={2}
              max={6}
              step={0.02}
              value={Math.log10(temperatureK)}
              onChange={(v) => setTemperatureK(10 ** v)}
            />
            <Slider
              label="electron density n_e"
              readout={`10^${logNe.toFixed(1)} cm⁻³`}
              anchor={describeDensity(logNe)}
              min={4}
              max={22}
              step={0.1}
              value={logNe}
              onChange={setLogNe}
            />
            <p className="caption">
              The ionized fraction here is{" "}
              <strong>{(100 * ionized).toFixed(1)}%</strong>
              {ionized > 0.99
                ? ", so almost no neutral atoms are left and every line stays faint no matter how hot it gets."
                : ionized < 0.01
                  ? ", essentially all neutral, so excitation alone sets the brightness."
                  : "."}
            </p>
          </>
        )}
      </ControlGroup>
      </div>

      <ControlGroup
        title="What shape is a line"
        hint="A spectral line is not infinitely thin. Turn this on to synthesize the shape it really has and look at one up close."
        tone={profile ? "active" : "plain"}
      >
        <Toggle
          label="Synthesize line profiles"
          checked={profile}
          onChange={(v) => {
            setProfile(v);
            setKeepFull(false);
            if (!v) setProfileZoom(null);
          }}
          why={
            profile
              ? "Voigt profiles: the upper level's finite lifetime gives Lorentzian wings, thermal motion gives a Gaussian core. Click a bar above to open one."
              : "Two things widen every line: the upper level cannot live forever, and the atoms are moving. Turn this on, then click a line to see its shape."
          }
        />
        {profile && (
          <Slider
            label="spectrograph resolving power R"
            readout={
              logResolvingPower === null
                ? "none"
                : `${(10 ** logResolvingPower).toExponential(1)} (λ/Δλ)`
            }
            anchor={describeResolvingPower(logResolvingPower)}
            min={2}
            max={7}
            step={0.05}
            value={logResolvingPower ?? 2}
            disabled={logResolvingPower === null}
            atRest={logResolvingPower === null}
            onChange={setLogResolvingPower}
          />
        )}
        {profile && (
          <button
            className="link-button"
            type="button"
            onClick={() => setLogResolvingPower(logResolvingPower === null ? 4 : null)}
          >
            {logResolvingPower === null
              ? "add a spectrograph, and see the line as an instrument would"
              : "remove the spectrograph, and see the line as it really is"}
          </button>
        )}
      </ControlGroup>

      {window_.splittable && (
        <ControlGroup title="What the axis covers" tone={fullRange ? "active" : "plain"}>
          <Toggle
            label="Include the within-n fine-structure components"
            checked={fullRange}
            onChange={setFullRange}
            why={
              fullRange
                ? "The axis now stretches to millimetre and metre wavelengths, which squeezes every optical line into a sliver at the left."
                : `${window_.hidden} component${window_.hidden === 1 ? "" : "s"} start and end on the same shell, so ${window_.hidden === 1 ? "it lies" : "they lie"} out at millimetre to metre wavelengths. They stay in the data either way.`
            }
          />
        </ControlGroup>
      )}

      {prof && profileZoom && (
        <ZoomPanel
          width={W}
          prof={prof}
          window_={profileZoom}
          onClear={() => {
            setKeepFull(true);
            setProfileZoom(null);
          }}
        />
      )}

      <ControlGroup
        title="Look through the gas instead of at it"
        tone={absorption && thermal ? "active" : "plain"}
      >
        <Toggle
          label="Absorption: put this gas in front of a continuum"
          checked={absorption}
          disabled={!thermal}
          disabledReason="This needs the LTE populations above: which level an atom is in is what its line absorbs with."
          onChange={setAbsorption}
          why="The same lines that glow when the gas is hot appear as dark gaps when you look through it at something brighter. That is how stellar spectra are read."
        />
        {absorption && thermal && (
          <Slider
            label="column density"
            readout={`10^${logColumn.toFixed(1)} m⁻²`}
            anchor="how much of the element is in the line of sight"
            min={14}
            max={26}
            step={0.1}
            value={logColumn}
            onChange={setLogColumn}
          />
        )}
      </ControlGroup>
      {absorption && thermal &&
        (absorptionData ? (
          <AbsorptionView abs={absorptionData} width={W} zoomed={profileZoom !== null} />
        ) : (
          <p className="hint-block">Computing the absorption spectrum…</p>
        ))}

      {spectrum.profile_note && (
        <p className="caption">No profile drawn: {spectrum.profile_note}</p>
      )}
      {prof?.stark_note && <p className="caption warn-note">{prof.stark_note}</p>}

      {(prof && trace) || strength || window_.hidden > 0 || spectrum.intensity_note ? (
        <Disclosure summary="Exactly what the heights and the curve are claiming" tone="caveat">
          {prof && trace && (
            <p className="caption">
              The curve is Voigt profiles, synthesized in the engine and summed
              onto an adaptive grid, drawn on log₁₀ intensity over{" "}
              {trace.decades} decades below the peak{" "}
              <Badge provenance={SPECTRUM_PROFILE_LIBERTY} />. It integrates to{" "}
              {prof.flux_closure.toFixed(4)}× the summed line strengths, which is
              the grid's own quadrature error, measured rather than assumed. On
              this log wavelength axis every line is a spike regardless of its
              real shape, so <strong>click a line</strong> to plot it linearly
              and see the profile itself.
            </p>
          )}
          {prof && !prof.stark_note && prof.stark_span_nm && (
            <p className="caption">
              Collisional broadening is not in this curve. At this density its
              linear Stark span would be{" "}
              <Notation>{prof.stark_span_nm.value.toExponential(2)}</Notation> nm, comfortably under
              the widths that are modelled, so the shape stands.
            </p>
          )}
          {strength && !isThermal && (
            <p className="caption">
              Bar height and opacity go as log₁₀ A over{" "}
              <Notation>{`10^${strength.lo.toFixed(1)} to 10^${strength.hi.toFixed(1)} s⁻¹`}</Notation>. That
              is the spontaneous emission <em>rate</em>, not a prediction of
              observed brightness. No level populations are modelled here: turn
              on LTE weighting for those.
            </p>
          )}
          {strength && isThermal && spectrum.thermal && (
            <p className="caption">
              Bar height and opacity go as log₁₀ ε over{" "}
              <Notation>{`10^${strength.lo.toFixed(1)} to 10^${strength.hi.toFixed(1)}`}</Notation> eV/s per
              atom, at T = {spectrum.thermal.temperature_k.toFixed(0)} K and{" "}
              <Notation>{`n_e = ${spectrum.thermal.electron_density_cm3.toExponential(0)} cm⁻³`}</Notation>. That is an LTE
              emissivity: level populations from Boltzmann, ionization from Saha, and the gas
              taken as <em>optically thin</em>. A real medium reabsorbs its own strong lines,
              which is why Lyman-α does not dominate an observed nebula the way it dominates
              this plot.
            </p>
          )}
          {window_.hidden > 0 && (
            <p className="caption">
              The axis covers the across-n lines ({window_.lo.toFixed(1)}-
              {window_.hi < 1e6
                ? `${window_.hi.toFixed(0)} nm`
                : `${(window_.hi / 1e6).toFixed(1)} mm`}
              ). {window_.hidden} within-n fine-structure component
              {window_.hidden === 1 ? " is" : "s are"} outside it, out at millimetre to metre
              wavelengths. They stay in the data and in the line list either way. Tick the
              box above to include them, which stretches the axis far enough that the optical
              lines collapse into a sliver.
            </p>
          )}
          {spectrum.intensity_note && (
            <p className="caption">{spectrum.intensity_note}</p>
          )}
        </Disclosure>
      ) : null}
    </div>
  );
}
