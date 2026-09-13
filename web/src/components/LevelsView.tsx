import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { isScreenedLevels } from "../api/client";
import type {
  FineLevel,
  GrossLevel,
  HFLevels,
  LevelsResponse,
  PauliCollapse,
  ScreenedLevels,
} from "../api/types";
import { HF_LADDER_AXIS_LIBERTY } from "../lib/liberties";
import { plotHeight, usePlotWidth } from "../lib/plotSize";
import { withinView } from "../lib/zoom";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { ControlGroup, Slider, Toggle } from "./Field";
import { Disclosure } from "./Disclosure";
import { OffWindowMarks, usePlotZoom, ZoomControls } from "./PlotZoom";
import { ViewIntro } from "./ViewIntro";

const SHAPE = { floor: 560, ratio: 0.676, min: 380, max: 560 };

const HARTREE_UEV = 27.211386245988e6;
const MU_B_UEV_PER_T = (0.5 / 2.35051756758e5) * HARTREE_UEV;

export function LevelsLadder({
  levels,
  activeN,
  maxL,
  onPick,
  width: W = 680,
}: {
  levels: LevelsResponse;
  activeN: number;
  maxL: number;
  onPick: (n: number, l: number) => void;
  width?: number;
}) {
  const H = plotHeight(W, SHAPE.ratio, SHAPE.min, SHAPE.max);
  const es = levels.gross.map((g) => g.energy_ev.value);
  const eMin = Math.min(...es);
  const y = scaleLinear([eMin, 0], [H - 40, 24]);
  const rungX1 = 70;
  const rungX2 = 320;
  let lastLabelY = Number.POSITIVE_INFINITY;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ minWidth: W }} role="img" className="levels-svg">
      <line x1={rungX1} x2={rungX2} y1={y(0)} y2={y(0)} className="zero" />
      <text x={rungX2 + 30} y={y(0)} dy="0.32em" className="tick">
        0: ionization limit
      </text>
      {levels.gross.map((g) => {
        const yr = y(g.energy_ev.value);
        const labelled = Math.abs(yr - lastLabelY) >= 16;
        if (labelled) lastLabelY = yr;
        const pick = () => onPick(g.n, Math.min(maxL, g.n - 1));
        return (
          <g
            key={g.n}
            className="rung-hit"
            role="button"
            tabIndex={0}
            aria-label={`select shell n=${g.n}`}
            onClick={pick}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                pick();
              }
            }}
          >
            <line
              x1={rungX1} x2={rungX2} y1={yr} y2={yr}
              className={g.n === activeN ? "rung rung-active" : "rung"}
            />
            {labelled && (
              <>
                <text
                  x={rungX1 - 32} y={yr} dy="0.32em"
                  textAnchor="end" className="tick"
                >
                  n={g.n}
                </text>
                <text x={rungX2 + 30} y={yr} dy="0.32em" className="tick">
                  {g.energy_ev.value.toFixed(2)} eV · 2n²={g.degeneracy}
                  {g.n === 1 ? " · ground state" : ""}
                </text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function FineFan({
  shell,
  grossEv,
  dirac,
}: {
  shell: FineLevel[];
  grossEv: number;
  dirac: boolean;
}) {
  if (shell.length === 0) return null;
  const splits = shell.map((f) => ({
    f,
    rel: dirac ? (f.energy_ev.value - grossEv) * 1e6 : f.shift_ev.value * 1e6,
  }));
  const rels = splits.map((s) => s.rel);
  const pad = Math.max(1, ...rels.map(Math.abs)) * 0.3;
  const y = scaleLinear(
    [Math.min(...rels, 0) - pad, Math.max(...rels, 0) + pad],
    [150, 30],
  );
  return (
    <svg viewBox="0 0 360 180" role="img" className="levels-zoom-svg">
      <line x1={30} x2={120} y1={y(0)} y2={y(0)} className="zero" />
      <text x={124} y={y(0)} dy="0.32em" className="tick">
        gross level
      </text>
      {splits.map((s, i) => (
        <g key={i}>
          <line x1={30} x2={120} y1={y(s.rel)} y2={y(s.rel)} className="rung" />
          <text x={124} y={y(s.rel)} dy="0.32em" className="tick">
            {`j=${s.f.j} · ${s.rel >= 0 ? "+" : ""}${s.rel.toFixed(1)} µeV`}
          </text>
        </g>
      ))}
      <text x={30} y={172} className="tick">
        {dirac ? "Dirac exact, (n,j) only" : "α² fine structure, µeV below gross"}
      </text>
    </svg>
  );
}

export function ZeemanFan({ fine, bField }: { fine: FineLevel; bField: number }) {
  const subs = fine.sublevels ?? [];
  if (subs.length === 0 || bField <= 0) return null;
  const parentEv = fine.energy_ev.value;
  const rels = subs.map((s) => (s.energy_ev.value - parentEv) * 1e6);
  const pad = Math.max(...rels.map(Math.abs), 1) * 0.3;
  const y = scaleLinear([Math.min(...rels) - pad, Math.max(...rels) + pad], [170, 30]);
  return (
    <svg viewBox="0 0 360 200" role="img" className="levels-zoom-svg">
      <line x1={30} x2={110} y1={y(0)} y2={y(0)} className="zero" />
      <text x={114} y={y(0)} dy="0.32em" className="tick">
        un-split j-level
      </text>
      {subs.map((s, i) => (
        <g key={i}>
          <line x1={30} x2={110} y1={y(rels[i])} y2={y(rels[i])} className="rung" />
          <text x={114} y={y(rels[i])} dy="0.32em" className="tick">
            {`m_j=${s.m_j} · ${rels[i] >= 0 ? "+" : ""}${rels[i].toFixed(1)} µeV`}
          </text>
        </g>
      ))}
      <text x={30} y={192} className="tick">
        {`B = ${bField} T · high field: ${subs[0]?.high_field_label ?? ""}`}
      </text>
    </svg>
  );
}

export function StarkFan({ gross, eField }: { gross: GrossLevel; eField: number }) {
  const subs = gross.sublevels ?? [];
  if (subs.length === 0 || eField <= 0) return null;
  const parentEv = gross.energy_ev.value;
  const rels = subs.map((s) => (s.energy_ev.value - parentEv) * 1000);
  const pad = Math.max(...rels.map(Math.abs), 1) * 0.3;
  const y = scaleLinear([Math.min(...rels) - pad, Math.max(...rels) + pad], [170, 30]);
  return (
    <svg viewBox="0 0 380 200" role="img" className="levels-zoom-svg">
      <line x1={30} x2={110} y1={y(0)} y2={y(0)} className="zero" />
      <text x={114} y={y(0)} dy="0.32em" className="tick">
        gross E_n
      </text>
      {subs.map((s, i) => (
        <g key={i}>
          <line x1={30} x2={110} y1={y(rels[i])} y2={y(rels[i])} className="rung" />
          <text x={114} y={y(rels[i])} dy="0.32em" className="tick">
            {`k=${s.k} · ${rels[i] >= 0 ? "+" : ""}${rels[i].toFixed(2)} meV`}
          </text>
        </g>
      ))}
      <text x={30} y={192} className="tick">
        {`F = ${eField} MV/m · linear fan (l-degenerate) + quadratic`}
      </text>
    </svg>
  );
}

export function ScreenedLadder({ levels, width: W = 680 }: {
  levels: ScreenedLevels;
  width?: number;
}) {
  const H = plotHeight(W, SHAPE.ratio, SHAPE.min, SHAPE.max);
  const es = levels.orbitals.map((o) => o.energy_ev.value);
  const eMin = Math.min(...es);
  const y = scaleLinear([eMin, 0], [H - 40, 24]);
  const rungX1 = 90;
  const rungX2 = 340;
  const labelled = new Set<string>();
  const placed: number[] = [];
  for (const o of [...levels.orbitals].sort((a, b) => {
    const fa = a.occupancy > 0 ? 0 : 1;
    const fb = b.occupancy > 0 ? 0 : 1;
    return fa !== fb ? fa - fb : a.energy_ev.value - b.energy_ev.value;
  })) {
    const yr = y(o.energy_ev.value);
    if (placed.every((p) => Math.abs(p - yr) >= 13)) {
      placed.push(yr);
      labelled.add(`${o.n}-${o.l}`);
    }
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ minWidth: W }} role="img" className="levels-svg">
      <line x1={rungX1} x2={rungX2} y1={y(0)} y2={y(0)} className="zero" />
      <text x={rungX2 + 30} y={Math.max(11, y(0) - 13)} dy="0.32em" className="tick">
        0: ionization limit
      </text>
      {levels.orbitals.map((o) => {
        const yr = y(o.energy_ev.value);
        const filled = o.occupancy > 0;
        const show = labelled.has(`${o.n}-${o.l}`);
        return (
          <g key={`${o.n}-${o.l}`}>
            <line
              x1={rungX1} x2={rungX2} y1={yr} y2={yr}
              className="rung"
              strokeWidth={filled ? 3 : 1.5}
              strokeDasharray={filled ? undefined : "4 4"}
              opacity={filled ? 1 : 0.5}
            />
            {show && (
              <>
                <text
                  x={rungX1 - 32} y={yr} dy="0.32em"
                  textAnchor="end" className="tick"
                >
                  {o.label}
                  {filled ? <tspan dy="-0.5em">{o.occupancy}</tspan> : ""}
                </text>
                <text x={rungX2 + 30} y={yr} dy="0.32em" className="tick">
                  {o.energy_ev.value.toFixed(2)} eV
                  {filled ? "" : " · virtual"}
                </text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function PauliComparison({ collapse }: { collapse: PauliCollapse }) {
  const gained = Math.abs(collapse.binding_change_ev.value);
  const noop = collapse.binding_change_ev.value === 0;
  const shrink = collapse.radius_ratio.value;
  return (
    <>
      <p className="caption">
        <strong>
          {noop
            ? "The exclusion principle costs this atom nothing"
            : `The exclusion principle costs this atom ${gained.toPrecision(4)} eV of binding`}
        </strong>{" "}
        {noop ? (
          <>
            , its ground configuration is already 1s
            <sup>{2}</sup>, so lifting the cap lifts nothing. Helium is the
            calibration case here, not the demonstration: the two solves agree
            to the last bit because they are the same solve.
          </>
        ) : (
          <>
            , the real atom ({collapse.real_config}) sits at{" "}
            {collapse.real_total_energy_ev.value.toPrecision(6)} eV and the
            collapsed one at{" "}
            {(collapse.real_total_energy_ev.value + collapse.binding_change_ev.value)
              .toPrecision(6)}{" "}
            eV, both on the same mesh. Drop the cap and nothing holds the
            electrons out of the deep well.
          </>
        )}
      </p>
      <p className="caption">
        <strong>And it is what makes the atom big.</strong> ⟨r⟩ falls from{" "}
        {collapse.real_radius.value.toFixed(3)} to{" "}
        {collapse.collapsed_radius.value.toFixed(3)} bohr, a factor of{" "}
        {shrink.toFixed(3)}. With the cap on, atomic size rises and falls across
        a period, and that rise and fall is the periodic table. With it off, ⟨r⟩
        just shrinks forever as Z grows: every element a smaller copy of the
        last one, and no chemistry left to do.
      </p>
      <Disclosure summary="Checked against a closed form, not only against itself">
        <p className="caption">
          N electrons in a single 1s of exponent ζ minimize at ζ* ={" "}
          {collapse.variational_zeta.value.toFixed(4)}, giving{" "}
          {collapse.variational_energy_ev.value.toPrecision(6)} eV. The solve
          above optimizes the whole radial function rather than one exponent, so
          it searches a bigger space and has to land at or below that number.
          It does. The formula is textbook: at Z = N = 2 it is the variational
          helium result.
        </p>
      </Disclosure>
    </>
  );
}

export function HFLadder({ levels, width: W = 680 }: {
  levels: HFLevels;
  width?: number;
}) {
  const H = plotHeight(W, SHAPE.ratio, SHAPE.min, SHAPE.max);
  const orbitals = levels.orbitals;
  const bind = orbitals.map((o) => Math.abs(o.energy_ev.value));
  const deepest = Math.max(...bind);
  const shallowest = Math.min(...bind.filter((b) => b > 0), deepest);
  const yRange: [number, number] = [40, H - 40];
  const yFull: [number, number] = [Math.log10(shallowest) - 0.25, Math.log10(deepest)];
  const zoom = usePlotZoom({ width: W, height: H, y: { domain: yFull, range: yRange } });
  const y = scaleLinear(zoom.y, yRange);
  const shown = withinView(orbitals, (o) => Math.log10(Math.abs(o.energy_ev.value)), zoom.y);
  const rungX1 = 100;
  const rungX2 = 360;
  const virial = levels.virial_ratio.value;
  const modelName = !levels.pauli
    ? "No Pauli exclusion (1s^N)"
    : levels.exchange
      ? "Hartree-Fock"
      : "Hartree (no exchange)";
  return (
    <div className="view-wrap">
      <ViewIntro
        lead={{
          title: `Energy levels: ${modelName}`,
          lead:
            "Each rung is one subshell, solved self-consistently: every electron " +
            "moves in the field of all the others, and the answer has to " +
            "reproduce itself.",
          notice:
            "The 1s sits more than two decades below the valence shell, so the " +
            "axis is logarithmic in binding energy. Deeper means further down.",
        }}
        badge={<Badge provenance={levels.provenance} />}
      >
        <p className="view-intro-config">
          {levels.symbol ?? `Z=${levels.z}`} {levels.config}
          {levels.is_ground
            ? levels.pauli
              ? " · ground configuration"
              : " · ground, with no cap left to obey"
            : " · excited, not the ground state"}
        </p>
      </ViewIntro>
      <svg
        viewBox={`0 0 ${W} ${H}`} style={{ minWidth: W }}
        role="img"
        className={`levels-svg plot-zoomable${zoom.dragging ? " plot-panning" : ""}`}
        ref={zoom.ref}
        {...zoom.handlers}
      >
        {}
        <text x={rungX1} y={16} className="tick" opacity={0.7}>
          ↑ 0 eV (ionization limit), off the top of a log axis
        </text>
        <OffWindowMarks
          above={orbitals.filter(
            (o) => Math.log10(Math.abs(o.energy_ev.value)) < Math.min(...zoom.y),
          ).length}
          below={orbitals.filter(
            (o) => Math.log10(Math.abs(o.energy_ev.value)) > Math.max(...zoom.y),
          ).length}
          x={rungX1} top={30} bottom={H - 14} noun="subshell"
        />
        {shown.map((o) => {
          const e = o.energy_ev.value;
          const yr = y(Math.log10(Math.abs(e)));
          return (
            <g key={`${o.n}-${o.l}`}>
              <line
                x1={rungX1} x2={rungX2} y1={yr} y2={yr}
                className="rung" strokeWidth={3}
              />
              <text x={rungX1 - 8} y={yr} dy="0.32em" textAnchor="end" className="tick">
                {o.label}
                <tspan dy="-0.5em">{o.occupancy}</tspan>
              </text>
              <text x={rungX2 + 8} y={yr} dy="0.32em" className="tick">
                {}
                {e.toPrecision(4)} eV
              </text>
            </g>
          );
        })}
      </svg>
      <ZoomControls zoom={zoom} what="the binding-energy axis" />
      <p className="caption">
        Total energy {levels.total_energy_ev.value.toFixed(2)} eV
        {levels.exchange
          ? ", and it is variational: the real atom sits at or below it."
          : ", stationary for this model, but not a bound on the real atom."}{" "}
        <Badge provenance={HF_LADDER_AXIS_LIBERTY} />
      </p>
      {!levels.exchange && levels.exchange_energy_ev !== null && (
        <p className="caption">
          <strong>
            Exchange is worth{" "}
            {Math.abs(levels.exchange_energy_ev.value).toFixed(2)} eV of binding
          </strong>{" "}
          to this atom: the gap between the energy above and the Hartree-Fock
          one, both solved on the same mesh. Exchange binds because same-spin
          electrons keep out of each other's way, so they repel each other less
          than distinguishable ones would.{" "}
          {levels.exchange_energy_ev.value === 0
            ? "Zero here, and that is the answer rather than a missing number: no two electrons in this configuration share a spin, so there is no pair to exchange."
            : "The Pauli occupancies are untouched, so nothing has piled into the 1s."}
        </p>
      )}
      {levels.collapse !== null && <PauliComparison collapse={levels.collapse} />}
      <Disclosure summary="What this model leaves out, and why the axis is logarithmic">
        <p className="caption">
          These are self-consistent-field orbital energies (
          {levels.exchange
            ? "APPROXIMATION, and the badge lists what Hartree-Fock leaves out, correlation above all"
            : "COUNTERFACTUAL, see the badge; this is not an approximation to the real atom"}
          ). The energy axis is logarithmic in binding energy{" "}
          <Badge provenance={HF_LADDER_AXIS_LIBERTY} /> because the 1s and the
          valence shell differ by more than two decades. The total above,{" "}
          {levels.total_energy_ev.value.toFixed(2)} eV,
          {levels.exchange
            ? " is variational, unlike the screened model's sum of orbital energies."
            : " is stationary for this model, but it is not a variational bound on the real atom: a product wavefunction is not antisymmetric, so it is not an admissible trial function for electrons, and the theorem simply does not apply to it."}
        </p>
      </Disclosure>
      <Disclosure summary="Solve diagnostics: how well the computation converged">
        <p className="caption">
          <strong>These describe the computation, not the atom</strong>{" "}
          (NUMERICAL): the solve{" "}
          {levels.converged ? "converged" : "DID NOT CONVERGE"} in{" "}
          {levels.coarse_iterations} coarse + {levels.iterations} fine SCF
          iterations on {levels.grid_points} radial points. The virial ratio
          −〈V〉/〈T〉 = {virial.toFixed(6)}, which is exactly 2 for a converged
          solution of this Hamiltonian. How far it sits from 2 measures the
          grid, not the element.
        </p>
      </Disclosure>
    </div>
  );
}

export function LevelsView() {
  const {
    n, l, system, levels,
    loadLevels, setQuantumNumbers,
    fineStructure, setFineStructure,
    dirac, setDirac,
    bField, setBField,
    eField, setEField,
    hyperfine, setHyperfine,
    model, config, exchange, pauli, hfLevels, hfStatus, loadHF, error,
  } = useAppStore();
  useEffect(() => {
    void loadLevels();
  }, [system, config, fineStructure, dirac, bField, eField, hyperfine, loadLevels]);
  const wantHF = model === "hf";
  useEffect(() => {
    if (wantHF && hfLevels === null && hfStatus === "idle") void loadHF();
  }, [wantHF, hfLevels, hfStatus, system, config, exchange, pauli, loadHF]);
  const { width: W, ref: wrapRef } = usePlotWidth(SHAPE.floor);

  if (wantHF) {
    if (hfStatus === "error") {
      return (
        <div className="view-wrap" ref={wrapRef}>
          <p className="hint-block">
            Hartree-Fock could not solve this atom: {error ?? "unknown reason"}
          </p>
        </div>
      );
    }
    if (hfLevels === null) {
      return (
        <div className="view-wrap" ref={wrapRef}>
          <p className="hint-block">Solving Hartree-Fock, this takes a few seconds…</p>
        </div>
      );
    }
    return <HFLadder levels={hfLevels} width={W} />;
  }

  if (!levels) {
    return (
      <div className="view-wrap" ref={wrapRef}>
        <p className="hint-block">Loading the levels…</p>
      </div>
    );
  }

  if (isScreenedLevels(levels)) {
    return (
      <div className="view-wrap" ref={wrapRef}>
        <ViewIntro
          lead={{
            title: `Energy levels of ${levels.system.name}`,
            lead:
              "Each rung is one subshell in a fitted central field. " +
              "Solid rungs hold electrons, dashed ones are empty.",
          }}
          badge={<Badge provenance={levels.orbitals[0].energy.provenance} />}
        >
          <p className="view-intro-config">
            {levels.config}
            {levels.is_ground ? " · ground configuration" : " · excited, not the ground state"}
          </p>
        </ViewIntro>
        <ScreenedLadder levels={levels} width={W} />
        <p className="caption">
          Total energy {levels.total_energy_ev.value.toFixed(2)} eV.
        </p>
      </div>
    );
  }

  const activeGross = levels.gross.find((g) => g.n === n) ?? levels.gross[0];
  const fineForN = levels.fine?.filter((f) => f.n === activeGross.n) ?? [];
  const hfShell = levels.hyperfine_shells?.find((s) => s.n === 1);

  return (
    <div className="view-wrap" ref={wrapRef}>
      <ViewIntro
        lead={{
          title: `Energy levels of ${levels.system.name}`,
          lead:
            "Each rung is one shell's gross energy under a reduced-mass Bohr model. " +
            "Click a rung and every view moves to that shell.",
          notice:
            "The rungs crowd toward the ionization limit at 0 because the energies go as −1/n².",
        }}
        badge={<Badge provenance={levels.gross[0].energy.provenance} />}
      />
      <ControlGroup title="Level detail">
        <div data-tour="fine-structure">
          <Toggle
            label="fine structure (α²)"
            checked={fineStructure}
            onChange={setFineStructure}
          />
        </div>
        {fineStructure && (
          <div data-tour="dirac-toggle">
            <Toggle
              label="Dirac exact (instead of α² perturbative)"
              checked={dirac}
              onChange={setDirac}
            />
          </div>
        )}
        {fineStructure && (
          <Slider
            label="magnetic field B"
            readout={`${bField.toFixed(1)} T`}
            min={0}
            max={20}
            step={0.1}
            value={bField}
            onChange={setBField}
          />
        )}
        {!fineStructure && (
          <p className="control-hint">turn on fine structure to add a magnetic field</p>
        )}
        <Slider
          label="electric field F"
          readout={`${eField.toFixed(1)} MV/m`}
          min={0}
          max={100}
          step={1}
          value={eField}
          onChange={setEField}
        />
        <Toggle
          label="hyperfine (nuclear spin, s-states)"
          checked={hyperfine}
          onChange={setHyperfine}
        />
        {hyperfine && hfShell !== undefined && !hfShell.available && (
          <p className="control-hint">{hfShell.reason}</p>
        )}
      </ControlGroup>
      <LevelsLadder
        levels={levels}
        activeN={n}
        maxL={l}
        onPick={(nn, ll) => setQuantumNumbers(nn, ll, 0)}
        width={W}
      />
      {fineStructure && fineForN.length > 0 && bField <= 0 && (
        <FineFan
          shell={fineForN}
          grossEv={activeGross.energy_ev.value}
          dirac={dirac}
        />
      )}
      {fineStructure && bField > 0 && (
        <>
          {fineForN.map((f) => (
            <ZeemanFan key={`${f.n}-${f.l}-${f.j}`} fine={f} bField={bField} />
          ))}
        </>
      )}
      {eField > 0 && <StarkFan gross={activeGross} eField={eField} />}
      <Disclosure summary="What the scale means">
        <p className="caption">
          {fineStructure && dirac
            ? "Dirac is exact for a point nucleus: the energy depends on n and j only, " +
              "so 2s₁/₂ and 2p₁/₂ coincide. Reality splits them by the Lamb shift, which " +
              "this model omits."
            : fineStructure
              ? "The α² fine structure splits each shell by (n, j): spin-orbit, " +
                "relativistic kinetic energy and the Darwin term. Exact Dirac energies " +
                "are one toggle away."
              : "Gross levels only: every shell exact under the reduced-mass model, no " +
                "fine structure yet. Turn it on above."}
          {eField > 0
            ? " With an electric field, each shell fans linearly in the parabolic quantum " +
              "number k — hydrogen's l-degeneracy signature. The model omits field ionization " +
              "(the perturbation series breaks down near F_ion)."
            : ""}
          {bField > 0
            ? " A magnetic field splits each j-level into 2j+1 m_j sublevels (anomalous " +
              "Zeeman); as B rises they reorganize into the Paschen-Back pattern where " +
              "(m_l, m_s) become the good labels. The diamagnetic B² term is omitted."
            : ""}
        </p>
      </Disclosure>
      {hyperfine && hfShell !== undefined && hfShell.available && (
        <div>
          <p className="caption">
            Hyperfine of 1s ({hfShell.nucleus}, I={hfShell.I}):
            {" "}
            {hfShell.levels.map((lv) => `F=${lv.F}`).join(", ")}. The F=1→F=0
            transition of hydrogen is the 21 cm line, 1420.4058 MHz.
          </p>
        </div>
      )}
      <MU_B_NOTE />
    </div>
  );
}

function MU_B_NOTE() {
  const bField = useAppStore((s) => s.bField);
  if (bField <= 0) return null;
  return (
    <p className="caption">
      Current field: µ_B·B = {(bField * MU_B_UEV_PER_T).toFixed(1)} µeV per m_j unit.
    </p>
  );
}
