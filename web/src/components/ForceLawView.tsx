import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import {
  PRESET_LABELS,
  PRESET_PARAMS,
  clampParam,
  validateExprClient,
  type ForcePreset,
} from "../lib/forceLaw";
import { useAppStore } from "../state/store";
import { plotHeight, usePlotWidth } from "../lib/plotSize";
import { Badge } from "./Badge";
import { Choice, ControlGroup, Select, Slider } from "./Field";
import { ViewIntro } from "./ViewIntro";

const SHAPE = { floor: 680, ratio: 0.676, min: 380, max: 560 };

export function ForceLawView() {
  const {
    system, systems, forcePreset, forceParams, forceL, forceExpr, forceLaw, forceStatus,
    setForcePreset, setForceParam, setForceL, setForceExpr, loadForceLaw,
  } = useAppStore();
  const oneElectron = systems.find((s) => s.key === system)?.kind === "hydrogenic";
  const systemsLoaded = systems.length > 0;
  useEffect(() => {
    // The force-law lab solves one-electron systems only; a multi-electron
    // selection here would be refused, so the view pins itself to hydrogen
    // until the picker is used.
    if (!systemsLoaded || !oneElectron) return;
    if (forceLaw === null && forceStatus === "idle") void loadForceLaw();
  }, [forceLaw, forceStatus, system, systemsLoaded, oneElectron, loadForceLaw]);

  const exprError = forcePreset === "custom" ? validateExprClient(forceExpr) : null;

  const { width: W, ref: wrapRef } = usePlotWidth(SHAPE.floor);

  if (!systemsLoaded) {
    return (
      <div className="view-wrap" ref={wrapRef}>
        <p className="hint-block">Loading the systems…</p>
      </div>
    );
  }

  if (!oneElectron) {
    return (
      <div className="view-wrap" ref={wrapRef}>
        <p className="hint-block">
          The force-law lab alters the 1/r law for a one-electron system, and
          {" "}{system} has many electrons: there is no single potential to
          bend. Pick hydrogen, deuterium, muonic hydrogen or He+ in the state
          panel and the counterfactual ladder comes back.
        </p>
      </div>
    );
  }

  if (forceStatus === "error" || (forceLaw === null && forceStatus !== "loading")) {
    if (forceLaw === null) {
      return (
        <div className="view-wrap" ref={wrapRef}>
          <p className="hint-block">
            {forceStatus === "error" ? "The solve failed." : "Preparing the solve…"}
          </p>
          <button type="button" className="primary" onClick={() => void loadForceLaw()}>
            Solve
          </button>
        </div>
      );
    }
  }

  const untrusted = forceLaw?.counterfactual.filter((c) => !c.trusted).length ?? 0;

  return (
    <div className="view-wrap" ref={wrapRef}>
      <ViewIntro
        lead={{
          title: "What if the force law were different?",
          lead:
            "Same solver, altered potential. Every level is computed rigorously under " +
            "the new rules and labelled COUNTERFACTUAL.",
        }}
        badge={
          forceLaw && forceLaw.counterfactual.length > 0 ? (
            <Badge provenance={forceLaw.counterfactual[0].energy.provenance} />
          ) : undefined
        }
      />
      {forceLaw && (
        <div className="counterfactual-banner">
          COUNTERFACTUAL · {forceLaw.preset}
          {forceLaw.expression ? ` · V(r) = ${forceLaw.expression}` : ""}
        </div>
      )}
      <ControlGroup title="Potential">
        <div data-tour="force-preset">
          <Choice<ForcePreset>
            legend="preset"
            value={forcePreset}
            onChange={setForcePreset}
            options={(Object.keys(PRESET_LABELS) as ForcePreset[]).map((p) => ({
              value: p,
              label: PRESET_LABELS[p],
            }))}
          />
        </div>
        {PRESET_PARAMS[forcePreset].map((spec) => (
          <Slider
            key={spec.name}
            label={`${spec.name}${spec.unit ? ` [${spec.unit}]` : ""}`}
            readout={String(forceParams[spec.name] ?? spec.default)}
            min={spec.min}
            max={spec.max}
            step={spec.step}
            value={forceParams[spec.name] ?? spec.default}
            onChange={(v) => setForceParam(spec.name, clampParam(spec, v))}
          />
        ))}
        {forcePreset === "custom" && (
          <label className="control-select">
            <span className="control-label">V(r) expression</span>
            <input
              type="text"
              value={forceExpr}
              onChange={(e) => setForceExpr(e.target.value)}
            />
            {exprError !== null && <span className="control-hint">{exprError}</span>}
          </label>
        )}
        <Select
          label="angular momentum l"
          value={String(forceL)}
          options={[0, 1, 2, 3].map((v) => ({ value: String(v), label: `l = ${v}` }))}
          onChange={(v) => setForceL(Number(v))}
        />
        <button
          type="button"
          className="primary"
          disabled={exprError !== null || forceStatus === "loading"}
          onClick={() => void loadForceLaw()}
        >
          {forceStatus === "loading" ? "Solving…" : "Solve"}
        </button>
      </ControlGroup>
      {forceLaw === null ? (
        <p className="hint-block">Press Solve to run the altered potential.</p>
      ) : (
        <ForceDiagram
          width={W}
          r={forceLaw.potential_curve.r}
          vEv={forceLaw.potential_curve.v_ev}
          levels={forceLaw.counterfactual.map((c) => ({
            e: c.energy_ev.value,
            trusted: c.trusted,
          }))}
          reference={forceLaw.reference.items.map((i) => ({
            label: i.label,
            e: i.energy_ev.value,
          }))}
        />
      )}
      {forceLaw && (
        <>
          <p className="caption">
            {forceLaw.bound_count} of {forceLaw.requested_count} requested levels bound.
            {untrusted > 0 &&
              ` ${untrusted} flagged untrusted: unconverged in box or grid, not real bound states.`}
          </p>
          <dl className="readouts">
            {forceLaw.counterfactual.map((c) => (
              <div key={c.radial_index} className="readout-row">
                <dt>
                  level {c.radial_index}
                  {!c.trusted && " · untrusted"}
                </dt>
                <dd>
                  {c.energy_ev.value.toFixed(3)} eV <Badge provenance={c.energy.provenance} />
                </dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </div>
  );
}

export function ForceDiagram({
  r,
  vEv,
  levels,
  reference,
  width: W,
}: {
  r: number[];
  vEv: number[];
  levels: { e: number; trusted: boolean }[];
  reference: { label: string; e: number }[];
  width: number;
}) {
  const H = plotHeight(W, SHAPE.ratio, SHAPE.min, SHAPE.max);
  const allE = [...levels.map((l) => l.e), ...reference.map((x) => x.e)];
  const lo = Math.min(...allE, ...vEv.filter((v) => Number.isFinite(v)));
  const hi = Math.max(...allE, 0);
  const pad = (hi - lo || 1) * 0.05;
  const x = scaleLinear([0, r[r.length - 1]], [70, W - 150]);
  const y = scaleLinear([lo - pad, hi + pad], [H - 30, 20]);
  const path = r
    .map((rv, i) => `${i === 0 ? "M" : "L"}${x(rv).toFixed(1)},${y(vEv[i]).toFixed(1)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ minWidth: W }} role="img" className="levels-svg">
      <line x1={70} x2={W - 150} y1={y(0)} y2={y(0)} className="zero" />
      <path d={path} className="curve" />
      {levels.map((lv, i) => (
        <line
          key={i}
          x1={70} x2={W - 150} y1={y(lv.e)} y2={y(lv.e)}
          className="rung"
          strokeDasharray={lv.trusted ? undefined : "5 4"}
          opacity={lv.trusted ? 1 : 0.55}
        />
      ))}
      {reference.map((ref) => (
        <g key={ref.label}>
          <line
            x1={W - 140} x2={W - 100} y1={y(ref.e)} y2={y(ref.e)}
            className="rung"
          />
          <text x={W - 94} y={y(ref.e)} dy="0.32em" className="tick">
            {ref.label}
          </text>
        </g>
      ))}
      <text
        x={(70 + W - 150) / 2} y={H - 6} textAnchor="middle"
        className="axis-title"
      >
        r [bohr]
      </text>
    </svg>
  );
}
