import { useEffect } from "react";
import { stateLabel } from "../lib/quantum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

export function InfoPanel() {
  const { n, l, m, system, systems, model, stateInfo, loadStateInfo } = useAppStore();

  const sys = systems.find((s) => s.key === system);
  const screened = sys?.kind === "screened";

  useEffect(() => {
    // Wait for the systems table: before it arrives we cannot tell a
    // one-electron system from a screened atom, and guessing sends the
    // hydrogenic state endpoint a system it must refuse.
    if (systems.length > 0 && !screened) void loadStateInfo();
  }, [n, l, m, system, systems.length, screened, loadStateInfo]);

  if (screened && sys) {
    return (
      <div className="info-panel" data-tour="state-card">
        <h3>
          {sys.name} <span className="info-formula">{stateLabel(n, l, m)}</span>
        </h3>
        <dl className="state-facts">
          <div>
            <dt>Z</dt>
            <dd>{sys.z}</dd>
          </div>
          <div>
            <dt>electrons</dt>
            <dd>{sys.n_electrons ?? "—"}</dd>
          </div>
          <div>
            <dt>model</dt>
            <dd>{model === "hf" ? "Hartree-Fock" : "screened (GSZ)"}</dd>
          </div>
          <div>
            <dt>fidelity</dt>
            <dd>
              <span className="badge badge-approximation">approximation</span>
            </dd>
          </div>
        </dl>
        <p className="panel-hint">
          No closed-form state; see Energy levels and Radial.
        </p>
      </div>
    );
  }

  if (!stateInfo) {
    return (
      <div className="info-panel" data-tour="state-card">
        <p className="hint-block">Loading the state…</p>
      </div>
    );
  }

  return (
    <div className="info-panel" data-tour="state-card">
      <h3>
        {stateInfo.system.name} {n}
        {["s", "p", "d", "f", "g", "h"][l] ?? `l=${l}`}
      </h3>
      <dl className="state-facts">
        <div>
          <dt>energy</dt>
          <dd>
            {stateInfo.energy_ev.value.toFixed(3)} eV <Badge provenance={stateInfo.energy.provenance} />
          </dd>
        </div>
        <div>
          <dt>mean radius</dt>
          <dd>{stateInfo.mean_radius.value.toFixed(3)} bohr</dd>
        </div>
        <div>
          <dt>nodes</dt>
          <dd>
            {stateInfo.radial_nodes} radial · {stateInfo.angular_nodes} angular
          </dd>
        </div>
        <div>
          <dt>|L|</dt>
          <dd>{stateInfo.angular_momentum.value.toFixed(3)} ħ</dd>
        </div>
      </dl>
    </div>
  );
}
