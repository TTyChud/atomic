import { useEffect, useState } from "react";
import { clampState } from "../lib/quantum";
import type { Basis, PlaneQuantity } from "../api/client";
import type { AtomModel, ColorMode, ViewMode } from "../lib/urlState";
import { compareAvailable, gszAvailable, subshellAvailable } from "../lib/hfModel";
import { NUCLEUS_MODES, type NucleusMode } from "../lib/nucleus";
import { isNarrow, useViewport } from "../lib/viewport";
import { useAppStore } from "../state/store";
import { Choice, ControlGroup, Select, Slider, Toggle } from "./Field";
import { ShowPhysics } from "./ShowPhysics";

const PRESET_SYSTEMS = ["h", "d", "t", "mu-h", "ps", "he+"];

export const COUNT_CHOICES = [10000, 100000, 300000];

export const VIEW_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: "cloud", label: "cloud" },
  { value: "plane", label: "plane" },
  { value: "radial", label: "radial" },
  { value: "levels", label: "levels" },
  { value: "spectrum", label: "spectrum" },
  { value: "whatif", label: "whatif" },
  { value: "forcelaw", label: "forcelaw" },
];
const CONFIG_RE = /^(\d[spdfgh]\d+)( \d[spdfgh]\d+)*$/;

export function Controls() {
  const {
    n, l, m, system, basis, view, colorMode, planeQuantity, count,
    systems, model, config, exchange, pauli, compare, hfLevels,
    nucleusMode,
    setQuantumNumbers, setSystem, setBasis, setView,
    setColorMode, setPlaneQuantity, setCount, setModel,
    setConfig, setExchange, setPauli, setCompare, ensureHF,
    setNucleusMode,
  } = useAppStore();

  const hasGsz = gszAvailable(systems, system);
  const canCompare = compareAvailable(systems, system);
  const isScreened = systems.find((s) => s.key === system)?.kind === "screened";
  const narrow = isNarrow(useViewport().width);

  useEffect(() => {
    if (model === "hf") void ensureHF();
  }, [model, ensureHF]);

  const [draft, setDraft] = useState(config ?? "");
  useEffect(() => setDraft(config ?? ""), [config]);
  const commitConfig = () => {
    const trimmed = draft.trim().replace(/\s+/, " ");
    if (trimmed === "" || CONFIG_RE.test(trimmed)) setConfig(trimmed === "" ? null : trimmed);
    else setDraft(config ?? "");
  };

  const pick = (nn: number, ll: number, mm: number) => {
    const c = clampState(nn, ll, mm);
    setQuantumNumbers(c.n, c.l, c.m);
  };

  const systemOptions =
    systems.length > 0
      ? systems.map((s) => ({ value: s.key, label: `${s.name} (${s.key})` }))
      : PRESET_SYSTEMS.map((k) => ({ value: k, label: k }));

  return (
    <div className="controls">
      <ControlGroup title="State">
        <div className="state-grid">
          <Select
            label="n"
            value={String(n)}
            options={[1, 2, 3, 4, 5, 6].map((v) => ({ value: String(v), label: `n = ${v}` }))}
            onChange={(v) => pick(Number(v), l, m)}
          />
          <div data-tour="l-picker">
            <Select
              label="l"
              value={String(l)}
              options={Array.from({ length: n }, (_, v) => ({
                value: String(v),
                label: `l = ${v}`,
                disabled: !subshellAvailable(hfLevels, model, n, v),
              }))}
              onChange={(v) => pick(n, Number(v), m)}
            />
          </div>
          <Select
            label="m"
            value={String(m)}
            options={Array.from({ length: 2 * l + 1 }, (_, i) => ({
              value: String(i - l),
              label: `m = ${i - l}`,
            }))}
            onChange={(v) => pick(n, l, Number(v))}
          />
        </div>
        <div data-tour="system-picker">
          <Select label="system" value={system} options={systemOptions} onChange={setSystem} />
        </div>
        {systems.length > 0 && systems.some((s) => s.kind === "screened") && (
          <div data-tour="model-picker">
            <Choice<AtomModel>
              legend="model"
              value={model}
              onChange={setModel}
              options={[
                { value: "gsz", label: "screened (GSZ)", disabled: !hasGsz },
                { value: "hf", label: "Hartree-Fock" },
              ]}
            />
          </div>
        )}
        {isScreened && (
          <p className="panel-hint">
            {model === "gsz"
              ? "A fitted central field: one potential for every electron, and no self-consistency."
              : !pauli
                ? "Counterfactual Hartree-Fock with the occupancy cap lifted: every electron in the 1s. Stationary for this altered model, not a variational bound on the real atom."
                : !exchange
                  ? "Counterfactual Hartree-Fock without exchange: distinguishable electrons that still obey the occupancy cap."
                  : "A self-consistent field, solved per subshell with no fitted parameters. What you see is one orbital, not the total density, which for these atoms is exactly spherical."}
          </p>
        )}
        {!hasGsz && systems.length > 0 && (
          <p className="panel-hint">
            Szydlik and Green never published neutral GSZ screening parameters
            for this element, so there is nothing to run the screened model on.
            Hartree-Fock builds its potential out of the orbitals it is solving
            for and needs no fitted table, which is the only reason this atom is
            here at all.
          </p>
        )}
        {isScreened && (
          <label className="control-text">
            <span className="control-label">configuration</span>
            <input
              type="text"
              value={draft}
              placeholder="Aufbau (ground)"
              spellCheck={false}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitConfig}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitConfig();
              }}
            />
          </label>
        )}
        {isScreened && (
          <div data-tour="compare-toggle">
            <Toggle
              label="Compare both models"
              checked={compare}
              disabled={!canCompare}
              disabledReason="This needs both models, and only one of them has parameters for this element."
              onChange={setCompare}
              why="Draws the total density under both models on one axis, with the number of electrons they place differently. The orbital plots stay on the model selected above."
            />
          </div>
        )}
        {isScreened && model === "hf" && (
          <>
            <div data-tour="exchange-toggle">
              <Toggle
                label="distinguishable electrons"
                checked={!exchange}
                disabled={!pauli}
                disabledReason="The switch below forces this on: exchange energy comes from antisymmetry, and antisymmetry is the exclusion principle."
                onChange={(v) => setExchange(!v)}
              why={
                !pauli
                  ? undefined
                  : exchange
                    ? "Exchange on. The wavefunction stays antisymmetric, as it is in this universe."
                    : "Counterfactual. Exchange is gone, so the wavefunction is a product instead of a determinant. The Pauli occupancies are untouched, so this is not electrons piling into 1s."
              }
              />
            </div>
            <div data-tour="pauli-toggle">
              <Toggle
                label="no Pauli exclusion"
                checked={!pauli}
                onChange={(v) => setPauli(!v)}
                why={
                  pauli
                    ? "Occupancies are capped at 2(2l+1), which is why the atom has shells and the periodic table has periods."
                    : "Counterfactual, and the stronger one. The cap is gone, so every electron falls into the 1s: one level, no shells, no chemistry. The Energy levels view compares the two energies and sizes."
                }
              />
            </div>
          </>
        )}
        <div data-tour="basis-picker">
          <Choice<Basis>
            legend="basis"
            value={basis}
            onChange={setBasis}
            options={[
              { value: "complex", label: "complex Y_lm" },
              { value: "real", label: "real orbitals" },
            ]}
          />
        </div>
      </ControlGroup>
      <ControlGroup title="View">
        {!narrow && (
          <div data-tour="view-list">
            <Choice<ViewMode>
              legend="view"
              value={view}
              onChange={setView}
              options={[
                { value: "cloud", label: "cloud" },
                { value: "plane", label: "plane" },
                { value: "radial", label: "radial" },
                { value: "levels", label: "levels" },
                { value: "spectrum", label: "spectrum" },
              ]}
            />
          </div>
        )}
        <Choice<ColorMode>
          legend="cloud color"
          value={colorMode}
          onChange={setColorMode}
          options={[
            { value: "solid", label: "solid" },
            { value: "density", label: "density" },
            { value: "phase", label: "phase", disabled: basis === "real" },
          ]}
        />
        <Choice<PlaneQuantity>
          legend="plane quantity"
          value={planeQuantity}
          onChange={setPlaneQuantity}
          options={[
            { value: "density", label: "density" },
            { value: "psi", label: "psi" },
          ]}
        />
        {narrow && (
          <p className="panel-hint">
            A phone opens at {COUNT_CHOICES[0].toLocaleString()} draws. The larger
            counts still work here and will be slower to draw, not less accurate:
            the sampler is the same one either way, and more draws is a finer
            picture of the same |ψ|².
          </p>
        )}
        <Slider
          label="cloud points"
          readout={count.toLocaleString()}
          min={1000}
          max={300000}
          step={1000}
          value={count}
          onChange={setCount}
        />
        <div data-tour="nucleus-picker">
          <Choice<NucleusMode>
            legend="nucleus"
            value={nucleusMode}
            onChange={setNucleusMode}
            options={NUCLEUS_MODES.map((o) => ({ value: o.value, label: o.label }))}
          />
        </div>
      </ControlGroup>
      <ShowPhysics />
    </div>
  );
}
