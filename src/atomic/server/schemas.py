import dataclasses
from typing import Literal

from pydantic import BaseModel

from atomic.atoms import Element, has_gsz_parameters
from atomic.broadening import SyntheticSpectrum
from atomic.classical import BohrOrbit, ClassicalGhost
from atomic.constants import BOHR_RADIUS_FM, HARTREE_EV
from atomic.constants_lab import ConstantsReport, DerivedObservable
from atomic.numerics.force_law import ForceLawResult
from atomic.populations import ThermalState
from atomic.provenance import Fidelity, Field, Provenance, Quantity
from atomic.systems import System
from atomic.transfer import AbsorbingLine, AbsorptionSpectrum, CurveOfGrowth

FidelityName = Literal[
    "exact", "numerical", "approximation", "counterfactual", "visual_liberty"
]

class ProvenanceModel(BaseModel):
    fidelity: FidelityName
    method: str
    assumptions: list[str]
    error_estimate: float | None
    refinement: str | None

    @classmethod
    def from_provenance(cls, p: Provenance) -> "ProvenanceModel":
        return cls(
            fidelity=p.fidelity.value,
            method=p.method,
            assumptions=list(p.assumptions),
            error_estimate=p.error_estimate,
            refinement=p.refinement,
        )

class ChannelModel(BaseModel):
    name: str
    dtype: str
    unit: str
    provenance: ProvenanceModel

class QuantityModel(BaseModel):
    value: float
    unit: str
    label: str
    provenance: ProvenanceModel

    @classmethod
    def from_quantity(cls, q: Quantity) -> "QuantityModel":
        return cls(
            value=q.value,
            unit=q.unit,
            label=q.label,
            provenance=ProvenanceModel.from_provenance(q.provenance),
        )

class FieldModel(BaseModel):
    values: list[float]
    grid: list[float]
    unit: str
    grid_unit: str
    label: str
    provenance: ProvenanceModel

    @classmethod
    def from_field(cls, f: Field) -> "FieldModel":
        if f.values.ndim != 1:
            raise ValueError(f"I only serialize 1-D fields in M1, got shape {f.values.shape}")
        return cls(
            values=f.values.tolist(),
            grid=f.grid.tolist(),
            unit=f.unit,
            grid_unit=f.grid_unit,
            label=f.label,
            provenance=ProvenanceModel.from_provenance(f.provenance),
        )

class DerivedObservableModel(BaseModel):
    quantity: QuantityModel
    ratio: float
    changed: bool

    @classmethod
    def from_observable(cls, o: DerivedObservable) -> "DerivedObservableModel":
        return cls(
            quantity=QuantityModel.from_quantity(o.quantity),
            ratio=o.ratio,
            changed=o.changed,
        )

class ConstantsReportModel(BaseModel):
    alpha: DerivedObservableModel
    bohr_radius_pm: DerivedObservableModel
    hartree_ev: DerivedObservableModel
    altered: bool

    @classmethod
    def from_report(cls, r: ConstantsReport) -> "ConstantsReportModel":
        return cls(
            alpha=DerivedObservableModel.from_observable(r.alpha),
            bohr_radius_pm=DerivedObservableModel.from_observable(r.bohr_radius_pm),
            hartree_ev=DerivedObservableModel.from_observable(r.hartree_ev),
            altered=r.altered,
        )

class BohrOrbitModel(BaseModel):
    n: int
    radius_bohr: QuantityModel
    radius_pm: QuantityModel

    @classmethod
    def from_orbit(cls, o: BohrOrbit) -> "BohrOrbitModel":
        return cls(
            n=o.n,
            radius_bohr=QuantityModel.from_quantity(o.radius_bohr),
            radius_pm=QuantityModel.from_quantity(o.radius_pm),
        )

class ClassicalGhostModel(BaseModel):
    n: int
    system_key: str
    z: int
    orbits: list[BohrOrbitModel]
    r0_bohr: QuantityModel
    collapse_time_s: QuantityModel
    orbital_period_s: QuantityModel
    orbit_count: QuantityModel

    @classmethod
    def from_ghost(cls, g: ClassicalGhost) -> "ClassicalGhostModel":
        return cls(
            n=g.n,
            system_key=g.system_key,
            z=g.z,
            orbits=[BohrOrbitModel.from_orbit(o) for o in g.orbits],
            r0_bohr=QuantityModel.from_quantity(g.r0_bohr),
            collapse_time_s=QuantityModel.from_quantity(g.collapse_time_s),
            orbital_period_s=QuantityModel.from_quantity(g.orbital_period_s),
            orbit_count=QuantityModel.from_quantity(g.orbit_count),
        )

def _to_fm(q: Quantity) -> Quantity:
    return Quantity(
        value=q.value * BOHR_RADIUS_FM,
        unit="fm",
        label=q.label + " [fm]",
        provenance=dataclasses.replace(
            q.provenance,
            method=q.provenance.method + "; converted to fm via the CODATA Bohr radius",
            error_estimate=(
                None if q.provenance.error_estimate is None
                else q.provenance.error_estimate * BOHR_RADIUS_FM
            ),
        ),
    )

class SystemModel(BaseModel):
    key: str
    name: str
    z: int
    mu_ratio: QuantityModel
    m_over_m_nucleus: float
    description: str
    nuclear_radius: QuantityModel | None
    nuclear_radius_fm: QuantityModel | None
    kind: Literal["hydrogenic", "screened"] = "hydrogenic"
    n_electrons: int | None = None
    has_gsz: bool = True

    @classmethod
    def from_system(cls, s: System) -> "SystemModel":
        r = s.nuclear_radius
        return cls(
            key=s.key,
            name=s.name,
            z=s.Z,
            mu_ratio=QuantityModel.from_quantity(s.mu_ratio),
            m_over_m_nucleus=s.m_over_M,
            description=s.description,
            nuclear_radius=None if r is None else QuantityModel.from_quantity(r),
            nuclear_radius_fm=None if r is None else QuantityModel.from_quantity(_to_fm(r)),
        )

    @classmethod
    def from_atom(cls, element: Element, n_electrons: int, description: str) -> "SystemModel":
        mu = Quantity(
            1.0, "m_e", f"mu/m_e ({element.name})",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="assumes infinite nuclear mass (screened-atom model)",
            ),
        )
        return cls(
            key=element.symbol.lower(), name=element.name, z=element.z,
            mu_ratio=QuantityModel.from_quantity(mu), m_over_m_nucleus=0.0,
            description=description, nuclear_radius=None, nuclear_radius_fm=None,
            kind="screened", n_electrons=n_electrons,
            has_gsz=has_gsz_parameters(element.z),
        )

class ScreenedOrbitalModel(BaseModel):
    n: int
    l: int
    label: str
    occupancy: int
    energy: QuantityModel
    energy_ev: QuantityModel

class ScreenedLevelsModel(BaseModel):
    system: SystemModel
    config: str
    is_ground: bool
    orbitals: list[ScreenedOrbitalModel]
    total_energy: QuantityModel
    total_energy_ev: QuantityModel

class HFOrbitalModel(BaseModel):

    n: int
    l: int
    label: str
    occupancy: int
    energy: QuantityModel
    energy_ev: QuantityModel
    channel: str

class PauliCollapseModel(BaseModel):

    binding_change: QuantityModel
    binding_change_ev: QuantityModel
    real_total_energy: QuantityModel
    real_total_energy_ev: QuantityModel
    real_config: str
    real_radius: QuantityModel
    collapsed_radius: QuantityModel
    radius_ratio: QuantityModel
    variational_zeta: QuantityModel
    variational_energy: QuantityModel
    variational_energy_ev: QuantityModel

class HFResultModel(BaseModel):

    kind: Literal["hf"] = "hf"
    z: int
    n_electrons: int
    symbol: str | None
    config: str
    is_ground: bool
    exchange: bool = True
    exchange_energy: QuantityModel | None = None
    exchange_energy_ev: QuantityModel | None = None
    pauli: bool = True
    collapse: PauliCollapseModel | None = None
    orbitals: list[HFOrbitalModel]
    total_energy: QuantityModel
    total_energy_ev: QuantityModel
    kinetic: QuantityModel
    potential: QuantityModel
    virial_ratio: QuantityModel
    iterations: int
    coarse_iterations: int
    converged: bool
    provenance: ProvenanceModel
    grid_channel: str
    grid_points: int
    channels: list[ChannelModel]

class ForceLawLevelModel(BaseModel):
    radial_index: int
    energy: QuantityModel
    energy_ev: QuantityModel
    trusted: bool = True

class LineModel(BaseModel):
    n_upper: int
    l_upper: int
    j_upper: float | None
    n_lower: int
    l_lower: int
    j_lower: float | None
    energy_ev: QuantityModel
    wavelength_nm: QuantityModel
    einstein_a_s: QuantityModel | None = None
    oscillator_strength: QuantityModel | None = None
    emissivity: QuantityModel | None = None

    @classmethod
    def from_line(cls, ln) -> "LineModel":
        return cls(
            n_upper=ln.n_upper, l_upper=ln.l_upper, j_upper=ln.j_upper,
            n_lower=ln.n_lower, l_lower=ln.l_lower, j_lower=ln.j_lower,
            energy_ev=QuantityModel.from_quantity(ln.energy),
            wavelength_nm=QuantityModel.from_quantity(ln.wavelength),
            einstein_a_s=(
                None if ln.einstein_a is None
                else QuantityModel.from_quantity(ln.einstein_a)
            ),
            oscillator_strength=(
                None if ln.oscillator_strength is None
                else QuantityModel.from_quantity(ln.oscillator_strength)
            ),
            emissivity=(
                None if ln.emissivity is None
                else QuantityModel.from_quantity(ln.emissivity)
            ),
        )

class ThermalModel(BaseModel):

    temperature_k: float
    electron_density_cm3: float
    ionized_fraction: QuantityModel
    partition_function: QuantityModel

    @classmethod
    def from_state(cls, state: ThermalState) -> "ThermalModel":
        return cls(
            temperature_k=state.conditions.temperature_k,
            electron_density_cm3=state.conditions.electron_density_cm3,
            ionized_fraction=QuantityModel.from_quantity(state.ionized_fraction),
            partition_function=QuantityModel.from_quantity(state.partition_function),
        )

class LineWidthModel(BaseModel):

    label: str
    wavelength_nm: float
    n_upper: int
    n_lower: int
    sigma_nm: float
    gamma_nm: float
    fwhm_nm: float
    terms: list[str]

class ProfileModel(BaseModel):

    wavelength_nm: list[float]
    intensity: list[float]
    unit: str
    weight_kind: str
    resolving_power: float | None
    flux_closure: float
    widths: list[LineWidthModel]
    stark_span_nm: QuantityModel | None
    stark_note: str | None
    provenance: ProvenanceModel

    @classmethod
    def from_synthetic(cls, syn: SyntheticSpectrum) -> "ProfileModel":
        return cls(
            wavelength_nm=[float(x) for x in syn.spectrum.grid],
            intensity=[float(v) for v in syn.spectrum.values],
            unit=syn.spectrum.unit,
            weight_kind=syn.weight_kind,
            resolving_power=syn.resolving_power,
            flux_closure=syn.flux_closure,
            widths=[
                LineWidthModel(
                    label=p.label, wavelength_nm=p.wavelength_nm,
                    n_upper=p.n_upper, n_lower=p.n_lower,
                    sigma_nm=p.sigma_nm, gamma_nm=p.gamma_nm,
                    fwhm_nm=p.fwhm_nm, terms=list(p.terms),
                )
                for p in syn.profiles
            ],
            stark_span_nm=(
                None if syn.stark_span is None
                else QuantityModel.from_quantity(syn.stark_span)
            ),
            stark_note=syn.stark_note,
            provenance=ProvenanceModel.from_provenance(syn.spectrum.provenance),
        )

class CurveOfGrowthModel(BaseModel):

    label: str
    wavelength_nm: float
    oscillator_strength: float
    sigma_nm: float
    gamma_nm: float
    damping_parameter: float
    column_density_m2: list[float]
    equivalent_width_nm: list[float]
    regime: list[str]
    slope: list[float]
    tau_centre: list[float]
    window_nm: float
    provenance: ProvenanceModel

    @classmethod
    def from_curve(
        cls, c: CurveOfGrowth, label: str, sigma_nm: float, gamma_nm: float
    ) -> "CurveOfGrowthModel":
        return cls(
            label=label,
            wavelength_nm=c.wavelength_nm,
            oscillator_strength=c.oscillator_strength,
            sigma_nm=sigma_nm,
            gamma_nm=gamma_nm,
            damping_parameter=c.damping_parameter,
            column_density_m2=[float(v) for v in c.column_density],
            equivalent_width_nm=[float(v) for v in c.equivalent_width],
            regime=list(c.regime),
            slope=[float(v) for v in c.slope],
            tau_centre=[float(v) for v in c.tau_centre],
            window_nm=c.window_nm,
            provenance=ProvenanceModel.from_provenance(c.provenance),
        )

class AbsorbingLineModel(BaseModel):

    wavelength_nm: float
    label: str
    oscillator_strength: float
    lower_column_m2: float
    tau_centre: float
    regime: str
    thin_width_nm: float
    fwhm_nm: float

    @classmethod
    def from_line(cls, d: AbsorbingLine) -> "AbsorbingLineModel":
        return cls(
            wavelength_nm=d.wavelength_nm,
            label=d.label,
            oscillator_strength=d.oscillator_strength,
            lower_column_m2=d.lower_column_m2,
            tau_centre=d.tau_centre,
            regime=d.regime,
            thin_width_nm=d.thin_width_nm,
            fwhm_nm=d.fwhm_nm,
        )

class AbsorptionSpectrumModel(BaseModel):

    wavelength_nm: list[float]
    transmission: list[float]
    optical_depth: list[float]
    lines: list[AbsorbingLineModel]
    thermal: ThermalModel | None
    column_density_m2: float
    equivalent_width_nm: float
    thin_limit_width_nm: float
    saturation: float
    blends: list[tuple[str, str]]
    flux_closure: float
    provenance: ProvenanceModel
    column_provenance: ProvenanceModel

    @classmethod
    def from_spectrum(
        cls, a: AbsorptionSpectrum, thermal: ThermalState | None = None
    ) -> "AbsorptionSpectrumModel":
        return cls(
            wavelength_nm=[float(v) for v in a.transmission.grid],
            transmission=[float(v) for v in a.transmission.values],
            optical_depth=[float(v) for v in a.optical_depth.values],
            lines=[AbsorbingLineModel.from_line(d) for d in a.lines],
            thermal=(
                None if thermal is None else ThermalModel.from_state(thermal)
            ),
            column_density_m2=a.column_density.value,
            equivalent_width_nm=a.equivalent_width.value,
            thin_limit_width_nm=a.thin_limit_width.value,
            saturation=a.saturation.value,
            blends=[tuple(b) for b in a.blends],
            flux_closure=a.flux_closure,
            provenance=ProvenanceModel.from_provenance(
                a.transmission.provenance
            ),
            column_provenance=ProvenanceModel.from_provenance(
                a.column_density.provenance
            ),
        )

class ComparisonModel(BaseModel):
    wavelength_nm: float
    reference_nm: float
    reference_uncertainty_nm: float | None
    delta_nm: float
    relative_error: float
    within_tolerance: bool

    @classmethod
    def from_comparison(cls, c) -> "ComparisonModel":
        return cls(
            wavelength_nm=c.line.wavelength.value,
            reference_nm=c.reference_nm,
            reference_uncertainty_nm=c.reference_uncertainty_nm,
            delta_nm=c.delta_nm,
            relative_error=c.relative_error,
            within_tolerance=c.within_tolerance,
        )

class ShellPeakModel(BaseModel):
    label: str
    gsz_radius: float | None
    hf_radius: float | None
    gsz_depth: float | None
    hf_depth: float | None

class DensityComparisonModel(BaseModel):
    gsz: FieldModel
    hf: FieldModel
    displaced_charge: QuantityModel
    shells: list[ShellPeakModel]
    provenance: ProvenanceModel

    @classmethod
    def from_comparison(cls, c) -> "DensityComparisonModel":
        return cls(
            gsz=FieldModel.from_field(c.gsz),
            hf=FieldModel.from_field(c.hf),
            displaced_charge=QuantityModel.from_quantity(c.displaced_charge),
            shells=[
                ShellPeakModel(
                    label=s.label,
                    gsz_radius=s.gsz_radius, hf_radius=s.hf_radius,
                    gsz_depth=s.gsz_depth, hf_depth=s.hf_depth,
                )
                for s in c.shells
            ],
            provenance=ProvenanceModel.from_provenance(c.provenance),
        )

class ReferenceItemModel(BaseModel):
    label: str
    energy: QuantityModel
    energy_ev: QuantityModel

class ReferenceModel(BaseModel):
    kind: Literal["levels", "markers"]
    items: list[ReferenceItemModel]

class PotentialCurveModel(BaseModel):
    r: list[float]
    v_ev: list[float]
    provenance: ProvenanceModel

class ForceLawModel(BaseModel):
    preset: str
    params: dict[str, float]
    l: int
    z: int
    system: SystemModel
    counterfactual: list[ForceLawLevelModel]
    bound_count: int
    requested_count: int
    reference: ReferenceModel
    potential_curve: PotentialCurveModel
    expression: str | None = None

    @classmethod
    def from_result(cls, result: ForceLawResult, system: SystemModel, to_ev) -> "ForceLawModel":
        curve = result.potential_curve
        return cls(
            preset=result.preset_key,
            params=result.params,
            l=result.l,
            z=result.z,
            system=system,
            counterfactual=[
                ForceLawLevelModel(
                    radial_index=c.radial_index,
                    energy=QuantityModel.from_quantity(c.energy),
                    energy_ev=QuantityModel.from_quantity(to_ev(c.energy)),
                    trusted=c.trusted,
                )
                for c in result.counterfactual
            ],
            bound_count=result.bound_count,
            requested_count=result.requested_count,
            reference=ReferenceModel(
                kind=result.reference.kind,
                items=[
                    ReferenceItemModel(
                        label=item.label,
                        energy=QuantityModel.from_quantity(item.energy),
                        energy_ev=QuantityModel.from_quantity(to_ev(item.energy)),
                    )
                    for item in result.reference.items
                ],
            ),
            potential_curve=PotentialCurveModel(
                r=curve.grid.tolist(),
                v_ev=(curve.values * HARTREE_EV).tolist(),
                provenance=ProvenanceModel.from_provenance(
                    dataclasses.replace(
                        curve.provenance,
                        method=curve.provenance.method
                        + "; converted to eV via CODATA Hartree-eV factor",
                    )
                ),
            ),
            expression=result.expression,
        )
