import dataclasses
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator
from pydantic import Field as PydanticField

import atomic
from atomic.analytic.dirac import dirac_energy
from atomic.analytic.fine_structure import fine_structure_shift, level_energy
from atomic.analytic.hydrogen import (
    angular_momentum_magnitude,
    energy,
    mean_radius,
    radial_wavefunction,
    validate_quantum_numbers,
)
from atomic.analytic.hyperfine import hyperfine_report
from atomic.analytic.stark import stark_sublevels
from atomic.analytic.wavefunction import WavefunctionValues, evaluate_state
from atomic.analytic.zeeman import zeeman_sublevels
from atomic.atoms import (
    ATOM_KEYS,
    SUBSHELL_LABELS,
    Configuration,
    atom_for_key,
    aufbau_configuration,
    element_by_z,
    format_config,
    has_gsz_parameters,
    is_atom_key,
    parse_config,
    total_electrons,
    validate_config,
)
from atomic.broadening import synthesize
from atomic.classical import classical_ghost
from atomic.constants import ALPHA, BOHR_RADIUS_PM, HARTREE_EV
from atomic.constants_lab import analyze_constants
from atomic.density_compare import compare_total_densities
from atomic.hf_atom import (
    HFResult,
    PauliCollapse,
    evaluate_hf_state,
    hf_exchange_energy,
    hf_radial,
    hf_total_radial_density,
    pauli_collapse,
    solve_hartree_fock,
)
from atomic.isosurface import (
    GRID_SIZES,
    Isosurface,
    hf_isosurface,
    isosurface,
    screened_isosurface,
)
from atomic.numerics.expression import ExpressionError
from atomic.numerics.force_law import PRESETS, force_law_levels, free_form_levels
from atomic.plane import PlaneGrid, hf_plane_grid, plane_grid, screened_plane_grid
from atomic.populations import ThermalConditions
from atomic.provenance import Fidelity, Field, Provenance, Quantity
from atomic.sampling import SampleCloud, sample_density, sample_hf_density, sample_screened_density
from atomic.screened_atom import (
    evaluate_screened_state,
    screened_radial,
    screened_total_radial_density,
    solve_screened_atom,
)
from atomic.server.jobs import Job, JobStatus, JobStore
from atomic.server.ratelimit import DEFAULT_CAPACITY, DEFAULT_PERIOD, TokenBucket
from atomic.server.schemas import (
    AbsorptionSpectrumModel,
    ChannelModel,
    ClassicalGhostModel,
    ComparisonModel,
    ConstantsReportModel,
    CurveOfGrowthModel,
    DensityComparisonModel,
    FieldModel,
    ForceLawModel,
    HFOrbitalModel,
    HFResultModel,
    LineModel,
    PauliCollapseModel,
    ProfileModel,
    ProvenanceModel,
    QuantityModel,
    ScreenedLevelsModel,
    ScreenedOrbitalModel,
    SystemModel,
    ThermalModel,
)
from atomic.server.thumbnails import render_thumbnail
from atomic.spectra import (
    compare_lines,
    load_reference,
    screened_transition_lines,
    subshell_label,
    transition_lines,
)
from atomic.systems import (
    element_emitter_mass,
    emitter_mass,
    get_system,
    hydrogen_like,
    list_systems,
)
from atomic.transfer import absorb, curve_of_growth, default_columns

_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

def _allowed_origins() -> list[str]:
    """CORS allow-list: dev origins plus anything configured for split deploys.

    A static UI hosted away from the engine (e.g. Vercel in front of a Fly
    API) sets ATOMIC_ALLOWED_ORIGINS on the engine, comma-separated. Origins
    must be scheme+host, no trailing slash. CORS only matters for browsers —
    nothing else here ever trusts an Origin header.
    """
    raw = os.environ.get("ATOMIC_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return _DEV_ORIGINS
    extra = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return _DEV_ORIGINS + [o for o in extra if o not in _DEV_ORIGINS]

logger = logging.getLogger(__name__)

def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

def _web_dist() -> Path:
    override = os.environ.get("ATOMIC_WEB_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "web" / "dist"

def _job_worker_count() -> int:
    override = os.environ.get("ATOMIC_JOB_WORKERS")
    if override:
        return max(1, int(override))
    return max(2, min(4, os.cpu_count() or 2))

async def _lifespan(app: FastAPI):
    yield
    app.state.executor.shutdown(wait=False, cancel_futures=True)

def _build_rate_limiter() -> TokenBucket | None:
    if os.environ.get("ATOMIC_RATE_LIMIT", "").lower() in ("off", "0", "false"):
        return None
    return TokenBucket(
        capacity=int(os.environ.get("ATOMIC_RATE_LIMIT_BURST", DEFAULT_CAPACITY)),
        period=float(os.environ.get("ATOMIC_RATE_LIMIT_PERIOD", DEFAULT_PERIOD)),
    )

def _client_key(request, header: str | None) -> str:
    if header:
        forwarded = request.headers.get(header)
        if forwarded:
            candidate = forwarded.split(",")[-1].strip()
            if candidate:
                return candidate
    return request.client.host if request.client else "unknown"

class StarkSublevelModel(BaseModel):
    n1: int
    n2: int
    m: int
    k: int
    energy: QuantityModel
    energy_ev: QuantityModel

class GrossLevelModel(BaseModel):
    n: int
    degeneracy: int
    energy: QuantityModel
    energy_ev: QuantityModel
    sublevels: list[StarkSublevelModel] | None = None

class ZeemanSublevelModel(BaseModel):
    m_j: float
    branch: str
    j_label: float
    high_field_label: str
    energy: QuantityModel
    energy_ev: QuantityModel

class FineLevelModel(BaseModel):
    n: int
    l: int
    j: float
    energy: QuantityModel
    energy_ev: QuantityModel
    shift: QuantityModel
    shift_ev: QuantityModel
    sublevels: list[ZeemanSublevelModel] | None = None

class HyperfineLevelModel(BaseModel):
    F: float
    energy: QuantityModel
    energy_ev: QuantityModel
    shift: QuantityModel
    shift_ev: QuantityModel

class HyperfineShellModel(BaseModel):
    n: int
    available: bool
    nucleus: str | None = None
    I: float | None = None
    A: QuantityModel | None = None
    A_ev: QuantityModel | None = None
    levels: list[HyperfineLevelModel] = []
    note: str | None = None
    reason: str | None = None

class LevelsResponse(BaseModel):
    system: SystemModel
    n_max: int
    fine_structure: bool
    alpha: float
    gross: list[GrossLevelModel]
    fine: list[FineLevelModel] | None
    dirac: bool = False
    b_field: float = 0.0
    e_field: float = 0.0
    hyperfine: bool = False
    hyperfine_shells: list[HyperfineShellModel] | None = None

class StateResponse(BaseModel):
    n: int
    l: int
    m: int
    system: SystemModel
    energy: QuantityModel
    energy_ev: QuantityModel
    mean_radius: QuantityModel
    mean_radius_pm: QuantityModel
    angular_momentum: QuantityModel
    radial_nodes: int
    angular_nodes: int

class SystemsResponse(BaseModel):
    systems: list[SystemModel]

class RadialResponse(BaseModel):
    n: int
    l: int
    system: SystemModel
    r_wavefunction: FieldModel
    radial_probability: FieldModel
    total_density: FieldModel | None = None
    density_comparison: DensityComparisonModel | None = None

class SpectrumResponse(BaseModel):
    system: SystemModel
    n_max: int
    fine_structure: bool
    lines: list[LineModel]
    comparison: list[ComparisonModel] | None
    reference_citation: str | None
    tolerance_relative: float | None
    intensity_note: str | None = None
    thermal: ThermalModel | None = None
    profile: ProfileModel | None = None
    profile_note: str | None = None

class ManyElectronRequest(BaseModel):
    model: Literal["gsz", "hf"] = "gsz"
    config: str | None = None
    exchange: bool = True
    pauli: bool = True

    @model_validator(mode="after")
    def _pauli_off_implies_exchange_off(self) -> "ManyElectronRequest":
        if self.model == "hf" and not self.pauli and self.exchange:
            raise ValueError(
                "pauli=false requires exchange=false: exchange energy is a "
                "consequence of antisymmetry and the exclusion principle IS "
                "antisymmetry, so with the principle off there is nothing for "
                "an exchange integral to act on"
            )
        return self

class SampleRequest(ManyElectronRequest):
    n: int
    l: int
    m: int
    count: int = PydanticField(default=100_000, ge=1_000, le=1_000_000)
    seed: int = 0
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"

class JobModel(BaseModel):
    id: str
    status: str
    progress: float
    error: str | None

class SampleMetaModel(BaseModel):
    kind: Literal["sample"] = "sample"
    count: int
    dtype: str
    layout: str
    unit: str
    n: int
    l: int
    m: int
    basis: str
    system: str
    model: str = "hydrogenic"
    provenance: ProvenanceModel
    channels: list[ChannelModel]

class PlaneRequest(ManyElectronRequest):
    n: int
    l: int
    m: int
    quantity: Literal["density", "psi"] = "density"
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"
    resolution: int = 256

@dataclasses.dataclass(frozen=True)
class SampleJobResult:
    cloud: SampleCloud
    psi: WavefunctionValues

@dataclasses.dataclass(frozen=True)
class HFJobResult:
    result: HFResult
    exchange_energy: Quantity | None
    collapse: PauliCollapse | None

class HFRequest(BaseModel):

    z: int
    n_electrons: int | None = None
    config: str | None = None
    exchange: bool = True
    pauli: bool = True

    @model_validator(mode="after")
    def _pauli_off_implies_exchange_off(self) -> "HFRequest":
        if not self.pauli and self.exchange:
            raise ValueError(
                "pauli=false requires exchange=false: exchange energy is a "
                "consequence of antisymmetry and the exclusion principle IS "
                "antisymmetry, so with the principle off there is nothing for "
                "an exchange integral to act on"
            )
        return self

_HF_MAX_N = 3
_HF_MAX_Z = 36

def _parse_config_or_422(text: str, pauli: bool = True):
    try:
        cfg = parse_config(text)
        validate_config(cfg, pauli)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=f"bad config: {exc}") from exc
    return cfg

def _validate_hf_request(z: int, n_electrons: int, config, pauli: bool = True) -> None:
    if not 1 <= z <= _HF_MAX_Z:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Z must be in [1, {_HF_MAX_Z}], got {z}; the Hartree-Fock "
                f"solver has not been exercised above {_HF_MAX_Z}, and a "
                f"non-relativistic model is a poor description of a heavier atom"
            ),
        )
    if not 1 <= n_electrons <= z + 1:
        raise HTTPException(
            status_code=400,
            detail=f"electron count must be in [1, Z+1] = [1, {z + 1}], got {n_electrons}",
        )
    try:
        validate_config(config, pauli)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if total_electrons(config) != n_electrons:
        raise HTTPException(
            status_code=400,
            detail=(
                f"configuration holds {total_electrons(config)} electrons, "
                f"not the {n_electrons} requested"
            ),
        )
    n_top = max(n for (n, _), _ in config)
    if n_top > _HF_MAX_N:
        raise HTTPException(
            status_code=400,
            detail=(
                f"this configuration occupies n = {n_top}, and convergence "
                f"reaches only n = {_HF_MAX_N}. The {n_top}s guess starts from a "
                f"bare-nucleus potential, which for a diffuse outer shell is far "
                f"too contracted for the self-consistent loop to recover; the "
                f"eigensolver stagnates rather than converging slowly. Ions "
                f"whose outermost shell is n <= {_HF_MAX_N} are fine at any Z up "
                f"to {_HF_MAX_Z}"
            ),
        )

def _hf_channel(n: int, l: int) -> str:
    return f"P_{n}{SUBSHELL_LABELS[l]}"

def _many_electron_target(
    system: str, config: str | None, pauli: bool
) -> tuple[int, int, Configuration]:
    if not is_atom_key(system):
        raise HTTPException(
            status_code=422,
            detail=(
                f"this model needs an atom with a known electron count, "
                f"and {system!r} is a one-electron system; its exact "
                f"wavefunction is already in the other views, so there "
                f"is nothing a self-consistent field would add"
            ),
        )
    element = atom_for_key(system)
    n_electrons = element.z
    cfg = (
        aufbau_configuration(n_electrons, pauli)
        if config is None
        else _parse_config_or_422(config, pauli)
    )
    _validate_hf_request(element.z, n_electrons, cfg, pauli)
    return element.z, n_electrons, cfg

def _hf_view_target(req) -> tuple[int, int, Configuration]:
    z, n_electrons, config = _many_electron_target(
        req.system, req.config, req.pauli
    )
    if (req.n, req.l) not in [nl for nl, _ in config]:
        held = ", ".join(f"{n}{SUBSHELL_LABELS[l]}" for (n, l), _ in config)
        why = (
            "the occupancy cap is lifted, so every electron is in the 1s and no "
            "other orbital exists to be an eigenfunction of anything"
            if not req.pauli
            else "one Fock operator is built per occupied subshell, so there is "
            "no operator for an empty one"
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"subshell {req.n}{SUBSHELL_LABELS[req.l]} is not occupied in "
                f"Z={z}, N={n_electrons} (which holds {held}); {why}"
            ),
        )
    return z, n_electrons, config

def _hf_symbol(z: int) -> str | None:
    try:
        return element_by_z(z).symbol
    except KeyError:
        return None

def _pauli_collapse_model(collapse: PauliCollapse) -> PauliCollapseModel:
    return PauliCollapseModel(
        binding_change=QuantityModel.from_quantity(collapse.binding_change),
        binding_change_ev=QuantityModel.from_quantity(_to_ev(collapse.binding_change)),
        real_total_energy=QuantityModel.from_quantity(collapse.real.total_energy),
        real_total_energy_ev=QuantityModel.from_quantity(
            _to_ev(collapse.real.total_energy)
        ),
        real_config=format_config(collapse.real.config),
        real_radius=QuantityModel.from_quantity(collapse.real_radius),
        collapsed_radius=QuantityModel.from_quantity(collapse.collapsed_radius),
        radius_ratio=QuantityModel.from_quantity(collapse.radius_ratio),
        variational_zeta=QuantityModel.from_quantity(collapse.variational_zeta),
        variational_energy=QuantityModel.from_quantity(collapse.variational_energy),
        variational_energy_ev=QuantityModel.from_quantity(
            _to_ev(collapse.variational_energy)
        ),
    )

def _hf_result_model(
    result: HFResult,
    exchange_energy: Quantity | None = None,
    collapse: PauliCollapse | None = None,
) -> HFResultModel:
    return HFResultModel(
        z=result.z,
        n_electrons=result.n_electrons,
        symbol=_hf_symbol(result.z),
        config=format_config(result.config),
        is_ground=result.is_ground,
        exchange=result.exchange,
        exchange_energy=(
            None if exchange_energy is None
            else QuantityModel.from_quantity(exchange_energy)
        ),
        exchange_energy_ev=(
            None if exchange_energy is None
            else QuantityModel.from_quantity(_to_ev(exchange_energy))
        ),
        pauli=result.pauli,
        collapse=None if collapse is None else _pauli_collapse_model(collapse),
        orbitals=[
            HFOrbitalModel(
                n=o.n, l=o.l, label=f"{o.n}{SUBSHELL_LABELS[o.l]}",
                occupancy=o.occupancy,
                energy=QuantityModel.from_quantity(o.energy),
                energy_ev=QuantityModel.from_quantity(_to_ev(o.energy)),
                channel=_hf_channel(o.n, o.l),
            )
            for o in result.orbitals
        ],
        total_energy=QuantityModel.from_quantity(result.total_energy),
        total_energy_ev=QuantityModel.from_quantity(_to_ev(result.total_energy)),
        kinetic=QuantityModel.from_quantity(result.kinetic),
        potential=QuantityModel.from_quantity(result.potential),
        virial_ratio=QuantityModel.from_quantity(result.virial_ratio),
        iterations=result.iterations,
        coarse_iterations=result.coarse_iterations,
        converged=result.converged,
        provenance=ProvenanceModel.from_provenance(result.provenance),
        grid_channel="grid",
        grid_points=len(result.orbitals[0].P.grid),
        channels=[
            ChannelModel(
                name="grid", dtype="float32", unit="bohr",
                provenance=ProvenanceModel.from_provenance(
                    result.orbitals[0].P.provenance
                ),
            ),
            *(
                ChannelModel(
                    name=_hf_channel(o.n, o.l), dtype="float32", unit=o.P.unit,
                    provenance=ProvenanceModel.from_provenance(o.P.provenance),
                )
                for o in result.orbitals
            ),
        ],
    )

class PlaneMetaModel(BaseModel):
    kind: Literal["plane"] = "plane"
    resolution: int
    dtype: str
    layout: str
    quantity: str
    unit: str
    label: str
    half_extent: float
    axis_unit: str
    n: int
    l: int
    m: int
    basis: str
    system: str
    model: str = "hydrogenic"
    provenance: ProvenanceModel

class IsoRequest(ManyElectronRequest):
    n: int
    l: int
    m: int
    fraction: float = PydanticField(default=0.9, gt=0.0, lt=1.0)
    resolution: int = 96
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"

    @model_validator(mode="after")
    def _resolution_is_offered(self) -> "IsoRequest":
        if self.resolution not in GRID_SIZES:
            raise ValueError(
                f"resolution must be one of {GRID_SIZES}, got {self.resolution}"
            )
        return self

class IsoMetaModel(BaseModel):
    kind: Literal["isosurface"] = "isosurface"
    vertex_count: int
    triangle_count: int
    channels: list[ChannelModel]
    target_fraction: float
    enclosed_fraction: QuantityModel
    outside_fraction: float
    level: QuantityModel
    escaped_fraction: QuantityModel
    mesh_volume: QuantityModel
    voxel_volume: QuantityModel
    area: QuantityModel
    components: int
    half_width: float
    resolution: int
    axis_unit: str
    n: int
    l: int
    m: int
    basis: str
    system: str
    model: str = "hydrogenic"
    label: str
    provenance: ProvenanceModel

def _validate_state(n: int, l: int, m: int) -> None:
    try:
        validate_quantum_numbers(n, l)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if abs(m) > l:
        raise HTTPException(status_code=422, detail=f"|m| must be <= l, got m={m}, l={l}")

def _resolve_system(key: str):
    try:
        return get_system(key)
    except KeyError:
        match = re.fullmatch(r"z(\d+)", key.strip().lower())
        if match:
            try:
                return hydrogen_like(int(match.group(1)))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        available = [s.key for s in list_systems()]
        raise HTTPException(
            status_code=404, detail=f"unknown system {key!r}; available: {available}"
        ) from None

def _screened_element(key: str):
    element = atom_for_key(key)
    if not has_gsz_parameters(element.z):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{element.name} has no published GSZ screening parameters: "
                f"Szydlik and Green, Phys. Rev. A 9, 1885 (1974), tabulate "
                f"neutral He to P and Ar and skip Z = 16 and 17. Inventing "
                f"them would mean shipping physics with no source. Ask "
                f"for model='hf' instead: Hartree-Fock builds its potential "
                f"out of the orbitals it is solving for and needs no fitted "
                f"table, which is why this atom is offered at all"
            ),
        )
    return element

def _to_ev(q: Quantity) -> Quantity:
    return Quantity(
        value=q.value * HARTREE_EV,
        unit="eV",
        label=q.label + " [eV]",
        provenance=dataclasses.replace(
            q.provenance,
            method=q.provenance.method + "; converted to eV via CODATA Hartree-eV factor",
        ),
    )

def _to_pm(q: Quantity) -> Quantity:
    return Quantity(
        value=q.value * BOHR_RADIUS_PM,
        unit="pm",
        label=q.label + " [pm]",
        provenance=dataclasses.replace(
            q.provenance,
            method=q.provenance.method + "; converted to pm via CODATA Bohr radius",
        ),
    )

def _hyperfine_shell_model(rep) -> HyperfineShellModel:
    return HyperfineShellModel(
        n=rep.n,
        available=True,
        nucleus=rep.nucleus_name,
        I=rep.I,
        A=QuantityModel.from_quantity(rep.A) if rep.A is not None else None,
        A_ev=QuantityModel.from_quantity(_to_ev(rep.A)) if rep.A is not None else None,
        levels=[
            HyperfineLevelModel(
                F=lv.F,
                energy=QuantityModel.from_quantity(lv.energy),
                energy_ev=QuantityModel.from_quantity(_to_ev(lv.energy)),
                shift=QuantityModel.from_quantity(lv.shift),
                shift_ev=QuantityModel.from_quantity(_to_ev(lv.shift)),
            )
            for lv in rep.levels
        ],
        note=rep.note,
    )

def _job_model(job: Job) -> JobModel:
    return JobModel(id=job.id, status=job.status.value, progress=job.progress, error=job.error)

def _finished_result(jobs: JobStore, job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    if job.status is not JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"job is {job.status.value}, not done")
    return job.result

def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(title="atomic", version=atomic.__version__, lifespan=_lifespan)
    app.state.job_systems = {}
    app.state.job_models = {}

    def _forget_job(job_id: str) -> None:
        app.state.job_systems.pop(job_id, None)
        app.state.job_models.pop(job_id, None)

    jobs = JobStore(on_evict=_forget_job)
    app.state.jobs = jobs
    app.state.executor = ThreadPoolExecutor(
        max_workers=_job_worker_count(), thread_name_prefix="atomic-job"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.rate_limit = _build_rate_limiter()
    client_ip_header = os.environ.get("ATOMIC_CLIENT_IP_HEADER")

    @app.middleware("http")
    async def _limit_job_creation(request, call_next):
        limiter = app.state.rate_limit
        if (
            limiter is not None
            and request.method == "POST"
            and request.url.path.startswith("/api/jobs/")
        ):
            charged = _client_key(request, client_ip_header)
            wait = limiter.check(charged)
            if wait is not None:
                retry = max(1, math.ceil(wait))
                logger.warning("rate limit refused %s; retry in %ds", charged, retry)
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                    content={
                        "detail": (
                            f"too many compute jobs from this client; retry in "
                            f"{retry}s. Each job is seconds of solver time, "
                            f"so the rate is capped to stay responsive for everyone."
                        )
                    },
                )
        return await call_next(request)

    def _dispatch(job: Job, work) -> JobModel:
        app.state.executor.submit(jobs.run, job.id, work)
        return _job_model(jobs.get(job.id))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": atomic.__version__}

    @app.get("/api/systems", response_model=SystemsResponse)
    def systems() -> SystemsResponse:
        def describe(element) -> str:
            if has_gsz_parameters(element.z):
                return (
                    f"{element.name}: GSZ screened central-field model "
                    f"(APPROXIMATION), or self-consistent Hartree-Fock."
                )
            return (
                f"{element.name}: Hartree-Fock only (APPROXIMATION). Szydlik "
                f"and Green never published neutral GSZ screening parameters "
                f"for Z = {element.z}, and Hartree-Fock needs none."
            )

        hydrogenic = [SystemModel.from_system(s) for s in list_systems()]
        screened = [
            SystemModel.from_atom(
                atom_for_key(k), atom_for_key(k).z, describe(atom_for_key(k)),
            )
            for k in ATOM_KEYS
        ]
        return SystemsResponse(systems=hydrogenic + screened)

    @app.get("/api/state/{n}/{l}/{m}", response_model=StateResponse)
    def state(n: int, l: int, m: int, system: str = "h") -> StateResponse:
        _validate_state(n, l, m)
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        e = energy(n, Z=sys_.Z, mu_ratio=mu)
        mr = mean_radius(n, l, Z=sys_.Z, mu_ratio=mu)
        return StateResponse(
            n=n, l=l, m=m,
            system=SystemModel.from_system(sys_),
            energy=QuantityModel.from_quantity(e),
            energy_ev=QuantityModel.from_quantity(_to_ev(e)),
            mean_radius=QuantityModel.from_quantity(mr),
            mean_radius_pm=QuantityModel.from_quantity(_to_pm(mr)),
            angular_momentum=QuantityModel.from_quantity(angular_momentum_magnitude(l)),
            radial_nodes=n - l - 1,
            angular_nodes=l,
        )

    @app.get("/api/levels", response_model=LevelsResponse | ScreenedLevelsModel)
    def levels(
        system: str = "h",
        n_max: int = 6,
        fine_structure: bool = False,
        alpha: float | None = None,
        config: str | None = None,
        dirac: bool = False,
        b_field: float = 0.0,
        e_field: float = 0.0,
        hyperfine: bool = False,
    ) -> LevelsResponse | ScreenedLevelsModel:
        if not 1 <= n_max <= 20:
            raise HTTPException(status_code=422, detail="n_max must be in [1, 20]")
        if alpha is not None and not 0.0 < alpha <= 0.5:
            raise HTTPException(status_code=422, detail="alpha must be in (0, 0.5]")
        if b_field < 0.0:
            raise HTTPException(status_code=422, detail="b_field must be >= 0")
        if e_field < 0.0:
            raise HTTPException(status_code=422, detail="e_field must be >= 0")
        if is_atom_key(system):
            element = _screened_element(system)
            cfg = (
                aufbau_configuration(element.z)
                if config is None
                else _parse_config_or_422(config)
            )
            if total_electrons(cfg) != element.z:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"config has {total_electrons(cfg)} electrons; "
                        f"{element.symbol} needs {element.z}"
                    ),
                )
            result = solve_screened_atom(
                element.z, total_electrons(cfg), cfg
            )
            return ScreenedLevelsModel(
                system=SystemModel.from_atom(
                    element, element.z, f"{element.name}: GSZ screened central-field model",
                ),
                config=format_config(result.config),
                is_ground=result.is_ground,
                orbitals=[
                    ScreenedOrbitalModel(
                        n=o.n, l=o.l,
                        label=f"{o.n}{SUBSHELL_LABELS[o.l]}{o.occupancy}",
                        occupancy=o.occupancy,
                        energy=QuantityModel.from_quantity(o.energy),
                        energy_ev=QuantityModel.from_quantity(_to_ev(o.energy)),
                    )
                    for o in result.orbitals
                ],
                total_energy=QuantityModel.from_quantity(result.total_energy),
                total_energy_ev=QuantityModel.from_quantity(_to_ev(result.total_energy)),
            )
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        alpha_used = ALPHA if alpha is None else alpha
        gross = []
        for n in range(1, n_max + 1):
            e = energy(n, Z=sys_.Z, mu_ratio=mu)
            gsubs = None
            if e_field > 0.0:
                sss = stark_sublevels(
                    n, Z=sys_.Z, mu_ratio=mu, field_mv_per_m=e_field,
                )
                gsubs = [
                    StarkSublevelModel(
                        n1=s.n1, n2=s.n2, m=s.m, k=s.k,
                        energy=QuantityModel.from_quantity(s.energy),
                        energy_ev=QuantityModel.from_quantity(_to_ev(s.energy)),
                    )
                    for s in sss
                ]
            gross.append(GrossLevelModel(
                n=n, degeneracy=2 * n * n,
                energy=QuantityModel.from_quantity(e),
                energy_ev=QuantityModel.from_quantity(_to_ev(e)),
                sublevels=gsubs,
            ))
        fine = None
        if dirac or fine_structure:
            fine = []
            for n in range(1, n_max + 1):
                for l in range(n):
                    for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
                        if dirac:
                            try:
                                le = dirac_energy(
                                    n, j, Z=sys_.Z, mu_ratio=mu, alpha=alpha_used
                                )
                            except ValueError as exc:
                                raise HTTPException(status_code=422, detail=str(exc)) from exc
                            e_bohr = energy(n, Z=sys_.Z, mu_ratio=mu)
                            sh = dataclasses.replace(
                                le,
                                value=le.value - e_bohr.value,
                                label=f"dE_Dirac {n},{l},j={j:g}",
                            )
                        else:
                            le = level_energy(
                                n, l, j, Z=sys_.Z, mu_ratio=mu,
                                m_over_M=sys_.m_over_M, alpha=alpha_used,
                            )
                            sh = fine_structure_shift(
                                n, l, j, Z=sys_.Z, mu_ratio=mu,
                                m_over_M=sys_.m_over_M, alpha=alpha_used,
                            )
                        subs = None
                        if b_field > 0.0:
                            zss = zeeman_sublevels(
                                n, l, Z=sys_.Z, mu_ratio=mu, m_over_M=sys_.m_over_M,
                                alpha=alpha_used, b_tesla=b_field, dirac=dirac,
                            )
                            subs = [
                                ZeemanSublevelModel(
                                    m_j=z.m_j, branch=z.branch, j_label=z.j_label,
                                    high_field_label=z.high_field_label,
                                    energy=QuantityModel.from_quantity(z.energy),
                                    energy_ev=QuantityModel.from_quantity(_to_ev(z.energy)),
                                )
                                for z in zss
                                if z.j_label == j
                            ]
                        fine.append(FineLevelModel(
                            n=n, l=l, j=j,
                            energy=QuantityModel.from_quantity(le),
                            energy_ev=QuantityModel.from_quantity(_to_ev(le)),
                            shift=QuantityModel.from_quantity(sh),
                            shift_ev=QuantityModel.from_quantity(_to_ev(sh)),
                            sublevels=subs,
                        ))
        hf_shells = None
        if hyperfine:
            first = hyperfine_report(1, sys_)
            if not first.available:
                hf_shells = [HyperfineShellModel(
                    n=1, available=False, reason=first.reason,
                )]
            else:
                hf_shells = [
                    _hyperfine_shell_model(hyperfine_report(n, sys_))
                    for n in range(1, n_max + 1)
                ]
        return LevelsResponse(
            system=SystemModel.from_system(sys_), n_max=n_max,
            fine_structure=fine_structure, alpha=alpha_used, gross=gross, fine=fine,
            dirac=dirac, b_field=b_field, e_field=e_field,
            hyperfine=hyperfine, hyperfine_shells=hf_shells,
        )

    @app.get("/api/spectrum", response_model=SpectrumResponse)
    def spectrum(
        system: str = "h", n_max: int = 6,
        fine_structure: bool = False,
        intensities: bool = False,
        temperature_k: float | None = None,
        electron_density_cm3: float | None = None,
        profile: bool = False,
        resolving_power: float | None = None,
        full_range: bool = False,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> SpectrumResponse:
        thermal = _resolve_thermal(temperature_k, electron_density_cm3)
        if resolving_power is not None and not 1e2 <= resolving_power <= 1e7:
            raise HTTPException(
                status_code=422, detail="resolving_power must be in [1e2, 1e7]"
            )
        zoom = _resolve_zoom(lambda_min, lambda_max)
        if is_atom_key(system):
            element = _screened_element(system)
            result = solve_screened_atom(
                element.z, element.z, aufbau_configuration(element.z)
            )
            lines = screened_transition_lines(
                result, intensities=intensities, thermal=thermal
            )
            reference = load_reference(system)
            comparison = citation = tol = None
            if reference is not None:
                tol = 0.05
                comparison = [
                    ComparisonModel.from_comparison(c)
                    for c in compare_lines(
                        lines, reference, tolerance_relative=tol, window_relative=0.25
                    )
                ]
                citation = reference.citation
            prof = note = None
            if profile:
                prof, note = _synthesize_profile(
                    lines, element_emitter_mass(element), hydrogenic=False,
                    resolving_power=resolving_power, full_range=full_range,
                    zoom=zoom,
                )
            return SpectrumResponse(
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: GSZ screened central-field model (APPROXIMATION).",
                ),
                n_max=lines.n_max, fine_structure=False,
                lines=[LineModel.from_line(ln) for ln in lines.lines],
                comparison=comparison, reference_citation=citation,
                tolerance_relative=tol, intensity_note=lines.intensity_note,
                thermal=(
                    None if lines.thermal is None
                    else ThermalModel.from_state(lines.thermal)
                ),
                profile=prof, profile_note=note,
            )
        if not 2 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [2, 10]")
        sys_ = _resolve_system(system)
        lines = transition_lines(
            sys_, n_max=n_max, fine_structure=fine_structure,
            intensities=intensities, thermal=thermal,
        )
        reference = load_reference(sys_.key)
        comparison = None
        citation = None
        tol = None
        if reference is not None:
            tol = 1e-5 if fine_structure else 3e-5
            comparison = [
                ComparisonModel.from_comparison(c)
                for c in compare_lines(lines, reference, tolerance_relative=tol)
            ]
            citation = reference.citation
        prof = note = None
        if profile:
            prof, note = _synthesize_profile(
                lines, emitter_mass(sys_), hydrogenic=True,
                resolving_power=resolving_power, full_range=full_range,
                zoom=zoom,
            )
        return SpectrumResponse(
            system=SystemModel.from_system(sys_),
            n_max=n_max,
            fine_structure=fine_structure,
            lines=[LineModel.from_line(ln) for ln in lines.lines],
            comparison=comparison,
            reference_citation=citation,
            tolerance_relative=tol,
            intensity_note=lines.intensity_note,
            thermal=(
                None if lines.thermal is None
                else ThermalModel.from_state(lines.thermal)
            ),
            profile=prof, profile_note=note,
        )

    def _resolve_thermal(
        temperature_k: float | None, electron_density_cm3: float | None
    ) -> ThermalConditions | None:
        if temperature_k is None and electron_density_cm3 is None:
            return None
        if temperature_k is None or electron_density_cm3 is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "temperature_k and electron_density_cm3 must be given "
                    "together: ionization depends on both"
                ),
            )
        if not 1e2 <= temperature_k <= 1e6:
            raise HTTPException(
                status_code=422, detail="temperature_k must be in [1e2, 1e6]"
            )
        if not 1e4 <= electron_density_cm3 <= 1e22:
            raise HTTPException(
                status_code=422, detail="electron_density_cm3 must be in [1e4, 1e22]"
            )
        return ThermalConditions(temperature_k, electron_density_cm3)

    def _resolve_zoom(
        lambda_min: float | None, lambda_max: float | None
    ) -> tuple[float, float] | None:
        if lambda_min is None and lambda_max is None:
            return None
        if lambda_min is None or lambda_max is None:
            raise HTTPException(
                status_code=422,
                detail="lambda_min and lambda_max must be given together",
            )
        if not 0.0 < lambda_min < lambda_max:
            raise HTTPException(
                status_code=422, detail="need 0 < lambda_min < lambda_max"
            )
        return (lambda_min, lambda_max)

    def _profile_window(lines) -> tuple[float, float] | None:
        across = [
            ln.wavelength.value for ln in lines if ln.n_upper != ln.n_lower
        ]
        if not across or len(across) == len(lines):
            return None
        return (min(across), max(across))

    def _synthesize_profile(
        lines, mass, hydrogenic: bool, resolving_power: float | None,
        full_range: bool, zoom: tuple[float, float] | None,
    ) -> tuple[ProfileModel | None, str | None]:
        window = zoom if zoom else (None if full_range else _profile_window(lines.lines))
        try:
            syn = synthesize(
                lines, emitter_mass=mass, hydrogenic=hydrogenic,
                resolving_power=resolving_power, window_nm=window,
                max_points=6000,
            )
        except ValueError as exc:
            return None, str(exc)
        return ProfileModel.from_synthetic(syn), None

    def _check_resolving_power(resolving_power: float | None) -> None:
        if resolving_power is not None and not 1e2 <= resolving_power <= 1e7:
            raise HTTPException(
                status_code=422, detail="resolving_power must be in [1e2, 1e7]"
            )

    def _lines_with_strengths(
        system: str, n_max: int, fine_structure: bool, thermal
    ):
        if is_atom_key(system):
            element = _screened_element(system)
            result = solve_screened_atom(
                element.z, element.z, aufbau_configuration(element.z)
            )
            lines = screened_transition_lines(result, intensities=True, thermal=thermal)
            return lines, element_emitter_mass(element), False
        if not 2 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [2, 10]")
        sys_ = _resolve_system(system)
        lines = transition_lines(
            sys_, n_max=n_max, fine_structure=fine_structure,
            intensities=True, thermal=thermal,
        )
        return lines, emitter_mass(sys_), True

    @app.get("/api/absorption", response_model=AbsorptionSpectrumModel)
    def absorption_endpoint(
        system: str = "h", n_max: int = 6, fine_structure: bool = False,
        temperature_k: float = 10000.0, electron_density_cm3: float = 1e13,
        column_density_m2: float = 1e20,
        resolving_power: float | None = None,
        lambda_min: float | None = None, lambda_max: float | None = None,
    ) -> AbsorptionSpectrumModel:
        thermal = _resolve_thermal(temperature_k, electron_density_cm3)
        if not 0.0 <= column_density_m2 <= 1e30:
            raise HTTPException(
                status_code=422,
                detail="column_density_m2 must be in [0, 1e30]",
            )
        _check_resolving_power(resolving_power)
        window = _resolve_zoom(lambda_min, lambda_max)
        lines, mass, hydrogenic = _lines_with_strengths(
            system, n_max, fine_structure, thermal
        )
        try:
            spectrum = absorb(
                lines, column_density_m2, emitter_mass=mass,
                hydrogenic=hydrogenic, resolving_power=resolving_power,
                window_nm=window,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AbsorptionSpectrumModel.from_spectrum(spectrum, lines.thermal)

    @app.get("/api/curve-of-growth", response_model=CurveOfGrowthModel)
    def curve_of_growth_endpoint(
        system: str = "h", n_max: int = 6, fine_structure: bool = False,
        temperature_k: float = 10000.0, electron_density_cm3: float = 1e13,
        lambda_nm: float = 656.28, resolving_power: float | None = None,
    ) -> CurveOfGrowthModel:
        thermal = _resolve_thermal(temperature_k, electron_density_cm3)
        if lambda_nm <= 0.0:
            raise HTTPException(status_code=422, detail="lambda_nm must be > 0")
        _check_resolving_power(resolving_power)
        lines, mass, hydrogenic = _lines_with_strengths(
            system, n_max, fine_structure, thermal
        )

        syn = synthesize(
            lines, emitter_mass=mass, hydrogenic=hydrogenic,
            resolving_power=resolving_power, max_points=2000,
        )
        if not syn.profiles:
            raise HTTPException(status_code=404, detail="no lines in this spectrum")

        paired = list(zip(syn.profiles, syn.lines, strict=True))
        target = min(
            paired, key=lambda pl: abs(pl[0].wavelength_nm - lambda_nm)
        )[0].wavelength_nm
        width, line = max(
            (pl for pl in paired if pl[0].wavelength_nm == target),
            key=lambda pl: (
                pl[1].oscillator_strength.value
                if pl[1].oscillator_strength is not None else 0.0
            ),
        )
        if line.oscillator_strength is None or line.oscillator_strength.value <= 0.0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the {width.label} line has no oscillator strength, so "
                    "there is no absorption cross-section for it and no curve "
                    "of growth"
                ),
            )
        f = line.oscillator_strength.value
        try:
            columns = default_columns(f, width.wavelength_nm, width.sigma_nm,
                                      width.gamma_nm)
            curve = curve_of_growth(
                f, width.wavelength_nm, width.sigma_nm, width.gamma_nm, columns
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CurveOfGrowthModel.from_curve(
            curve, f"{width.label} ({subshell_label(line)})",
            width.sigma_nm, width.gamma_nm
        )

    @app.get("/api/radial/{n}/{l}", response_model=RadialResponse)
    def radial(
        n: int, l: int, system: str = "h", points: int = 400,
        model: Literal["gsz", "hf"] = "gsz",
        config: str | None = None,
        exchange: bool = True,
        pauli: bool = True,
        compare: bool = False,
    ) -> RadialResponse:
        _validate_state(n, l, 0)
        if not 50 <= points <= 2000:
            raise HTTPException(status_code=422, detail="points must be in [50, 2000]")

        def _comparison() -> DensityComparisonModel | None:
            if not compare:
                return None
            z, n_electrons, cfg = _many_electron_target(system, config, pauli)
            _screened_element(system)
            return DensityComparisonModel.from_comparison(
                compare_total_densities(
                    z, n_electrons, config=cfg, exchange=exchange, pauli=pauli,
                )
            )

        if model == "hf":
            if not pauli and exchange:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "pauli=false requires exchange=false: exchange energy "
                        "is a consequence of antisymmetry and the exclusion "
                        "principle IS antisymmetry, so with the principle off "
                        "there is nothing for an exchange integral to act on"
                    ),
                )
            hf_z, hf_n, hf_config = _hf_view_target(
                SimpleNamespace(system=system, config=config, pauli=pauli, n=n, l=l)
            )
            rw, p = hf_radial(
                hf_z, hf_n, n, l, points=points,
                config=hf_config, exchange=exchange, pauli=pauli,
            )
            density = hf_total_radial_density(
                hf_z, hf_n, config=hf_config, exchange=exchange, pauli=pauli,
                points=points,
            )
            element = atom_for_key(system)
            return RadialResponse(
                n=n, l=l,
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: self-consistent Hartree-Fock "
                    f"(APPROXIMATION; COUNTERFACTUAL with a switch thrown).",
                ),
                r_wavefunction=FieldModel.from_field(rw),
                radial_probability=FieldModel.from_field(p),
                total_density=FieldModel.from_field(density),
                density_comparison=_comparison(),
            )
        if is_atom_key(system):
            element = _screened_element(system)
            rw, prob = screened_radial(element.z, element.z, n, l, points=points)
            cfg = (
                aufbau_configuration(element.z)
                if config is None
                else _parse_config_or_422(config)
            )
            density = screened_total_radial_density(
                element.z, element.z, config=cfg, points=points,
            )
            return RadialResponse(
                n=n, l=l,
                system=SystemModel.from_atom(
                    element, element.z, f"{element.name}: GSZ screened central-field model",
                ),
                r_wavefunction=FieldModel.from_field(rw),
                radial_probability=FieldModel.from_field(prob),
                total_density=FieldModel.from_field(density),
                density_comparison=_comparison(),
            )
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        r_max = min(400.0, 60.0 * n * n / (sys_.Z * mu))
        grid = np.linspace(0.0, r_max, points)
        rw = radial_wavefunction(n, l, grid, Z=sys_.Z, mu_ratio=mu)
        prob = Field(
            values=(np.abs(rw.values) ** 2 * grid**2),
            grid=grid,
            unit="bohr^-1",
            grid_unit="bohr",
            label=f"P_{n},{l}(r) (Z={sys_.Z})",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method="P(r) = |R_nl(r)|^2 r^2 from the closed-form R_nl",
                assumptions=rw.provenance.assumptions,
            ),
        )
        return RadialResponse(
            n=n, l=l,
            system=SystemModel.from_system(sys_),
            r_wavefunction=FieldModel.from_field(rw),
            radial_probability=FieldModel.from_field(prob),
            density_comparison=_comparison(),
        )

    @app.get("/api/constants", response_model=ConstantsReportModel)
    def constants_endpoint(
        hbar: float = 1.0, e: float = 1.0, m_e: float = 1.0,
        eps0: float = 1.0, c: float = 1.0,
    ) -> ConstantsReportModel:
        for name, mult in (("hbar", hbar), ("e", e), ("m_e", m_e),
                           ("eps0", eps0), ("c", c)):
            if not 0.25 <= mult <= 4.0:
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} multiplier must be in [0.25, 4], got {mult}",
                )
        return ConstantsReportModel.from_report(
            analyze_constants(hbar=hbar, e=e, m_e=m_e, eps0=eps0, c=c)
        )

    @app.get("/api/classical", response_model=ClassicalGhostModel)
    def classical_endpoint(system: str = "h", n: int = 1) -> ClassicalGhostModel:
        if n < 1:
            raise HTTPException(status_code=422, detail=f"n must be >= 1, got {n}")
        sys_ = _resolve_system(system)
        return ClassicalGhostModel.from_ghost(classical_ghost(n=n, system=sys_))

    @app.get("/api/forcelaw", response_model=ForceLawModel)
    def forcelaw_endpoint(
        preset: str = "powerlaw",
        l: int = 0,
        system: str = "h",
        n_states: int = 4,
        p: float = 1.0,
        lambda_: float = Query(default=3.0, alias="lambda"),
        omega: float = 0.3,
        v0: float = 2.0,
        a: float = 3.0,
        core: float = 0.2,
        expr: str | None = None,
    ) -> ForceLawModel:
        if l < 0:
            raise HTTPException(status_code=422, detail=f"l must be >= 0, got {l}")
        if not 1 <= n_states <= 8:
            raise HTTPException(
                status_code=422, detail=f"n_states must be in [1, 8], got {n_states}"
            )
        sys_ = _resolve_system(system)

        if preset == "custom":
            if not expr or not expr.strip():
                raise HTTPException(status_code=422, detail="custom preset requires 'expr'")
            try:
                result = free_form_levels(expr, l=l, system=sys_, n_states=n_states)
            except (ExpressionError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            if preset not in PRESETS:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown preset {preset!r}; known: {sorted(PRESETS)}",
                )
            supplied = {
                "p": p, "lambda": lambda_, "omega": omega, "v0": v0, "a": a, "core": core,
            }
            params = {spec.name: supplied[spec.name] for spec in PRESETS[preset].params}
            try:
                result = force_law_levels(preset, params, l=l, system=sys_, n_states=n_states)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        return ForceLawModel.from_result(result, SystemModel.from_system(sys_), _to_ev)

    @app.get("/api/thumbnail/{n}/{l}/{m}")
    def thumbnail(n: int, l: int, m: int, system: str = "h",
                  basis: str = "complex", size: int = 120,
                  model: str = "gsz", config: str | None = None,
                  exchange: bool = True, pauli: bool = True) -> Response:
        _validate_state(n, l, m)
        if basis not in ("complex", "real"):
            raise HTTPException(status_code=422, detail=f"unknown basis {basis!r}")
        if not 32 <= size <= 256:
            raise HTTPException(status_code=422, detail="size must be in [32, 256]")
        if model not in ("gsz", "hf"):
            raise HTTPException(status_code=422, detail=f"unknown model {model!r}")
        stored_config: str | None = None
        if is_atom_key(system):
            element = atom_for_key(system)
            if model == "hf":
                if not pauli and exchange:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "pauli=false requires exchange=false: exchange energy is a "
                            "consequence of antisymmetry and the exclusion principle IS "
                            "antisymmetry, so with the principle off there is nothing for "
                            "an exchange integral to act on"
                        ),
                    )
                cfg = (
                    aufbau_configuration(element.z, pauli)
                    if config is None
                    else _parse_config_or_422(config, pauli)
                )
                _validate_hf_request(element.z, element.z, cfg, pauli)
                if (n, l) not in [nl for nl, _ in cfg]:
                    held = ", ".join(
                        f"{nn}{SUBSHELL_LABELS[ll]}" for (nn, ll), _ in cfg
                    )
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"subshell {n}{SUBSHELL_LABELS[l]} is not occupied in "
                            f"Z={element.z}, N={element.z} (which holds {held}); one "
                            f"Fock operator is built per occupied subshell, so an empty "
                            f"one has no operator to be an eigenfunction of"
                        ),
                    )
                stored_config = format_config(cfg)
            else:
                _screened_element(system)
        else:
            try:
                _resolve_system(system)
            except HTTPException as exc:
                raise HTTPException(status_code=422, detail=exc.detail) from exc
        png = render_thumbnail(
            n, l, m, system, basis, size,
            "hf" if model == "hf" else "gsz", stored_config, exchange, pauli,
        )
        return Response(
            content=png, media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/api/jobs/sample", response_model=JobModel)
    async def create_sample_job(req: SampleRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        hf_target = _hf_view_target(req) if req.model == "hf" else None
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        if hf_target is not None:
            hf_z, hf_n, hf_config = hf_target
            app.state.job_models[job.id] = "hf"

            def work(progress):
                cloud = sample_hf_density(
                    hf_z, hf_n, req.n, req.l, req.m, req.count,
                    seed=req.seed, progress=lambda f: progress(0.9 * f),
                    basis=req.basis, config=hf_config,
                    exchange=req.exchange, pauli=req.pauli,
                )
                psi = evaluate_hf_state(
                    hf_z, hf_n, req.n, req.l, req.m,
                    cloud.positions.astype(np.float64), basis=req.basis,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )
                progress(1.0)
                return SampleJobResult(cloud=cloud, psi=psi)

            return _dispatch(job, work)

        if is_atom_key(req.system):
            element = _screened_element(req.system)
            app.state.job_models[job.id] = "screened"

            def work(progress):
                cloud = sample_screened_density(
                    element.z, element.z, req.n, req.l, req.m, req.count,
                    seed=req.seed, progress=lambda f: progress(0.9 * f), basis=req.basis,
                )
                psi = evaluate_screened_state(
                    element.z, element.z, req.n, req.l, req.m,
                    cloud.positions.astype(np.float64), basis=req.basis,
                )
                progress(1.0)
                return SampleJobResult(cloud=cloud, psi=psi)

            return _dispatch(job, work)

        sys_ = _resolve_system(req.system)
        app.state.job_models[job.id] = "hydrogenic"

        def work(progress):
            cloud = sample_density(
                req.n, req.l, req.m, req.count,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                seed=req.seed, progress=lambda f: progress(0.9 * f), basis=req.basis,
            )
            psi = evaluate_state(
                req.n, req.l, req.m, cloud.positions.astype(np.float64),
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, basis=req.basis,
            )
            progress(1.0)
            return SampleJobResult(cloud=cloud, psi=psi)

        return _dispatch(job, work)

    @app.post("/api/jobs/plane", response_model=JobModel)
    async def create_plane_job(req: PlaneRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        if not 16 <= req.resolution <= 1024:
            raise HTTPException(status_code=422, detail="resolution must be in [16, 1024]")
        hf_target = _hf_view_target(req) if req.model == "hf" else None
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        if hf_target is not None:
            hf_z, hf_n, hf_config = hf_target
            app.state.job_models[job.id] = "hf"

            def work(progress):
                return hf_plane_grid(
                    hf_z, hf_n, req.n, req.l, req.m,
                    quantity=req.quantity, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )

            return _dispatch(job, work)

        if is_atom_key(req.system):
            element = _screened_element(req.system)
            app.state.job_models[job.id] = "screened"

            def work(progress):
                return screened_plane_grid(
                    element.z, element.z, req.n, req.l, req.m,
                    quantity=req.quantity, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                )

            return _dispatch(job, work)

        sys_ = _resolve_system(req.system)
        app.state.job_models[job.id] = "hydrogenic"

        def work(progress):
            return plane_grid(
                req.n, req.l, req.m, quantity=req.quantity, basis=req.basis,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                resolution=req.resolution, progress=progress,
            )

        return _dispatch(job, work)

    @app.post("/api/jobs/isosurface", response_model=JobModel)
    async def create_iso_job(req: IsoRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        hf_target = _hf_view_target(req) if req.model == "hf" else None
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        if hf_target is not None:
            hf_z, hf_n, hf_config = hf_target
            app.state.job_models[job.id] = "hf"

            def work(progress):
                return hf_isosurface(
                    hf_z, hf_n, req.n, req.l, req.m,
                    target_fraction=req.fraction, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )

            return _dispatch(job, work)

        if is_atom_key(req.system):
            element = _screened_element(req.system)
            app.state.job_models[job.id] = "screened"

            def work(progress):
                return screened_isosurface(
                    element.z, element.z, req.n, req.l, req.m,
                    target_fraction=req.fraction, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                )

            return _dispatch(job, work)

        sys_ = _resolve_system(req.system)
        app.state.job_models[job.id] = "hydrogenic"

        def work(progress):
            return isosurface(
                req.n, req.l, req.m,
                target_fraction=req.fraction, basis=req.basis,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                resolution=req.resolution, progress=progress,
            )

        return _dispatch(job, work)

    @app.post("/api/jobs/hf", response_model=JobModel)
    async def create_hf_job(req: HFRequest) -> JobModel:
        n_electrons = req.z if req.n_electrons is None else req.n_electrons
        config = (
            aufbau_configuration(n_electrons, req.pauli)
            if req.config is None
            else _parse_config_or_422(req.config, req.pauli)
        )
        _validate_hf_request(req.z, n_electrons, config, req.pauli)
        comparable = not req.pauli and config == aufbau_configuration(
            n_electrons, pauli=False
        )

        job = jobs.create()

        def work(progress):
            result = solve_hartree_fock(
                req.z, n_electrons, config, req.exchange, req.pauli
            )
            delta = (
                None if req.exchange or not req.pauli
                else hf_exchange_energy(req.z, n_electrons, config)
            )
            collapse = pauli_collapse(req.z, n_electrons) if comparable else None
            progress(1.0)
            return HFJobResult(result, delta, collapse)

        return _dispatch(job, work)

    @app.get("/api/jobs/{job_id}", response_model=JobModel)
    def job_status(job_id: str) -> JobModel:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
        return _job_model(job)

    def _sample_meta(
        res: SampleJobResult, system_key: str, model_key: str
    ) -> SampleMetaModel:
        cloud = res.cloud
        channels = [
            ChannelModel(
                name="positions", dtype="float32", unit="bohr",
                provenance=ProvenanceModel.from_provenance(cloud.provenance),
            ),
            ChannelModel(
                name="density", dtype="float32", unit="bohr^-3",
                provenance=ProvenanceModel.from_provenance(res.psi.provenance),
            ),
        ]
        if cloud.basis == "complex":
            channels.append(
                ChannelModel(
                    name="phase", dtype="float32", unit="rad",
                    provenance=ProvenanceModel.from_provenance(res.psi.provenance),
                )
            )
        return SampleMetaModel(
            count=cloud.positions.shape[0], dtype="float32", layout="xyz-interleaved",
            unit="bohr", n=cloud.n, l=cloud.l, m=cloud.m, basis=cloud.basis,
            system=system_key, model=model_key,
            provenance=ProvenanceModel.from_provenance(cloud.provenance),
            channels=channels,
        )

    def _plane_meta(
        pg: PlaneGrid, system_key: str, model_key: str
    ) -> PlaneMetaModel:
        return PlaneMetaModel(
            resolution=pg.values.shape[0], dtype="float32",
            layout="row-major float32; row i = z ascending, col j = x ascending",
            quantity=pg.quantity, unit=pg.unit, label=pg.label,
            half_extent=float(pg.axis[-1]), axis_unit="bohr",
            n=pg.n, l=pg.l, m=pg.m, basis=pg.basis, system=system_key,
            model=model_key,
            provenance=ProvenanceModel.from_provenance(pg.provenance),
        )

    def _iso_meta(
        surf: Isosurface, system_key: str, model_key: str
    ) -> IsoMetaModel:
        prov = ProvenanceModel.from_provenance(surf.provenance)
        return IsoMetaModel(
            vertex_count=int(surf.vertices.shape[0]),
            triangle_count=int(surf.triangles.shape[0]),
            channels=[
                ChannelModel(
                    name="vertices", dtype="float32", unit="bohr", provenance=prov
                ),
                ChannelModel(
                    name="triangles", dtype="uint32", unit="1", provenance=prov
                ),
                ChannelModel(
                    name="phase", dtype="float32", unit="rad",
                    provenance=ProvenanceModel.from_provenance(surf.level.provenance),
                ),
            ],
            target_fraction=surf.target_fraction,
            enclosed_fraction=QuantityModel.from_quantity(surf.enclosed_fraction),
            outside_fraction=surf.outside_fraction,
            level=QuantityModel.from_quantity(surf.level),
            escaped_fraction=QuantityModel.from_quantity(surf.escaped_fraction),
            mesh_volume=QuantityModel.from_quantity(surf.mesh_volume),
            voxel_volume=QuantityModel.from_quantity(surf.voxel_volume),
            area=QuantityModel.from_quantity(surf.area),
            components=surf.components,
            half_width=surf.half_width,
            resolution=surf.resolution,
            axis_unit="bohr",
            n=surf.n, l=surf.l, m=surf.m, basis=surf.basis, system=system_key,
            model=model_key,
            label=surf.label,
            provenance=prov,
        )

    @app.get(
        "/api/jobs/{job_id}/meta",
        response_model=SampleMetaModel | PlaneMetaModel | IsoMetaModel | HFResultModel,
    )
    def job_meta(
        job_id: str,
    ) -> SampleMetaModel | PlaneMetaModel | IsoMetaModel | HFResultModel:
        res = _finished_result(jobs, job_id)
        system_key = app.state.job_systems.get(job_id, "h")
        model_key = app.state.job_models.get(job_id, "hydrogenic")
        if isinstance(res, PlaneGrid):
            return _plane_meta(res, system_key, model_key)
        if isinstance(res, Isosurface):
            return _iso_meta(res, system_key, model_key)
        if isinstance(res, HFJobResult):
            return _hf_result_model(res.result, res.exchange_energy, res.collapse)
        return _sample_meta(res, system_key, model_key)

    def _iso_channel_payload(surf: Isosurface, channel: str | None) -> np.ndarray:
        if channel is None or channel == "vertices":
            return surf.vertices.astype(np.float32)
        if channel == "triangles":
            return surf.triangles.astype(np.uint32)
        if channel == "phase":
            return surf.vertex_phase.astype(np.float32)
        raise HTTPException(
            status_code=422,
            detail=f"no channel {channel!r} on this job; it has vertices, triangles, phase",
        )

    @app.get("/api/jobs/{job_id}/data")
    def job_data(job_id: str, channel: str | None = None) -> Response:
        res = _finished_result(jobs, job_id)
        if isinstance(res, PlaneGrid):
            if channel is not None:
                raise HTTPException(
                    status_code=422, detail="plane jobs have a single channel"
                )
            payload = res.values.astype(np.float32)
        elif isinstance(res, Isosurface):
            payload = _iso_channel_payload(res, channel)
        elif isinstance(res, HFJobResult):
            hf_res = res.result
            if channel is None or channel == "grid":
                payload = hf_res.orbitals[0].P.grid.astype(np.float32)
            else:
                for o in hf_res.orbitals:
                    if _hf_channel(o.n, o.l) == channel:
                        payload = o.P.values.astype(np.float32)
                        break
                else:
                    known = ", ".join(
                        ["grid", *(_hf_channel(o.n, o.l) for o in hf_res.orbitals)]
                    )
                    raise HTTPException(
                        status_code=422,
                        detail=f"no channel {channel!r} on this job; it has {known}",
                    )
        elif (channel or "positions") == "positions":
            payload = res.cloud.positions
        elif channel == "density":
            payload = (np.abs(res.psi.values) ** 2).astype(np.float32)
        elif channel == "phase" and res.cloud.basis == "complex":
            payload = np.angle(res.psi.values).astype(np.float32)
        else:
            raise HTTPException(status_code=422, detail=f"no channel {channel!r} on this job")
        return Response(
            content=payload.tobytes(), media_type="application/octet-stream"
        )

    web_dist = _web_dist()
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
        logger.info("Mounted the UI from %s", web_dist)
    else:
        logger.warning("No UI found at %s; serving the API only", web_dist)

    return app
