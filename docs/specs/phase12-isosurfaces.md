# Phase 12 — Isosurfaces (spec)

Adapted from `../AtomSim` Phase 25 (`2026-08-03-phase25-isosurfaces-design.md`)
plus the Phase-26 HF-3D seam that unblocks `hf_isosurface` over HTTP.
`s/atomsim/atomic/`. Code is current truth; this explains what and why.

## What and why

Requirements spec §5, the one v1 visualization item never built:
*3D orbitals/densities: Monte-Carlo point-cloud sampling of |ψ|² plus
isosurfaces (textbook lobes).* The cloud shipped in Phase 1. This is the
other half.

A lobe is the least honest picture in chemistry. Textbooks draw one without
saying what it is a picture of, and the reader concludes the orbital has a
boundary. It does not. A surface through |ψ|² is a contour choice; the only
thing making it a claim about an atom is the fraction of the electron it
contains. So the control is that fraction: ask for 0.9, the engine solves for
the contour enclosing 90%, and the picture states the complement — outside
one time in ten.

## Engine deltas (port of the reference, wholesale)

- `numerics/marching_tets.py` (new, 311 ln): Kuhn-decomposition marching
  tetrahedra, not cubes. Six tets per cell sharing the main diagonal, 16
  derived cases built at import (`_TET_CASES` from the cut-edge statement,
  not transcribed), outward orientation, welded vertices, `Mesh`,
  `marching_tets(field, level, origin, spacing, chunk)`, `enclosed_volume`
  (divergence theorem), `surface_area`, `edge_use_counts` (watertight = all
  twos), `connected_components` (union-find over welded mesh). Chunked in x
  slabs so 128³ never materializes 12M tets at once.
- `isosurface.py` (new, 561 ln): `GRID_SIZES = (48, 64, 96, 128)`,
  `default_half_width(n, Z, mu_ratio)`, `solve_level` (descending sort +
  cumulative `rho dV` crossing, interpolated; target is fraction of the whole
  electron, refuses what the box cannot hold), `fraction_above`,
  `radial_mass` (Gauss-Legendre `l+2` × uniform-phi `2|m|+2` spherical
  average + 1-D radial quadrature — the tail lives in r, not on the cubic
  grid), `_fit_box` (inscribed sphere to `_BOX_CAPTURE = 0.999`, escaped mass
  is an upper bound), `_density_on_grid` (chunked evaluator), `_build`
  (level solve, mesh, phase via `arg(psi)`, mesh-vs-voxel volumes, halved-grid
  fraction + volume error bars), `isosurface` (`NUMERICAL`),
  `screened_isosurface` (`APPROXIMATION`), `hf_isosurface`
  (`config`/`exchange`/`pauli` threaded, tier read off the solve:
  `APPROXIMATION`, `COUNTERFACTUAL` when switched).
- Reference drift notes ported as behaviour, not prose: tetrahedra instead of
  the 256-case table; box sized by radial integral (cubic-grid sums collapse
  under box growth: H 1s 0.997 at 8 bohr → 0.0005 at 134); fraction error bar
  nearly blind so volume carries its own; p-lobe component count reported as
  measured with cell-size caveat; request names fraction only (no level, no
  box); triangles travel as uint32 with their own decoder; `hf_isosurface`
  exists in-engine with HTTP wiring for all three models via the shared
  `model/config/exchange/pauli` fields (the reference's Phase-25 deferral is
  resolved here by reusing the Phase-11 request shape, not a new endpoint).

## Server

`IsoRequest(ManyElectronRequest)`: `n/l/m`, `fraction (0,1)` default 0.9,
`resolution` in `GRID_SIZES` (validator, 422), `basis`, `system`.
`POST /api/jobs/isosurface` branches `hf` → `hf_isosurface` (with
`_hf_view_target` occupancy + pauli + antisymmetry refusals),
screened atom → `screened_isosurface`, else hydrogenic `isosurface`
(Z + mu_ratio). `IsoMetaModel`: `vertex/triangle_count`, `channels`
(vertices float32 bohr / triangles uint32 / phase float32 rad),
`target/enclosed/outside/level/escaped/mesh_volume/voxel_volume/area`,
`components`, `half_width`, `resolution`, `axis_unit`, `n/l/m/basis/system/
model/label/provenance`. `job_data` serves the three channels as raw
octet-stream; default channel is vertices; unknown channel is 422 naming the
three that exist.

## Web

- `lib/isoSurface.ts`: `buildSurfaceColors` (shared `phaseColor` with the
  cloud — same lobes, same colours), `enclosedCaption` (measured enclosed +
  complement, never the request), `surfaceExtent` (farthest vertex, not the
  box), `componentsCaption` (plain count for s, cell caveat for l>0).
- `components/IsoSurface.tsx`: `BufferGeometry` from engine vertices +
  triangles + colors, `computeVertexNormals` (disclosed smoothing liberty),
  `DoubleSide`, `PHYSICS_TO_SCREEN` rotation shared with the cloud.
- `lib/frame.ts`: the single `PHYSICS_TO_SCREEN = [-π/2, 0, 0]` convention.
- `api/types.ts`: `IsoMeta`; `api/client.ts`: `IsoParams`/`createIsoJob`,
  `getIndexChannel`/`decodeIndices` (uint32, multiple-of-12 guard — float
  decoding of index bytes is plausible garbage, not an error).

## Expected physics (tests assert)

Hydrogen 1s 90% contour at 2.6612 bohr (vertex radii mean, mesh volume, area);
radius tracks fraction (0.5/0.95); tighter contour inside looser; KS-validated
sampler lands inside at the stated rate (40k draws, ±0.01); box holds 99.9%+
for 1s and 4f; too-small hand box refused; mesh-vs-voxel agreement;
refinement reduces error; watertight lobed orbital; s = 1 piece, real-basis
2p 50% = 2 pieces with measured gap, 90% fused with caveat; opposite-sign
lobes via vertex phase; complex m=±1 phase winds as `arg(ψ)−mφ` constant;
`NUMERICAL` tier for exact ψ; complement + no-boundary + inscribed-box +
halving disclosures; fraction bar <1e-3 with real geometric bar beside it;
unsupported resolutions refused; screened (`APPROXIMATION`, Na 3s) and HF
(Be 1s < 2s volume ordering, H 1s → closed form, He bit-identical with/without
exchange, multi-shell differs, Pauli collapse two-measure ordering) surfaces.
Server: end-to-end 1s geometry through the wire, default channel, 422 channel
list, provenance survival, screened tier, muonic scale (÷186), fraction/reso/
quantum-number refusals, tighter-fraction-smaller-surface, HF
counterfactual badge over HTTP.
