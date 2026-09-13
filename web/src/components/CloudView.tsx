import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import type * as THREE from "three";
import { formatSeconds, slowMotionFactor } from "../lib/classical";
import { Notation } from "../lib/mathText";
import { stateLabel } from "../lib/quantum";
import { systemKind } from "../lib/systemKind";
import { buildCloudColors } from "../lib/cloudColors";
import {
  buildSurfaceColors,
  componentsCaption,
  enclosedCaption,
  surfaceExtent,
} from "../lib/isoSurface";
import {
  CLASSICAL_SLOWMO,
  NUCLEUS_MARKER_LIBERTY,
  RENDER_LIBERTIES,
} from "../lib/liberties";
import { nucleusCaption, nucleusSphere } from "../lib/nucleus";
import type { SurfaceMode } from "../lib/urlState";
import { ISO_FRACTIONS, useAppStore } from "../state/store";
import { AxisTriad, axisArmLength } from "./AxisTriad";
import { Badge } from "./Badge";
import { GhostClock, GhostOverlay } from "./GhostOverlay";
import { IsoSurface } from "./IsoSurface";
import { Legend } from "./Legend";
import { PointCloud } from "./PointCloud";
import { ViewIntro } from "./ViewIntro";

function formatArm(length: number): string {
  return length.toFixed(length >= 100 ? 0 : length >= 10 ? 1 : 2);
}

function CameraRig({ distance }: { distance: number }) {
  const camera = useThree((s) => s.camera as THREE.PerspectiveCamera);
  useEffect(() => {
    camera.position.set(distance * 0.7, distance * 0.45, distance);
    camera.near = distance / 100;
    camera.far = distance * 100;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, distance]);
  return null;
}

const WEBGL = (() => {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
})();

export function CloudView() {
  const {
    n, l, m, system, systems, basis, count,
    positions, density, phase, meta, status, error,
    colorMode, sample, stateInfo,
    nucleusMode,
    ghostOn, setGhostOn, ghost, ghostStatus, loadGhost,
    surfaceMode, setSurfaceMode, isoFraction, setIsoFraction,
    iso, isoStatus, loadIso,
  } = useAppStore();
  const kind = systemKind(systems, system);
  useEffect(() => {
    if (kind === "hydrogenic" && ghostOn && ghostStatus === "idle") void loadGhost();
  }, [kind, ghostOn, ghostStatus, loadGhost]);
  const ghostTauRef = useRef(0);
  const showSurface = surfaceMode !== "cloud";
  const showCloud = surfaceMode !== "surface";
  useEffect(() => {
    if (showCloud) void sample();
  }, [n, l, m, system, basis, count, showCloud, sample]);
  useEffect(() => {
    if (showSurface && isoStatus === "idle") void loadIso();
  }, [showSurface, isoStatus, loadIso, n, l, m, system, basis, isoFraction]);

  const colors = useMemo(
    () => buildCloudColors(colorMode, density, phase),
    [colorMode, density, phase],
  );
  const surfaceColors = useMemo(
    () => (iso ? buildSurfaceColors(iso.phase) : null),
    [iso],
  );

  const meanRadiusDistance = stateInfo
    ? Math.max(6 * stateInfo.mean_radius.value, 1e-3)
    : 5 * n * n + 3;
  const distance =
    surfaceMode === "surface" && iso
      ? Math.max(2.6 * surfaceExtent(iso.vertices), 1e-3)
      : meanRadiusDistance;
  const sysInfo = stateInfo?.system ?? null;
  const nucleus = nucleusSphere(
    nucleusMode,
    sysInfo?.nuclear_radius?.value ?? null,
    distance,
  );
  const caption = nucleusCaption(nucleusMode, sysInfo, nucleus);

  if (status === "error" && showCloud) {
    return (
      <div className="view-wrap">
        <p className="hint-block">Sampling failed: {error ?? "unknown reason"}</p>
      </div>
    );
  }
  if ((showCloud && (!positions || !meta)) || (showSurface && isoStatus === "loading" && !iso)) {
    return (
      <div className="view-wrap">
        <p className="hint-block">
          {showSurface && !showCloud ? "Solving for the level…" : "Sampling the cloud…"}
        </p>
      </div>
    );
  }

  if (!WEBGL) {
    return (
      <div className="view-wrap">
        <p className="hint-block">
          This browser has no WebGL, so there is no 3-D point cloud to draw. The
          other views are all 2-D and work here: the cross-section shows the
          same orbital, and the radial plots show where the electron is without
          needing a renderer at all.
        </p>
      </div>
    );
  }

  return (
    <div className="view-wrap">
      <ViewIntro
        lead={{
          title: `The ${stateLabel(n, l, m)} cloud in ${meta?.system ?? system}`,
          lead:
            `${(meta?.count ?? count).toLocaleString()} Monte-Carlo points of |psi|², drawn where ` +
            "the electron is likely to be. Denser dots, denser electron.",
        }}
        badge={meta ? <Badge provenance={meta.provenance} /> : undefined}
      />
      <div className="stage-3d">
        <Canvas camera={{ fov: 50 }} dpr={[1, 2]}>
          <color attach="background" args={["#080c0e"]} />
          <CameraRig distance={distance} />
          <AxisTriad distance={distance} />
          {showCloud && positions && (
            <PointCloud
              positions={positions}
              pointSize={distance / 350}
              colors={colors}
            />
          )}
          {ghostOn && ghost && (
            <GhostOverlay ghost={ghost} distance={distance} tauRef={ghostTauRef} />
          )}
          {showSurface && iso && surfaceColors && (
            <>
              <ambientLight intensity={0.65} />
              <directionalLight position={[1, 1, 1]} intensity={1.1} />
              <IsoSurface
                vertices={iso.vertices}
                triangles={iso.triangles}
                colors={surfaceColors}
              />
            </>
          )}
          {nucleus && (
            <mesh>
              <sphereGeometry args={[nucleus.radius, 32, 16]} />
              <meshBasicMaterial
                color={nucleus.kind === "marker" ? "#ffb86b" : "#ffd9a0"}
              />
            </mesh>
          )}
          <OrbitControls />
        </Canvas>
        {meta && (
          <div className="stage-caption">
            |ψ|² Monte-Carlo · {meta.count.toLocaleString()} draws
          </div>
        )}
        <div className="canvas-overlay">
          <Badge provenance={RENDER_LIBERTIES} />
          {nucleus?.kind === "marker" && <Badge provenance={NUCLEUS_MARKER_LIBERTY} />}
          {caption && <span className="nucleus-caption">{caption}</span>}
          <span className="ghost-readout">
            axes ±{formatArm(axisArmLength(distance))} a{"₀"} · z is the
            quantization axis
          </span>
          <Legend mode={colorMode} />
        </div>
        {kind === "hydrogenic" && (
          <label className="ghost-toggle">
            <input
              type="checkbox"
              checked={ghostOn}
              onChange={(e) => setGhostOn(e.target.checked)}
            />
            Classical ghost
          </label>
        )}
        {kind === "screened" && (
          <span className="ghost-readout">
            No classical ghost here: a Kepler orbit needs a 1/r field, and
            screening is exactly what this model adds.
          </span>
        )}
        {ghostOn && ghostStatus === "loading" && (
          <span className="ghost-readout">loading classical orbits…</span>
        )}
        {ghostOn && ghost && (
          <div className="ghost-hud">
            <div className="ghost-banner">
              Counterfactual: a classical electron would spiral in; real atoms do not
            </div>
            <GhostClock
              tauRef={ghostTauRef}
              collapseSeconds={ghost.collapse_time_s.value}
            />
            <div className="ghost-readout">
              collapse in {formatSeconds(ghost.collapse_time_s.value)}{" "}
              <Badge provenance={ghost.collapse_time_s.provenance} />
            </div>
            <div className="ghost-readout">
              {Math.round(ghost.orbit_count.value).toLocaleString()} orbits before
              collapse <Badge provenance={ghost.orbit_count.provenance} />
            </div>
            <div className="ghost-readout">
              playing at about{" "}
              <Notation>{slowMotionFactor(ghost.collapse_time_s.value).toExponential(1)}</Notation>×
              slow motion <Badge provenance={CLASSICAL_SLOWMO} />
            </div>
          </div>
        )}
      </div>
      <div className="surface-controls" data-tour="surface-controls">
        <label>
          Draw
          <select
            value={surfaceMode}
            onChange={(e) => setSurfaceMode(e.target.value as SurfaceMode)}
          >
            <option value="cloud">point cloud</option>
            <option value="surface">enclosing surface</option>
            <option value="both">both</option>
          </select>
        </label>
        {showSurface && (
          <label>
            Enclosing
            <select
              value={isoFraction}
              onChange={(e) => setIsoFraction(Number(e.target.value))}
            >
              {ISO_FRACTIONS.map((f) => (
                <option key={f} value={f}>
                  {(f * 100).toFixed(0)}%
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {showSurface && iso && (
        <div className="surface-hud">
          <p className="caption">
            {enclosedCaption(iso.meta)}{" "}
            <Badge provenance={iso.meta.enclosed_fraction.provenance} />
          </p>
          <p className="caption">
            |psi|² = {iso.meta.level.value.toExponential(3)} bohr^-3 on the{" "}
            {iso.meta.resolution}^3 grid, {componentsCaption(iso.meta)}
          </p>
          <p className="caption">
            {iso.meta.escaped_fraction.value.toExponential(1)} of the electron sits
            outside the box entirely
          </p>
        </div>
      )}
      <p className="caption">
        {(meta?.count ?? count).toLocaleString()} points · {meta?.basis ?? basis} basis · drag to orbit, scroll to zoom
      </p>
    </div>
  );
}
