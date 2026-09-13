import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { PHYSICS_TO_SCREEN } from "../lib/frame";

interface Props {
  positions: Float32Array;
  pointSize: number;
  colors?: Float32Array | null;
}

export function PointCloud({ positions, pointSize, colors }: Props) {
  const useVertexColors = Boolean(colors && colors.length === positions.length);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (colors && colors.length === positions.length) {
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    }
    return g;
  }, [positions, colors]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <points geometry={geometry} rotation={PHYSICS_TO_SCREEN}>
      <pointsMaterial
        key={useVertexColors ? "vertex-colors" : "solid"}
        size={pointSize}
        sizeAttenuation
        color={useVertexColors ? "#ffffff" : "#7cffb2"}
        vertexColors={useVertexColors}
        transparent
        opacity={useVertexColors ? 0.55 : 0.35}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
