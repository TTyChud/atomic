import { Html, Line } from "@react-three/drei";
import { PHYSICS_TO_SCREEN } from "../lib/frame";

const AXES: { key: "x" | "y" | "z"; dir: [number, number, number]; color: string }[] = [
  { key: "x", dir: [1, 0, 0], color: "#ff7a6b" },
  { key: "y", dir: [0, 1, 0], color: "#6cd98a" },
  { key: "z", dir: [0, 0, 1], color: "#34e0a1" },
];

const ARM = 0.42;

export function axisArmLength(distance: number): number {
  return distance * ARM;
}

export function AxisTriad({ distance }: { distance: number }) {
  const length = axisArmLength(distance);
  return (
    <group rotation={PHYSICS_TO_SCREEN}>
      {AXES.map((a) => (
        <Line
          key={a.key}
          points={[
            [-a.dir[0] * length, -a.dir[1] * length, -a.dir[2] * length],
            [a.dir[0] * length, a.dir[1] * length, a.dir[2] * length],
          ]}
          color={a.color}
          lineWidth={1}
          transparent
          opacity={0.5}
          depthWrite={false}
        />
      ))}
      {AXES.map((a) => (
        <Html
          key={a.key}
          position={[
            a.dir[0] * length * 1.07,
            a.dir[1] * length * 1.07,
            a.dir[2] * length * 1.07,
          ]}
          center
          style={{ pointerEvents: "none" }}
        >
          <span className="axis-label" style={{ color: a.color }}>
            {a.key}
          </span>
        </Html>
      ))}
    </group>
  );
}
