import { useEffect, useMemo, type Ref } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { BodyMesh, FitResponse } from "../api";
import { FIELD_COLORS, type MeasurementKey } from "../fields";

interface Props {
  mesh: BodyMesh | null;
  tapes: FitResponse["tapes"] | null;
  height: number; // metres, for framing the camera
  showTapes: boolean;
  meshRef?: Ref<THREE.Mesh>;
}

function Body({ mesh, meshRef }: { mesh: BodyMesh; meshRef?: Ref<THREE.Mesh> }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
    g.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
    g.computeVertexNormals();
    return g;
  }, [mesh]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <mesh ref={meshRef} geometry={geometry} name="body">
      <meshStandardMaterial color="#d9b8a0" roughness={0.75} metalness={0} />
    </mesh>
  );
}

function Tapes({ tapes }: { tapes: NonNullable<FitResponse["tapes"]> }) {
  return (
    <group>
      {Object.entries(tapes).flatMap(([key, polylines]) =>
        (polylines ?? []).map((points, i) => (
          <Line
            key={`${key}-${i}`}
            points={points}
            color={FIELD_COLORS[key as MeasurementKey] ?? "#222"}
            lineWidth={2.5}
            // Lines that run inside the body (arm bones, inseam) are drawn on top.
            depthTest={!(key === "arm" || key === "inseam")}
            renderOrder={2}
          />
        )),
      )}
    </group>
  );
}

export function BodyViewer({ mesh, tapes, height, showTapes, meshRef }: Props) {
  return (
    <Canvas camera={{ position: [0.9, 1.1, 4.2], fov: 30, near: 0.05, far: 50 }} dpr={[1, 2]}>
      <hemisphereLight args={["#ffffff", "#8a8a8a", 1.6]} />
      <directionalLight position={[2, 4, 3]} intensity={1.6} />
      <directionalLight position={[-3, 2, -2]} intensity={0.6} />
      <Grid
        args={[2, 2]}
        cellSize={0.1}
        sectionSize={0.5}
        cellColor="#9a9a9a"
        sectionColor="#7a7a7a"
        fadeDistance={6}
        infiniteGrid
      />
      {mesh && <Body mesh={mesh} meshRef={meshRef} />}
      {mesh && tapes && showTapes && <Tapes tapes={tapes} />}
      <OrbitControls target={[0, height * 0.52, 0]} enableDamping makeDefault />
    </Canvas>
  );
}
