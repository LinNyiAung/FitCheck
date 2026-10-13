import { useMemo, useRef, useState } from "react";
import type * as THREE from "three";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { decodeMesh, fitBody, type FitResponse } from "./api";
import { EMPTY_VALUES, presetValues, type FormValues, type Gender } from "./fields";
import { MeasurementForm } from "./components/MeasurementForm";
import { BodyViewer } from "./components/BodyViewer";
import { ResultsPanel } from "./components/ResultsPanel";

function downloadGlb(mesh: THREE.Mesh) {
  new GLTFExporter().parse(
    mesh,
    (glb) => {
      const url = URL.createObjectURL(new Blob([glb as ArrayBuffer], { type: "model/gltf-binary" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = "fitcheck-body.glb";
      a.click();
      URL.revokeObjectURL(url);
    },
    (err) => console.error(err),
    { binary: true },
  );
}

export default function App() {
  const [gender, setGender] = useState<Gender>("female");
  const [values, setValues] = useState<FormValues>(() => presetValues("female"));
  const [result, setResult] = useState<FitResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showTapes, setShowTapes] = useState(true);
  const meshRef = useRef<THREE.Mesh>(null);

  const mesh = useMemo(() => (result ? decodeMesh(result.mesh) : null), [result]);
  const height = (result?.measurements.find((m) => m.key === "height")?.achieved ?? 170) / 100;

  async function submit() {
    setLoading(true);
    setError(null);
    try {
      setResult(await fitBody(gender, values));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <MeasurementForm
        gender={gender}
        values={values}
        loading={loading}
        onGenderChange={setGender}
        onChange={(key, value) => setValues((v) => ({ ...v, [key]: value }))}
        onPreset={(g) => {
          setGender(g);
          setValues(presetValues(g));
        }}
        onClear={() => setValues(EMPTY_VALUES)}
        onSubmit={submit}
      />

      <main className="stage">
        <BodyViewer
          mesh={mesh}
          tapes={result?.tapes ?? null}
          height={height}
          showTapes={showTapes}
          meshRef={meshRef}
        />
        {!mesh && (
          <div className="overlay">
            {loading ? "Fitting body…" : error ? "Something went wrong." : "Your body will appear here."}
          </div>
        )}
        <div className="toolbar">
          <button className="ghost" type="button" onClick={() => setShowTapes((s) => !s)}>
            {showTapes ? "Hide tape lines" : "Show tape lines"}
          </button>
          <button
            className="ghost"
            type="button"
            disabled={!mesh}
            onClick={() => meshRef.current && downloadGlb(meshRef.current)}
          >
            Download .glb
          </button>
        </div>
      </main>

      <ResultsPanel result={result} error={error} />
    </div>
  );
}
