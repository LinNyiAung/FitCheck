import type { FormValues, Gender, MeasurementKey } from "./fields";

export interface MeasurementRow {
  key: MeasurementKey;
  label: string;
  unit: "cm" | "kg";
  target: number | null;
  achieved: number;
  error?: number;
}

type Point = [number, number, number];

export interface FitResponse {
  measurements: MeasurementRow[];
  warnings: string[];
  seconds: number;
  iterations: number;
  shape: {
    phenotypes: Record<string, number>;
    local_changes: Record<string, number>;
  };
  mesh: { positions: string; indices: string }; // base64 float32 / uint32
  tapes: Partial<Record<MeasurementKey, Point[][]>>;
}

export interface BodyMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

function decodeBase64(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

export function decodeMesh(mesh: FitResponse["mesh"]): BodyMesh {
  return {
    positions: new Float32Array(decodeBase64(mesh.positions)),
    indices: new Uint32Array(decodeBase64(mesh.indices)),
  };
}

export async function fitBody(gender: Gender, values: FormValues): Promise<FitResponse> {
  const payload: Record<string, unknown> = { gender };
  for (const [k, v] of Object.entries(values)) {
    if (v.trim() !== "") payload[k] = Number(v);
  }
  const res = await fetch("/api/fit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    // FastAPI validation errors are a list; our own errors are a string.
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d: { loc: string[]; msg: string }) => `${d.loc.at(-1)}: ${d.msg}`).join("\n")
      : String(data.detail ?? res.statusText);
    throw new Error(detail);
  }
  return data as FitResponse;
}
