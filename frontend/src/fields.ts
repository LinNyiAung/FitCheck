export type MeasurementKey =
  | "height" | "chest" | "waist" | "hip" | "neck" | "shoulder" | "arm" | "inseam" | "weight";

export type Gender = "female" | "male";

export interface FieldDef {
  key: MeasurementKey;
  label: string;
  unit: "cm" | "kg";
  color: string;
  hint: string;
  required?: boolean;
}

// Colours are shared by the form dots and the tape lines on the 3D body.
export const FIELDS: FieldDef[] = [
  { key: "height", label: "Height", unit: "cm", color: "#888888", required: true, hint: "Standing straight, no shoes, floor to top of head." },
  { key: "chest", label: "Chest / bust", unit: "cm", color: "#e4572e", hint: "Around the fullest part of the chest, under the armpits, tape level." },
  { key: "waist", label: "Waist", unit: "cm", color: "#f3a712", hint: "Around the narrowest part of the torso, usually just above the navel." },
  { key: "hip", label: "Hip", unit: "cm", color: "#a8c686", hint: "Around the fullest part of the buttocks, feet together." },
  { key: "neck", label: "Neck", unit: "cm", color: "#669bbc", hint: "Around the base of the neck, where a shirt collar sits." },
  { key: "shoulder", label: "Shoulder width", unit: "cm", color: "#b07cc6", hint: "Across the back, from one shoulder tip to the other." },
  { key: "arm", label: "Arm length", unit: "cm", color: "#5b6abf", hint: "Shoulder tip, over the elbow, to the wrist bone." },
  { key: "inseam", label: "Inseam", unit: "cm", color: "#3e8989", hint: "Crotch to floor, inside of the leg, no shoes." },
  { key: "weight", label: "Weight", unit: "kg", color: "#888888", hint: "Optional. Helps with the overall build." },
];

export const FIELD_COLORS = Object.fromEntries(FIELDS.map((f) => [f.key, f.color])) as Record<
  MeasurementKey,
  string
>;

export type FormValues = Record<MeasurementKey, string>;

export const EMPTY_VALUES = Object.fromEntries(FIELDS.map((f) => [f.key, ""])) as FormValues;

export const PRESETS: Record<Gender, Record<MeasurementKey, number>> = {
  female: { height: 165, chest: 90, waist: 72, hip: 98, neck: 33, shoulder: 39, arm: 55, inseam: 76, weight: 60 },
  male: { height: 178, chest: 100, waist: 86, hip: 100, neck: 39, shoulder: 46, arm: 61, inseam: 81, weight: 78 },
};

export function presetValues(gender: Gender): FormValues {
  return Object.fromEntries(
    Object.entries(PRESETS[gender]).map(([k, v]) => [k, String(v)]),
  ) as FormValues;
}
