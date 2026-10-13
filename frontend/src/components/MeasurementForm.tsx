import { FIELDS, type FormValues, type Gender, type MeasurementKey } from "../fields";

interface Props {
  gender: Gender;
  values: FormValues;
  loading: boolean;
  onGenderChange: (g: Gender) => void;
  onChange: (key: MeasurementKey, value: string) => void;
  onPreset: (g: Gender) => void;
  onClear: () => void;
  onSubmit: () => void;
}

export function MeasurementForm({
  gender, values, loading, onGenderChange, onChange, onPreset, onClear, onSubmit,
}: Props) {
  return (
    <form
      className="panel"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <h1>FitCheck · Body</h1>
      <p className="sub">
        Enter your measurements to generate a 3D body that matches them. Height is required;
        every other measurement you add makes the body more accurate.
      </p>

      <div className="seg" role="radiogroup" aria-label="Gender">
        {(["female", "male"] as const).map((g) => (
          <button
            key={g}
            type="button"
            role="radio"
            aria-checked={gender === g}
            className={gender === g ? "on" : ""}
            onClick={() => onGenderChange(g)}
          >
            {g === "female" ? "Female" : "Male"}
          </button>
        ))}
      </div>

      <h2>Measurements</h2>
      {FIELDS.map((f) => (
        <div className="field" key={f.key}>
          <label htmlFor={`f-${f.key}`}>
            <span className="dot" style={{ background: f.color }} />
            {f.label}
            {f.required && <span className="req">*</span>}
          </label>
          <div className="inp">
            <input
              id={`f-${f.key}`}
              type="number"
              step="0.5"
              inputMode="decimal"
              required={f.required}
              value={values[f.key]}
              onChange={(e) => onChange(f.key, e.target.value)}
            />
            <span>{f.unit}</span>
          </div>
          <div className="hint">{f.hint}</div>
        </div>
      ))}

      <div className="presets">
        <button className="ghost" type="button" onClick={() => onPreset("female")}>Example female</button>
        <button className="ghost" type="button" onClick={() => onPreset("male")}>Example male</button>
        <button className="ghost" type="button" onClick={onClear}>Clear</button>
      </div>
      <button className="primary" type="submit" disabled={loading}>
        {loading ? "Fitting body…" : "Generate my body"}
      </button>
    </form>
  );
}
