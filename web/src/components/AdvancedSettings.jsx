function NumberField({ label, name, value, min, max, step = 1, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        name={name}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(name, event.target.value)}
      />
    </label>
  );
}

export default function AdvancedSettings({ form, updateField, models, activeModel, onModelChange, faceReference }) {
  return (
    <details className="expert-settings">
      <summary>Parámetros del modo experto</summary>
      <div className="settings-grid">
        {models.length > 0 && (
          <label className="field field-wide">
            <span>Modelo / checkpoint</span>
            <select value={activeModel?.value || ""} onChange={onModelChange}>
              {models.map((model) => <option key={model.value} value={model.value}>{model.label}</option>)}
            </select>
          </label>
        )}
        <NumberField label="Pasos" name="steps" value={form.steps} min={1} max={60} onChange={updateField} />
        <NumberField label="CFG" name="cfg_scale" value={form.cfg_scale} min={0} max={20} step={0.5} onChange={updateField} />
        <NumberField label="Ancho" name="width" value={form.width} min={256} max={512} onChange={updateField} />
        <NumberField label="Alto" name="height" value={form.height} min={256} max={512} onChange={updateField} />
        <NumberField label="Seed (-1 aleatoria)" name="seed" value={form.seed} min={-1} onChange={updateField} />
        <NumberField label="Batch size" name="batch_size" value={form.batch_size} min={1} max={1} onChange={updateField} />
        {faceReference && (
          <>
            <NumberField label="Fuerza de identidad" name="identity_strength" value={form.identity_strength} min={0} max={1.5} step={0.05} onChange={updateField} />
            <NumberField label="Fuerza de estructura" name="structure_strength" value={form.structure_strength} min={0} max={1.5} step={0.05} onChange={updateField} />
          </>
        )}
        <label className="field field-wide">
          <span>Prompt negativo</span>
          <textarea value={form.negative_prompt} onChange={(event) => updateField("negative_prompt", event.target.value)} rows={2} />
        </label>
      </div>
      <p className="muted small">El backend limita el tamaño máximo a 512 × 512 y el batch a 1.</p>
    </details>
  );
}
