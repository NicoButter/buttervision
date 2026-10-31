function NumberField({ label, name, value, min, max, step = 1, disabled = false, onChange, dimensions = [256, 320, 384, 448, 512] }) {
  return (
    <label className="field">
      <span>{label}</span>
      {name === "width" || name === "height" ? (
        <select value={value} disabled={disabled} onChange={(event) => onChange(name, Number(event.target.value))}>
          {dimensions.map((dimension) => <option key={dimension} value={dimension}>{dimension}</option>)}
        </select>
      ) : (
        <input
          type="number"
          name={name}
          value={value}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onChange={(event) => onChange(name, event.target.value)}
        />
      )}
    </label>
  );
}

export default function AdvancedSettings({ form, updateField, faceReference, busy, loras, onLoraChange, dimensions }) {
  return (
    <details className="expert-settings">
      <summary>Parámetros del modo experto</summary>
      <div className="settings-grid">
        <div className="field field-wide">
          <strong>LoRA locales</strong>
          {loras.length === 0 && <span>No hay LoRA registrados.</span>}
          {loras.map((lora) => (
            <div key={lora.value}>
              <label>
                <input type="checkbox" checked={lora.active} disabled={busy || (!lora.compatible && !lora.active)} onChange={(event) => onLoraChange(lora, event.target.checked, lora.weight)} />
                {lora.label} · {lora.family}
              </label>
              {lora.active && <label> Peso <input key={`${lora.value}-${lora.weight}`} type="number" min="0" max="2" step="0.05" defaultValue={lora.weight} disabled={busy} onBlur={(event) => onLoraChange(lora, true, Number(event.target.value))} /></label>}
              {!lora.compatible && <p className="muted small">{lora.reason}</p>}
            </div>
          ))}
        </div>
        <NumberField label="Pasos" name="steps" value={form.steps} min={1} max={60} onChange={updateField} />
        <NumberField label="CFG" name="cfg_scale" value={form.cfg_scale} min={0} max={20} step={0.5} onChange={updateField} />
        <NumberField label="Ancho" name="width" value={faceReference ? 512 : form.width} dimensions={dimensions} disabled={faceReference} onChange={updateField} />
        <NumberField label="Alto" name="height" value={faceReference ? 512 : form.height} dimensions={dimensions} disabled={faceReference} onChange={updateField} />
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
      <p className="muted small">
        {faceReference ? "Face Reference solo admite formato cuadrado de 512 × 512." : `Resoluciones disponibles para este perfil: ${dimensions.join(", ")} px. Batch máximo: 1.`}
      </p>
    </details>
  );
}
