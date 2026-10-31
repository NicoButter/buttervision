function familyName(family) {
  return family === "sd15" ? "SD 1.5" : family === "sdxl" ? "SDXL" : "Desconocida";
}

export default function ModelSelector({ models, activeModel, busy, onChange, onRefresh, error, engine }) {
  const selected = activeModel || models.find((model) => model.active);
  const previousFamily = selected?.family;
  return (
    <section className="model-selector" aria-labelledby="model-selector-title">
      <div className="section-label"><strong id="model-selector-title">Modelo de imagen</strong><span>{busy ? "Actualizando…" : "Catálogo local"}</span></div>
      {selected ? <p className="model-active"><strong>{selected.label}</strong><span>{familyName(selected.family)} · seleccionado</span></p> : <p className="muted">No hay un checkpoint activo disponible.</p>}
      <label className="field">
        <span>Checkpoint registrado</span>
        <select aria-label="Checkpoint registrado" value={selected?.value || ""} onChange={(event) => onChange(event, previousFamily)} disabled={busy || engine?.status === "busy"}>
          {!models.length && <option value="">No hay modelos registrados</option>}
          {models.map((model) => <option key={model.value} value={model.value} disabled={model.family === "unknown"}>{model.label} · {familyName(model.family)}{model.active ? " · activo" : ""}</option>)}
        </select>
      </label>
      <button type="button" className="secondary-button" onClick={onRefresh} disabled={busy || engine?.status === "busy"}>Actualizar catálogo</button>
      {models.length === 1 && <p className="muted small">Para registrar más modelos, configura <code>BUTTERVISION_CHECKPOINT_DIRS</code> y reinicia el servicio. Solo se muestran archivos locales registrados.</p>}
      {selected && <p className="muted small">Seleccionado en el catálogo no confirma que los pesos estén cargados en VRAM ni que la generación supere la validación de memoria.</p>}
      {selected?.family === "sdxl" && <p className="model-warning">SDXL en la GTX 1650 de 4 GB puede no superar los controles de memoria. PIZARRA no ejecuta inferencia fuera de memoria.</p>}
      {previousFamily && models.some((model) => model.family !== previousFamily && model.family !== "unknown") && <p className="muted small">Face Reference solo admite SD 1.5. Los LoRA de otra familia no se podrán activar. Las imágenes de referencia cargadas se conservan.</p>}
      {error && <p className="field-error" role="alert">{error}</p>}
    </section>
  );
}
