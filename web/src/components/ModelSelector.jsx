function familyName(family) {
  return family === "sd15" ? "SD 1.5" : family === "sdxl" ? "SDXL" : "Desconocida";
}

const STATE_LABELS = {
  detected: "Detectado",
  selected: "Seleccionado · pendiente de carga",
  loading: "Cargando",
  loaded: "Cargado · generación aún no validada",
  generating: "Generando",
  error: "Error recuperable",
  incompatible: "Incompatible",
};

export default function ModelSelector({ models, activeModel, busy, onChange, onRefresh, error, engine, expert = false, compact = false }) {
  const selected = activeModel || models.find((model) => model.active);
  const previousFamily = selected?.family;
  const modelState = engine?.model_state || selected?.state || "selected";
  return (
    <section className={`model-selector${compact ? " model-selector-compact" : ""}`} aria-labelledby="model-selector-title">
      <div className="section-label"><strong id="model-selector-title">Modelo</strong><span>{busy ? "Actualizando…" : "Catálogo local"}</span></div>
      {selected ? <p className="model-active"><strong>{selected.label}</strong><span>{expert ? `${familyName(selected.family)} · ${STATE_LABELS[modelState] || "Seleccionado · pendiente de carga"}` : modelState === "loading" ? "Preparando modelo" : modelState === "generating" ? "Generando" : modelState === "error" ? "Error" : modelState === "incompatible" ? "Incompatible" : "Seleccionado"}</span></p> : <p className="muted">No hay un checkpoint activo disponible.</p>}
      <label className="field">
        <span>Checkpoint registrado</span>
        <select aria-label="Checkpoint registrado" value={selected?.value || ""} onChange={(event) => onChange(event, previousFamily)} disabled={busy || engine?.status === "busy"}>
          {!models.length && <option value="">No hay modelos registrados</option>}
          {models.map((model) => <option key={model.value} value={model.value} disabled={model.family === "unknown"}>{expert ? `${model.label} · ${familyName(model.family)} · ${model.origin || "Origen desconocido"} · ${STATE_LABELS[model.state] || "Detectado"}` : `${model.label} · ${familyName(model.family)}${model.family === "unknown" ? " · Incompatible" : ""}`}</option>)}
        </select>
      </label>
      <button type="button" className="secondary-button" onClick={onRefresh} disabled={busy || engine?.status === "busy"}>Actualizar catálogo</button>
      {expert && models.length === 1 && <p className="muted small">Forge se detecta automáticamente. También puedes agregar otras carpetas con <code>BUTTERVISION_CHECKPOINT_DIRS</code>. Solo se muestran modelos locales.</p>}
      {expert && selected && ["selected", "detected"].includes(modelState) && <p className="muted small">El checkpoint está seleccionado localmente. Se cargará al iniciar una generación; aún no está listo para generar.</p>}
      {expert && selected?.family === "sdxl" && <p className="model-warning">SDXL en la GTX 1650 de 4 GB puede no superar los controles de memoria. PIZARRA no ejecuta inferencia fuera de memoria.</p>}
      {expert && previousFamily && models.some((model) => model.family !== previousFamily && model.family !== "unknown") && <p className="muted small">Face Reference solo admite SD 1.5. Los LoRA de otra familia no se podrán activar. Las imágenes de referencia cargadas se conservan.</p>}
      {error && <p className="field-error" role="alert">{error}</p>}
    </section>
  );
}
