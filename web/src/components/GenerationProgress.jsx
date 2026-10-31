const STATUS_LABELS = {
  queued: "En cola",
  running: "Procesando",
  completed: "Completado",
  failed: "No se pudo completar",
};

export default function GenerationProgress({ job, mode }) {
  if (!job) return null;
  const progress = Number.isFinite(Number(job.progress)) ? Number(job.progress) : 0;
  const label = STATUS_LABELS[job.status] || job.status;
  return (
    <section className="status-card" aria-live="polite">
      <div className="status-heading">
        <strong>{job.message || label}</strong>
        <span className={`status-tag ${job.status}`}>{label}</span>
      </div>
      {(job.status === "queued" || job.status === "running") && (
        <progress value={progress} max="1" aria-label="Progreso informado por el servidor" />
      )}
      {mode === "expert" && job.info && <pre>{job.info}</pre>}
      {job.status === "failed" && (
        <p className="field-error" role="alert">
          {mode === "expert" ? (job.error || "Error sin detalles.") : "La generación no se pudo completar. Revisa la imagen de referencia y vuelve a intentarlo."}
        </p>
      )}
    </section>
  );
}
