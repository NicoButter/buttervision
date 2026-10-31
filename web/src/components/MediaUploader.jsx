import { useEffect, useState } from "react";

export default function MediaUploader({ accept, kind, onFile, preview, title, description }) {
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function handleChange(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!file.type.startsWith(`${kind}/`)) {
      setError(kind === "image" ? "Elige un archivo de imagen válido." : "Elige un archivo de video válido.");
      return;
    }
    setError("");
    onFile(file);
  }

  return (
    <div className="upload-field">
      <label className="upload-card">
        <input type="file" accept={accept} onChange={handleChange} aria-label={title} />
        {preview ? (
          kind === "image" ? <img src={preview} alt="Vista previa del archivo seleccionado" /> :
            <video src={preview} controls aria-label="Vista previa del video seleccionado" />
        ) : (
          <span className="upload-placeholder">
            <span className="upload-icon" aria-hidden="true">＋</span>
            <strong>{title}</strong>
            <span>{description}</span>
          </span>
        )}
      </label>
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  );
}
