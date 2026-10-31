import { useEffect, useMemo, useState } from "react";
import AdvancedSettings from "../components/AdvancedSettings.jsx";
import BeforeAfterSlider from "../components/BeforeAfterSlider.jsx";
import GenerationProgress from "../components/GenerationProgress.jsx";
import MediaUploader from "../components/MediaUploader.jsx";
import { usePreferences } from "../hooks/usePreferences.js";
import { api, outputUrl, userFacingError } from "../services/api.js";

const INITIAL_FORM = {
  prompt: "",
  negative_prompt: "",
  steps: 20,
  cfg_scale: 5,
  width: 512,
  height: 512,
  seed: -1,
  batch_size: 1,
  identity_strength: 0.8,
  structure_strength: 0.8,
};

const TOOLS = [
  { id: "create-image", icon: "✦", title: "Crear imagen", description: "Imagina algo y conviértelo en una imagen." },
  { id: "create-video", icon: "▶", title: "Crear video", description: "Prepara una idea para darle movimiento." },
  { id: "edit-image", icon: "◈", title: "Editar imagen", description: "Cuéntanos qué te gustaría cambiar." },
  { id: "edit-video", icon: "▣", title: "Editar video", description: "Describe cómo quieres transformar tu video." },
];

function releasePreview(media) {
  if (media?.preview?.startsWith("blob:")) URL.revokeObjectURL(media.preview);
}

function HomePage({ navigate }) {
  return (
    <section className="home-page">
      <div className="welcome-copy">
        <p className="eyebrow">TU ESPACIO CREATIVO</p>
        <h1>¿Qué te gustaría crear hoy?</h1>
        <p>Elige una opción para empezar. Puedes describirlo con tus propias palabras.</p>
      </div>
      <div className="tool-grid">
        {TOOLS.map((tool) => (
          <button className="tool-card" key={tool.id} onClick={() => navigate(tool.id)}>
            <span className="tool-icon" aria-hidden="true">{tool.icon}</span>
            <span className="tool-card-copy"><strong>{tool.title}</strong><span>{tool.description}</span></span>
            <span className="card-arrow" aria-hidden="true">→</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function CreateImagePage({
  form, updateField, reference, setReference, onReferenceFile, onGenerate, busy, error, job,
  mode, models, activeModel, onModelChange, navigate,
}) {
  const [format, setFormat] = useState("square");
  const dimensions = { portrait: [384, 512], square: [512, 512], landscape: [512, 384] };
  const faceReference = Boolean(reference);
  const result = job?.result_images?.[0];

  function chooseFormat(nextFormat) {
    setFormat(nextFormat);
    const [width, height] = dimensions[nextFormat];
    updateField("width", width);
    updateField("height", height);
  }

  return (
    <section className="page-content">
      <button className="back-link" onClick={() => navigate("home")}>← Inicio</button>
      <div className="page-heading">
        <p className="eyebrow">CREACIÓN</p>
        <h1>Crear imagen</h1>
        <p>Describe lo que tienes en mente. No necesitas conocer términos técnicos.</p>
      </div>
      <div className="creation-layout">
        <div className="form-card">
          <label className="field">
            <span>¿Qué imagen quieres crear?</span>
            <textarea
              value={form.prompt}
              onChange={(event) => updateField("prompt", event.target.value)}
              placeholder="Por ejemplo: Una casa cálida en el bosque durante el otoño..."
              rows={5}
            />
          </label>
          <fieldset className="format-options">
            <legend>Formato</legend>
            {[
              ["portrait", "Vertical"],
              ["square", "Cuadrado"],
              ["landscape", "Horizontal"],
            ].map(([value, label]) => (
              <button
                type="button"
                key={value}
                className={`format-choice ${format === value ? "selected" : ""}`}
                aria-pressed={format === value}
                onClick={() => chooseFormat(value)}
              >
                <span className={`format-shape ${value}`} aria-hidden="true" />{label}
              </button>
            ))}
          </fieldset>
          <div className="reference-section">
            <div className="section-label"><strong>Foto de rostro</strong><span>Opcional</span></div>
            {!reference ? (
              <MediaUploader
                accept="image/*"
                kind="image"
                title="Añadir una foto de referencia"
                description="La imagen se procesa localmente en Buttervision."
                onFile={onReferenceFile}
              />
            ) : (
              <div className="reference-preview">
                <img src={reference.dataUrl} alt="Vista previa de la foto de referencia" />
                <div><strong>{reference.file.name}</strong><span>La foto se usará como referencia, sin prometer una preservación exacta de identidad.</span></div>
                <button className="text-button" onClick={() => setReference(null)}>Quitar foto</button>
              </div>
            )}
            {faceReference && (
              <p className="notice">La referencia facial se genera en formato cuadrado (512 × 512) por los límites actuales del backend.</p>
            )}
          </div>
          {mode === "expert" && (
            <AdvancedSettings
              form={form}
              updateField={updateField}
              models={models}
              activeModel={activeModel}
              onModelChange={onModelChange}
              faceReference={faceReference}
            />
          )}
          <button className="primary-button" disabled={busy || !form.prompt.trim()} onClick={onGenerate}>
            {busy ? "Creando imagen…" : "Crear imagen"}
          </button>
          {error && <p className="field-error" role="alert">{error}</p>}
        </div>
        <div className="result-column">
          <div className="result-card">
            {result ? (
              <img className="generated-image" src={outputUrl(result)} alt="Imagen generada" />
            ) : (
              <div className="result-placeholder">
                <span aria-hidden="true">✦</span>
                <strong>Tu creación aparecerá aquí</strong>
                <p>Cuando termine la generación, podrás guardar la imagen.</p>
              </div>
            )}
          </div>
          {result && (
            <div className="result-actions">
              <a className="secondary-button" href={outputUrl(result)} download>Descargar imagen</a>
              <button className="secondary-button" onClick={onGenerate} disabled={busy || !form.prompt.trim()}>Crear otra</button>
            </div>
          )}
          <GenerationProgress job={job} mode={mode} />
        </div>
      </div>
    </section>
  );
}

function PendingPage({
  tool, navigate, media, onMediaFile, prompt, setPrompt, videoImage, setVideoImageFile, videoFace, setVideoFaceFile, removeVideoFace,
}) {
  const isVideo = tool.includes("video");
  const title = TOOLS.find((item) => item.id === tool)?.title;
  const preview = media?.preview || "";
  const [videoDuration, setVideoDuration] = useState("3");
  const [videoFormat, setVideoFormat] = useState("square");
  return (
    <section className="page-content">
      <button className="back-link" onClick={() => navigate("home")}>← Inicio</button>
      <div className="page-heading">
        <p className="eyebrow">EN PREPARACIÓN</p>
        <h1>{title}</h1>
        <p>{isVideo ? "Esta experiencia se está preparando para conectarse con Buttervision." : "Carga una imagen y describe el cambio que te gustaría ver."}</p>
      </div>
      <div className="pending-layout">
        {tool === "create-video" && (
          <div className="form-card">
            <div className="section-label"><strong>Imagen inicial</strong><span>Opcional</span></div>
            <MediaUploader
              accept="image/*"
              kind="image"
              title="Añadir imagen inicial"
              description="La generación de video aún no está conectada."
              preview={videoImage?.preview}
              onFile={setVideoImageFile}
            />
            <div className="section-label"><strong>Foto de rostro</strong><span>Opcional</span></div>
            {!videoFace ? (
              <MediaUploader
                accept="image/*"
                kind="image"
                title="Añadir foto de referencia"
                description="No se enviará a ningún servicio."
                onFile={setVideoFaceFile}
              />
            ) : (
              <div className="reference-preview">
                <img src={videoFace.preview} alt="Vista previa de foto de rostro" />
                <button className="text-button" onClick={removeVideoFace}>Quitar foto</button>
              </div>
            )}
            <label className="field">
              <span>Describe tu video</span>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} placeholder="Describe qué debería suceder en el video…" />
            </label>
            <label className="field">
              <span>Duración</span>
              <select value={videoDuration} onChange={(event) => setVideoDuration(event.target.value)}>
                <option value="3">3 segundos</option>
                <option value="5">5 segundos</option>
                <option value="8">8 segundos</option>
              </select>
            </label>
            <label className="field">
              <span>Formato</span>
              <select value={videoFormat} onChange={(event) => setVideoFormat(event.target.value)}>
                <option value="square">Cuadrado</option>
                <option value="portrait">Vertical</option>
                <option value="landscape">Horizontal</option>
              </select>
            </label>
            <p className="notice">Duración y formato son opciones de diseño; el backend todavía no puede procesarlos.</p>
          </div>
        )}
        {tool === "edit-image" && (
          <div className="form-card">
            <MediaUploader
              accept="image/*"
              kind="image"
              title="Subir imagen"
              description="El archivo original no se modificará."
              preview={preview}
              onFile={onMediaFile}
            />
            <label className="field">
              <span>¿Qué te gustaría cambiar?</span>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} placeholder="Por ejemplo: Cambiá el fondo y mejorá la iluminación…" />
            </label>
          </div>
        )}
        {tool === "edit-video" && (
          <div className="form-card">
            <MediaUploader
              accept="video/*"
              kind="video"
              title="Subir video"
              description="Puedes previsualizar el archivo aquí."
              preview={preview}
              onFile={onMediaFile}
            />
            <label className="field">
              <span>¿Qué te gustaría cambiar?</span>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} placeholder="Describe los cambios que quieres…" />
            </label>
          </div>
        )}
        <div className="pending-result">
          <div className="pending-badge">Pendiente de integración</div>
          {tool === "edit-image" && <BeforeAfterSlider original={preview} />}
          {tool !== "edit-image" && (
            <div className="result-placeholder">
              <span aria-hidden="true">{isVideo ? "▶" : "✦"}</span>
              <strong>El resultado aparecerá aquí</strong>
              <p>No se enviará ninguna solicitud ni se mostrarán resultados de demostración.</p>
            </div>
          )}
          <p className="notice">Esta función todavía no está conectada a una API de procesamiento. No se realizará ninguna acción al pulsar el botón.</p>
          <button className="primary-button" disabled>{title}</button>
        </div>
      </div>
    </section>
  );
}

function GalleryPage({ history, loading, error, onRefresh, navigate, onOpen }) {
  return (
    <section className="page-content">
      <button className="back-link" onClick={() => navigate("home")}>← Inicio</button>
      <div className="page-heading gallery-heading">
        <div><p className="eyebrow">TU BIBLIOTECA</p><h1>Mis creaciones</h1><p>Imágenes guardadas por Buttervision en este dispositivo.</p></div>
        <button className="secondary-button" onClick={onRefresh} disabled={loading}>Actualizar</button>
      </div>
      {error && <p className="field-error" role="alert">{error}</p>}
      {loading ? <p className="muted">Cargando tus imágenes…</p> : history.length ? (
        <div className="gallery-grid">
          {history.map((src) => (
            <article className="gallery-item" key={src}>
              <button className="gallery-image-button" onClick={() => onOpen(src)} aria-label="Abrir imagen">
                <img src={outputUrl(src)} alt="Creación guardada" loading="lazy" />
              </button>
              <a className="download-link" href={outputUrl(src)} download>Descargar</a>
            </article>
          ))}
        </div>
      ) : (
        !error && <div className="empty-gallery"><span aria-hidden="true">✦</span><strong>Aún no hay creaciones</strong><p>Las imágenes que generes aparecerán aquí.</p></div>
      )}
    </section>
  );
}

export default function App() {
  const { mode, toggleMode, storageError } = usePreferences();
  const [page, setPage] = useState("home");
  const [form, setForm] = useState(INITIAL_FORM);
  const [reference, setReference] = useState(null);
  const [models, setModels] = useState([]);
  const [runtime, setRuntime] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [job, setJob] = useState(null);
  const [generationError, setGenerationError] = useState("");
  const [metadataError, setMetadataError] = useState("");
  const [selectedImage, setSelectedImage] = useState("");
  const [imageEditMedia, setImageEditMedia] = useState(null);
  const [videoEditMedia, setVideoEditMedia] = useState(null);
  const [imageEditPrompt, setImageEditPrompt] = useState("");
  const [videoEditPrompt, setVideoEditPrompt] = useState("");
  const [videoPrompt, setVideoPrompt] = useState("");
  const [videoImage, setVideoImage] = useState(null);
  const [videoFace, setVideoFace] = useState(null);
  const [pollError, setPollError] = useState("");

  useEffect(() => {
    if (!selectedImage) return undefined;
    function handleKeyDown(event) {
      if (event.key === "Escape") setSelectedImage("");
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedImage]);

  const activeModel = useMemo(() => models.find((model) => model.active), [models]);
  const busy = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    let active = true;
    Promise.allSettled([api("/api/runtime"), api("/api/models")]).then((results) => {
      if (!active) return;
      const [runtimeResult, modelResult] = results;
      if (runtimeResult.status === "fulfilled") {
        setRuntime(runtimeResult.value);
        const defaults = runtimeResult.value.defaults || {};
        setForm((current) => ({
          ...current,
          steps: defaults.steps ?? current.steps,
          cfg_scale: defaults.cfg_scale ?? current.cfg_scale,
          width: defaults.width ?? current.width,
          height: defaults.height ?? current.height,
          batch_size: defaults.batch_size ?? current.batch_size,
        }));
      } else setMetadataError("No se pudo cargar la información del servidor.");
      if (modelResult.status === "fulfilled") setModels(modelResult.value);
      else setMetadataError("No se pudieron cargar los modelos disponibles.");
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!job?.id || !["queued", "running"].includes(job.status)) return undefined;
    const timeout = window.setTimeout(async () => {
      try {
        const nextJob = await api(`/api/jobs/${job.id}`);
        setJob(nextJob);
        setPollError("");
        if (nextJob.status === "completed" || nextJob.status === "failed") {
          try {
            const data = await api("/api/history");
            setHistory(data.images || []);
          } catch {
            setHistoryError("La generación terminó, pero no se pudo actualizar la galería.");
          }
        }
      } catch (error) {
        setJob((current) => ({ ...current, status: "failed" }));
        setPollError(userFacingError(error));
      }
    }, 1200);
    return () => window.clearTimeout(timeout);
  }, [job]);

  useEffect(() => {
    if (page === "gallery") refreshHistory();
  }, [page]);

  function updateField(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function navigate(target) {
    if (page === "edit-image" && target !== page) {
      releasePreview(imageEditMedia);
      setImageEditMedia(null);
    }
    if (page === "edit-video" && target !== page) {
      releasePreview(videoEditMedia);
      setVideoEditMedia(null);
    }
    if (page === "create-video" && target !== page) {
      releasePreview(videoImage);
      releasePreview(videoFace);
      setVideoImage(null);
      setVideoFace(null);
    }
    setPage(target);
  }

  function setImageEditFile(file) {
    releasePreview(imageEditMedia);
    setImageEditMedia({ file, preview: URL.createObjectURL(file) });
  }

  function setVideoEditFile(file) {
    releasePreview(videoEditMedia);
    setVideoEditMedia({ file, preview: URL.createObjectURL(file) });
  }

  function setVideoImageFile(file) {
    releasePreview(videoImage);
    setVideoImage({ file, preview: URL.createObjectURL(file) });
  }

  function setVideoFaceFile(file) {
    releasePreview(videoFace);
    setVideoFace({ file, preview: URL.createObjectURL(file) });
  }

  function removeVideoFace() {
    releasePreview(videoFace);
    setVideoFace(null);
  }

  async function loadReference(file) {
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("No se pudo leer la foto."));
        reader.readAsDataURL(file);
      });
      setReference({ file, dataUrl });
    } catch (error) {
      setGenerationError(userFacingError(error));
    }
  }

  async function handleGenerate() {
    setGenerationError("");
    setPollError("");
    const isFaceReference = Boolean(reference);
    const payload = {
      prompt: form.prompt.trim(),
      negative_prompt: form.negative_prompt,
      steps: Number(isFaceReference ? Math.min(Number(form.steps), 20) : form.steps),
      cfg_scale: Number(form.cfg_scale),
      width: Number(isFaceReference ? 512 : form.width),
      height: Number(isFaceReference ? 512 : form.height),
      seed: Number(form.seed),
      batch_size: isFaceReference ? 1 : Number(form.batch_size),
    };
    const endpoint = isFaceReference ? "/api/generate/face-reference" : "/api/generate/txt2img";
    if (isFaceReference) {
      payload.reference_image = reference.dataUrl;
      payload.identity_strength = Number(form.identity_strength);
      payload.structure_strength = Number(form.structure_strength);
    }

    setJob({ id: "", status: "queued", message: "Preparando tu imagen", progress: 0, result_images: [] });
    try {
      const created = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      setJob({ id: created.job_id, status: created.status, message: "En cola", progress: 0, result_images: [] });
    } catch (error) {
      setJob(null);
      setGenerationError(userFacingError(error));
    }
  }

  async function refreshHistory() {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const data = await api("/api/history");
      setHistory(data.images || []);
    } catch (error) {
      setHistoryError(userFacingError(error));
    } finally {
      setHistoryLoading(false);
    }
  }

  async function handleModelChange(event) {
    try {
      const nextModels = await api("/api/models/active", {
        method: "POST",
        body: JSON.stringify({ model_id: event.target.value }),
      });
      setModels(nextModels);
      setMetadataError("");
    } catch (error) {
      setMetadataError(userFacingError(error));
    }
  }

  function pageContent() {
    if (page === "home") return <HomePage navigate={navigate} />;
    if (page === "create-image") {
      return (
        <CreateImagePage
          form={form} updateField={updateField} reference={reference} setReference={setReference} onReferenceFile={loadReference}
          onGenerate={handleGenerate} busy={busy} error={generationError || pollError} job={job}
          mode={mode} models={models} activeModel={activeModel} onModelChange={handleModelChange} navigate={navigate}
        />
      );
    }
    if (page === "gallery") {
      return <GalleryPage history={history} loading={historyLoading} error={historyError} onRefresh={refreshHistory} navigate={navigate} onOpen={setSelectedImage} />;
    }
    return (
      <PendingPage
        tool={page}
        navigate={navigate}
        media={page === "edit-image" ? imageEditMedia : page === "edit-video" ? videoEditMedia : null}
        onMediaFile={page === "edit-image" ? setImageEditFile : setVideoEditFile}
        videoImage={videoImage}
        setVideoImageFile={setVideoImageFile}
        videoFace={videoFace}
        setVideoFaceFile={setVideoFaceFile}
        removeVideoFace={removeVideoFace}
        prompt={page === "edit-image" ? imageEditPrompt : page === "edit-video" ? videoEditPrompt : videoPrompt}
        setPrompt={page === "edit-image" ? setImageEditPrompt : page === "edit-video" ? setVideoEditPrompt : setVideoPrompt}
      />
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => navigate("home")} aria-label="Buttervision, ir al inicio">
          <span className="brand-mark" aria-hidden="true">b</span> buttervision
        </button>
        <nav className="main-nav" aria-label="Navegación principal">
          <button className={page === "home" ? "active" : ""} aria-current={page === "home" ? "page" : undefined} onClick={() => navigate("home")}>Inicio</button>
          <button className={page === "gallery" ? "active" : ""} aria-current={page === "gallery" ? "page" : undefined} onClick={() => navigate("gallery")}>Mis creaciones</button>
        </nav>
        <button className={`mode-toggle ${mode === "expert" ? "enabled" : ""}`} onClick={toggleMode} aria-pressed={mode === "expert"}>
          {mode === "expert" ? "Modo experto · Activado" : "Modo experto"}
        </button>
      </header>
      {storageError && <p className="global-notice" role="status">{storageError}</p>}
      {metadataError && <p className="global-notice" role="status">{metadataError}</p>}
      {runtime && mode === "expert" && (
        <div className="runtime-note">Modelo de hardware: {runtime.profile} · El estado del dispositivo se consulta durante la generación.</div>
      )}
      {pageContent()}
      <footer className="app-footer">Buttervision procesa tus archivos localmente en esta instalación.</footer>
      {selectedImage && (
        <div className="image-modal" role="dialog" aria-modal="true" aria-label="Vista de la imagen" onClick={() => setSelectedImage("")}>
          <button className="modal-close" onClick={() => setSelectedImage("")} aria-label="Cerrar imagen">×</button>
          <img src={outputUrl(selectedImage)} alt="Creación seleccionada" onClick={(event) => event.stopPropagation()} />
          <a href={outputUrl(selectedImage)} download onClick={(event) => event.stopPropagation()}>Descargar imagen</a>
        </div>
      )}
    </main>
  );
}
