import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";

const defaultForm = {
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

async function api(path, options) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function App() {
  const [tool, setTool] = useState("txt2img");
  const [form, setForm] = useState(defaultForm);
  const [referenceImage, setReferenceImage] = useState(null);
  const [runtime, setRuntime] = useState(null);
  const [models, setModels] = useState([]);
  const [history, setHistory] = useState([]);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  const activeModel = useMemo(() => models.find((model) => model.active), [models]);
  const busy = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    refreshMeta();
    return () => clearInterval(pollRef.current);
  }, []);

  async function refreshMeta() {
    const [runtimeData, modelData, historyData] = await Promise.all([
      api("/api/runtime"),
      api("/api/models"),
      api("/api/history"),
    ]);
    setRuntime(runtimeData);
    setModels(modelData);
    setHistory(historyData.images || []);
    setForm((current) => ({
      ...current,
      steps: runtimeData.defaults.steps,
      cfg_scale: runtimeData.defaults.cfg_scale,
      width: runtimeData.defaults.width,
      height: runtimeData.defaults.height,
      batch_size: runtimeData.defaults.batch_size,
    }));
  }

  function updateField(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  async function handleModelChange(event) {
    const model_id = event.target.value;
    const nextModels = await api("/api/models/active", {
      method: "POST",
      body: JSON.stringify({ model_id }),
    });
    setModels(nextModels);
  }

  async function handleReferenceFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setReferenceImage(await fileToDataUrl(file));
  }

  async function submit() {
    setError("");
    clearInterval(pollRef.current);
    try {
      const payload = {
        prompt: form.prompt,
        negative_prompt: form.negative_prompt,
        steps: Number(tool === "face_reference" ? Math.min(form.steps, 20) : form.steps),
        cfg_scale: Number(form.cfg_scale),
        width: Number(form.width),
        height: Number(form.height),
        seed: Number(form.seed),
        batch_size: Number(form.batch_size),
      };

      const endpoint = tool === "face_reference" ? "/api/generate/face-reference" : "/api/generate/txt2img";
      if (tool === "face_reference") {
        if (!referenceImage) throw new Error("Sube una imagen de referencia.");
        payload.reference_image = referenceImage;
        payload.identity_strength = Number(form.identity_strength);
        payload.structure_strength = Number(form.structure_strength);
        payload.batch_size = 1;
      }

      const created = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      setJob({ id: created.job_id, status: created.status, progress: 0, message: "Queued", result_images: [] });
      pollRef.current = setInterval(() => pollJob(created.job_id), 1200);
      await pollJob(created.job_id);
    } catch (err) {
      setError(err.message);
    }
  }

  async function pollJob(jobId) {
    const nextJob = await api(`/api/jobs/${jobId}`);
    setJob(nextJob);
    if (nextJob.status === "completed" || nextJob.status === "failed") {
      clearInterval(pollRef.current);
      const historyData = await api("/api/history");
      setHistory(historyData.images || []);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">Butter<span>Vision</span></div>
        <nav className="tool-picker" aria-label="Tools">
          <select value={tool} onChange={(event) => setTool(event.target.value)}>
            <option value="txt2img">Text to Image</option>
            <option value="face_reference">Face Reference</option>
          </select>
        </nav>
        <select className="model-select" value={activeModel?.value || ""} onChange={handleModelChange}>
          {models.map((model) => (
            <option key={model.value} value={model.value}>{model.label}</option>
          ))}
        </select>
        <div className={`status-pill ${runtime?.cuda_available === true ? "ok" : "warn"}`}>
          {runtime?.cuda_available === true ? runtime.device_name : runtime?.device_name || "Runtime pending"}
        </div>
      </header>

      <section className="workspace">
        <aside className="panel controls-panel">
          <div className="panel-title">
            <span>{tool === "face_reference" ? "Face Reference" : "Text to Image"}</span>
            <small>{runtime?.profile || "loading"}</small>
          </div>

          {tool === "face_reference" && (
            <label className="upload-box">
              {referenceImage ? <img src={referenceImage} alt="Reference preview" /> : <span>Upload reference face</span>}
              <input type="file" accept="image/*" onChange={handleReferenceFile} />
            </label>
          )}

          <label>
            Prompt
            <textarea value={form.prompt} onChange={(event) => updateField("prompt", event.target.value)} rows={5} />
          </label>
          <label>
            Negative Prompt
            <textarea value={form.negative_prompt} onChange={(event) => updateField("negative_prompt", event.target.value)} rows={3} />
          </label>

          <div className="grid two">
            <NumberField label="Steps" value={form.steps} min={1} max={60} onChange={(value) => updateField("steps", value)} />
            <NumberField label="CFG" value={form.cfg_scale} min={0} max={20} step={0.5} onChange={(value) => updateField("cfg_scale", value)} />
            <NumberField label="Width" value={form.width} min={512} max={512} onChange={(value) => updateField("width", value)} />
            <NumberField label="Height" value={form.height} min={512} max={512} onChange={(value) => updateField("height", value)} />
            <NumberField label="Seed" value={form.seed} step={1} onChange={(value) => updateField("seed", value)} />
            <NumberField label="Batch" value={tool === "face_reference" ? 1 : form.batch_size} min={1} max={1} onChange={(value) => updateField("batch_size", value)} />
          </div>

          {tool === "face_reference" && (
            <div className="grid two">
              <NumberField label="Identity" value={form.identity_strength} min={0} max={1.5} step={0.05} onChange={(value) => updateField("identity_strength", value)} />
              <NumberField label="Structure" value={form.structure_strength} min={0} max={1.5} step={0.05} onChange={(value) => updateField("structure_strength", value)} />
            </div>
          )}

          <button className="generate-button" disabled={busy || !form.prompt.trim()} onClick={submit}>
            {busy ? "Generating..." : "Generate"}
          </button>
          {error && <div className="error-box">{error}</div>}
        </aside>

        <section className="output-zone">
          <div className="hero-output">
            {job?.result_images?.[0] ? (
              <img src={job.result_images[0]} alt="Generated result" />
            ) : (
              <div className="empty-output">Generated image will appear here</div>
            )}
          </div>
          <div className="job-card">
            <div>
              <strong>{job?.message || "Idle"}</strong>
              <span>{job?.status || "ready"}</span>
            </div>
            <progress value={job?.progress || 0} max="1" />
            {job?.info && <pre>{job.info}</pre>}
            {job?.error && <div className="error-box">{job.error}</div>}
          </div>
        </section>
      </section>

      <section className="history-section">
        <div className="section-title">Recent Generations</div>
        <div className="history-grid">
          {history.map((src) => <img key={src} src={src} alt="Recent generation" />)}
        </div>
      </section>
    </main>
  );
}

function NumberField({ label, value, onChange, min, max, step = 1 }) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

createRoot(document.getElementById("root")).render(<App />);
