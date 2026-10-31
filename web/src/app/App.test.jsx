import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import BeforeAfterSlider from "../components/BeforeAfterSlider.jsx";

const runtime = {
  profile: "gtx1650_lowvram",
  cuda_available: null,
  device_name: "Runtime checked during generation",
  defaults: { steps: 20, cfg_scale: 5, width: 512, height: 512, batch_size: 1 },
  upload_limits: { image_bytes: 10 * 1024 * 1024, video_bytes: 100 * 1024 * 1024 },
};

function installApiMock({ history = [], onGenerate, uploadLimits, jobWarnings = [], busyMessage, models = [], engine = null, loras = [], pizarra = { enabled: false, available: null, metrics_enabled: false, storage: null }, changeError, changedModels, changedLoras } = {}) {
  let currentModels = models;
  let currentLoras = loras;
  let currentEngine = engine;
  globalThis.fetch = vi.fn(async (url, options = {}) => {
    if (url === "/api/runtime") return response({
      ...runtime,
      upload_limits: uploadLimits || runtime.upload_limits,
    });
    if (url === "/api/models") return response(currentModels);
    if (url === "/api/engine") return response(currentEngine);
    if (url === "/api/pizarra/status") return response(pizarra);
    if (url === "/api/loras/active") return response(currentLoras);
    if (url === "/api/history") return response({ images: history });
    if (url.startsWith("/api/generate/")) {
      if (busyMessage) return { ok: false, status: 409, text: async () => JSON.stringify({ detail: busyMessage }) };
      onGenerate?.(url, JSON.parse(options.body));
      return response({ job_id: "job-1", status: "queued" });
    }
    if (url === "/api/jobs/job-1") {
      return response({
        id: "job-1", status: "running", progress: 0.25, message: "Generando",
        result_images: [], error: null, info: null,
        warnings: jobWarnings,
      });
    }
    if (url === "/api/models/active") {
      if (changeError) return { ok: false, status: changeError.status, text: async () => JSON.stringify({ detail: changeError.detail }) };
      currentModels = changedModels || models;
      currentLoras = changedLoras || loras;
      const selected = currentModels.find((model) => model.active);
      if (selected && currentEngine) currentEngine = { ...currentEngine, model: selected.label, family: selected.family };
      return response(currentModels);
    }
    if (url === "/api/loras") return response(currentLoras);
    throw new Error(`Unexpected API request: ${url}`);
  });
}

function response(data) {
  return { ok: true, json: async () => data, text: async () => JSON.stringify(data) };
}

async function openImageCreation() {
  render(<App />);
  await screen.findByRole("button", { name: /Crear imagen/ });
  fireEvent.click(screen.getByRole("button", { name: /Crear imagen/ }));
  await screen.findByRole("heading", { name: "Crear imagen" });
}

beforeEach(() => {
  localStorage.clear();
  installApiMock();
  Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:test-preview") });
});
afterEach(cleanup);

describe("Buttervision UX", () => {
  it("muestra modelo, familia, LoRA activos y estado del motor", async () => {
    installApiMock({
      models: [{ label: "CyberRealistic", value: "cyber", family: "sd15", active: true }],
      engine: { model: "CyberRealistic", family: "sd15", status: "idle" },
      loras: [{ label: "Retrato", value: "lora", family: "sd15", active: true, compatible: true, weight: 0.8 }],
    });
    await openImageCreation();
    await screen.findByText(/LoRA activos \(1\)/);
    expect(screen.getByText(/Modelo: CyberRealistic/)).toBeTruthy();
    expect(screen.getByText(/Familia: SD 1.5 · Estado del modelo: Seleccionado · pendiente de carga/)).toBeTruthy();
    expect(screen.getByText(/Retrato/)).toBeTruthy();
  });

  it("muestra el selector de modelo en modo simple y avisa cuando solo hay un registrado", async () => {
    installApiMock({ models: [{ label: "CyberRealistic", value: "cyber", family: "sd15", active: true }] });
    await openImageCreation();
    expect(screen.getByLabelText("Checkpoint registrado").value).toBe("cyber");
    expect(screen.getByText(/BUTTERVISION_CHECKPOINT_DIRS/)).toBeTruthy();
    expect(screen.getByText(/Se cargará al iniciar una generación; aún no está listo/)).toBeTruthy();
  });

  it("muestra cada estado del checkpoint sin llamar listo al seleccionado", async () => {
    const labels = [
      ["detected", "Detectado"], ["selected", "Seleccionado · pendiente de carga"],
      ["loading", "Cargando"], ["loaded", "Cargado · generación aún no validada"],
      ["generating", "Generando"], ["error", "Error recuperable"], ["incompatible", "Incompatible"],
    ];
    for (const [state, label] of labels) {
      cleanup();
      installApiMock({
        models: [{ label: "Juggernaut XL", value: "xl", family: "sdxl", active: true, state }],
        engine: { model: "Juggernaut XL", family: "sdxl", status: "idle", model_state: state },
      });
      await openImageCreation();
      expect(screen.getAllByText(new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))).length).toBeGreaterThan(0);
    }
  });

  it("actualiza el catálogo desde el endpoint registrado", async () => {
    installApiMock({ models: [{ label: "CyberRealistic", value: "cyber", family: "sd15", active: true }] });
    await openImageCreation();
    const before = globalThis.fetch.mock.calls.filter(([url]) => url === "/api/models").length;
    fireEvent.click(screen.getByRole("button", { name: "Actualizar catálogo" }));
    await waitFor(() => expect(globalThis.fetch.mock.calls.filter(([url]) => url === "/api/models").length).toBeGreaterThan(before));
  });

  it("expone los estados PIZARRA usando solo la respuesta del backend", async () => {
    const states = [
      [{ enabled: true, available: true, metrics_enabled: true, storage: { total_bytes: 800_000_000_000, quota_bytes: 400_000_000_000, managed_bytes: 100_000_000_000, quota_remaining_bytes: 300_000_000_000, free_bytes: 600_000_000_000, available_bytes: 580_000_000_000, min_free_bytes: 20_000_000_000 } }, "Disponible", /400.0 GB/],
      [{ enabled: false, available: null, metrics_enabled: false, storage: null }, "Deshabilitado", null],
      [{ enabled: true, available: false, metrics_enabled: true, storage: null }, "No disponible", null],
    ];
    for (const [pizarra, label, metric] of states) {
      cleanup(); installApiMock({ pizarra }); render(<App />);
      expect(await screen.findByText(`PIZARRA: ${label}`)).toBeTruthy();
      if (metric) { fireEvent.click(screen.getByText(`PIZARRA: ${label}`)); expect(await screen.findByText(metric)).toBeTruthy(); }
    }
  });

  it("distingue backend PIZARRA desconectado como estado desconocido", async () => {
    installApiMock();
    const original = globalThis.fetch;
    globalThis.fetch = vi.fn(async (url, options) => url === "/api/pizarra/status" ? Promise.reject(new Error("offline")) : original(url, options));
    render(<App />);
    expect(await screen.findByText("PIZARRA: Desconocido")).toBeTruthy();
  });

  it("confirma cambio SD 1.5 a SDXL y refresca motor y LoRA", async () => {
    const models = [
      { label: "CyberRealistic", value: "cyber", family: "sd15", active: true },
      { label: "Juggernaut XL v9", value: "xl", family: "sdxl", active: false },
    ];
    const changed = models.map((model) => ({ ...model, active: model.value === "xl" }));
    installApiMock({ models, engine: { model: "CyberRealistic", family: "sd15", status: "idle" }, changedModels: changed, changedLoras: [{ value: "xl-lora", label: "LoRA XL", family: "sdxl", active: false, compatible: true }] });
    await openImageCreation();
    fireEvent.change(screen.getByLabelText("Checkpoint registrado"), { target: { value: "xl" } });
    await waitFor(() => expect(screen.getByLabelText("Checkpoint registrado").value).toBe("xl"));
    expect(globalThis.fetch.mock.calls.some(([url, options]) => url === "/api/models/active" && options.method === "POST")).toBe(true);
    expect(globalThis.fetch.mock.calls.filter(([url]) => ["/api/models", "/api/loras", "/api/engine"].includes(url)).length).toBeGreaterThan(3);
  });

  it("mantiene el modelo anterior y explica HTTP 409 durante un cambio", async () => {
    installApiMock({ models: [{ label: "CyberRealistic", value: "cyber", family: "sd15", active: true }, { label: "Juggernaut", value: "xl", family: "sdxl", active: false }], changeError: { status: 409, detail: "busy" } });
    await openImageCreation();
    fireEvent.change(screen.getByLabelText("Checkpoint registrado"), { target: { value: "xl" } });
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", "El motor está ocupado. Espera a que termine la generación para cambiar de modelo.");
    expect(screen.getByText("CyberRealistic")).toBeTruthy();
  });

  it("permite volver de SDXL a SD 1.5 y bloquea familias desconocidas", async () => {
    const models = [
      { label: "CyberRealistic", value: "cyber", family: "sd15", active: false },
      { label: "Juggernaut XL v9", value: "xl", family: "sdxl", active: true },
      { label: "Checkpoint sin identificar", value: "unknown", family: "unknown", active: false },
    ];
    const changed = models.map((model) => ({ ...model, active: model.value === "cyber" }));
    installApiMock({ models, engine: { model: "Juggernaut XL v9", family: "sdxl", status: "idle" }, changedModels: changed });
    await openImageCreation();
    expect(screen.getByRole("option", { name: /Checkpoint sin identificar/ }).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Checkpoint registrado"), { target: { value: "cyber" } });
    await waitFor(() => expect(screen.getByLabelText("Checkpoint registrado").value).toBe("cyber"));
    expect(await screen.findByText(/Face Reference solo admite SD 1.5/)).toBeTruthy();
  });

  it("conserva checkpoint ante HTTP 400 y muestra el motivo del backend", async () => {
    installApiMock({ models: [{ label: "CyberRealistic", value: "cyber", family: "sd15", active: true }, { label: "Juggernaut", value: "xl", family: "sdxl", active: false }], changeError: { status: 400, detail: "Modelo no registrado" } });
    await openImageCreation();
    fireEvent.change(screen.getByLabelText("Checkpoint registrado"), { target: { value: "xl" } });
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", "No se pudo cambiar el modelo: Modelo no registrado");
    expect(screen.getByLabelText("Checkpoint registrado").value).toBe("cyber");
  });

  it("muestra restricciones SDXL y deshabilita cambios con el motor ocupado", async () => {
    installApiMock({
      models: [{ label: "Juggernaut", value: "xl", family: "sdxl", active: true }],
      engine: { model: "Juggernaut", family: "sdxl", status: "busy", resolutions: [512, 640, 768] },
      loras: [{ label: "SD15", value: "lora", family: "sd15", active: false, compatible: false, reason: "Familia incompatible" }],
    });
    await openImageCreation();
    expect(await screen.findByText(/Face Reference solo admite SD 1.5/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Modo experto" }));
    expect(screen.getByLabelText("Checkpoint registrado").disabled).toBe(true);
    expect(screen.getByLabelText(/SD15/).disabled).toBe(true);
    expect(screen.getByText("Familia incompatible")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Actualizar catálogo" }).disabled).toBe(true);
  });
  it("muestra el rechazo del motor ocupado enviado por la API", async () => {
    const busyMessage = "El motor está ocupado generando o cambiando de modelo.";
    installApiMock({ busyMessage });
    await openImageCreation();
    fireEvent.change(screen.getByLabelText("¿Qué imagen quieres crear?"), { target: { value: "Un paisaje" } });
    fireEvent.click(screen.getByRole("button", { name: "Crear imagen" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", busyMessage);
  });

  it("muestra la advertencia de truncamiento del trabajo", async () => {
    const warning = "El prompt supera el límite CLIP y se truncará. El texto excedente no se procesará.";
    installApiMock({ jobWarnings: [warning] });
    await openImageCreation();
    fireEvent.change(screen.getByLabelText("¿Qué imagen quieres crear?"), { target: { value: "Un paisaje" } });
    fireEvent.click(screen.getByRole("button", { name: "Crear imagen" }));
    expect(await screen.findByText(warning, {}, { timeout: 3000 })).toBeTruthy();
  });

  it("permite comparar original y resultado cuando ambos están disponibles", () => {
    render(<BeforeAfterSlider original="original.png" result="resultado.png" />);
    const slider = screen.getByRole("slider", { name: "Comparar imagen original y resultado" });
    expect(slider.value).toBe("50");
    fireEvent.change(slider, { target: { value: "72" } });
    expect(slider.value).toBe("72");
    expect(screen.getByAltText("Imagen original")).toBeTruthy();
    expect(screen.getByAltText("Imagen editada")).toBeTruthy();
  });

  it("muestra las cuatro herramientas y navega a sus pantallas", async () => {
    render(<App />);
    expect(await screen.findByRole("heading", { name: "¿Qué te gustaría crear hoy?" })).toBeTruthy();
    for (const name of ["Crear imagen", "Crear video", "Editar imagen", "Editar video"]) {
      expect(screen.getByRole("button", { name: new RegExp(name) })).toBeTruthy();
    }
    fireEvent.click(screen.getByRole("button", { name: /Editar video/ }));
    expect(await screen.findByRole("heading", { name: "Editar video" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Editar video" }).disabled).toBe(true);
  });

  it("usa modo sencillo por defecto y persiste el cambio de modo", async () => {
    render(<App />);
    const toggle = await screen.findByRole("button", { name: "Modo experto" });
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(toggle);
    expect(screen.getByRole("button", { name: "Modo experto · Activado" }).getAttribute("aria-pressed")).toBe("true");
    expect(localStorage.getItem("buttervision-experience-mode")).toBe("expert");
    cleanup();
    render(<App />);
    expect(await screen.findByRole("button", { name: "Modo experto · Activado" })).toBeTruthy();
  });

  it("conserva la descripción escrita al alternar modos", async () => {
    await openImageCreation();
    const prompt = screen.getByLabelText("¿Qué imagen quieres crear?");
    fireEvent.change(prompt, { target: { value: "Un jardín al atardecer" } });
    fireEvent.click(screen.getByRole("button", { name: "Modo experto" }));
    expect(screen.getByLabelText("¿Qué imagen quieres crear?").value).toBe("Un jardín al atardecer");
    expect(screen.getByText("Parámetros del modo experto")).toBeTruthy();
  });

  it("envía la creación sin referencia al endpoint txt2img con formato elegido", async () => {
    const onGenerate = vi.fn();
    installApiMock({ onGenerate });
    await openImageCreation();
    fireEvent.change(screen.getByLabelText("¿Qué imagen quieres crear?"), { target: { value: "Un paisaje" } });
    fireEvent.click(screen.getByRole("button", { name: "Vertical" }));
    fireEvent.click(screen.getByRole("button", { name: "Crear imagen" }));
    await waitFor(() => expect(onGenerate).toHaveBeenCalled());
    expect(onGenerate.mock.calls[0][0]).toBe("/api/generate/txt2img");
    expect(onGenerate.mock.calls[0][1]).toMatchObject({ prompt: "Un paisaje", width: 384, height: 512, batch_size: 1 });
    await waitFor(() => expect(screen.getAllByText("En cola").length).toBeGreaterThan(0));
  });

  it("usa face-reference al cargar una fotografía y muestra su vista previa", async () => {
    const onGenerate = vi.fn();
    installApiMock({ onGenerate });
    await openImageCreation();
    fireEvent.click(screen.getByRole("button", { name: "Modo experto" }));
    const file = new File(["image"], "rostro.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Añadir una foto de referencia"), { target: { files: [file] } });
    expect(await screen.findByAltText("Vista previa de la foto de referencia")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cuadrado" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Vertical" }).disabled).toBe(true);
    expect(screen.getByLabelText("Ancho").disabled).toBe(true);
    expect(screen.getByLabelText("Ancho").value).toBe("512");
    expect(screen.getByLabelText("Alto").value).toBe("512");
    fireEvent.change(screen.getByLabelText("¿Qué imagen quieres crear?"), { target: { value: "Retrato" } });
    fireEvent.click(screen.getByRole("button", { name: "Crear imagen" }));
    await waitFor(() => expect(onGenerate).toHaveBeenCalled());
    expect(onGenerate.mock.calls[0][0]).toBe("/api/generate/face-reference");
    expect(onGenerate.mock.calls[0][1]).toMatchObject({ prompt: "Retrato", width: 512, height: 512 });
    expect(onGenerate.mock.calls[0][1].reference_image).toContain("data:image/png;base64,");
  });

  it("mantiene la orientación experta sincronizada con ancho y alto", async () => {
    const onGenerate = vi.fn();
    installApiMock({ onGenerate });
    await openImageCreation();
    fireEvent.click(screen.getByRole("button", { name: "Modo experto" }));
    fireEvent.click(screen.getByText("Parámetros del modo experto"));
    fireEvent.change(screen.getByLabelText("Alto"), { target: { value: "384" } });
    expect(screen.getByRole("button", { name: "Horizontal" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.change(screen.getByLabelText("¿Qué imagen quieres crear?"), { target: { value: "Costa" } });
    fireEvent.click(screen.getByRole("button", { name: "Crear imagen" }));
    await waitFor(() => expect(onGenerate).toHaveBeenCalled());
    expect(onGenerate.mock.calls[0][1]).toMatchObject({ width: 512, height: 384 });
  });

  it("valida formatos y tamaño de archivo usando límites enviados por FastAPI", async () => {
    installApiMock({ uploadLimits: { image_bytes: 4, video_bytes: 6 } });
    render(<App />);
    await screen.findByRole("button", { name: /Editar imagen/ });
    fireEvent.click(screen.getByRole("button", { name: /Editar imagen/ }));
    const imageInput = await screen.findByLabelText("Subir imagen");
    fireEvent.change(imageInput, {
      target: { files: [new File(["bad"], "animated.gif", { type: "image/gif" })] },
    });
    expect((await screen.findByRole("alert")).textContent).toContain("JPEG, PNG o WebP");
    fireEvent.change(imageInput, {
      target: { files: [new File(["12345"], "large.png", { type: "image/png" })] },
    });
    expect((await screen.findByRole("alert")).textContent).toContain("supera el límite de 4 bytes");
    expect(globalThis.fetch.mock.calls.some(([url]) => url.includes("/api/generate/"))).toBe(false);
  });

  it("valida formato y tamaño de videos antes de conservarlos en pantalla", async () => {
    installApiMock({ uploadLimits: { image_bytes: 8, video_bytes: 6 } });
    render(<App />);
    await screen.findByRole("button", { name: /Editar video/ });
    fireEvent.click(screen.getByRole("button", { name: /Editar video/ }));
    const videoInput = await screen.findByLabelText("Subir video");
    fireEvent.change(videoInput, {
      target: { files: [new File(["bad"], "movie.avi", { type: "video/x-msvideo" })] },
    });
    expect((await screen.findByRole("alert")).textContent).toContain("Usa MP4, WebM o MOV");
    fireEvent.change(videoInput, {
      target: { files: [new File(["1234567"], "large.mp4", { type: "video/mp4" })] },
    });
    expect((await screen.findByRole("alert")).textContent).toContain("supera el límite de 6 bytes");
  });

  it("rechaza archivos que no sean imágenes y deja operativas las pantallas vacías", async () => {
    render(<App />);
    await screen.findByRole("button", { name: /Crear imagen/ });
    fireEvent.click(screen.getByRole("button", { name: /Editar imagen/ }));
    fireEvent.change(await screen.findByLabelText("Subir imagen"), {
      target: { files: [new File(["not an image"], "nota.txt", { type: "text/plain" })] },
    });
    expect((await screen.findByRole("alert")).textContent).toContain("JPEG, PNG o WebP");
    fireEvent.click(screen.getByRole("button", { name: "Mis creaciones" }));
    expect(await screen.findByText("Aún no hay creaciones")).toBeTruthy();
  });

  it("muestra los resultados de la galería y no activa procesamiento de video", async () => {
    installApiMock({ history: ["/outputs/sample.png"] });
    render(<App />);
    await screen.findByRole("button", { name: /Crear video/ });
    fireEvent.click(screen.getByRole("button", { name: "Mis creaciones" }));
    expect(await screen.findByAltText("Creación guardada")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Inicio" }));
    fireEvent.click(screen.getByRole("button", { name: /Crear video/ }));
    expect(await screen.findByText("Pendiente de integración")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Crear video" }).disabled).toBe(true);
    expect(globalThis.fetch.mock.calls.some(([url]) => url.includes("/api/generate/"))).toBe(false);
  });

  it("expone los errores HTTP sin dejar la pantalla inutilizable", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (url === "/api/runtime") return response(runtime);
      if (url === "/api/models") return response([]);
      if (url === "/api/history") return response({ images: [] });
      if (url.startsWith("/api/generate/")) {
        return {
          ok: false,
          status: 422,
          statusText: "Unprocessable Entity",
          text: async () => JSON.stringify({ detail: "La descripción no es válida." }),
        };
      }
      throw new Error(`Unexpected API request: ${url} ${options.method || "GET"}`);
    });
    await openImageCreation();
    fireEvent.change(screen.getByLabelText("¿Qué imagen quieres crear?"), { target: { value: "Paisaje" } });
    fireEvent.click(screen.getByRole("button", { name: "Crear imagen" }));
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("La descripción no es válida.");
    expect(screen.getByRole("button", { name: "Inicio" })).toBeTruthy();
  });
});
