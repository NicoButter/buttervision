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

function installApiMock({ history = [], onGenerate, uploadLimits } = {}) {
  globalThis.fetch = vi.fn(async (url, options = {}) => {
    if (url === "/api/runtime") return response({
      ...runtime,
      upload_limits: uploadLimits || runtime.upload_limits,
    });
    if (url === "/api/models") return response([]);
    if (url === "/api/history") return response({ images: history });
    if (url.startsWith("/api/generate/")) {
      onGenerate?.(url, JSON.parse(options.body));
      return response({ job_id: "job-1", status: "queued" });
    }
    if (url === "/api/jobs/job-1") {
      return response({
        id: "job-1", status: "running", progress: 0.25, message: "Generando",
        result_images: [], error: null, info: null,
      });
    }
    if (url === "/api/models/active") return response([]);
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
