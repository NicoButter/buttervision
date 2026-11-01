import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PizarraStatus from "./PizarraStatus.jsx";

const available = {
  enabled: true, available: true, metrics_enabled: true,
  activity: { confirmed: false, last_io_at: null },
  storage: { total_bytes: 800_000_000_000, quota_bytes: 400_000_000_000, managed_bytes: 100_000_000_000 },
};
const response = (data = available) => ({ ok: true, json: async () => data });
const fullRequests = () => globalThis.fetch.mock.calls.filter(([url]) => url === "/api/pizarra/status");

beforeEach(() => { globalThis.fetch = vi.fn(async () => response()); });
afterEach(() => { cleanup(); vi.useRealTimers(); });

async function openPanel() {
  const trigger = await screen.findByRole("button", { name: /PIZARRA: Disponible/ });
  fireEvent.click(trigger);
  const panel = screen.getByRole("dialog", { name: "Estado del banco PIZARRA" });
  await waitFor(() => expect(fullRequests()).toHaveLength(1));
  await waitFor(() => expect(screen.getByRole("button", { name: "Actualizar" }).disabled).toBe(false));
  return { trigger, panel };
}

describe("Popover PIZARRA", () => {
  it("abre un diálogo no modal con nombre, foco y dos indicadores; Escape devuelve el foco", async () => {
    render(<PizarraStatus navigationKey="home" />);
    expect(screen.queryByRole("dialog")).toBeNull();
    const { trigger, panel } = await openPanel();
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    expect(trigger.getAttribute("aria-controls")).toBe(panel.id);
    expect(trigger.getAttribute("aria-haspopup")).toBe("dialog");
    expect(panel.getAttribute("aria-modal")).toBe("false");
    expect(document.activeElement).toBe(panel);
    expect(trigger.querySelectorAll(".pizarra-dot")).toHaveLength(2);
    expect(trigger.querySelector(".available")).toBeTruthy();
    expect(screen.getAllByText("Offload de inferencia: no implementado por estos contadores.")).toHaveLength(1);
    fireEvent.keyDown(panel, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(trigger);
    expect(fullRequests()).toHaveLength(1);
  });

  it("cierra al pulsar fuera y permite cambiar el selector de modelos sin recuperar el foco", async () => {
    const onChange = vi.fn();
    render(<><select aria-label="Modelo global" onChange={onChange}><option value="sd15">SD 1.5</option><option value="sdxl">SDXL</option></select><PizarraStatus /></>);
    await openPanel();
    const selector = screen.getByRole("combobox", { name: "Modelo global" });
    fireEvent.pointerDown(selector);
    act(() => selector.focus());
    fireEvent.change(selector, { target: { value: "sdxl" } });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(selector);
    expect(selector.value).toBe("sdxl");
    expect(onChange).toHaveBeenCalledOnce();
  });

  it("permite salir con Tab y cierra cuando el foco pasa a otro control", async () => {
    render(<><PizarraStatus /><button>Otro control</button></>);
    await openPanel();
    const outside = screen.getByRole("button", { name: "Otro control" });
    act(() => outside.focus());
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(outside);
  });

  it("cierra con el botón explícito y restaura el foco", async () => {
    render(<PizarraStatus />);
    const { trigger } = await openPanel();
    fireEvent.click(screen.getByRole("button", { name: "Cerrar panel PIZARRA" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it("cierra al cambiar de página sin nuevas consultas", async () => {
    const { rerender } = render(<PizarraStatus navigationKey="home" />);
    await openPanel();
    const requests = globalThis.fetch.mock.calls.length;
    rerender(<PizarraStatus navigationKey="gallery" />);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(globalThis.fetch.mock.calls).toHaveLength(requests);
  });

  it("reutiliza los detalles recientes al reabrir y consulta solo al actualizar explícitamente", async () => {
    render(<PizarraStatus />);
    const { trigger } = await openPanel();
    fireEvent.keyDown(document, { key: "Escape" });
    fireEvent.click(trigger);
    expect(screen.getByRole("dialog")).toBeTruthy();
    expect(fullRequests()).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Actualizar" }));
    await waitFor(() => expect(fullRequests()).toHaveLength(2));
    expect(globalThis.fetch.mock.calls.filter(([url]) => url.endsWith("?light=true"))).toHaveLength(1);
  });

  it("espera el sondeo pendiente sin duplicar peticiones y no solicita detalles si ya se cerró", async () => {
    let resolveLight;
    globalThis.fetch.mockImplementationOnce(() => new Promise((resolve) => { resolveLight = resolve; }));
    render(<PizarraStatus />);
    fireEvent.click(screen.getByRole("button", { name: /PIZARRA:/ }));
    expect(globalThis.fetch).toHaveBeenCalledOnce();
    fireEvent.keyDown(document, { key: "Escape" });
    await act(async () => resolveLight(response()));
    expect(fullRequests()).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: /PIZARRA: Disponible/ }));
    await waitFor(() => expect(fullRequests()).toHaveLength(1));
  });

  it("ajusta el alto disponible cuando cambia la altura de la pantalla", async () => {
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ top: 120 });
    const viewport = vi.spyOn(window, "innerHeight", "get").mockReturnValue(400);
    render(<PizarraStatus />);
    const { panel } = await openPanel();
    expect(panel.style.getPropertyValue("--pizarra-available-height")).toBe("264px");
    viewport.mockReturnValue(320);
    fireEvent(window, new Event("resize"));
    expect(panel.style.getPropertyValue("--pizarra-available-height")).toBe("184px");
  });

  it("caduca la actividad confirmada aunque todavía no haya otro sondeo", async () => {
    vi.useFakeTimers();
    globalThis.fetch.mockResolvedValue(response({ ...available, activity: { confirmed: true, last_io_at: Date.now() / 1000 } }));
    render(<PizarraStatus />);
    await act(async () => {});
    expect(screen.getByLabelText("Actividad PIZARRA confirmada")).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(3100); });
    expect(screen.getByLabelText("Sin actividad PIZARRA confirmada")).toBeTruthy();
  });
});
