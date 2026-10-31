import { useEffect, useRef, useState } from "react";
import { api } from "../services/api.js";

function formatGB(bytes) {
  return bytes == null ? "No informado" : `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

export default function PizarraStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const inFlight = useRef(false);

  useEffect(() => {
    let active = true;
    let timer;
    async function refresh() {
      if (inFlight.current) { timer = window.setTimeout(refresh, 30000); return; }
      inFlight.current = true;
      try {
        const next = await api("/api/pizarra/status");
        if (active) { setStatus(next); setError(false); }
      } catch {
        if (active) setError(true);
      } finally {
        inFlight.current = false;
        if (active) timer = window.setTimeout(refresh, 30000);
      }
    }
    refresh();
    return () => { active = false; window.clearTimeout(timer); };
  }, []);

  async function refreshNow() {
    if (inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    try { setStatus(await api("/api/pizarra/status")); setError(false); }
    catch { setError(true); }
    finally { inFlight.current = false; setLoading(false); }
  }

  const state = error || !status ? "Desconocido" : !status.enabled ? "Deshabilitado" : status.available ? "Disponible" : "No disponible";
  const storage = status?.storage;
  const used = storage?.managed_bytes;
  const quota = storage?.quota_bytes;
  const occupancy = Number.isFinite(used) && Number.isFinite(quota) && quota > 0 ? Math.min(100, (used / quota) * 100) : null;

  return <details className="pizarra-panel">
    <summary><span className={`pizarra-dot ${error || !status ? "unknown" : !status.enabled ? "disabled" : status.available ? "available" : "unavailable"}`} />PIZARRA: {state}</summary>
    <div className="pizarra-content">
      <div className="section-label"><strong>Estado del banco</strong><button type="button" className="text-button" onClick={refreshNow} disabled={loading}>{loading ? "Actualizando…" : "Actualizar"}</button></div>
      {error || !status ? <p>No se pudo consultar el estado del backend.</p> : !status.enabled ? <p>PIZARRA está deshabilitado por configuración.</p> : !status.available ? <p>La función está habilitada, pero el volumen no está disponible o autorizado.</p> : <>
        <p className="muted">Disponible. Las métricas {status.metrics_enabled ? "están habilitadas" : "están deshabilitadas"}.</p>
        <dl className="storage-metrics">
          <div><dt>Capacidad informada del volumen</dt><dd>{formatGB(storage?.total_bytes)}</dd></div>
          <div><dt>Cuota lógica configurada</dt><dd>{formatGB(quota)}</dd></div>
          <div><dt>Ocupado por PIZARRA</dt><dd>{formatGB(used)}</dd></div>
          <div><dt>Restante en cuota</dt><dd>{formatGB(storage?.quota_remaining_bytes)}</dd></div>
          <div><dt>Libre físicamente</dt><dd>{formatGB(storage?.free_bytes)}</dd></div>
          <div><dt>Utilizable según el servicio</dt><dd>{formatGB(storage?.available_bytes)}</dd></div>
          <div><dt>Reserva mínima de seguridad</dt><dd>{formatGB(storage?.min_free_bytes)}</dd></div>
        </dl>
        {occupancy != null && <div className="quota-meter" role="img" aria-label={`Cuota PIZARRA ocupada ${occupancy.toFixed(0)} por ciento`}><span style={{ width: `${occupancy}%` }} /></div>}
      </>}
    </div>
  </details>;
}
