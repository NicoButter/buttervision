import { useEffect, useRef, useState } from "react";
import { api } from "../services/api.js";

function formatGB(bytes) {
  return bytes == null ? "No informado" : `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

export default function PizarraStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activity, setActivity] = useState(false);
  const inFlight = useRef(false);

  useEffect(() => {
    let active = true;
    let timer;
    async function refresh() {
      if (inFlight.current) { timer = window.setTimeout(refresh, 30000); return; }
      inFlight.current = true;
      let nextDelay = 30000;
      try {
        const next = await api("/api/pizarra/status?light=true");
        nextDelay = next.activity?.confirmed ? 1000 : 30000;
        if (active) { setStatus(next); setError(false); setActivity(Boolean(next.activity?.confirmed)); }
      } catch {
        if (active) { setError(true); setActivity(false); }
      } finally {
        inFlight.current = false;
        if (active) timer = window.setTimeout(refresh, nextDelay);
      }
    }
    refresh();
    return () => { active = false; window.clearTimeout(timer); };
  }, []);

  async function refreshNow() {
    if (inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    try { const next = await api("/api/pizarra/status"); setStatus(next); setActivity(Boolean(next.activity?.confirmed)); setError(false); }
    catch { setError(true); setActivity(false); }
    finally { inFlight.current = false; setLoading(false); }
  }

  const state = error || !status ? "Desconocido" : status.pending_authorization ? "Pendiente de autorización" : !status.enabled ? "Deshabilitado" : status.available ? "Disponible" : "No disponible";
  const storage = status?.storage;
  const used = storage?.managed_bytes;
  const quota = storage?.quota_bytes;
  const occupancy = Number.isFinite(used) && Number.isFinite(quota) && quota > 0 ? Math.min(100, (used / quota) * 100) : null;

  const availability = error || !status ? "error" : (!status.enabled && !status.pending_authorization) ? "disabled" : status.pending_authorization ? "pending" : status.available ? "available" : "unavailable";
  return <details className="pizarra-panel" onToggle={(event) => { if (event.currentTarget.open) refreshNow(); }}>
    <summary><span className={`pizarra-dot ${availability}`} />PIZARRA: {state}<span className={`pizarra-dot activity-dot ${activity ? "active" : ""}`} aria-label={activity ? "Actividad PIZARRA confirmada" : "Sin actividad PIZARRA confirmada"} /></summary>
    <div className="pizarra-content">
      <div className="section-label"><strong>Estado del banco</strong><button type="button" className="text-button" onClick={refreshNow} disabled={loading}>{loading ? "Actualizando…" : "Actualizar"}</button></div>
      {error || !status ? <p>No se pudo consultar el estado del backend.</p> : status.pending_authorization ? <p>Se detectó un montaje en la ruta configurada. Configura su UUID autorizado en `.env.local` y reinicia; el sondeo no escribió en el volumen.</p> : !status.enabled ? <p>PIZARRA está deshabilitado por configuración.</p> : !status.available ? <p>La función está habilitada, pero el volumen no está disponible o autorizado.</p> : <>
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
        <p className="muted">Operaciones medidas — metadatos: {formatGB(storage?.metadata_read_bytes)} leídos · {formatGB(storage?.metadata_written_bytes)} escritos; transferencias tensoriales: {formatGB(storage?.tensor_read_bytes)} leídos · {formatGB(storage?.tensor_written_bytes)} escritos.</p>
        <p className="muted">E/S medida por PizarraStorage: {formatGB(storage?.read_bytes)} leídos · {formatGB(storage?.written_bytes)} escritos. Incluye metadatos y transferencias tensoriales gestionadas; no mide actividad externa ni confirma offload de inferencia.</p>
        <p className="muted">Offload de inferencia: no implementado por estos contadores.</p>
        <p className="muted">Offload de inferencia: no implementado por estos contadores.</p>
        {occupancy != null && <div className="quota-meter" role="img" aria-label={`Cuota PIZARRA ocupada ${occupancy.toFixed(0)} por ciento`}><span style={{ width: `${occupancy}%` }} /></div>}
      </>}
    </div>
  </details>;
}
