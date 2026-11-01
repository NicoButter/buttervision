import { useCallback, useEffect, useId, useRef, useState } from "react";
import { api } from "../services/api.js";

function formatGB(bytes) {
  return bytes == null ? "No informado" : `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

export default function PizarraStatus({ navigationKey }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activity, setActivity] = useState(false);
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const triggerRef = useRef(null);
  const panelRef = useRef(null);
  const rootRef = useRef(null);
  const mounted = useRef(false);
  const inFlight = useRef(null);
  const detailedCache = useRef(null);
  const previousNavigation = useRef(navigationKey);

  const requestStatus = useCallback(async function request(detailed = false, force = false) {
    if (inFlight.current) {
      const pending = inFlight.current;
      if (!detailed || pending.detailed) return pending.promise;
      await pending.promise.catch(() => {});
      return mounted.current ? request(detailed, force) : null;
    }
    if (detailed && !force && detailedCache.current && Date.now() - detailedCache.current.time < 30000) {
      return detailedCache.current.data;
    }
    if (detailed && mounted.current) setLoading(true);
    const promise = api(detailed ? "/api/pizarra/status" : "/api/pizarra/status?light=true")
      .then((next) => {
        if (detailed) detailedCache.current = { time: Date.now(), data: next };
        if (mounted.current) {
          setStatus((previous) => ({ ...next, storage: !detailed && next.available && previous?.available
            ? { ...previous.storage, ...next.storage } : next.storage }));
          setError(false);
        }
        return next;
      })
      .catch((error) => {
        if (mounted.current) setError(true);
        throw error;
      })
      .finally(() => {
        inFlight.current = null;
        if (mounted.current && detailed) setLoading(false);
      });
    inFlight.current = { detailed, promise };
    return promise;
  }, []);

  useEffect(() => {
    mounted.current = true;
    let active = true;
    let timer;
    async function refresh() {
      let nextDelay = 30000;
      try {
        const next = await requestStatus();
        nextDelay = next?.activity?.confirmed ? 1000 : 30000;
      } catch { /* The request reports backend errors without extending activity. */ }
      finally {
        if (active) timer = window.setTimeout(refresh, nextDelay);
      }
    }
    refresh();
    return () => { active = false; mounted.current = false; window.clearTimeout(timer); };
  }, [requestStatus]);

  useEffect(() => {
    const confirmed = !error && status?.enabled && status?.available && status?.activity?.confirmed;
    setActivity(Boolean(confirmed));
    if (!confirmed) return undefined;
    const remaining = status.activity.last_io_at == null ? 3000
      : Math.max(0, Math.min(3000, status.activity.last_io_at * 1000 + 3000 - Date.now()));
    const timer = window.setTimeout(() => setActivity(false), remaining);
    return () => window.clearTimeout(timer);
  }, [error, status?.enabled, status?.available, status?.activity?.confirmed, status?.activity?.last_io_at]);

  function close(restoreFocus = false) {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  }

  useEffect(() => {
    if (previousNavigation.current !== navigationKey) {
      previousNavigation.current = navigationKey;
      if (panelRef.current?.contains(document.activeElement)) triggerRef.current?.focus();
      setOpen(false);
    }
  }, [navigationKey]);

  useEffect(() => {
    if (!open) return undefined;
    let active = true;
    panelRef.current?.focus();
    async function loadDetails() {
      if (inFlight.current && !inFlight.current.detailed) await inFlight.current.promise.catch(() => {});
      if (active) requestStatus(true).catch(() => {});
    }
    loadDetails();
    function fitViewport() {
      const panel = panelRef.current;
      if (panel) panel.style.setProperty("--pizarra-available-height", `${Math.max(0, window.innerHeight - panel.getBoundingClientRect().top - 16)}px`);
    }
    fitViewport();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fitViewport);
    if (rootRef.current) observer?.observe(rootRef.current.closest(".topbar") || rootRef.current);
    function dismissOutside(event) {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    }
    function dismissEscape(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        setOpen(false);
        triggerRef.current?.focus();
      }
    }
    document.addEventListener("pointerdown", dismissOutside, true);
    document.addEventListener("focusin", dismissOutside);
    document.addEventListener("keydown", dismissEscape);
    window.addEventListener("resize", fitViewport);
    window.addEventListener("scroll", fitViewport, { passive: true });
    return () => {
      active = false;
      document.removeEventListener("pointerdown", dismissOutside, true);
      document.removeEventListener("focusin", dismissOutside);
      document.removeEventListener("keydown", dismissEscape);
      window.removeEventListener("resize", fitViewport);
      window.removeEventListener("scroll", fitViewport);
      observer?.disconnect();
    };
  }, [open, requestStatus]);

  function refreshNow() { requestStatus(true, true).catch(() => {}); }

  const state = error || !status ? "Desconocido" : status.pending_authorization ? "Pendiente de autorización" : !status.enabled ? "Deshabilitado" : status.available ? "Disponible" : "No disponible";
  const storage = status?.storage;
  const used = storage?.managed_bytes;
  const quota = storage?.quota_bytes;
  const occupancy = Number.isFinite(used) && Number.isFinite(quota) && quota > 0 ? Math.min(100, (used / quota) * 100) : null;

  const availability = error || !status ? "error" : (!status.enabled && !status.pending_authorization) ? "disabled" : status.pending_authorization ? "pending" : status.available ? "available" : "unavailable";
  return <div className="pizarra-panel" ref={rootRef}>
    <button type="button" ref={triggerRef} className="pizarra-trigger" aria-haspopup="dialog" aria-expanded={open} aria-controls={open ? panelId : undefined} onClick={() => open ? close(true) : setOpen(true)}><span className={`pizarra-dot ${availability}`} />PIZARRA: {state}<span className={`pizarra-dot activity-dot ${activity ? "active" : ""}`} aria-label={activity ? "Actividad PIZARRA confirmada" : "Sin actividad PIZARRA confirmada"} /></button>
    {open && <div id={panelId} ref={panelRef} className="pizarra-content" role="dialog" aria-modal="false" aria-labelledby={`${panelId}-title`} tabIndex={-1}>
      <div className="section-label"><strong id={`${panelId}-title`}>Estado del banco PIZARRA</strong><button type="button" className="text-button" onClick={() => close(true)} aria-label="Cerrar panel PIZARRA">Cerrar</button></div>
      <button type="button" className="text-button" onClick={refreshNow} disabled={loading}>{loading ? "Actualizando…" : "Actualizar"}</button>
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
        {occupancy != null && <div className="quota-meter" role="img" aria-label={`Cuota PIZARRA ocupada ${occupancy.toFixed(0)} por ciento`}><span style={{ width: `${occupancy}%` }} /></div>}
      </>}
    </div>}
  </div>;
}
