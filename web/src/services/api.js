const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const body = await response.text();
    let message = body || response.statusText;
    try {
      const parsed = JSON.parse(body);
      message = parsed.detail || parsed.message || message;
    } catch {
      // Keep plain-text API errors readable.
    }
    throw new Error(message || `Error HTTP ${response.status}`);
  }
  return response.json();
}

export function outputUrl(path) {
  return `${API_BASE}${path}`;
}

export function userFacingError(error) {
  const message = error instanceof Error ? error.message : "";
  if (/networkerror|failed to fetch|load failed/i.test(message)) {
    return "No se pudo conectar con Buttervision. Comprueba que el servidor esté en marcha.";
  }
  return message || "Ocurrió un error inesperado. Inténtalo de nuevo.";
}
