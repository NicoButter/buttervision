import { useState } from "react";

const STORAGE_KEY = "buttervision-experience-mode";

function readPreference() {
  try {
    return {
      mode: window.localStorage.getItem(STORAGE_KEY) === "expert" ? "expert" : "simple",
      error: "",
    };
  } catch {
    return { mode: "simple", error: "No se pudo acceder al almacenamiento local; el modo no se conservará al cerrar la página." };
  }
}

export function usePreferences() {
  const [preference, setPreference] = useState(readPreference);
  const { mode, error: storageError } = preference;

  function toggleMode() {
    const next = mode === "simple" ? "expert" : "simple";
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
      setPreference({ mode: next, error: "" });
    } catch {
      setPreference({ mode: next, error: "No se pudo guardar la preferencia; el modo solo se conservará durante esta sesión." });
    }
  }

  return { mode, toggleMode, storageError };
}
