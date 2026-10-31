# Arquitectura de Buttervision

## Interfaces

React, en `web/`, es la interfaz principal para el desarrollo futuro. Presenta los flujos de crear imagen, crear video, editar imagen y editar video, además de la galería y los modos sencillo/experto. La generación de imágenes y la galería usan la API local; video y edición son experiencias visuales pendientes de integración y no simulan procesamiento.

La interfaz Gradio de `ui/interface.py` se conserva para compatibilidad y diagnóstico. No comparte navegación ni estado con React y no se modifica en esta iteración.

## Flujo de React y API

En desarrollo, Vite sirve React en `http://localhost:5173` y FastAPI debe estar disponible en `http://localhost:7861`. En producción local, `npm run build` crea `web/dist`; `api/app.py` entrega esos archivos cuando están presentes.

```text
React (web/src)
  ├── páginas y componentes: presentación y estado local
  ├── services/api.js: llamadas HTTP y rutas de archivos
  ├── hooks/usePreferences.js: modo sencillo/experto en localStorage
  └── FastAPI (api/app.py)
        ├── /api/generate/txt2img
        ├── /api/generate/face-reference
        ├── /api/jobs/{job_id}
        ├── /api/history
        ├── /api/models y /api/models/active
        └── /api/runtime
              └── GenerationService
                    ├── core/advanced_pipeline.py
                    └── core/face_reference_pipeline.py
```

React no cambia los contratos de la API. El navegador envía descripciones y parámetros existentes; la referencia facial se codifica como Data URL y solo se conserva en memoria durante la sesión. Las preferencias de modo, no los archivos ni los prompts, se guardan en el almacenamiento local del navegador.

La API solo soporta generación de imágenes. El backend limita la resolución a 512 × 512 y el batch a 1; la referencia facial también se procesa a 512 × 512. La selección de orientación en la interfaz no puede superar estos límites. El progreso mostrado proviene del trabajo real, sin porcentajes inventados.

## Responsabilidades del backend y core

- `api/app.py`: rutas FastAPI, CORS local, archivos estáticos del frontend y montaje de `/outputs`.
- `api/schemas.py`: validación de parámetros y respuestas.
- `api/generation.py`: trabajos asíncronos, polling, historial y guardado de resultados.
- `core/advanced_pipeline.py` y `core/face_reference_pipeline.py`: motores existentes de generación. La UX v2 no los modifica.
- `ui/interface.py`: interfaz Gradio heredada.

## Límites, privacidad y siguientes integraciones

- No hay endpoints de API para crear videos, editar imágenes o editar videos; las pantallas correspondientes no envían solicitudes.
- La UI de edición no modifica ni sobrescribe archivos originales.
- `api/app.py` expone todo el directorio `outputs/` mediante `/outputs` sin autenticación. La API de generación actual también guarda `reference_face.png` junto a los resultados. La API y esos archivos deben mantenerse en una instalación local confiable; acceso controlado, retención mínima y endurecimiento de privacidad quedan pendientes.
- No se añade telemetría, almacenamiento remoto ni servicio externo. La tipografía y los elementos gráficos usan recursos locales del sistema y CSS.
- Un futuro flujo de edición requiere un contrato de API propio antes de conectar cualquier procesamiento.
