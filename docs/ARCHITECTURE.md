# Arquitectura de Buttervision

## Interfaces

React, en `web/`, es la interfaz principal para el desarrollo futuro. Presenta los flujos de crear imagen, crear video, editar imagen y editar video, además de la galería y los modos sencillo/experto. La generación de imágenes y la galería usan la API local; video y edición son experiencias visuales pendientes de integración y no simulan procesamiento.

La interfaz Gradio de `ui/interface.py` se conserva para compatibilidad y diagnóstico. No comparte navegación ni estado con React. Su flujo de referencia facial ya no persiste la fotografía y la galería excluye también las referencias históricas.

## Flujo de React y API

`./run.sh` inicia React en `http://127.0.0.1:5173` y FastAPI en `http://127.0.0.1:7861`. `./run_gradio.sh` conserva la interfaz Gradio anterior en `http://127.0.0.1:7860`. Los lanzadores individuales de React y FastAPI se reservan para desarrollo. Los tres servicios se enlazan a loopback por defecto. En producción local, `npm run build` crea `web/dist`; `api/app.py` entrega esos archivos cuando están presentes.

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

React no cambia las rutas de generación existentes. El navegador envía descripciones y parámetros existentes; la referencia facial se codifica como Data URL, se valida en FastAPI y no se escribe en disco en nuevas solicitudes. Las referencias generadas anteriormente permanecen en el directorio local, sin borrado automático, y el endpoint estático bloquea nombres conocidos de referencia, `prompt.txt` y `metadata.json`. Las preferencias de modo, no los archivos ni los prompts, se guardan en el almacenamiento local del navegador.

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
- `api/app.py` expone imágenes generadas de `outputs/` mediante `/outputs` sin autenticación; no expone archivos de referencia con nombre conocido, `prompt.txt` ni `metadata.json`. Archivos antiguos se conservan en disco, pero no se entregan por las rutas bloqueadas. PNG de resultados pueden contener metadata incrustada y continúan siendo accesibles localmente.
- Los tamaños máximos configurables son 10 MiB para imágenes y 100 MiB para videos por defecto. Se validan JPEG/PNG/WebP; para el flujo actualmente integrado, FastAPI valida también el contenido de la foto facial. Videos aún no se envían a backend.
- Loopback reduce exposición de red, pero no sustituye autenticación ni control de acceso local. Cualquier proceso/usuario con acceso al servicio local puede consultar resultados de imagen; no hay expiración o limpieza automatizada.
- `core/face_reference_pipeline.py` conserva dependencias de assets y rutas de instalación de Forge. No se resuelven en UX v2.1; queda pendiente para la siguiente fase.
- No se añade telemetría, almacenamiento remoto ni servicio externo. La tipografía y los elementos gráficos usan recursos locales del sistema y CSS.
- Un futuro flujo de edición requiere un contrato de API propio antes de conectar cualquier procesamiento.
