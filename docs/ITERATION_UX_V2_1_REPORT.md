# Informe de estabilización — Buttervision UX v2.1

> **Actualización del lanzador:** después de esta iteración, `./run.sh` quedó como el único punto de entrada principal para React + FastAPI. Gradio se inicia con `./run_gradio.sh`; `run_api.sh` y `run_web_dev.sh` son herramientas auxiliares para desarrollo. Las referencias a `run_ux.sh` en el historial describen el lanzador usado durante la validación original.

## A. Resumen

Se estabilizó el arranque local de React y FastAPI, manteniendo Gradio como interfaz heredada. Los tres servicios tienen puertos fijos por defecto y escuchan en loopback: React `127.0.0.1:5173`, FastAPI `127.0.0.1:7861` y Gradio `127.0.0.1:7860`.

Las nuevas referencias faciales ya no se escriben en `outputs/`. El servidor estático bloquea las rutas de archivos conocidos de referencia facial, prompts y metadatos. Los cuatro prompts/metadatos que ya estaban versionados se quitaron del índice de Git, pero sus copias locales se conservaron.

Se alinearon selector de orientación, resolución experta y límites de Face Reference. Se añadieron validación MIME/tamaño en el frontend y validación de contenido/tamaño en el endpoint facial. No se generaron imágenes o videos con motores de IA, no se instalaron modelos y no se modificaron los motores.

## B. Cambios realizados

### Inicio y puertos

- `run_ux.sh` inicia React y FastAPI juntos en loopback.
- `run_ux.bat` inicia ambos servicios en ventanas locales separadas.
- `run_api.sh` / `run_api.bat` inician FastAPI en `127.0.0.1:7861`.
- `run_web_dev.sh` / `run_web_dev.bat` inician React en `127.0.0.1:5173`.
- `web/vite.config.js` fija host, puerto y `strictPort`, y reenvía `/api` y `/outputs` a FastAPI local.
- `main.py` y `config.ServerConfig` usan loopback y el puerto Gradio 7860 por defecto; compartir Gradio requiere pedirlo explícitamente.
- Se corrigieron los ejemplos de inicio en README, guías, ejemplos y mensajes de instaladores.

### Privacidad y Git

- `api/generation.py` deja de guardar `reference_face.png` con las nuevas generaciones y elimina el Data URL de la carga del trabajo una vez que empieza a procesarlo.
- `ui/interface.py` también deja de guardar la foto facial y excluye las referencias históricas de su galería.
- `api/app.py` devuelve 404 para `reference_face.png`, nombres que empiezan por `reference_`, `prompt.txt` y `metadata.json` al servir `/outputs`; la galería de FastAPI también excluye las referencias.
- `.gitignore` excluye el directorio completo `outputs/`.
- Se retiraron del índice de Git cuatro archivos históricos `metadata.json` y `prompt.txt` bajo `outputs/`. `git rm --cached` preservó los archivos locales; no se borraron del disco.

### Formatos y cargas

- El formato visible de Crear imagen deriva de la resolución efectiva, incluidos los cambios del modo experto.
- Face Reference limita la selección a Cuadrado; ancho y alto expertos muestran 512 × 512 y no se pueden editar mientras haya referencia facial.
- Los tamaños expertos se eligen entre 256, 320, 384, 448 y 512 px por dimensión, dentro del límite del backend.
- Formatos admitidos: JPEG, PNG y WebP para imágenes; MP4, WebM y MOV para videos.
- Límites configurables por variables de entorno: `BUTTERVISION_MAX_IMAGE_UPLOAD_BYTES` (10 MiB por defecto) y `BUTTERVISION_MAX_VIDEO_UPLOAD_BYTES` (100 MiB por defecto). FastAPI comunica los límites mediante `/api/runtime`.
- El endpoint de Face Reference valida base64, MIME declarado, formato real y tamaño antes de crear un trabajo, con errores HTTP explícitos.
- Los límites del video solo protegen la selección/preview de frontend: todavía no hay endpoint de video que reciba esos archivos.

## C. Pruebas y validación

### Pruebas automatizadas

- `cd web && npm test`: **12 pruebas aprobadas**. Incluyen alineación formato/resolución, bloqueo de orientación facial, límites y MIME para imagen/video, navegación y ausencia de solicitudes a funciones pendientes.
- `venv/bin/python -m unittest discover -s tests -v`: **6 pruebas aprobadas**. Se validan respuestas reales de FastAPI mediante `TestClient`, límites publicados, rechazo MIME/tamaño antes de crear trabajos, bloqueo estático y exclusión de referencias en las galerías, comprobando que los archivos bloqueados siguen existiendo.
- Total: **18 pruebas aprobadas**.

### Build, auditoría y humo HTTP

- `cd web && npm run build`: aprobado.
- `cd web && npm audit --audit-level=high`: 0 vulnerabilidades reportadas.
- `bash -n run_ux.sh run_api.sh run_web_dev.sh run.sh setup.sh install.sh`: aprobado.
- `venv/bin/python -m py_compile api/app.py api/uploads.py api/output_files.py api/generation.py api/schemas.py config.py main.py tests/test_ux_stabilization.py ui/interface.py`: aprobado.
- Iniciado `./run_ux.sh`; `curl http://127.0.0.1:5173/api/runtime` confirmó comunicación React/Vite → FastAPI y devolvió los límites configurados.
- `ss -ltn` confirmó listeners en `127.0.0.1:5173` y `127.0.0.1:7861`.
- Una petición HTTP a una foto facial histórica en `/outputs` devolvió **404**; se comprobó que el archivo original permaneció en disco.
- No se iniciaron Gradio ni los motores de generación y no se generaron imágenes de prueba.

### Corrección del informe UX v2

`docs/ITERATION_UX_V2_REPORT.md` ahora indica nueve pruebas frontend aprobadas, corrigiendo la mención anterior a ocho frente a nueve.

## D. Riesgos y pendientes

- El servicio no incorpora autenticación. Loopback evita exponerlo a la red por defecto, pero otros procesos/usuarios locales pueden acceder a la API y a las imágenes generadas.
- Las imágenes PNG generadas continúan accesibles por `/outputs` y pueden contener prompts/metadatos incrustados. Los metadatos y prompts como archivos independientes ya no se sirven desde esa ruta, pero permanecen localmente para compatibilidad.
- No se implementó expiración, borrado automático ni control de acceso por propietario. No se borraron fotos, resultados, prompts o metadatos históricos.
- Los lanzadores admiten configuración de host/puerto mediante variables o argumentos en los servicios individuales; establecer hosts públicos sigue siendo una decisión explícita del operador y no es adecuado en redes no confiables.
- Los lanzadores `.bat` se revisaron, pero no se ejecutaron en Windows durante esta validación en Linux.
- **Pendiente para la siguiente fase:** `core/face_reference_pipeline.py` depende de assets y rutas específicas de una instalación Forge. Esta estabilización no modifica ni intenta resolver esa dependencia.
- También quedan fuera de alcance las APIs de edición, video y selección automática de regiones, así como cualquier refactor de los motores.

## E. Archivos principales

- Arranque actual: `run.sh`, `run.bat`, `run_gradio.sh`, `run_gradio.bat`; lanzadores individuales de desarrollo: `run_api.sh`, `run_api.bat`, `run_web_dev.sh`, `run_web_dev.bat`.
- API y límites: `api/app.py`, `api/uploads.py`, `api/output_files.py`, `api/schemas.py`, `api/generation.py`, `config.py`.
- Interfaz y pruebas: `web/vite.config.js`, `web/src/app/App.jsx`, `web/src/components/AdvancedSettings.jsx`, `web/src/components/MediaUploader.jsx`, `web/src/app/App.test.jsx`, `tests/test_ux_stabilization.py`.
- Documentación: `README.md`, `docs/ARCHITECTURE.md`, `docs/README_SETUP.md`, `docs/EXAMPLES.md`, `docs/QUICKSTART.md` y este informe.
