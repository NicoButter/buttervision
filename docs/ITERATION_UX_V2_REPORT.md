# Informe de implementación — Buttervision UX v2

## A. Resumen ejecutivo

La iteración reemplaza la pantalla técnica única de React por una experiencia en español, responsive y con dos niveles de uso. La pantalla de inicio presenta las cuatro herramientas solicitadas. Crear imagen y Mis creaciones se conectan a las rutas existentes de FastAPI; crear video, editar imagen y editar video presentan su interfaz, pero no simulan procesamiento ni llaman a rutas inexistentes.

El modo sencillo es el predeterminado. El modo experto expone los controles que admite la API actual. Se conservaron los contratos HTTP, los motores de inferencia y la interfaz Gradio. La compilación de producción termina correctamente y las nueve pruebas automatizadas del frontend pasan.

## B. Auditoría inicial

### Problemas identificados

- `web/src/main.jsx` concentraba navegación, llamadas HTTP, formularios, selección de modelo, polling e historial en un único componente.
- Los textos y controles estaban principalmente en inglés y exponían parámetros técnicos desde el inicio.
- La UI solo podía seleccionar text-to-image o referencia facial; no había una pantalla principal para las cuatro herramientas ni una galería navegable.
- La generación aplicaba dimensiones cuadradas y rígidas sin una explicación para formatos alternativos.
- El polling usaba `setInterval`, con posibilidad de solicitudes solapadas y sin tratamiento de error diferenciado.
- El frontend no tenía scripts ni pruebas automatizadas.
- FastAPI dispone de generación txt2img, referencia facial, estado de trabajos, modelos, runtime e historial de imágenes. No existen rutas para generación de video ni para edición de imagen o video.
- `api/generation.py` limita ancho/alto a 512 y el batch efectivo a 1; la referencia facial se fuerza a 512 × 512.

### Componentes y capacidades reutilizados

- Se conservan las rutas `/api/generate/txt2img`, `/api/generate/face-reference`, `/api/jobs/{job_id}`, `/api/history`, `/api/models`, `/api/models/active` y `/api/runtime`.
- Se reutilizan las capacidades actuales de Face Reference y el directorio de resultados expuesto por la API.
- El seguimiento de trabajos presenta estados y progreso devueltos por la API; no calcula porcentajes ficticios.

### Decisiones de arquitectura

- App React compacta con pantallas internas explícitas y componentes reutilizables para carga, parámetros expertos, progreso y comparación antes/después; no se añade un framework de rutas.
- Un servicio centraliza solicitudes HTTP y la construcción de URLs de resultados.
- Los archivos seleccionados y las instrucciones se mantienen en el estado de React, no en almacenamiento persistente del navegador.
- Vite limita su servidor de desarrollo a loopback y reenvía `/api` y `/outputs` a FastAPI local.

## C. Cambios implementados

### Archivos creados

- `web/src/app/App.jsx`: navegación, pantallas y coordinación del estado.
- `web/src/app/App.test.jsx`: pruebas de interacción e integración simulada.
- `web/src/components/AdvancedSettings.jsx`: controles técnicos existentes.
- `web/src/components/BeforeAfterSlider.jsx`: comparación reutilizable, con resultado pendiente claramente indicado.
- `web/src/components/GenerationProgress.jsx`: presentación del estado real de trabajos.
- `web/src/components/MediaUploader.jsx`: carga, vista previa y validación de tipo de archivo.
- `web/src/hooks/usePreferences.js`: preferencia persistente del modo de experiencia.
- `web/src/services/api.js`: cliente HTTP, URL de archivos y mensajes de error.
- `web/vite.config.js`: proxy de API y outputs en desarrollo local.
- `web/vitest.config.js`: configuración del entorno de pruebas.
- `docs/ITERATION_UX_V2_REPORT.md`: este informe.

### Archivos modificados

- `web/src/main.jsx`: punto de entrada React modular.
- `web/src/styles.css`: identidad oscura original, responsive, estados de foco y controles accesibles.
- `web/index.html`: idioma español y metadatos de Buttervision.
- `web/package.json` y `web/package-lock.json`: scripts de test y herramientas de pruebas; se aplicaron actualizaciones compatibles para dejar la auditoría npm sin vulnerabilidades reportadas.
- `README.md`: descripción de React como interfaz principal, instrucciones locales de desarrollo y advertencia de privacidad.
- `docs/ARCHITECTURE.md`: responsabilidades de React, API y Gradio, límites y riesgos actuales.

No se modificaron `api/`, `core/`, `config.py`, los algoritmos de inferencia ni la interfaz Gradio.

### Componentes y pantallas

- Inicio: cuatro tarjetas de acción principales.
- Crear imagen: prompt, orientación, foto de referencia opcional, preview, estados de trabajo y descarga.
- Crear video: prompt, imagen de inicio y referencia facial opcionales, selectores visuales de duración/formato y aviso de integración pendiente.
- Editar imagen: carga de original, preview, instrucción y espacio de comparación antes/después.
- Editar video: carga y preview del original, instrucción y espacio de resultado.
- Mis creaciones: miniaturas de imágenes de la API, apertura, descarga y estado vacío.
- Modo experto: controles expandibles en la pantalla de creación de imagen.

Las pantallas son estados de navegación de React y no agregan rutas URL.

## D. Funcionalidades

| Herramienta | UI implementada | Integración real disponible | Pendiente |
|---|---|---|---|
| Crear imagen | Prompt, selección de orientación, referencia opcional con preview/quitar, resultado, polling, error, guardar y repetir. | Sin foto usa `POST /api/generate/txt2img`; con foto usa `POST /api/generate/face-reference`; consulta `GET /api/jobs/{job_id}` y actualiza `GET /api/history`. | El formato vertical/horizontal solo puede usar dimensiones compatibles de hasta 512; con referencia se envían 512 × 512. La preservación facial exacta no está garantizada. |
| Crear video | Prompt, imagen inicial opcional, foto facial opcional, selectores de duración/formato y espacio de resultado. | Ninguna: no hay ruta de generación de video conectada. | Inferencia e integración de video. Los selectores son solo visuales y no se envían. |
| Editar imagen | Carga y preview del original, instrucciones naturales y espacio de comparación. | Ninguna: no hay endpoint de edición conectado. | API de edición, interpretación de instrucciones y selección automática de zonas. El original no se sobrescribe. |
| Editar video | Carga, reproductor de preview, instrucciones y espacio de resultado. | Ninguna: no hay endpoint de edición conectado. | Procesamiento e integración de edición de video. |

La galería usa exclusivamente las rutas de imágenes entregadas por `GET /api/history`; no crea metadatos, fechas ni contenido de video.

## E. Modo experto

### Parámetros disponibles

- Modelo/checkpoint, mediante `GET /api/models` y `POST /api/models/active`.
- Pasos, CFG, seed, ancho, alto y batch size.
- Prompt negativo.
- Fuerza de identidad y fuerza de estructura al usar referencia facial.

Los máximos y límites reflejan la API existente. El backend actual clampa resolución a 512 × 512 y batch efectivo a 1. No se muestran como funcionales LoRA, sampler, scheduler, denoising, ControlNet ni optimizaciones de VRAM; quedan para una fase futura.

### Persistencia

La preferencia sencilla/experto se guarda como `buttervision-experience-mode` en `localStorage`. Las instrucciones y archivos permanecen en estado de la sesión React y no se guardan en localStorage. Cambiar de modo no desmonta ni reinicia los formularios.

## F. Validación

### Comandos

- `cd web && npm install`: completado.
- `cd web && npm test`: **9 pruebas aprobadas**.
- `cd web && npm run build`: **compilación aprobada** con Vite 8.3.5.
- `cd web && npm audit --audit-level=high`: **0 vulnerabilidades reportadas**.
- `git diff --check`: sin errores de espacios en blanco.

### Cobertura de pruebas

Las pruebas validan las cuatro herramientas y navegación, modo sencillo por defecto y persistencia de preferencia, conservación del prompt al alternar modos, payloads y endpoints correctos para txt2img y Face Reference, preview facial, comparación antes/después, rechazo de archivo inválido, estado vacío y contenido de galería, falta de solicitudes de video y presentación de errores HTTP.

No se descargaron modelos ni se realizó inferencia. Las llamadas API se probaron con mocks.

### Smoke test visual y limitaciones

La página de desarrollo se abrió en navegador y se comprobó el layout responsive a 535 px; las tarjetas cambian a una sola columna. Como FastAPI no estaba iniciado durante esa prueba visual, `/api/runtime`, `/api/models` y `/api/history` respondieron con 502 desde el proxy. Esto no afecta los tests con mocks, pero deja sin validar una ejecución real contra un proceso FastAPI en esta sesión.

El endpoint de archivos `/outputs` no tiene autenticación. Además, el backend actual guarda `reference_face.png` junto a los resultados y expone el directorio mediante esa ruta estática. Se documentó el riesgo; el endurecimiento del almacenamiento/servidor queda fuera de esta iteración.

## G. Pendientes para la siguiente iteración

1. Definir e implementar API para edición de imágenes con instrucciones naturales y selección automática de zonas.
2. Conectar generación de video y establecer capacidades reales de duración, formato e imagen inicial.
3. Definir e implementar edición de video.
4. Evaluar y documentar los límites de preservación de identidad facial sin prometer exactitud.
5. Mejorar gestión de VRAM y controles técnicos solo cuando exista soporte real en la API.
6. Proteger `/outputs`, evitar exposición de fotos de referencia y definir retención mínima de archivos privados.
7. Validar en una instalación local completa con FastAPI iniciado y hardware/modelos disponibles, sin ejecutar inferencias durante tests de frontend.
