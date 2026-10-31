# Buttervision — UX v2.4: Centro de modelos y estado PIZARRA

Fecha: 31 de octubre de 2026. Rama: `main`.

## Resultado

Crear imagen muestra el checkpoint activo y ofrece el catálogo local registrado también en modo simple. Los checkpoints desconocidos permanecen visibles, pero deshabilitados. Cambiar de checkpoint espera la respuesta del backend y vuelve a consultar modelos, motor y LoRA; HTTP 409 explica que el motor está ocupado y HTTP 400 conserva la selección anterior y muestra el motivo. Las dimensiones se ajustan al confirmar un cambio de familia y las referencias cargadas no se eliminan. Se informa que selección no significa pesos residentes en VRAM y se mantiene la advertencia de memoria SDXL en la GTX 1650 de 4 GB.

PIZARRA queda visible en la barra superior en ambos modos. Disponible, deshabilitado, volumen no disponible y error de consulta se representan de forma distinta. El panel plegable muestra únicamente cifras devueltas por la API: capacidad informada, cuota, uso, restante de cuota, espacio libre físico, espacio utilizable, reserva mínima y estado de métricas. La barra representa uso/capacidad de cuota, sin sustituir la lectura de espacio físico. Las cantidades usan GB decimales. El polling es de 30 segundos y no solapa solicitudes; se detiene al desmontar.

## Archivos

- `web/src/components/ModelSelector.jsx`: selector compartido y avisos del catálogo/memoria.
- `web/src/components/PizarraStatus.jsx`: estado global y panel de almacenamiento.
- `web/src/app/App.jsx`: integración, confirmación de cambio y refresco posterior.
- `web/src/components/AdvancedSettings.jsx`: conserva controles expertos, elimina el segundo selector.
- `web/src/services/api.js`: adjunta el estado HTTP a los errores.
- `web/src/styles.css`: estilos oscuros responsive para selector, indicador y panel.
- `web/src/app/App.test.jsx`: cobertura para catálogo, cambio, 409 y estados PIZARRA.

No se modificaron endpoints, gestión de modelos, presupuesto SDXL ni código de inferencia. No se ejecutaron generaciones ni se accedió al HDD PIZARRA.

## Verificación

- `cd web && npm test -- --reporter=dot`: **21 pruebas aprobadas**.
- `cd web && npm run build`: aprobado (24 módulos transformados).
- `git diff --check`: aprobado.
- `venv/bin/python -m unittest discover -s tests -q` y los módulos Python relevantes: intentados, pero las ejecuciones con pruebas API/concurrencia no produjeron resumen antes de que terminara el proceso en este sandbox. No se considera resultado aprobado.
- `venv/bin/python -m compileall -q api core tests`: comprobación sintáctica ejecutada por separado.

## Pasos operativos

Para añadir modelos, el operador debe registrar directorios locales mediante `BUTTERVISION_CHECKPOINT_DIRS` y reiniciar Buttervision; la UI no admite rutas arbitrarias ni descarga pesos. Si Juggernaut XL v9 está presente en el directorio registrado de Forge, el refresco del catálogo lo mostrará como SDXL. Seleccionarlo no omite `check_budget`, no cambia precisión/offload y no demuestra que la carga o una generación puedan completarse. PIZARRA sigue siendo almacenamiento y telemetría; no implementa inferencia out-of-core.
