# Informe de iteración v2.5

## Diagnóstico inicial

El catálogo detectaba Buttervision y las carpetas explícitas en variables de entorno, pero no recorría instalaciones conocidas de Forge. El selector de checkpoints estaba dentro de Crear imagen. PIZARRA requería activación manual y el panel mostraba un único estado, sin separar actividad medida.

## Cambios realizados

- `core/model_catalog.py`, `core/model_manager.py` y `api/generation.py`: descubrimiento local Forge, combinación con Buttervision y carpetas declaradas, descarte de archivos vacíos, resolución de rutas reales, clasificación de origen y etiquetas diferenciadas para nombres repetidos. El catálogo se consulta al iniciar la interfaz/backend y al actualizar; no lee tensores, descarga, copia ni crea enlaces.
- `run.sh`, `.env.example` y `core/pizarra/config.py`: `./run.sh` lee opcionalmente `.env.local` como pares literales `BUTTERVISION_*=valor` (no lo ejecuta). El entorno del proceso tiene precedencia. `.env.local` es la fuente persistente por instalación; `.env.example` es referencia para crearla.
- Configuración PIZARRA en modo `AUTO`: se reconoce un punto de montaje dedicado cuando coincide el UUID configurado o contiene el marcador de bloqueo PIZARRA válido ya existente. El modo no crea marcadores. Un montaje sin autorización se muestra pendiente y no se escribe. `BUTTERVISION_PIZARRA_ENABLED=false` o `BUTTERVISION_PIZARRA_MODE=OFF` deshabilitan explícitamente.
- `core/pizarra/storage.py` y `/api/pizarra/status`: actividad reciente basada en incrementos de bytes medidos por operaciones PizarraStorage. El sondeo del navbar usa `light=true` y no recorre árboles ni consulta ocupación; el panel completo mide ocupación al abrirse o al actualizar. El estado de disponibilidad y las métricas no producen actividad contada. La medida no cubre operaciones externas o accesos directos al HDD.
- `web/src/components/ModelSelector.jsx`, `PizarraStatus.jsx`, `web/src/app/App.jsx` y `web/src/styles.css`: selector global en navbar, detalles avanzados en modo experto y formulario de creación simplificado. Los cambios conservan los estados React y archivos de entrada. Indicadores distintos de disponibilidad y actividad, con respeto a movimiento reducido.

## Procedimiento de descubrimiento Forge

Se exploran `$HOME/stable-diffusion-webui-forge/models/Stable-diffusion` y `$HOME/proyectos_2024/stable-diffusion-webui-forge/models/Stable-diffusion`, además de `BUTTERVISION_FORGE_HOME/models/Stable-diffusion`. Los LoRA se buscan en `models/Lora`. `BUTTERVISION_CHECKPOINT_DIRS` y `BUTTERVISION_LORA_DIRS` agregan otras raíces. Las inexistentes o inaccesibles se omiten sin impedir el arranque. Los checkpoints y LoRA se autorizan contra el catálogo actualizado antes de ser usados.

## Procedimiento PIZARRA

AUTO comprueba el punto de montaje real y el dispositivo. Con UUID configurado compara el dispositivo de bloque. Sin UUID exige el marcador `.pizarra.lock` preexistente con la firma de Buttervision. No hay inicialización automática. La primera autorización se realiza configurando el UUID del volumen en `.env.local` y reiniciando; el sondeo pendiente no escribe en ese volumen. Storage mantiene sus comprobaciones de montaje, identidad, permisos, cuota y estructura.

## Indicadores

El punto de disponibilidad viene de `/api/pizarra/status`: verde solo si `enabled=true` y `available=true`; gris para deshabilitado; amarillo para montaje detectado pendiente de autorización; rojo si la función habilitada no puede validar el volumen. El punto de actividad se alimenta de `activity.confirmed`, que requiere E/S reciente contada por PizarraStorage; no representa el HDD en general. Métricas separan disponibilidad, bytes gestionados, metadatos y transferencias de registros tensoriales por la API de almacenamiento. No se afirma offload de inferencia.

## Estados esperados de API (ejemplos)

```json
{"enabled":true,"available":true,"metrics_enabled":true,"activity":{"confirmed":false,"read_bytes":128,"written_bytes":96},"storage":{"managed_bytes":96,"quota_bytes":400000000000}}
```

```json
{"enabled":false,"available":false,"pending_authorization":true,"metrics_enabled":false,"storage":null}
```

```json
{"enabled":true,"available":false,"metrics_enabled":true,"storage":null,"message":"Volumen PIZARRA no disponible o no autorizado."}
```

## Pruebas ejecutadas

`source venv/bin/activate && python -m unittest discover -s tests -q`: 154 pruebas correctas. `npm test`: 26 pruebas correctas. `npm run build`: compilación correcta. `bash -n run.sh`: sintaxis correcta. También se arrancó `./run.sh`: Vite y Uvicorn quedaron listos; `/api/models` devolvió el checkpoint Buttervision SD 1.5 y dos Forge (SD 1.5 y SDXL), y `/api/pizarra/status?light=true` devolvió `enabled=true`, `available=true`, `activity.confirmed=false`. No se cargaron checkpoints reales ni se generaron imágenes.

## Limitaciones pendientes

- La actividad incluye lecturas/escrituras de metadatos y transferencias por la API PizarraStorage, pero no detecta E/S externa no instrumentada ni atribuye actividad física general al volumen.
- El campo de UUID debe ser configurado por la persona que conoce el dispositivo autorizado. AUTO no elige ni confía en discos por sí mismo.
- DISK_OFFLOAD y Tensor Store experimental conservan su alcance actual.
