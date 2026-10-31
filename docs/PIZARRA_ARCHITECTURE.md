# PIZARRA Engine v0.1 — arquitectura

PIZARRA prepara infraestructura local para experimentar con almacenamiento persistente, CPU, RAM y GPU. En esta versión el scheduler de demostración **no participa en la inferencia Diffusers**. No hay carga parcial de pesos, ejecución out-of-core, almacenamiento de latentes de difusión ni reanudación matemática. El tamaño del disco no aumenta por sí solo el tamaño de una operación que cabe en VRAM.

## Componentes

- `core/pizarra/config.py`: configuración optativa por entorno; deshabilitado por defecto y sin crear rutas al arrancar.
- `storage.py`: validación del volumen, espacios por UUID, cuota, reserva de espacio libre, registros inmutables con checksum, lectura verificada, detección y limpieza explícita de temporales, y recuperación de manifiestos.
- `job_store.py`: SQLite local en `data/pizarra/jobs.sqlite3`, separado del HDD. Guarda únicamente el estado administrativo, referencias públicas de resultados y métricas, nunca el payload. Un `application_id` identifica la base propia; archivos inseguros o de otra aplicación se rechazan.
- `metrics.py`: muestreo de recursos del proceso/dispositivo, duraciones de fases observables y contadores de almacenamiento. `core/engine_memory.py` vincula el colector al hilo del trabajo sin sincronizar CUDA en cada operación.
- `memory_budget.py`: calcula márgenes y presupuestos informativos a partir de mediciones; no cambia precisión, admisión ni resolución.
- `scheduler.py`: DAG local de operaciones CPU pequeñas, con estado SQLite y resultados confirmados en el banco. Es un demostrador aislado del bucle de difusión.
- `benchmarks/pizarra_benchmark.py`: mediciones explícitas contra servidores locales y comparación posterior.

## Ciclo de trabajo

Deshabilitado, no se crea la base ni se consulta el banco; la generación conserva el camino anterior y `worker_lock` mantiene el rechazo HTTP 409. Habilitado, el servicio registra el trabajo localmente, valida el montaje y prepara su espacio antes de iniciar el hilo. El worker registra `running`, mide, usa el pipeline existente, valida el banco al finalizar y realiza la limpieza. La transición final y las métricas se confirman juntas en SQLite antes de publicar `completed` en memoria. Se elimina el payload de memoria al terminar.

Los estados administrativos son `queued`, `running`, `completed`, `failed` e `interrupted`. Al abrir la base, los trabajos sin dueño vivo pasan a `interrupted`: se compara PID y hora de inicio del proceso, y la falta de permiso para observarlo no se interpreta como muerte. Un trabajo completado conserva sus referencias de salida; no se vuelve a ejecutar nada automáticamente. Un SIGKILL impide cualquier actualización inmediata: la detección se hace al abrir nuevamente la base. El registro no contiene los estados internos del scheduler de difusión y no permite reanudarla.

Si falla el banco antes de aceptar el trabajo se devuelve 503 sin iniciar inferencia. Si se desconecta durante una llamada Diffusers, v0.1 lo detecta en el siguiente control de almacenamiento, al terminar esa llamada; no interrumpe de forma forzada una operación GPU en curso. El estado se guarda como fallido en SQLite si es posible; una imagen ya generada puede permanecer en los outputs existentes. Si también falla SQLite, el estado en memoria indica el fallo y se intenta persistirlo, sin prometer que el proceso pueda registrar un error en una base inaccesible.

## Banco y seguridad

La raíz configurada debe ser **el punto de montaje exacto** de un volumen distinto del disco raíz. Se comprueban `/proc/self/mountinfo`, el dispositivo y su identidad durante las operaciones; se rechazan symlinks en la ruta, escapes, submontajes y archivos especiales. Para vincular el dispositivo a una identidad entre reinicios se recomienda configurar `BUTTERVISION_PIZARRA_VOLUME_UUID`: se compara el dispositivo del montaje con `/dev/disk/by-uuid/<UUID>`. Sin esta variable se verifica el montaje y su identidad durante la ejecución, pero no se distingue otro disco montado en la misma ruta después de reiniciar.

```text
<volumen>/
  .pizarra.lock
  jobs/<uuid>/manifest.json
  jobs/<uuid>/checkpoints/
  jobs/<uuid>/tensors/
  jobs/<uuid>/temporary/
  cache/                 # reservado
  benchmarks/            # reservado
```

El manifiesto versionado incluye identificador y checksum. Los resultados se escriben como registros autocontenidos con versión, longitud y SHA-256, primero en un temporal identificado; se sincronizan archivo y directorio y se publican con `link` atómico que falla si el destino existe. La lectura verifica tamaño y checksum. Un temporal truncado no aparece como resultado confirmado. La política de limpieza exige antigüedad explícita y sólo toca pares temporales propios identificados por su manifiesto; no borra resultados ni archivos ajenos. `recover_manifests()` devuelve únicamente manifiestos válidos; no implica que haya checkpoints de inferencia recuperables.

El conteo de cuota comprende los archivos bajo las áreas administradas y el bloqueo, incluso temporales; las operaciones de escritura cooperantes se serializan con `flock`. También se comprueba el espacio físico disponible antes y después de escribir. El espacio usado por otros programas y la asignación de bloques del filesystem pueden cambiar entre comprobaciones: ni la cuota lógica ni el margen pueden reservar espacio físico frente a procesos externos no cooperantes. Este formato actual lee/escribe cada registro completo en RAM; todavía no es un tensor store para archivos enormes. La implementación es Linux y requiere un filesystem local que soporte hardlinks, `fsync` y `flock`; se recomienda ext4.

## Métricas y presupuestos

Las métricas se guardan como JSON en la base local y están disponibles aunque el HDD deje de responder. La RAM disponible y el porcentaje CPU son del sistema; RSS y tiempo CPU corresponden al proceso; VRAM libre/total es del dispositivo y asignada/reservada es del proceso PyTorch. El pico RSS y el pico de VRAM del trabajo son **máximos observados por muestreo**, pueden perder picos breves. También se expone el máximo de asignación CUDA desde el inicio del proceso, claramente diferenciado; no se reinician contadores globales CUDA. El tiempo GPU exclusivo no puede medirse correctamente sin sincronización y se informa como `null`.

Las fases disponibles incluyen preparación, carga de modelo cuando ocurre, `inference_and_decoding` para la llamada completa al pipeline, guardado, limpieza y `pipeline_execution` como intervalo general. Son tiempos de pared y pueden solaparse; no deben sumarse como si fueran exclusivos. No se inventa una duración de decoding separada. Los bytes/tiempos PIZARRA son de operaciones explícitas de esa instancia durante el intervalo del trabajo, no E/S física de todo el dispositivo; pueden incluir otra operación que comparta la instancia. Ausencia de medición se expresa con `null`, no con cero ficticio.

`inspect_budget()` calcula márgenes configurables sobre RAM/VRAM disponibles. La suma de RSS de otros procesos visibles puede contar páginas compartidas más de una vez y ser incompleta; memoria de GPU fuera de la reserva PyTorch no es exclusivamente de otros procesos. El plan es sólo observación y no sustituye las comprobaciones existentes de SDXL ni garantiza evitar OOM.

## Scheduler experimental

Las tareas declaran ID, operación, dependencias, referencias de entrada/salida, presupuesto y estado. El registro valida operaciones, entradas, dependencias y ciclos antes de insertar el DAG. SQLite reclama una tarea mediante transacción; sólo se ejecuta si sus dependencias terminaron. Una salida se confirma en storage antes de marcar la tarea completa. Un dueño muerto deja tareas `interrupted`, sin reejecución automática. Si el proceso cae entre confirmar la salida y actualizar SQLite puede quedar un resultado huérfano; no se declara éxito ni se repite la tarea. El demostrador multiplica matrices CPU pequeñas por bloques, confirma parciales y reconstruye el resultado; no crea workers CUDA.

## Observabilidad y privacidad

- `GET /api/pizarra/status`: habilitación, disponibilidad y cuotas, sin rutas físicas. `?include_budget=true` agrega el presupuesto observacional.
- `GET /api/pizarra/jobs/{job_id}/metrics`: métricas locales; 404 si no existen y 503 si no se puede leer SQLite.

El banco no se publica como estático. Las métricas y manifiestos no incluyen prompts, fotos ni credenciales. El historial previo de imágenes y metadata de outputs conserva su comportamiento existente; PIZARRA no elimina esos resultados ni cambia la política previa de archivos de salida.
