# PIZARRA Engine v0.1 — informe de implementación

Fecha: 31 de octubre de 2026. Estado de trabajo: rama `main`.

## Resultado

PIZARRA incorpora infraestructura opcional para almacenamiento local, registros persistentes, observabilidad, presupuestos informativos y un scheduler experimental CPU. Deshabilitado por defecto, mantiene el servicio de generación existente. Habilitado, registra trabajos en SQLite local, valida el volumen antes de aceptarlos, crea un espacio aislado, toma métricas y confirma estado final y métricas en una transacción local.

**No se implementó inferencia out-of-core ni reanudación de difusión.** Tampoco se ejecutó generación real en la GTX 1650, ni se midió rendimiento en un HDD físico. La compatibilidad real con ese hardware y un montaje Fedora debe validarse manualmente. El tamaño del disco no cambia las limitaciones matemáticas de las operaciones ni garantiza evitar OOM.

## Verificación de la arquitectura anterior

Se verificó que los trabajos se conservaban en `self.jobs`, que `worker_lock` protege generación, cambios de checkpoint y LoRA, y que las solicitudes ocupadas devuelven 409. Siguen activos CPU offload, attention slicing, VAE slicing y los límites y comprobaciones de memoria de SDXL. Los resultados existentes siguen guardándose en `outputs`; la API no incorporó generación funcional de video. No se modificaron algoritmos de inferencia, precisión global, xFormers ni las políticas de admisión SDXL.

## Cambios y archivos

- `core/pizarra/config.py` e `__init__.py`: configuración optativa, apagada por defecto, incluida la verificación opcional de UUID autorizado.
- `core/pizarra/storage.py`: comprobación de montaje, identidad del volumen, cuota y espacio libre; espacios de trabajo aislados; escritura temporal, publicación inmutable atómica, checksum, recuperación de manifiestos y limpieza explícita de temporales.
- `core/pizarra/job_store.py`: registro SQLite local privado, propietario PID/hora de inicio, recuperación de estados y transacción conjunta de estado y métricas. Rechaza symlinks y bases ajenas en su archivo administrado.
- `core/pizarra/metrics.py` y `core/engine_memory.py`: métricas de recursos y fases observables, con datos desconocidos como `null` y sin sincronización CUDA por operación.
- `core/pizarra/memory_budget.py`: plan observacional con márgenes configurables y mediciones disponibles; no interviene en decisiones del motor.
- `core/pizarra/scheduler.py`: DAG CPU serial con dependencias y estado persistente, más demostración verificable de multiplicación por bloques. No está conectado a Diffusers.
- `api/generation.py`, `api/app.py`: integración condicional, preservación de contratos, rechazo controlado si falta el banco y endpoints de observabilidad. Las llamadas de los pipelines SD 1.5, facial y SDXL se instrumentan sin alterar sus argumentos o cálculos.
- `benchmarks/pizarra_benchmark.py`: ejecución explícita y comparación de dos mediciones, sin ejecutarse automáticamente. `.env.example` documenta las variables y `.gitignore` excluye la base local, resultados de benchmark y archivos de entorno locales.
- `tests/test_pizarra_*.py`: pruebas con mocks, directorios temporales, montajes simulados y datos CPU pequeños. Se conservan las pruebas anteriores.
- Documentación: `docs/PIZARRA_ARCHITECTURE.md`, `docs/PIZARRA_FEDORA_SETUP.md` y este informe.

## Comportamiento y recuperación

El banco exige un punto de montaje exacto, distinto del disco raíz, y rechaza symlinks y submontajes no autorizados. Se recomienda especificar `BUTTERVISION_PIZARRA_VOLUME_UUID`, además de montar mediante UUID en Fedora: así se rechaza otro dispositivo montado en esa ruta. No se crea automáticamente `/mnt/pizarra` ni se manipulan discos físicos. Las escrituras se restringen a los espacios propios, se comprueban cuota y reserva física y no se reemplazan archivos existentes. La reserva y la cuota no son una reserva física exclusiva frente a otros procesos que escriben fuera del banco.

El archivo temporal se sincroniza y se confirma mediante hardlink atómico que falla ante un destino existente; cada registro contiene longitud y SHA-256. Las lecturas verifican integridad; los temporales incompletos no se presentan como resultados. La limpieza requiere una antigüedad explícita y sólo elimina temporales propios. Los manifiestos recuperados identifican espacios válidos, no pasos reanudables de difusión.

Los estados `queued` y `running` de un dueño que ya no está vivo pasan a `interrupted` al reabrir la base; no se reanudan automáticamente. La identidad del proceso combina PID y hora de inicio para reducir errores por reutilización de PID. Los resultados válidos existentes se conservan. Las imágenes faciales y prompts no se guardan en el registro administrativo ni en métricas; las salidas y su metadata previa conservan su política existente.

Si el disco no está disponible, la API rechaza el nuevo trabajo con 503 y la base local sigue disponible para diagnóstico. Si se pierde durante la inferencia, el motor detecta el fallo al volver a comprobar el banco, marca el trabajo como fallido cuando es posible y no afirma que se pueda reanudar. La llamada de difusión en curso no puede cancelarse de manera segura desde esta infraestructura.

## Configuración y API

Las variables principales son `BUTTERVISION_PIZARRA_ENABLED`, `BUTTERVISION_PIZARRA_ROOT`, `BUTTERVISION_PIZARRA_VOLUME_UUID`, `BUTTERVISION_PIZARRA_QUOTA_GB`, `BUTTERVISION_PIZARRA_MIN_FREE_GB` y `BUTTERVISION_PIZARRA_METRICS_ENABLED`. La cuota y el margen usan GB decimales. Los márgenes de presupuesto RAM/VRAM son fracciones opcionales y sólo informativas. Las instrucciones concretas para identificar y montar un volumen existente, configurar UUID, permisos y verificar en Fedora están en `PIZARRA_FEDORA_SETUP.md`.

Se añadieron `GET /api/pizarra/status`, con presupuesto opcional mediante `?include_budget=true`, y `GET /api/pizarra/jobs/{job_id}/metrics`. No exponen el path del banco ni publican sus archivos. Se mantienen los endpoints existentes de generación, trabajos e historial. Sin PIZARRA no se crea la base, no se valida el montaje y no se inicia el colector.

## Pruebas y benchmark

- `venv/bin/python -m unittest discover -s tests -v`: **114 pruebas aprobadas**, incluidas las existentes y las nuevas de persistencia, privacidad, concurrencia, cuotas, fallos, symlinks, checksum, montajes simulados, DAG, matrices por bloques, métricas, presupuesto, API y benchmark simulado.
- `venv/bin/python -m compileall -q api core tests benchmarks`, `bash -n run.sh` y `git diff --check`: aprobados.
- `cd web && npm test -- --reporter=dot`: **16 pruebas aprobadas**. `npm run build`: aprobado.
- No se ejecutaron benchmarks ni inferencias reales; no hay resultados de rendimiento para comparar. El benchmark requiere `--execute`, dos sesiones controladas y el PID del servidor; rechaza comparaciones con parámetros, scheduler, modelo o LoRA distintos. Las pruebas del benchmark usan respuestas simuladas.

La suite TestClient se ejecutó fuera de la restricción del sandbox que bloquea su comunicación entre hilos. Los tests no descargan modelos y no necesitan una GPU ni un HDD real. La validación de montaje y pérdida de acceso usa mocks; no sustituye pruebas con el filesystem y el dispositivo de destino.

## Limitaciones y riesgos pendientes

- El bloqueo de inferencia existente es por proceso. Los lanzadores usan un worker; no ejecutar servidores independientes ni Forge/Gradio generando a la vez sobre la GPU. El presupuesto observacional no fuerza límites ni evita por sí mismo un OOM.
- Los picos de RSS y VRAM por trabajo son muestreados; la asignación CUDA máxima de proceso tiene alcance de toda su vida. El tiempo GPU exclusivo no se calcula y figura como `null`. Las fases pueden solaparse y la llamada Diffusers agrupa inferencia y decoding. Los contadores del banco son de llamadas de la instancia, no de E/S física total.
- La cuota es lógica y se coordina entre escritores PIZARRA cooperantes. Otro proceso puede consumir espacio entre controles; fallos de energía, desmontajes y semántica real del filesystem requieren pruebas en Fedora/ext4. No hay garantías de transacción distribuida entre la base local, los outputs existentes y el HDD.
- El almacenamiento actual materializa registros completos en RAM; no sirve todavía para tensores mayores que la RAM. `cache/` y `benchmarks/` son áreas reservadas, no un caché de modelos ni un almacén de benchmarks integrado. Un fallo entre confirmar un resultado del scheduler y actualizar SQLite puede dejarlo huérfano; el scheduler no lo vuelve a ejecutar automáticamente.
- Sin `BUTTERVISION_PIZARRA_VOLUME_UUID`, una unidad distinta montada en la misma ruta antes de iniciar se considera el volumen configurado. El uso del UUID y la inspección manual del montaje reducen este riesgo. Los identificadores y métricas no constituyen checkpoints de latentes o estado del scheduler de difusión.

## Próxima fase y validación Fedora

Seguir `PIZARRA_FEDORA_SETUP.md`: verificar UUID y montaje existente, arrancar con PIZARRA deshabilitado y habilitado, consultar estado y cuota, y sólo con autorización ejecutar la comparación controlada. En una prueba de interrupción posterior, verificar `interrupted`, la integridad del banco y la ausencia de reejecución; no desmontar físicamente el disco mientras se escribe.

Para v0.2 quedan como investigación: tensor store con lectura/escritura parcial y límites verificables, carga fragmentada de pesos, administración correcta de activaciones, prefetch HDD → RAM → GPU, recomputación frente a lectura y persistencia matemáticamente suficiente de latentes y estado del scheduler. Antes de conectarlo a difusión se necesitarán pruebas de equivalencia numérica, fallos e interrupción y mediciones reales de memoria y rendimiento.
