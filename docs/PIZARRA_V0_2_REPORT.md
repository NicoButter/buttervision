# Buttervision — PIZARRA Engine v0.2

Fecha: 31 de octubre de 2026. Alcance: prototipo CPU de almacenamiento y operaciones tensoriales por bloques.

## Resultado

Se añadió Tensor Store v1 para `float32` y `float16`, con forma/versionado/orden C en manifiesto persistente y registros inmutables por fragmento. Cada bloque usa checksum y publicación atómica de `PizarraStorage`. Una referencia puede reabrirse por `job_id` + `tensor:<id>` tras recrear `TensorStore`; `missing_blocks` detecta huecos y `write_block` permite completarlos después de reiniciar. El bloque final incompleto no se publica.

`tensor_ops` implementa escala por fragmentos, reducción acumulada en float64 y matmul 2D por regiones, con resultado float32. Las pruebas usan arreglos pequeños NumPy como referencia e incluyen dimensiones irregulares y lectura 2D que cruza límites de fragmentos. El ejecutor es CPU-only.

El presupuesto de buffers limita tamaño de bloque, buffers de operación y resultado parcial, y contabiliza reservas actuales/pico. El benchmark muestrea RSS y RAM disponible. Estos valores no imponen un límite duro al RSS: el intérprete, NumPy, almacenamiento, el caché del filesystem y otras bibliotecas pueden usar más. CUDA no se inicializa.

## Compatibilidad y seguridad

No se tocaron `api/generation.py`, los endpoints, el scheduler administrativo v0.1 ni los pipelines de imagen. Las pruebas físicas usan `PizarraStorage` con montaje simulado en directorio temporal. No se leyó ni escribió `/mnt/pizarra`, no se borraron datos, no se modificaron archivos safetensors y no se ejecutaron inferencias. El Tensor Store está separado del flujo de difusión; no declara compatibilidad out-of-core de SDXL.

La cuota todavía inspecciona los árboles PIZARRA para cada comprobación/publicación. Esto mantiene reconciliación exacta pero puede encarecer volúmenes con muchos bloques; falta medir sobre HDD real antes de implementar un ledger incremental.

## Pruebas y build

- Nuevo `test_pizarra_tensor_store.py`: **9 aprobadas**.
- PIZARRA v0.1 `test_pizarra_storage.py`: **17 aprobadas**.
- PIZARRA v0.1 `test_pizarra_scheduler.py`: **11 aprobadas**.
- PIZARRA v0.1 jobs, métricas y benchmark: **21 aprobadas** (13 + 4 + 4).
- Frontend React: **24 aprobadas**; build Vite: **aprobado**.
- `compileall -q api core tests benchmarks` y `git diff --check`: aprobados.
- `test_pizarra_api.py`: no obtuvo resultado dentro de 15 segundos en este sandbox y se detuvo por timeout; no se declara aprobado. No cambiaron endpoint ni contrato API.

No se ejecutó el benchmark físico. Por eso no hay medición observada de RSS con un tensor de 1 GiB, rendimiento del HDD ni crecimiento de cuota por muchos bloques. Una prueba automatizada sí procesa por almacenamiento temporal un tensor de 16 KiB con presupuesto de buffers de 1 KiB (16 veces menor) en fragmentos de 128 bytes; comprueba suma/escala exactas y los topes contables del ejecutor. Esta prueba de fragmentos no sustituye la medición física de 1 GiB ni promete un límite duro al RSS total.

## Archivos

- `core/pizarra/tensor_store.py`: formato, escritura/lectura incremental, referencias, detección y recuperación de huecos, política y limitador de buffers.
- `core/pizarra/tensor_ops.py`: operaciones CPU, acumulación y muestreo de recursos.
- `benchmarks/pizarra_tensor_benchmark.py`: escritura explícita, lectura, escala, reducción, comprobación e informe JSON.
- `tests/test_pizarra_tensor_store.py`: pruebas del formato y exactitud.
- `docs/PIZARRA_V0_2_ARCHITECTURE.md`, `docs/PIZARRA_V0_2_BENCHMARK.md` y este informe.

## Prueba física pendiente

En Fedora, con el servicio PIZARRA configurado y disponible, primero se recomienda una prueba de 0.01 GiB. Cuando el operador confirme espacio suficiente y acepte conservar fuente y resultado, repetir con el caso propuesto de 1 GiB y presupuesto de buffers de 128 MiB. El benchmark requiere `--execute` y guarda ambos tensores en el volumen; el caso grande consume más de 2 GiB de cuota/volumen contando entrada, salida y metadatos. Consultar el procedimiento detallado en `PIZARRA_V0_2_BENCHMARK.md`.

## Recomendación técnica para v0.3

1. Ejecutar el benchmark físico progresivamente y conservar JSON, número de bloques, RSS pico, espacio consumido, latencia y throughput.
2. Medir el costo de cuota al aumentar los bloques antes de diseñar un ledger transaccional incremental; reconciliar siempre contra los archivos reales.
3. Ampliar el scheduler administrativo para transportar referencias tensoriales sin convertirlas a bytes completos, con estados explícitos de bloque incompleto y recuperación.
4. Agregar ejecución CUDA aislada solo después de fijar presupuesto de transferencia y comprobar exclusión de inferencia. No conectar aún al pipeline Diffusers.
5. Evaluar por separado la lectura de formatos de pesos y el streaming de capas antes de proponer una integración SDXL.
