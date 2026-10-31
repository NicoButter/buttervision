# PIZARRA v0.2 — Tensor Store y ejecutor por bloques

## Alcance

v0.2 incorpora persistencia y cálculo experimental de tensores fuera de memoria, solo CPU. No se conecta a FastAPI, al scheduler administrativo, a `GenerationService`, Diffusers ni a la generación de imágenes. El scheduler y el formato JSON de v0.1 siguen disponibles con sus límites anteriores. No implica compatibilidad out-of-core de SDXL.

## Formato de Tensor Store v1

Cada tensor pertenece a un `job_id` ya creado por `PizarraStorage.create_workspace`. Se registra en `jobs/<job_id>/tensors/` mediante los registros inmutables existentes:

- `tensor-<id>-manifest.json`: versión, ID, forma, dtype, orden C, tamaño en elementos por bloque, número de bloques y número total de elementos.
- `tensor-<id>-block-<n>.bin`: bytes IEEE little-endian de un fragmento lineal; la última pieza puede ser menor.

Formatos admitidos: `float32` y `float16`. El manifiesto persiste dentro del sobre de checksum PIZARRA y cada fragmento tiene su propio checksum SHA-256. `PizarraStorage.commit` escribe el registro temporal, sincroniza y publica con enlace atómico sin reemplazar un bloque existente. La validación de montaje/UUID, rutas sin symlinks, cuota lógica y reserva física se mantienen en Storage v0.1.

La referencia que entrega la API Python tiene la forma `TensorRef(job_id, tensor_id, ...)`; su forma serializable breve es `tensor:<id>` junto al `job_id`. Un ID suministrado al crear el tensor permite reabrirlo aunque el proceso se interrumpa antes de devolver el resultado. `open_tensor`, `missing_blocks` y `write_block` permiten inspeccionar y completar explícitamente bloques faltantes. Un bloque corrupto provoca error, nunca se toma como confirmado. Los bloques existentes son inmutables.

La escritura acepta un iterable de arrays/bytes consecutivos; cada elemento entregado debe caber en `max_fragment_bytes`. El escritor une fragmentos pequeños hasta el tamaño configurado, y publica uno por uno. La lectura devuelve un array por fragmento. `read_block` lee una región 2D recorriendo los fragmentos lineales que la cruzan; el tamaño del resultado está limitado por la política.

## Presupuesto y medición

`BufferPolicy` define tamaño máximo de fragmento, buffers simultáneos, resultado parcial y reserva de seguridad. Su validador impide que `max_fragment_bytes × max_simultaneous_buffers + result_partial_bytes + safety_margin_bytes` exceda el presupuesto configurado. `BufferLimiter` contabiliza reservas regulares y parciales, bloquea solicitudes concurrentes cuando alcanzan el límite y libera en `finally`.

El ejecutor limita inputs, resultados y conversiones de matmul. Estos límites gobiernan los fragmentos administrados por el ejecutor; no son un límite duro del proceso. Python, NumPy, almacenamiento, allocator y otras bibliotecas pueden reservar memoria adicional. Se muestrea RSS del proceso y RAM disponible del sistema; CUDA solo se observa si PyTorch ya estaba importado, CUDA está disponible y su contexto ya está inicializado. El ejecutor v0.2 no inicializa ni utiliza CUDA.

## Operaciones disponibles

- `tensor_ops.transform` / `scale`: operación elemento a elemento que conserva forma y dtype.
- `tensor_ops.reduce_sum`: suma incremental con acumulador float64.
- `tensor_ops.matmul`: multiplicación 2D por filas, columnas y K en bloques; float16 acumula y produce float32.

Las matrices de referencia en pruebas caben en RAM y se comparan con NumPy. Matmul tiene una implementación CPU deliberadamente serial para mantener memoria previsible. La eficiencia puede ser baja para matrices grandes; el benchmark grande cubre lectura, escala, reducción y escritura, no un producto cuadrático gigante.

## Cuotas y rendimiento

No se cambió la contabilización de cuota. Cada publicación sigue reconciliándose con los árboles del banco y el espacio físico, lo cual conserva exactitud e integridad, pero puede costar más conforme crece el número de bloques. Se necesita medición sobre el HDD y una cantidad representativa de fragmentos antes de justificar una cuenta incremental persistente. El benchmark reporta duración y bytes transferidos; no hay resultados físicos en este informe.

## Seguridad y compatibilidad

PIZARRA continúa deshabilitado por defecto. El ejecutor no ofrece API HTTP y no participa en el lock de inferencia porque no puede ser iniciado desde Buttervision ni comparte GPU. `PizarraStorage` rechaza un volumen distinto/no autorizado. No hay borrado automático de tensores ni escrituras en el HDD desde las pruebas automatizadas. Los `.safetensors` originales no se abren ni modifican.
