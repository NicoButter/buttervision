# PIZARRA v0.1 — configuración en Fedora

Estos pasos son **manuales**: el programa no formatea, monta, desmonta ni modifica discos. No ejecutar una generación real o benchmark sin autorización del operador. Utilizar un filesystem Linux local con hardlinks, `flock` y `fsync`, por ejemplo ext4, y disponer de una copia de seguridad apropiada.

## Montaje persistente por UUID

1. Identificar el volumen existente con `lsblk -f`, `sudo blkid` y `findmnt`. Anotar su UUID, filesystem y destino; confirmar cuidadosamente que es el disco autorizado. No formatearlo para esta prueba.
2. El operador puede crear el punto de montaje vacío: `sudo mkdir -p /mnt/pizarra`. El directorio por sí solo **no constituye un montaje** y la aplicación lo rechazará si el volumen está ausente.
3. Editar `/etc/fstab` con la herramienta administrativa elegida y añadir una entrada para el UUID verificado, adaptando las opciones a la instalación:

   ```text
   UUID=<UUID_DEL_FILESYSTEM>  /mnt/pizarra  ext4  defaults,nofail  0  2
   ```

   Con `nofail`, el sistema puede iniciar sin el disco, pero PIZARRA no usará silenciosamente el directorio del disco raíz. El operador puede montar manualmente con `sudo mount /mnt/pizarra` y debe comprobar `findmnt -no TARGET,UUID,FSTYPE /mnt/pizarra`. Si ya está montado, comprobarlo antes de cualquier cambio. La raíz configurada debe coincidir exactamente con el punto de montaje y ser un dispositivo distinto del de `/`; un subdirectorio o un bind mount del disco raíz se rechaza.
4. Verificar que el usuario de Buttervision tenga permisos de escritura sobre el punto de montaje. Si es necesario, el administrador ajusta permisos del directorio del volumen de forma explícita; no usar cambios recursivos sobre archivos ajenos.

## Variables de entorno

`.env.example` es un ejemplo, no contiene secretos y no se carga automáticamente. Definir las variables en la terminal de arranque o mediante el mecanismo de servicios elegido. Por ejemplo, reemplazar el UUID después de verificarlo:

```bash
export BUTTERVISION_PIZARRA_ENABLED=true
export BUTTERVISION_PIZARRA_ROOT=/mnt/pizarra
export BUTTERVISION_PIZARRA_VOLUME_UUID=UUID_DEL_FILESYSTEM
export BUTTERVISION_PIZARRA_QUOTA_GB=400
export BUTTERVISION_PIZARRA_MIN_FREE_GB=20
export BUTTERVISION_PIZARRA_METRICS_ENABLED=true
bash run.sh
```

Las cuotas están expresadas en GB decimales de 1.000.000.000 bytes. El UUID adicional es muy recomendable: sin él se valida que exista un volumen separado y que no cambie durante el proceso, pero no se comprueba qué otro disco fue montado en esa ruta antes del arranque. La cuota limita el espacio lógico contabilizado en las áreas del banco y el margen usa el espacio físico disponible. Elegir valores menores que la capacidad disponible y dejar margen para otros programas. `BUTTERVISION_PIZARRA_RAM_MARGIN_FRACTION` y `BUTTERVISION_PIZARRA_GPU_MARGIN_FRACTION`, por defecto `0.1`, sólo afectan el presupuesto observacional.

Para deshabilitar PIZARRA, detener el servidor y arrancar con `BUTTERVISION_PIZARRA_ENABLED=false`. El motor anterior no escribe en el banco ni crea la base PIZARRA. La base administrativa del modo habilitado permanece en `data/pizarra/`, dentro del proyecto, con directorio privado y fuera del HDD; el banco nunca se sirve por HTTP.

## Verificación sin inferencia

Con el servidor activo, consultar `curl http://127.0.0.1:7861/api/pizarra/status` y, opcionalmente, `curl 'http://127.0.0.1:7861/api/pizarra/status?include_budget=true'`. La respuesta debe indicar `enabled` y `available` verdaderos, con espacio libre y cuota razonables. Si no está montado, hay un UUID incorrecto o faltan permisos, `available` será falso y la API rechazará nuevos trabajos PIZARRA con HTTP 503. La respuesta no expone la ruta física.

Para probar recuperación, el operador puede detener el proceso durante una generación de prueba autorizada y reiniciar. `GET /api/jobs/<job_id>` debe mostrar `interrupted` para un proceso propietario terminado, sin ejecución automática; `GET /api/pizarra/jobs/<job_id>/metrics` sólo estará disponible si se pudo confirmar una medición. No retirar físicamente ni desmontar un volumen en escritura para forzar una prueba. Una desconexión real se detecta en el siguiente control; si sucede durante Diffusers, puede terminar la llamada en curso antes de registrar el fallo.

## Benchmark con autorización

El benchmark genera una imagen real **sólo** con `--execute`; no se ejecutó automáticamente en esta entrega. Requiere dos sesiones sucesivas del servidor, una con PIZARRA deshabilitado y otra habilitado, el mismo modelo activo y sin otros generadores. Identificar el PID real de Uvicorn con `ps -eo pid,args` y reemplazar `1234` en los ejemplos:

```bash
mkdir -p benchmarks/results
venv/bin/python benchmarks/pizarra_benchmark.py run --execute --mode legacy --server-pid 1234 --output benchmarks/results/legacy.pizarra.json
# Reiniciar el servidor con PIZARRA habilitado; seleccionar el mismo modelo.
venv/bin/python benchmarks/pizarra_benchmark.py run --execute --mode pizarra --server-pid 1234 --output benchmarks/results/pizarra.pizarra.json
venv/bin/python benchmarks/pizarra_benchmark.py compare benchmarks/results/legacy.pizarra.json benchmarks/results/pizarra.pizarra.json
```

Se usa el mismo prompt de prueba fijo, seed, dimensiones, pasos y CFG por defecto; pueden establecerse con opciones idénticas en ambas sesiones. La comparación exige el mismo identificador de modelo, familia, scheduler, LoRA activos y parámetros. Registra tiempo de pared, estado, hash de resultados, RSS y VRAM por proceso muestreados y E/S del proceso si se puede medir. `nvidia-smi` es opcional: sin datos, VRAM queda `null`. La instrumentación de PIZARRA agrega métricas propias. Los archivos de resultado no guardan el prompt ni imágenes; las imágenes reales de prueba se guardan en outputs según el comportamiento habitual. No se sobrescribe un archivo de benchmark existente.

Los hashes corresponden al archivo completo, incluida su metadata y fecha; una diferencia no demuestra por sí sola que hayan cambiado los píxeles. Los picos muestreados pueden omitir máximos breves. Cachés, temperaturas, procesos ajenos, estado de los modelos y el propio monitor pueden influir; no atribuir una diferencia directamente a PIZARRA ni declarar mejora sin mediciones repetidas y controladas.
