# PIZARRA v0.5E: imagen SDXL experimental

Pendiente de validación física: esta fase solo queda validada cuando se comprueben el PNG real y la limpieza del job PIZARRA en Fedora. El probe es independiente de producción, CPU y offline. La imagen de 64×64 es una prueba técnica, no de calidad. Requiere checkpoint SDXL registrado, configuración SDXL local completa, volumen PIZARRA configurado y systemd de usuario. Usa el bloqueo de inferencia, un scope con 10 GiB y swap cero, y una única sesión para `down_blocks.0.resnets.0`. La reserva de memoria anterior a cada restauración es conservadora y no garantiza evitar OOM.

Primera ejecución física (el directorio actual debe existir y el archivo no debe existir):

```bash
venv/bin/python scripts/pizarra_sdxl_image_probe.py --execute --model "$HOME/stable-diffusion-webui-forge/models/Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" --output "$PWD/pizarra-sdxl-64.png" --size 64 --timeout-seconds 3600
```

Para una prueba posterior independiente de 128×128, cambiar `--size 64` por `--size 128` y usar otra ruta de salida. No se ejecuta automáticamente. Conservar los eventos JSON: `session` identifica job y token; `cleaned` confirma la limpieza al finalizar. Verificar también el PNG y el volumen. En caso de fallo se intenta restaurar sin borrar la sesión y se registran el estado y las credenciales para recuperación; un OOM o SIGKILL puede impedir estos eventos. Usar el procedimiento de recuperación de v0.5C antes de retirar cualquier sesión.

Las instantáneas de RSS, USS, cgroup, VRAM y `/proc/self/io` describen el proceso en esos instantes, no sus picos. Los bytes PIZARRA son contadores de API, no tráfico físico del HDD: el caché del sistema puede absorber lecturas y escrituras. Para medir actividad física usar herramientas del sistema por separado. Los tiempos y eventos identifican carga, codificación, denoising, VAE, persistencia, generación y salida; la restauración final de cierre también lee datos después del evento de generación.

Hotfix de memoria: el probe codifica el prompt original con ambos codificadores y conserva los embeddings de secuencia y agrupados SDXL. Luego elimina las referencias de la pipeline a ambos codificadores, ejecuta `gc.collect()` y solicita `malloc_trim` cuando glibc lo permite. Diffusers 0.36.0 admite esos embeddings y usa su dimensión agrupada cuando falta el segundo codificador. El prompt, seed, pasos, guidance y dimensiones no cambian. La reserva preventiva para cada restauración permanece igual; si todavía no alcanza, se conserva la sesión.

Los eventos `memory_before_prompt_encoding`, `memory_after_prompt_encoding`, `memory_after_encoder_release` y `memory_before_restore` incluyen RSS, USS, memoria anónima y carga del cgroup, además de `anon`, `file` e `inactive_file` de `memory.stat` cuando estén disponibles. Son instantáneas, y el caché de archivos puede impedir que la carga total del cgroup baje tanto como el RSS. La reducción con tensores sintéticos y la generación con una pipeline SDXL mínima no demuestran que Juggernaut complete la ejecución física; comprobarlo mediante el comando anterior en Fedora.
