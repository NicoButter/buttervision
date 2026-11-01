# PIZARRA v0.5E: imagen SDXL experimental

Pendiente de validación física: esta fase solo queda validada cuando se comprueben el PNG real y la limpieza del job PIZARRA en Fedora. El probe es independiente de producción, CPU y offline. La imagen de 64×64 es una prueba técnica, no de calidad. Requiere checkpoint SDXL registrado, configuración SDXL local completa, volumen PIZARRA configurado y systemd de usuario. Usa el bloqueo de inferencia, un scope con 10 GiB y swap cero, y una única sesión para `down_blocks.0.resnets.0`. La reserva de memoria anterior a cada restauración es conservadora y no garantiza evitar OOM.

Primera ejecución física (el directorio actual debe existir y el archivo no debe existir):

```bash
venv/bin/python scripts/pizarra_sdxl_image_probe.py --execute --model "$HOME/stable-diffusion-webui-forge/models/Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" --output "$PWD/pizarra-sdxl-64.png" --size 64 --timeout-seconds 3600
```

Para una prueba posterior independiente de 128×128, cambiar `--size 64` por `--size 128` y usar otra ruta de salida. No se ejecuta automáticamente. Conservar los eventos JSON: `session` identifica job y token; `cleaned` confirma la limpieza al finalizar. Verificar también el PNG y el volumen. En caso de fallo se intenta restaurar sin borrar la sesión y se registran el estado y las credenciales para recuperación; un OOM o SIGKILL puede impedir estos eventos. Usar el procedimiento de recuperación de v0.5C antes de retirar cualquier sesión.

Las instantáneas de RSS, USS, cgroup, VRAM y `/proc/self/io` describen el proceso en esos instantes, no sus picos. Los bytes PIZARRA son contadores de API, no tráfico físico del HDD: el caché del sistema puede absorber lecturas y escrituras. Para medir actividad física usar herramientas del sistema por separado. Los tiempos y eventos identifican carga, codificación, denoising, VAE, persistencia, generación y salida; la restauración final de cierre también lee datos después del evento de generación.
