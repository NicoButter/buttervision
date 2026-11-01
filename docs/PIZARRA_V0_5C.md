# PIZARRA v0.5C: UNet SDXL progresivo aislado

Este experimento opt-in carga la pipeline SDXL mediante `load_progressive` existente y usa directamente `pipe.unet.down_blocks.0.resnets.0`. `ModuleOffload` rechaza hooks PyTorch o `_hf_hook` de Accelerate en todo el UNet, almacenamiento compartido y tensores incompatibles; no instala hooks. El objeto ResNet conserva su identidad, pero sus parámetros se sustituyen por `meta` y luego por nuevos objetos Parameter. La prueba usa exclusivamente CPU, `eval` e inferencia sin gradientes. Exige acceso exclusivo a la instancia; no admite entrenamiento ni optimizadores. Los pesos se escriben una sola vez, se verifica SHA-256 y se restauran explícitamente **antes** de invocar el bloque o UNet. Esto no implementa recuperación automática dentro del forward.

El modo predeterminado compara el ResNet antes y después con entradas sintéticas idénticas. `--full-forward` añade dos forward completos del UNet con entradas SDXL sintéticas y latentes 8×8; no genera imágenes. Cada comparación exige igualdad exacta y registra errores absoluto máximo y medio. Un fallo posterior a la persistencia intenta restaurar y conserva la sesión incluso si se restaura correctamente. Si la restauración falla no se borra nada. Guardar los eventos JSON con `job_id` y `token`; la limpieza sólo ocurre con la API segura tras el éxito. Si el proceso se interrumpe, usar `venv/bin/python scripts/pizarra_weight_offload_probe.py --execute --recover-job JOB --token TOKEN` para verificar y recuperar; `--cleanup-only` elimina explícitamente la sesión sin recuperarla.

## Fedora

Configurar y validar previamente el volumen PIZARRA y el catálogo local, y ejecutar desde el repositorio (ajustar la ruta y elegir el límite según los recursos disponibles):

```bash
venv/bin/python scripts/pizarra_unet_probe.py --execute --model "$HOME/stable-diffusion-webui-forge/models/Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" --memory-max-gib 10 --timeout-seconds 3600 --malloc-trim
```

Añadir `--full-forward` sólo si hay tiempo y memoria suficientes. El supervisor crea un scope systemd de usuario con `MemoryMax` elegido y `MemorySwapMax=0`, y aplica timeout; el worker comprueba el nombre de su scope cgroup v2 y sus límites efectivos antes de cargar. No eleva el límite automáticamente. Se necesita systemd de usuario, cgroup v2 y un volumen PIZARRA autorizado. Un OOM o SIGKILL puede impedir cualquier `finally`: conservar el último token emitido y consultar el journal de systemd. Una salida no cero puede deberse a límite de memoria, timeout, hooks o incompatibilidad; consultar el evento y el journal, sin atribuir automáticamente el fallo a OOM.

Los eventos incluyen RSS, USS, cgroup, VRAM disponible y contadores `/proc/self/io`; las instantáneas son puntuales, no picos continuos. `pizarra_write_bytes` y los contadores de lectura PIZARRA son bytes contabilizados por su API, no E/S física. `/proc/self/io` corresponde al proceso completo y a la contabilidad del kernel, con efectos de caché; tampoco mide directamente toda la actividad física del dispositivo. Para observar el dispositivo utilizar una medición externa como `iostat -dx`, teniendo en cuenta otras cargas. El modelo se ejecuta en CPU: los valores VRAM son contexto, no una prueba de ejecución en GPU.

Prueba automatizada: `venv/bin/python -m unittest discover -s tests -p 'test_pizarra_unet_probe.py' -q`. Usa un checkpoint SDXL diminuto de prueba cargado con el cargador progresivo; no constituye validación física de Juggernaut XL.
