# PIZARRA v0.5D experimental

El controlador opt-in envuelve temporalmente los métodos `forward` del UNet y de un único bloque, sin instalar hooks. `ModuleOffload` mantiene sus verificaciones de hooks, almacenamiento compartido e integridad. El UNet y todos sus submódulos deben estar en evaluación; cada llamada debe hacerse en `torch.inference_mode()`. Se rechazan llamadas concurrentes o recursivas al UNet y llamadas directas al bloque. El propietario debe garantizar acceso exclusivo a la instancia y sus tensores, sin modificar el modelo ni invocar descendientes directamente durante la sesión; Python no puede impedir accesos externos arbitrarios. No usar con entrenamiento, optimizadores, GPU ni hooks. Los wrappers se eliminan en `close`, que restaura los pesos antes de borrar la sesión. Un fallo de restore deja la sesión y los pesos meta; si falla la verificación al liberar, los pesos quedan residentes y se conserva la sesión. Resolver la causa antes de volver a cerrar. Un proceso terminado abruptamente requiere la recuperación externa documentada en v0.5C.

El probe usa el UNet progresivo y dos forwards completos sintéticos en CPU. Verifica igualdad exacta, estado meta y ausencia de escrituras adicionales. Registra tiempos, RSS, USS, cgroup, VRAM y contadores de E/S; son instantáneas y contabilidad del proceso, no mediciones continuas de pico ni necesariamente E/S física. Antes de cada restore exige espacio en el cgroup para dos veces el tamaño de pesos más 512 MiB; es una reserva conservadora, no una garantía contra OOM. El límite nunca se aumenta y swap está deshabilitado. El pico anterior de 9,91 GiB deja muy poco margen: si no alcanza, se rechaza el paso y se conserva el job y token. Un OOM o SIGKILL puede impedir limpieza; guardar los eventos y usar la recuperación de v0.5C.

En Fedora, con el volumen PIZARRA y modelo configurados:

```bash
venv/bin/python scripts/pizarra_unet_probe.py --execute --automatic --model "$HOME/stable-diffusion-webui-forge/models/Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" --memory-max-gib 10 --timeout-seconds 3600 --malloc-trim
```

La comparación del probe verifica esas entradas concretas, no demuestra generación completa ni equivalencia para todas las entradas. No se activa en producción.
