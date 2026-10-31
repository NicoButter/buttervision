# Buttervision — SDXL Low-VRAM v2.3.1

## Resultado

Se corrigió el flujo para que elegir Juggernaut XL solo lo deje en **Seleccionado**. El backend no lo presenta como listo hasta terminar la carga; durante una solicitud publica **Cargando**, **Cargado**, **Generando** y vuelve a **Cargado** al concluir. Los fallos publican **Error recuperable** y conservan únicamente el tipo de excepción, sin exponer rutas ni prompts. Los modelos desconocidos se muestran como **Incompatible** y no se pueden seleccionar.

La ruta SDXL carga archivos y configuraciones locales. `from_single_file` y `from_pretrained` reciben `low_cpu_mem_usage=True` y `local_files_only=True`; CUDA usa FP16 y offload secuencial de CPU cuando la configuración lo habilita. La carga sigue siendo diferida hasta la primera generación. CyberRealistic y Face Reference SD 1.5 conservan sus backends existentes.

No se integró PIZARRA Tensor Store en Diffusers. No se descargó ni copió ningún modelo. Los tests automatizados no generan imágenes.

## Presupuesto y mediciones

La política anterior multiplicaba el tamaño del checkpoint por dos con CUDA y por cuatro sin CUDA, y sumaba 2 GiB. Para el archivo local de Juggernaut XL medido, la regla con CUDA exigía alrededor de 15.24 GiB **libres**.

| Medición en este entorno | Resultado |
| --- | ---: |
| RAM total detectada | 16,369,364,992 bytes (15.245 GiB) |
| RAM disponible al medir | 4,576,755,712 bytes (4.262 GiB) |
| Juggernaut XL local | 7,105,348,188 bytes (6.617 GiB) |
| Política anterior, perfil CUDA | 16,358,180,024 bytes (15.235 GiB) requeridos |
| Nueva estimación para CUDA + FP16 | tamaño del checkpoint + margen fijo explícito de 2 GiB = 9,252,831,836 bytes (8.617 GiB) |
| Nueva estimación de fallback CPU + FP32 | 2 × tamaño del checkpoint + margen de 2 GiB = 16,358,180,024 bytes (15.235 GiB) |

Las cifras son **umbrales estimados**, no mediciones del pico de Juggernaut XL. La estimación CUDA se basa en pesos FP16; el fallback CPU usa FP32 y por eso cuenta dos veces los bytes del checkpoint. Diffusers documenta que la carga de archivo único con `low_cpu_mem_usage` y Accelerate intenta limitar el pico de RAM aproximadamente al tamaño del modelo; el margen de 2 GiB queda visible en el diagnóstico. La admisión requiere que Accelerate esté presente. El control también considera el límite y el uso del cgroup v2 cuando está disponible. Con 4.262 GiB libres, este entorno sigue bloqueando la carga; no se relajó el control para forzarla.

La instalación inspeccionada tiene PyTorch 2.5.1+cu121, Diffusers 0.36.0 y Accelerate 1.12.0, pero `torch.cuda.is_available()` devuelve falso y no hay driver NVIDIA accesible. Por ello estos datos no prueban el comportamiento de la GTX 1650. Tampoco demuestran que SDXL genere dentro de 4 GB de VRAM.

## Estrategia y límites

- `low_cpu_mem_usage=True` usa la ruta de inicialización de bajo uso de RAM de Diffusers/Accelerate cuando la combinación instalada la admite. No es una carga out-of-core ni una garantía de que el sistema tenga RAM suficiente.
- FP16 y `enable_sequential_cpu_offload()` se mantienen para CUDA. El offload secuencial debe habilitarse antes de mover el pipeline entero a CUDA. Puede reducir mucho la VRAM a costa de transferencias y velocidad; la GTX 1650 aún puede quedarse sin memoria durante atención, VAE u operaciones temporales.
- Se conserva el mínimo operativo previo de 1.5 GiB de VRAM libre para offload secuencial. Sin offload, se exige espacio de VRAM para el tamaño local del checkpoint más el margen de 2 GiB. El archivo en disco no predice el pico exacto de VRAM.
- Atención por slices, VAE slicing y VAE tiling siguen habilitados. No se añadió cuantización, se mantuvo FP16 y no se cambió la precisión de SD 1.5.
- Accelerate ofrece carga/distribución de modelos grandes con inicialización vacía y dispatch, pero esa API no reemplaza directamente la construcción de una pipeline Diffusers SDXL desde un checkpoint monolítico. No se implementó un cargador alternativo experimental.
- La prueba aislada manual usa `systemd-run --user --scope` con `MemoryHigh`, `MemoryMax` y `MemorySwapMax=0`, y pone un límite temporal al worker. El límite controla memoria RAM del proceso/cgroup, no reserva ni aísla la VRAM compartida. Si Fedora no delega esos controles, systemd-run debe fallar y no habrá fallback sin límites.
- El cgroup reduce el riesgo de que un OOM del worker mate FastAPI, pero no es una garantía absoluta: un OOM global del kernel, un límite de usuario mal configurado, otra aplicación usando la GPU o un error del driver pueden afectar al servidor o al host.
- El proceso hijo usa la configuración local de Buttervision, cachés locales y checkpoint registrado. No inicia descargas. La ejecución debe hacerse sin otra inferencia activa, especialmente Forge, ya que comparte RAM y VRAM.

Documentación consultada: [carga de archivos únicos Diffusers](https://huggingface.co/docs/diffusers/main/api/loaders/single_file), [optimización de memoria Diffusers](https://huggingface.co/docs/diffusers/optimization/memory), [visión general de pipelines Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/overview), [inferencia de modelos grandes Accelerate](https://huggingface.co/docs/accelerate/main/concept_guides/big_model_inference), [controles de recursos systemd](https://freedesktop.org/software/systemd/man/latest/systemd.resource-control.html), [systemd-run](https://www.freedesktop.org/software/systemd/man/254/systemd-run.html).

## Pruebas automatizadas

Las pruebas cubren admisión por RAM, falta de RAM antes de llamar al cargador, falta de espacio VRAM si se desactiva offload, `low_cpu_mem_usage`, FP16 y offload, selección sin carga, compatibilidad, errores de carga recuperables con limpieza y reintento, límites de la orden systemd y regresiones del catálogo/Face Reference SD 1.5. Los test doubles interceptan la carga y generación. Ninguna prueba genera imágenes ni llama a servicios de descarga.

Validación ejecutada en este entorno: 13 pruebas dirigidas SDXL/selección y 3 regresiones SD 1.5/Face Reference/OOM pasaron; la suite web completa pasó (25 pruebas) y `npm run build` pasó. `py_compile` y `git diff --check` pasaron. La suite Python amplia se atasca en la prueba HTTP preexistente `test_incompatible_lora_and_busy_mutations_do_not_load`, al hacer una petición con `TestClient` mientras el hilo principal mantiene el lock del motor. La selección de pruebas nuevas y las regresiones anteriores se ejecutaron aparte y pasaron. Esto limita la validación de la suite completa, no es una ejecución de SDXL.

## Validación manual en Fedora con la GTX 1650

1. Confirma que Juggernaut XL y la configuración SDXL requerida ya están en rutas registradas (`BUTTERVISION_CHECKPOINT_DIRS` y/o las rutas locales que usa la instalación). No descargues ni copies archivos para esta prueba.
2. Cierra Forge y cualquier otra inferencia. Verifica que haya espacio libre suficiente: la primera estimación requiere checkpoint + 2 GiB; el margen de RAM de systemd queda en al menos 4 GiB para el host.
3. Verifica que el servicio local tenga activado `enable_cpu_offload` y que CUDA funcione (`nvidia-smi`). Si CUDA no está disponible, el worker se niega a ejecutar.
4. Desde la raíz del proyecto y con el mismo entorno Python/configuración de Buttervision, revisa primero el modo informativo:

   ```bash
   venv/bin/python scripts/sdxl_isolated_probe.py
   ```

5. Inicia la prueba controlada explícitamente, usando la ruta ya registrada del checkpoint:

   ```bash
   venv/bin/python scripts/sdxl_isolated_probe.py \
     --execute \
     --model "$BUTTERVISION_CHECKPOINT_DIRS/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" \
     --memory-max-gib 10 \
     --timeout-seconds 3600
   ```

   Ajusta la ruta al nombre/ruta local real. `MemoryMax` no puede superar RAM total menos 4 GiB; el valor predeterminado es 10 GiB. El sondeo usa 512×512, lote 1, 10 pasos, semilla fija y guarda su única imagen de salida bajo `outputs/`.
6. En otra terminal observa `free -h`, `nvidia-smi -l 1` y los logs del servicio. Comprueba que la UI avance por Seleccionado → Cargando → Cargado → Generando → Cargado. Si el scope acaba por memoria o tiempo, el servidor debería seguir respondiendo; guarda el código de salida, logs, RAM pico, VRAM pico y versión del driver.
7. Si completa la prueba, registra expresamente ese resultado real y repite solo con los tamaños/pasos previstos para el uso. Una ejecución 512×512 no valida resoluciones mayores, LoRA, Face Reference en SDXL ni carga simultánea. La compatibilidad SDXL con esta GTX 1650 permanece **no comprobada** hasta completar una generación real satisfactoria en ese hardware.
