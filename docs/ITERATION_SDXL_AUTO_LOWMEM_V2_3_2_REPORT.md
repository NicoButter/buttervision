# Buttervision — SDXL Auto Low-Memory v2.3.2

## Resultado

Se implementó un planificador `AUTO` para SDXL. Antes de leer un checkpoint, compara la RAM disponible, RSS del proceso, memoria disponible bajo cgroup, VRAM y las capacidades detectadas de Diffusers/Accelerate. Si ninguna estrategia tiene el margen requerido, rechaza la carga y expone el motivo en el panel experto.

El entorno de desarrollo no tiene CUDA. En la última medición tenía 16,369,364,992 bytes de RAM, 8,980,762,624 bytes disponibles, Accelerate 1.12.0 y Diffusers 0.36.0. No se encontró cgroup v2 limitante. El checkpoint Juggernaut XL v9 registrado ocupa 7,105,348,188 bytes (6.62 GiB). En esta medición, el margen bajo de carga requerido (`tamaño + 2 GiB`) era 8.62 GiB, así que se rechazó con seguridad. No se ejecutó una inferencia ni se confirmó compatibilidad con la GTX 1650.

## Perfiles y umbrales

Los márgenes son presupuestos conservadores, no una garantía de que el modelo cargará. Se aplican a los recursos observados en el momento de la decisión; la RAM considerada es el mínimo entre memoria del host y disponibilidad del cgroup cuando existe.

| Perfil | Carga de pesos | Requisito de admisión | Ejecución |
| --- | --- | --- | --- |
| `STANDARD` | Carga convencional (`low_cpu_mem_usage=False`) | RAM libre ≥ 2 × checkpoint + 2 GiB; con CUDA, VRAM libre ≥ checkpoint + 2 GiB | FP16 en CUDA y pipeline en GPU; CPU FP32 si no hay CUDA y hay RAM suficiente |
| `LOW_RAM` | Accelerate `low_cpu_mem_usage=True` y safetensors con mmap habilitado | RAM libre ≥ checkpoint + 2 GiB; Accelerate y mmap disponibles | Mantiene FP16 en CUDA; si la VRAM no alcanza para el pipeline, activa CPU sequential offload cuando está configurado y hay al menos 1.5 GiB libres |
| `LOW_VRAM` | Accelerate + mmap | Mismo mínimo de RAM baja; CUDA, sequential CPU offload configurado y ≥ 1.5 GiB de VRAM libre | FP16 y `enable_sequential_cpu_offload()` |
| `DISK_OFFLOAD` | Igual que LOW_RAM para construir el pipeline | Opt-in experimental, CUDA, APIs detectadas, mínimo RAM baja, ≥ 1.5 GiB VRAM y ≥ 1.25 × checkpoint + 1 GiB en disco | Prueba manual con group offload; no lo selecciona `AUTO` |

`AUTO` prefiere `STANDARD` si pasa todos sus controles; si no, selecciona `LOW_RAM` o `LOW_VRAM` cuando su combinación de recursos es viable. Superar la VRAM por sí solo ya no bloquea el perfil bajo: cuando se cumplen RAM, CUDA, offload configurado y VRAM mínima, se activa CPU offload. Si falta RAM para inicializar desde el checkpoint, se rechaza; no se reduce el umbral a ciegas.

La selección de checkpoint sigue mostrando `Seleccionado`. Los estados de carga siguen diferenciados (`selected`, `loading`, `loaded`, `generating`, `error`, `incompatible`); la interfaz simple informa preparación/resultado/error y no presenta selección como modelo listo. El modo experto añade el perfil, RSS, RAM/cgroup, estimación de carga, VRAM, offload, fase y motivos de rechazo. CyberRealistic y Face Reference permanecen en sus rutas SD 1.5.

## Qué reduce cada técnica

- `low_cpu_mem_usage` inicializa componentes con meta-device/Accelerate y busca evitar la duplicación convencional del state dict durante su materialización. Ya estaba en la ruta anterior y ahora se elige mediante el plan.
- `disable_mmap=False` deja habilitado el acceso mmap de safetensors en la carga de checkpoint único. Diffusers 0.36.0 también usa mmap por defecto en esa ruta. No significa que desaparezca el coste de materializar pesos ni que el checkpoint entero deje de estar respaldado por RAM/cache.
- FP16 reduce el tamaño de pesos en GPU en la ruta CUDA. CPU offload secuencial reduce la VRAM de inferencia al mover módulos entre CPU y GPU; puede aumentar notablemente latencia.
- Attention slicing, VAE slicing y tiling reducen picos de activaciones/decodificación durante inferencia. No reducen el pico inicial necesario para interpretar el checkpoint.
- `from_single_file` todavía abre el state dict del checkpoint al construir componentes. La carga actual no valida una carga verdaderamente progresiva por componentes ni elimina toda coexistencia temporal de buffers y pesos. El presupuesto conservador de RAM permanece por esa razón.

Versiones detectadas localmente: PyTorch 2.5.1+cu121, Diffusers 0.36.0, Accelerate 1.12.0, safetensors 0.7.0. La firma instalada de `enable_group_offload` expone `offload_to_disk_path`. No se validó el comportamiento ni el rendimiento de disco con una SDXL real en CUDA; se mantiene experimental, explícito y deshabilitado en `AUTO`. Requiere `BUTTERVISION_SDXL_DISK_OFFLOAD_DIR`, comprueba permisos/espacio, rechaza raíces registradas de checkpoints y `models/Stable-diffusion`, y usa una subcarpeta privada única. El proceso padre elimina esa carpeta al finalizar, incluso si el worker termina con SIGKILL. Esta optimización actúa sobre offload después de construir el pipeline y **no elimina el pico inicial de RAM**. No se conecta PIZARRA Tensor Store con Diffusers.

Referencias técnicas consultadas: [Diffusers 0.36 — optimización de memoria](https://huggingface.co/docs/diffusers/v0.36.0/en/optimization/memory), [Diffusers — carga de archivos individuales](https://huggingface.co/docs/diffusers/api/loaders/single_file) y [Accelerate 1.12 — big model inference](https://huggingface.co/docs/accelerate/v1.12.0/usage_guides/big_modeling). La compatibilidad de APIs se comprobó con las instalaciones locales; no equivale a comprobar Juggernaut XL en la GPU objetivo.

## Aislamiento y límites

`scripts/sdxl_isolated_probe.py` mantiene el modo informativo sin inferencia. Con `--execute` valida un checkpoint SDXL registrado, configuración local completa, offload CPU habilitado, `systemd-run` y `timeout`; ejecuta 512×512, 10 pasos, lote 1 en un scope con `MemoryHigh`, `MemoryMax`, `MemorySwapMax=0` y timeout. El padre informa códigos SIGKILL/OOM de forma explícita. `--strategy DISK_OFFLOAD` es solo para evaluación manual y necesita el directorio local configurado.

El API y el sondeo adquieren un `flock` local no bloqueante en el directorio de caché compartido, además del lock de worker del proceso API. Impide que el sondeo comience mientras el API usa el motor y viceversa. El lock se libera en bloques `finally`; al morir el proceso, el sistema operativo cierra el descriptor. El fichero de lock puede permanecer, pero no es un lock persistente.

El límite cgroup contiene el worker si el sistema soporta los scopes de usuario configurados; reduce el riesgo, pero no es una sandbox ni protege frente a fallos de kernel/controlador, agotamiento de recursos del host fuera del scope o OOM global. Tampoco puede comprobar la compatibilidad de todas las capas SDXL sin una carga real. El worker puede guardar una imagen en `outputs/` solo en la validación manual opt-in; los tests no generan imágenes.

No se descargaron ni copiaron modelos/configuraciones, no se tocaron swap, controladores, hardware ni archivos de Forge, y no se integró PIZARRA con Diffusers.

## Pruebas ejecutadas

Sin carga de modelos ni generación de imágenes:

- `venv/bin/python -m unittest tests.test_model_manager_v2_3 -v`: 41 passed. Incluye AUTO, alternativa por encima de VRAM, rechazo por RAM, disco experimental/permisos/espacio, configuración local, offload/FP16, OOM simulado, jerarquía cgroup, bloqueo API-sondeo y recuperación del lock, estados de selección/carga, catálogo, Face Reference, LoRA y rutas SD 1.5. El mapeo HTTP del motor se ejercita directamente con `engine_call`; no se inicia un servidor de pruebas.
- `npm test -- --run`: 26 passed.
- `npm run build`: completó correctamente.
- `venv/bin/python -m compileall -q api core scripts tests`: completó correctamente.
- `git diff --check`: sin errores.

Se retiraron solicitudes `TestClient` de dos pruebas que se bloqueaban en este entorno y se sustituyeron por llamadas directas al adaptador `engine_call`, conservando las aserciones de estado 400/409. El servidor FastAPI no forma parte de esta prueba unitaria. Los resultados del host actual no verifican el comportamiento de CUDA/GTX 1650.

## Validación manual en Fedora (GTX 1650 4 GB / 16 GB RAM)

1. Asegurar que Juggernaut XL v9 y su configuración SDXL completa ya están registrados localmente. No mover ni copiar archivos.
2. Revisar `nvidia-smi`, `free -h`, el espacio del disco local elegido y `enable_cpu_offload` en la configuración existente. Cerrar otras cargas GPU/RAM y dejar al menos 4 GiB para el host.
3. Desde el entorno virtual y la raíz del repositorio, revisar primero sin inferencia: `venv/bin/python scripts/sdxl_isolated_probe.py`.
4. Ejecutar una única prueba limitada: `venv/bin/python scripts/sdxl_isolated_probe.py --execute --model RUTA_LOCAL_REGISTRADA --memory-max-gib 10 --timeout-seconds 3600 --strategy AUTO`. Seguir `free -h`, `nvidia-smi` y el journal del scope. El worker usa prompt fijo, 512×512, 10 pasos, semilla fija y un resultado.
5. Comprobar código de salida, imagen creada en `outputs/`, registros de Buttervision, RAM/VRAM máxima observada, tiempo de carga/inferencia y estado final del API. Si el preflight rechaza, registrar el perfil y los recursos observados; no relajar límites.
6. Solo para evaluar el backend experimental de disco: configurar `BUTTERVISION_SDXL_DISK_OFFLOAD_DIR` a un directorio local autorizado fuera de Forge/checkpoints, medir espacio y repetir con `--strategy DISK_OFFLOAD`. Registrar rendimiento, espacio temporal pico y limpieza. No promoverlo a `AUTO` hasta probar la compatibilidad SDXL completa.

Una generación exitosa en el equipo destino es condición necesaria antes de declarar Juggernaut XL compatible. Este cambio prepara la evaluación controlada, pero todavía no satisface esa comprobación de hardware.
