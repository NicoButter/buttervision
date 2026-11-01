# PIZARRA v0.3 — Inference Integration Spike

## Resumen de la iteración

Se agregó un adaptador experimental y separado para ejecutar **una** inferencia SDXL local mediante Diffusers, bajo `scripts/sdxl_isolated_probe.py` y dentro de un scope cgroup de usuario. El adaptador consulta el presupuesto observacional de PIZARRA y las capacidades de Diffusers, selecciona `standard` o `cpu_offload`, instrumenta la carga, inferencia y limpieza, y libera el pipeline al terminar.

El adaptador exige que PIZARRA esté habilitada y que `PizarraStorage` valide su volumen. No se conecta al API ni al botón habitual de generación. No modifica CyberRealistic, Face Reference, LoRA, React, Gradio, el scheduler ni el formato de Tensor Store.

**El offload a HDD de PIZARRA queda bloqueado en este spike.** La interfaz de almacenamiento actual no puede imponer su cuota a cada escritura mutable que Diffusers realiza directamente en `offload_to_disk_path`. Conectarla a una carpeta del banco permitiría saltarse el protocolo de publicación inmutable y las comprobaciones de cuota. El adaptador rechaza esa estrategia con una explicación explícita; no crea una carpeta de offload ni escribe en PIZARRA durante la inferencia.

## Auditoría de las versiones locales

Versiones detectadas en el entorno de desarrollo:

- PyTorch `2.5.1+cu121`
- Diffusers `0.36.0`
- Accelerate `1.12.0`
- safetensors `0.7.0`
- `torch.cuda.is_available()` devolvió `False`.

La firma instalada de `StableDiffusionXLPipeline.enable_group_offload()` incluye `offload_to_disk_path`. Su implementación recorre los componentes `torch.nn.Module` del pipeline. El UNet (`UNet2DConditionModel`) y el VAE (`AutoencoderKL`) declaran soporte group offload. Los encoders CLIP de Transformers no exponen ese método especializado; la ruta del pipeline les aplica el helper genérico de hooks. Esto verifica la disponibilidad estática de la ruta de código, **no** que todos los hooks funcionen con los pesos SDXL concretos.

En Diffusers 0.36.0, el helper rechaza group offload si ya hay hooks `AlignDevicesHook` o `CpuOffload` instalados por Accelerate. Por tanto, group offload y sequential CPU offload no se combinan en el mismo componente. La documentación también advierte que group offload puede fallar si la ejecución del modelo hace conversiones de dispositivo dependientes de los pesos. No se inicializó un pipeline SDXL en CUDA ni se verificaron esos hooks en runtime.

Referencias primarias consultadas: [Diffusers 0.36.0: optimización de memoria](https://huggingface.co/docs/diffusers/v0.36.0/en/optimization/memory), [API de StableDiffusionXLPipeline 0.36.0](https://huggingface.co/docs/diffusers/v0.36.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl), y código instalado de Diffusers/Accelerate en este entorno.

La ruta group offload a disco crea subdirectorios y escribe grupos de tensores con `safetensors.torch.save_file`; durante la lectura vuelve a cargar cada grupo desde esos archivos. Es E/S mutable directa y no usa leases, publicación atómica, UUID de workspace ni cuotas de `PizarraStorage`.

### Carga inicial desde el checkpoint monolítico

En Diffusers 0.36.0, `from_single_file` llama a `load_single_file_checkpoint`, que devuelve un state dict completo. Para safetensors, `disable_mmap=False` usa `safetensors.torch.load_file`; `low_cpu_mem_usage=True` crea el modelo con pesos meta y materializa pesos con las utilidades de carga de Diffusers/Accelerate. Eso reduce copias de la inicialización convencional, pero no convierte el checkpoint monolítico en carga por shards/componentes: el state dict sigue representando el checkpoint durante la conversión y el despacho de pesos.

Group offload se aplica después de construir los componentes del pipeline. Puede mover grupos ya materializados a su directorio durante la activación de hooks y no elimina el pico inicial de lectura/conversión. Tampoco se afirmó que group offload sea compatible con Juggernaut XL sin una ejecución real.

## Arquitectura del adaptador

`core/pizarra/inference_adapter.py` contiene el primer adaptador:

1. Acepta solo checkpoints SDXL que estén en el catálogo local aprobado.
2. Exige PIZARRA habilitada y valida el montaje/identidad del volumen mediante `PizarraStorage`.
3. Consulta `core.pizarra.memory_budget.inspect_budget()` y el planificador SDXL. Aplica el presupuesto objetivo RAM/GPU de PIZARRA y el mínimo efectivo de memoria disponible considerando cgroup cuando se detecta.
4. Selecciona `standard` cuando hay margen para colocar el pipeline; selecciona `cpu_offload` cuando el plan SDXL admite carga de bajo uso de RAM y sequential CPU offload. FP16 y CUDA siguen siendo responsabilidad de Diffusers; PyTorch ejecuta las operaciones tensoriales.
5. Bloquea `disk_offload_experimental` hasta que PIZARRA ofrezca una concesión temporal con cuota estricta para escrituras externas. No sustituye módulos Diffusers por Tensor Store.
6. Usa `PizarraMetrics` para medir el proceso y CUDA, y limpia el pipeline antes de devolver éxito o error.

El worker escribe una línea JSON de progreso antes de cargar pesos y antes de inferir. Si muere durante la carga, el padre puede informar la última fase, modelo, arquitectura, estrategia y presupuesto inicial que alcanzó a registrar. El proceso padre captura stdout/stderr, muestrea el árbol de procesos y registra timeout/señal/OOM posible. El muestreo del árbol es de mejor esfuerzo; el scope systemd aplica `MemoryHigh`, `MemoryMax` y `MemorySwapMax=0`, y `timeout` limita duración.

La protección contiene el worker bajo systemd/cgroups cuando Fedora permite crear el scope. No es una sandbox ni puede proteger frente a fallos de kernel/controlador, OOM global o una terminación de systemd fuera del control del proceso supervisor. El monitor del padre puede no alcanzar a muestrear todos los procesos del scope; el RSS del worker y VRAM de PyTorch se miden dentro del proceso mientras sigue vivo.

## Estrategias y política de admisión

| Estrategia | Estado | Comportamiento |
| --- | --- | --- |
| `standard` | Implementada en el adaptador | Usa el perfil estándar de Diffusers cuando las estimaciones RAM/VRAM y el margen de PIZARRA lo admiten. |
| `cpu_offload` | Implementada en el adaptador | Usa FP16 en CUDA con sequential CPU offload cuando la capacidad y la configuración local lo admiten. Reduce VRAM y puede aumentar mucho la latencia. |
| `disk_offload_experimental` | Bloqueada | Aunque Diffusers 0.36.0 expone `offload_to_disk_path`, PIZARRA no puede imponer cuota por escritura en esa ruta mutable. |

No se promete soporte para cualquier checkpoint SDXL. La arquitectura debe identificarse como SDXL y el plan debe admitir una estrategia antes de abrir los pesos. Sin margen de memoria o sin configuración de CPU offload cuando es necesaria, el adaptador falla antes de la carga.

## Almacenamiento PIZARRA y límites de seguridad

El adaptador solo usa las interfaces `validate()` y `stats()` de `PizarraStorage`. Con ello confirma volumen, montaje y cuota, pero no usa su HDD para pesos. En este spike:

- Bytes temporales de offload PIZARRA: `0`.
- Escrituras directas de Diffusers en PIZARRA: deshabilitadas.
- Bytes/segundos de lectura y escritura PIZARRA: delta de los contadores de la API de almacenamiento durante el intento. Para esta ruta sin escrituras se espera cero; si las métricas están deshabilitadas, se informa `null`.
- E/S física total del HDD: no atribuible desde `PizarraStorage` y no se presenta como métrica del experimento.
- Checkpoint Forge: solo lectura por Diffusers. No se copia, modifica ni mueve.

No se crea workspace persistente ni se deja un duplicado de pesos. La salida manual SDXL puede guardarse en `outputs/` como en el sondeo anterior. Los registros no incluyen el prompt ni rutas completas. El offload a disco solo podrá reconsiderarse con una API de almacenamiento que proporcione un directorio temporal privado y aislamiento efectivo de cuota mientras el backend escribe, además de validar limpieza tras SIGKILL.

## Métricas por ejecución

El registro JSON incluye:

- ID de ejecución, checkpoint local (nombre y tamaño) y arquitectura.
- Estrategia/perfil, RAM inicial disponible, RSS inicial/final y máximo muestreado.
- VRAM libre inicial/final, memoria PyTorch asignada/reservada, máximo muestreado y high-water mark del allocator cuando existe.
- Duración de carga, inferencia+decodificación, limpieza y duración total del worker.
- Delta de bytes/tiempo de la API PIZARRA, tamaño temporal de offload y disponibilidad/cuota del volumen.
- Estado y categoría/tipo del fallo.

RSS y VRAM muestreados no son picos instantáneos. El high-water mark de PyTorch es de vida del proceso y no se reinicia. Diffusers devuelve imágenes PIL decodificadas en la llamada del pipeline; el tiempo de decodificación separado se informa `null`, no se estima. La lectura inicial del checkpoint y las operaciones tensoriales no son E/S PIZARRA. El supervisor añade mediciones de RSS del árbol de procesos, que son separadas del RSS del worker.

## Mediciones y verificación

El entorno de desarrollo no reproduce el equipo destino: `CUDA=False`, PIZARRA deshabilitada, RAM total `16,369,364,992` bytes, RAM disponible `8,065,994,752` bytes en la última lectura y sin límite cgroup v2 visible. La instalación contiene las versiones indicadas arriba. No hay mediciones de carga, inferencia, VRAM, HDD PIZARRA ni tiempo real de Juggernaut XL.

Las pruebas verifican el planificador, llamadas de backend simuladas, selección CPU offload, rechazo de volumen o memoria insuficiente, bloqueo de group disk offload, captura de fallos y limpieza, métricas de proceso/CUDA simuladas, muestreo y timeout del supervisor, y las garantías de montaje/cuota/symlink de almacenamiento existentes. **No cargan un checkpoint, no inicializan CUDA y no generan imágenes.**

## Prueba manual en Fedora

1. Configura PIZARRA con `BUTTERVISION_PIZARRA_ENABLED=true`, `BUTTERVISION_PIZARRA_ROOT` apuntando al HDD montado y, si está configurado, `BUTTERVISION_PIZARRA_VOLUME_UUID`. Conserva la cuota y reserva de espacio existentes. `PizarraStorage.validate()` debe aceptar el montaje y la identidad; una carpeta ordinaria no es suficiente.
2. Comprueba que Juggernaut XL v9 y su configuración SDXL completa ya están registrados localmente, `enable_cpu_offload` está activo si la estrategia lo requiere, y que `nvidia-smi`, `free -h`, `zramctl` y el espacio del HDD muestran recursos disponibles. No copies ni descargues archivos.
3. Revisa el comando sin cargar modelos: `venv/bin/python scripts/sdxl_isolated_probe.py`.
4. Opta por la ruta del adaptador: `venv/bin/python scripts/sdxl_isolated_probe.py --execute --model RUTA_LOCAL_REGISTRADA --pizarra-adapter --pizarra-strategy AUTO --memory-max-gib 10 --timeout-seconds 3600`. Solo un worker corre; la resolución es 512×512, batch 1, 10 pasos, seed `230231` y un prompt corto fijo. No combines el adaptador PIZARRA con `--strategy DISK_OFFLOAD`.
5. Conserva la salida JSON del terminal (puedes redirigirla a una ubicación local autorizada), el journal del scope, `nvidia-smi`, `free -h`/`zramctl`, el consumo del montaje PIZARRA y el estado del API. Registra la fase de fallo, uso máximo muestreado, tiempos y código del proceso. La ejecución puede guardar un PNG en `outputs/`; el sondeo es opt-in.
6. Si la selección o carga es rechazada, conserva el presupuesto y la última fase. Si el worker muere con señal/OOM, el informe del padre incluye progreso previo y muestras de su árbol de procesos cuando fueron observables. No relajes límites para forzar la carga.

Un resultado `completed` en una prueba real de Fedora es necesario antes de declarar Juggernaut XL compatible. Este spike no satisface todavía esa validación.

## Pruebas automatizadas

Resultados del entorno de desarrollo, sin CUDA y sin cargar checkpoints:

- `venv/bin/python -m unittest discover -s tests -v`: **150 pruebas aprobadas**. Incluye el adaptador nuevo, selección de estrategias, preflight por memoria, rechazo del volumen y del offload HDD, errores/limpieza simulados, captura y timeout del worker, y los tests existentes de motores, catálogo, almacenamiento, scheduler, métricas y Tensor Store.
- `npm test -- --run` desde `web/`: **26 pruebas aprobadas**.
- `npm run build` desde `web/`: completó correctamente con Vite 8.3.5.
- `venv/bin/python -m compileall -q api core scripts tests`: completó correctamente.
- `git diff --check`: sin errores de whitespace.

Las llamadas de generación en pruebas usan componentes simulados. No se generaron imágenes ni se ejecutó inferencia SDXL. Las pruebas validan política y manejo de fallos en mocks; no sustituyen la medición real en Fedora/GTX 1650.

## Pendientes para v0.4

- Probar Juggernaut XL real con la GTX 1650 y registrar carga/inferencia/decodificación observadas.
- Diseñar una concesión de directorio temporal PIZARRA con aislamiento de escritura, cuota aplicada en tiempo real, identidad de montaje fijada y limpieza tras fallo abrupto.
- Verificar group offload para todos los componentes SDXL reales y cuantificar RAM, VRAM, HDD y latencia antes de habilitarlo.
- Decidir una política de persistencia/retención de registros de experimentos sin almacenar prompts, imágenes privadas ni pesos.
