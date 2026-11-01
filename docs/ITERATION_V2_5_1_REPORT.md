# Buttervision v2.5.1 — Popover PIZARRA y diagnóstico de Juggernaut XL

Fecha: 2026-11-01. Rama: `main`.

## Resultado y alcance

Se corrigió el panel PIZARRA. La investigación identifica los bloqueos de selección, admisión y carga de SDXL, y recomienda evaluar primero Forge local como vía de menor implementación para Juggernaut. No se integró ninguno de los motores propuestos ni se cambiaron los umbrales de memoria.

**Juggernaut XL sigue sin estar validado en la GTX 1650.** Esta entrega no incluye una generación real. La consulta final con acceso al driver sí confirma GTX 1650/CUDA y reproduce el rechazo por RAM antes de abrir pesos; no prueba carga ni inferencia en esa GPU.

## Corrección del popover

- `web/src/components/PizarraStatus.jsx`: botón con `aria-expanded`, `aria-haspopup=dialog` y referencia al diálogo abierto; diálogo no modal con nombre accesible y botón de cierre. Al abrir recibe foco. Escape y el cierre explícito devuelven el foco al botón. El clic exterior y la salida de foco cierran sin robar el foco al control de destino; Tab puede salir libremente.
- `web/src/app/App.jsx`: pasa la página actual mediante `navigationKey`; el panel se cierra al cambiar de página, incluso si la navegación es programática.
- `web/src/styles.css`: el panel se ancla debajo del navbar completo, también cuando este ocupa varias filas. Su ancho respeta el viewport; altura limitada por pantalla, navbar y 480 px, con scroll interno. Ajusta su altura al redimensionar o desplazar la página. No coloca una capa sobre el selector de modelos ni un fondo modal que intercepte sus clics.
- Se eliminó una de las dos líneas idénticas sobre offload. Se conservaron los indicadores independientes de disponibilidad y E/S confirmada, incluido movimiento reducido. La actividad caduca a los tres segundos aunque no llegue una nueva respuesta.
- El sondeo conserva `light=true` (sin recorrido del árbol del volumen): 30 segundos en reposo y 1 segundo si se informa actividad reciente. Los detalles completos se consultan al abrir si no hay una respuesta completa de menos de 30 segundos, o al pulsar Actualizar. Se comparten peticiones en curso; no hay nuevas consultas por cerrar o navegar. Un sondeo ligero no elimina los detalles ya obtenidos. Cerrar mientras se espera el primer sondeo cancela la petición de detalles aún no iniciada.

Pruebas nuevas en `web/src/components/PizarraStatus.test.jsx` y una prueba de integración en `web/src/app/App.test.jsx`: apertura/nombre/atributos ARIA, foco, Escape, cierre explícito, clic exterior sobre selector utilizable, salida de foco con teclado, navegación, caché/reapertura, petición inicial pendiente, altura disponible y caducidad de actividad. Son pruebas DOM con viewport simulado; no se realizó una inspección visual en navegador real ni en dispositivos físicos.

## Diagnóstico reproducido sin cargar pesos

Se revisaron `api/generation.py`, `core/sdxl_memory.py`, `core/sdxl_pipeline.py`, `core/pizarra/inference_adapter.py` y [PIZARRA v0.3](PIZARRA_V0_3_INFERENCE_REPORT.md). Se invocaron únicamente inspección de recursos, `inspect_load_budget()`, `check_budget()` y validación de configuración. Del checkpoint se leyeron los primeros **364.728 bytes** de cabecera/índice, sin materializar tensores.

Versiones locales: PyTorch `2.5.1+cu121`, Diffusers `0.36.0`, Accelerate `1.12.0`, safetensors `0.7.0`. El primer sondeo dentro del aislamiento devolvía CUDA inaccesible. Se repitió la consulta fuera del aislamiento: `nvidia-smi` confirmó **NVIDIA GeForce GTX 1650, 4096 MiB, 3699 MiB libres, driver 615.78.08** y PyTorch confirmó `cuda_available=true`. El diagnóstico definitivo de admisión usa este segundo sondeo, sin cargar modelos; CUDA inaccesible dentro del aislamiento no se atribuye a un fallo del driver físico.

| Dato observado o cálculo del código actual | Bytes | GiB |
| --- | ---: | ---: |
| RAM física visible | 16.369.369.088 | 15,245 |
| RAM disponible en el sondeo con acceso al driver | 7.489.413.120 | 6,975 |
| Checkpoint Juggernaut XL v9 | 7.105.348.188 | 6,617 |
| RAM mínima LOW_RAM / LOW_VRAM: checkpoint + 2 GiB | 9.252.831.836 | 8,617 |
| RAM mínima STANDARD: 2 × checkpoint + 2 GiB | 16.358.180.024 | 15,235 |
| VRAM disponible medida por PyTorch | 3.821.076.480 | 3,559 |
| VRAM libre mínima de admisión para offload secuencial | 1.610.612.736 | 1,500 |
| Espacio temporal mínimo calculado para DISK_OFFLOAD | 9.955.427.059 | 9,272 |

Faltaban **1.763.418.716 bytes de RAM disponible (1,642 GiB)** para la regla de bajo uso de RAM. La VRAM libre sí supera 1,5 GiB. No se observó un límite cgroup finito en este sondeo; el código considera el menor margen visible entre RAM disponible y jerarquía cgroup cuando existe. RSS del proceso de inspección: 640.118.784 bytes. La RAM libre cambia con otras aplicaciones; estas cifras son una captura, no constantes del hardware. La diferencia entre VRAM informada por NVIDIA y PyTorch corresponde a consultas en instantes distintos, con el contexto CUDA de inspección ya inicializado en la segunda.

La salida efectiva fue:

```json
{
  "requested_profile": "AUTO",
  "selected_profile": null,
  "admitted": false,
  "load_phase": "rejected",
  "cuda_available": true,
  "cpu_offload_enabled": false,
  "disk_offload_enabled": false,
  "reasons": [
    "La RAM disponible no alcanza el mínimo estimado para cargar los pesos en mmap con margen."
  ]
}
```

`check_budget()` lanzó `ValueError` con esa única razón en el host con CUDA. Las capacidades Accelerate, carga de bajo uso de RAM, safetensors y group offload a disco sí están presentes. **No hay estrategia seleccionada en este rechazo**; LOW_RAM es un candidato que no supera la admisión. Tener la opción CPU offload habilitada no significa que sus hooks se hayan instalado. El campo auxiliar `required_vram_bytes` muestra el cálculo STANDARD cuando no se selecciona ningún perfil; no debe interpretarse como el consumo medido de LOW_RAM.

En la GTX 1650, si CUDA es accesible, quedan al menos 1,5 GiB de VRAM libre, las capacidades están presentes y hay entre 8,617 y 15,235 GiB de RAM disponible, el `AUTO` actual puede escoger LOW_RAM con offload secuencial a CPU. Con RAM por debajo de 8,617 GiB tampoco se admite LOW_VRAM: ambos comparten la regla de carga inicial. El umbral de 1,5 GiB es una condición del planificador, no una garantía de que atención y VAE quepan durante una generación.

## Hasta dónde avanza realmente

1. **Descubrimiento**: catálogo, tamaño y arquitectura por cabecera; no implica selección ni carga.
2. **Selección**: `GenerationService.set_model()` exige `local_sdxl_config()` antes de sustituir el gestor anterior. En la configuración efectiva inspeccionada faltaba un directorio SDXL completo, con el error `Falta una configuración SDXL completa y local; configura BUTTERVISION_SDXL_CONFIG. No se descargará automáticamente.` En ese caso se rechaza la selección y se conserva el modelo anterior, antes de evaluar el trabajo.
3. **Configuración ya disponible**: se probó, solo en un proceso de inspección, `BUTTERVISION_SDXL_CONFIG=/home/lordcommander/stable-diffusion-webui-forge/backend/huggingface/stabilityai/stable-diffusion-xl-base-1.0`. `local_sdxl_config()` la acepta. Se puede declarar esa ruta existente en `.env.local` para resolver la selección sin copiar ni descargar configuraciones. No se modificó la configuración persistente del usuario. Esto no resuelve la RAM ni prueba la carga de esos componentes.
4. **Solicitud normal**: cuando SDXL está seleccionado y aún no hay pipeline, `create_job()` llama a `check_budget()` antes de crear/enviar un trabajo al worker o crear su workspace PIZARRA. El rechazo descrito termina aquí; el API lo traduce a HTTP 400. No se alcanza `from_single_file()`, instalación de hooks, denoising ni decodificación.
5. **Si superara ese control**: `SDXLPipeline._build_pipeline()` repite el preflight, carga con `from_single_file(... low_cpu_mem_usage=True, disable_mmap=False, local_files_only=True)` para LOW_RAM y luego activa offload. El código instalado de Diffusers obtiene primero un state dict del archivo completo y construye los componentes. El mapeo de memoria reduce copias y permite paginación, pero no garantiza un RSS pequeño durante conversión/asignación. No se ha medido ese pico para Juggernaut en la GTX 1650.

La API puede conservar un diagnóstico anterior en `self.load_budget`, y el manejador de error vuelve a inspeccionar recursos después del fallo. Esas cifras pueden diferir ligeramente del instante exacto de admisión. La fase genérica `loading` de un futuro intento tampoco probaría que todos los componentes se cargaron; harían falta logs y mediciones por fase.

## Por qué PIZARRA no desbloquea la generación normal

La ruta normal usa `GenerationService` → `SDXLPipeline`; no importa ni instancia `PizarraInferenceAdapter`. PIZARRA registra trabajos, espacios y métricas por las interfaces existentes, pero no sustituye los módulos tensoriales de Diffusers. El adaptador v0.3 se invoca exclusivamente desde `scripts/sdxl_isolated_probe.py --pizarra-adapter`, como experimento aislado con límites de memoria y limpieza propia.

Ese adaptador valida el volumen y aplica además el presupuesto observacional de PIZARRA al mismo planificador SDXL. No reduce el tamaño del checkpoint ni puede rescatar un plan rechazado por RAM. Su estrategia experimental de HDD se rechaza expresamente. Conectarlo al botón normal hoy añadiría otro preflight/medición y alteraría el ciclo de vida de trabajos sin aportar el mecanismo de carga que falta.

Los impedimentos concretos del HDD son:

- `AUTO` nunca elige DISK_OFFLOAD. La ruta experimental exige solicitud y autorización explícitas, CUDA, API de group offload, RAM de carga suficiente, VRAM mínima, directorio válido y espacio libre. En este sondeo no existe `BUTTERVISION_SDXL_DISK_OFFLOAD_DIR` y tampoco se cumple la RAM de carga.
- `enable_group_offload()` se ejecuta **después** de construir el pipeline. No elimina el pico de conversión/carga inicial; incluso su admisión actual exige checkpoint + 2 GiB de RAM.
- El adaptador PIZARRA rechaza `DISK_OFFLOAD_EXPERIMENTAL`: los hooks de Diffusers escriben/leen archivos directamente en un directorio mutable, mientras que PizarraStorage ofrece leases, registros inmutables y cuotas comprobadas por su propia API. Una ruta dentro de `/mnt/pizarra` permitiría eludir ese control. El directorio temporal privado del sondeo ordinario no equivale a una cuota PIZARRA impuesta por el sistema de archivos.
- Group offload y hooks de offload CPU de Accelerate no deben superponerse en los mismos componentes. La presencia de la firma de la API no demuestra compatibilidad de UNet, ambos CLIP y VAE concretos. Ver [implementación Diffusers 0.36.0](https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/hooks/group_offloading.py).
- Tensor Store no implementa las operaciones completas de UNet/atención/CLIP/VAE ni está conectado al bucle de denoising. El HDD disponible y los bytes de metadatos escritos no demuestran offload de inferencia.

## Alternativa A — Forge local como backend

La instalación local inspeccionada corresponde al commit `dfdcbab685e57677014f05a3309b48cc87383167` de `lllyasviel/stable-diffusion-webui-forge`. Sus [opciones de arranque](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/dfdcbab685e57677014f05a3309b48cc87383167/modules/cmd_args.py) incluyen `--nowebui` para API sin interfaz, `--port`, `--skip-load-model-at-start`, `--no-download-sd-model` y desactivación de extensiones. No se arrancó Forge ni se editaron sus archivos.

La [API instalada](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/dfdcbab685e57677014f05a3309b48cc87383167/modules/api/api.py) ofrece listado de modelos, txt2img, progreso, interrupción y liberación del checkpoint cargado. txt2img devuelve imágenes codificadas y permite omitir su guardado por Forge. Su cola serializa las solicitudes. **El puerto predeterminado API-only de este checkout es 7861**, que colisiona con FastAPI de Buttervision: una prueba futura debe usar otro, por ejemplo 7862, limitado a loopback.

Propuesta de integración futura, todavía no implementada:

1. Mantener React, validaciones, catálogo, trabajos y resultados en Buttervision. Añadir un único backend SDXL configurable que hable desde FastAPI con la API Forge local; el navegador no accede directamente a Forge.
2. Resolver el checkpoint del catálogo contra el título/hash/ruta autorizada que publica Forge, sin permitir destinos HTTP ni rutas arbitrarias suministradas por el cliente. Usar el archivo existente en su ubicación actual.
3. Descargar de RAM/VRAM el pipeline nativo antes de delegar y mantener exclusión entre motores. El `flock` Buttervision actual no es adquirido por Forge: deben coordinarse desde un único supervisor y evitar solicitudes externas concurrentes. Mantener timeout, cancelación, límites del worker y recuperación tras desconexión.
4. Traducir prompt/negativo, seed, dimensiones y sampler admitidos; solicitar `save_images=false` y guardar las salidas en Buttervision. No persistir opciones globales de Forge para cambiar modelos. Deshabilitar inicialmente las funciones sin equivalencia validada, como Face Reference y LoRA de la ruta nativa.

La gestión [CPU/GPU de Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/dfdcbab685e57677014f05a3309b48cc87383167/backend/memory_management.py) dispone de carga parcial según memoria disponible. Es una base ya desarrollada para probar 4 GB. Sin embargo, su `backend/utils.py::load_torch_file()` también llama a `safetensors.torch.load_file()` para el archivo monolítico y `split_state_dict()` reparte sus claves. No garantiza evitar el pico de RAM del checkpoint ni constituye offload PIZARRA.

## Alternativa B — Carga incremental propia con Diffusers/PyTorch

Es técnicamente investigable, con trabajo sustancial de conversión y ciclo de vida. `low_cpu_mem_usage=True` por sí solo no implementa esta alternativa. [PyTorch documenta modelos meta y asignación de pesos](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html); para este archivo safetensors corresponde usar [acceso selectivo con `safe_open`](https://huggingface.co/docs/safetensors/en/index), no aplicar `torch.load(mmap=True)` a un formato distinto.

Diseño propuesto (inferencia de ingeniería, no capacidad existente): instanciar por configuración en `meta`; leer/asignar solo los tensores del componente o bloque activo, incluyendo los cambios de nombres y particiones necesarios. Ejecutar CLIP 1 y CLIP 2 por turnos conservando embeddings; liberar encoders antes del denoising; ejecutar UNet con descarga por submódulos/bloques; liberar UNet antes de VAE con tiling. Deben conservarse las dos tokenizaciones, pooled embeddings, time IDs, CFG, scheduler y políticas de precisión. Los hooks y referencias Python no pueden retener inadvertidamente todos los componentes.

Los [checkpoints por shards de Diffusers](https://huggingface.co/docs/diffusers/v0.36.0/en/optimization/memory) reducen el pico adicional al shard mayor, pero cargar primero Juggernaut entero y luego llamar a `save_pretrained()` no soluciona la conversión inicial. Una ruta sin duplicar el archivo requeriría el lector/conversor selectivo anterior; una conversión futura por shards exigiría autorización de almacenamiento y límites durante la propia conversión. [Accelerate documenta inicialización vacía y despacho](https://huggingface.co/docs/accelerate/v1.12.0/en/concept_guides/big_model_inference), cuya integración con módulos SDXL y hooks debe validarse, no asumirse.

La cabecera del archivo permite calcular el tamaño de pesos, sin ejecutar el modelo:

| Componente | GiB en el checkpoint | Tipo observado |
| --- | ---: | --- |
| UNet | 4,782 | FP16 |
| CLIP 1 | 0,229 | FP16 |
| CLIP 2 | 1,294 | FP16 |
| VAE | 0,312 | FP32 |

El UNet completo supera los 4 GiB de la GTX 1650, antes de activaciones. Mover componentes completos CPU→GPU no basta; hacen falta submódulos/bloques y presupuestos de activaciones. El VAE puede requerir FP32 por estabilidad. Dividir componentes reduce simultaneidad, pero no hace desaparecer el coste del UNet o del denoising.

Para usar PIZARRA en el futuro haría falta una concesión temporal por trabajo con cuota dura efectiva para escritores externos, identidad de montaje fijada, acceso limitado al área concedida, tratamiento de ENOSPC/desconexión y limpieza segura tras SIGKILL. Reservar bytes en Python o limitar RSS con cgroups no limita por sí solo cuánto escribe Diffusers al HDD. Debe probarse aislamiento de directorios/dispositivo y cuota del sistema de archivos o un broker que medie todas las escrituras; no bastan nombres de carpetas. Las transferencias repetidas por paso y los accesos del HDD aumentan mucho la latencia.

## Comparación y recomendación

Los siguientes son presupuestos orientativos de diseño para **un solo trabajo**, no mediciones de Juggernaut. No cambian la admisión del motor actual.

| Aspecto | A: Forge local | B: Carga incremental propia |
| --- | --- | --- |
| Viabilidad inmediata | Alta para construir un adaptador API; resultado en GTX 1650 pendiente | API base disponible, pero lector/conversor y fases SDXL pendientes |
| RAM estimada | Pesos de 6,617 GiB más runtime, conversión y buffers: planificar aproximadamente 8–12 GiB de proceso, más Buttervision y SO. No es un máximo garantizado | Con UNet residente: 4,782 GiB + buffer limitado de 0,25–0,5 GiB + runtime/activaciones; reservar inicialmente 7–10 GiB para el prototipo. Menos RAM con HDD requiere otro mecanismo aún no implementado |
| VRAM estimada | Objetivo encajar bajo 4 GiB mediante gestión parcial; pico real de atención/VAE sin medir | Bloque activo + activaciones + contexto CUDA deben caber en VRAM libre; UNet completo de 4,782 GiB no cabe. No hay garantía hasta medir |
| Complejidad | Moderada: contratos API, supervisor, bloqueo, cancelación y resultados | Alta: mapping de pesos, meta/materialización, precisión, fases, hooks y recuperación |
| Mantenimiento | Fijar versión de Forge y probar su contrato API; segundo entorno/proceso | Mantener compatibilidad Diffusers/PyTorch/Transformers y cada variante SDXL |
| PIZARRA HDD | No necesario para el primer ensayo; no aporta offload por esta integración | Posible evolución con cuotas/aislamiento efectivos y lector/hooks compatibles |

**Recomendación: priorizar A como próximo prototipo, después de una única prueba controlada en la GTX 1650.** No integrar A y B simultáneamente. Forge ya aporta API y gestión de VRAM; B conserva mayor control pero exige resolver mucho más código antes de la primera imagen verificable.

Antes de ese prototipo conviene comprobar en el equipo destino: acceso real a CUDA y VRAM libre, ausencia de otros motores cargados, RAM suficiente y configuración SDXL existente. La ruta de configuración Forge aceptada arriba permite retirar el bloqueo de selección de Buttervision sin descargas; no justifica bajar su presupuesto. Si la ruta nativa llega a admitirse, el sondeo aislado existente permite una validación futura con los límites actuales.

Para declarar funcional la vía elegida hace falta una ejecución real, en la GTX 1650, de Juggernaut identificado por hash, lote 1 y 512×512, inicialmente sin LoRA ni referencias. Registrar carga por etapas, RSS/VRAM pico, tiempo, resultado decodificado válido y limpieza; probar después error/cancelación y retorno al modelo anterior. La prueba debe excluir simultaneidad entre Forge y Buttervision y conservar margen para el SO. No se ejecutó en esta entrega.

## Tests y build

- `venv/bin/python -m unittest discover -s tests -q`: **154 pruebas aprobadas**; incluye admisión SDXL, rechazo antes de carga, estrategias, adaptador PIZARRA, seguridad de almacenamiento y regresiones existentes.
- `npm test` desde `web/`: **36 pruebas aprobadas en 2 archivos**, incluidas las 10 comprobaciones nuevas del popover y navegación.
- `npm run build` desde `web/`: compilación correcta con Vite 8.3.5.
- `git diff --check`: sin errores.
- Preflight de solo lectura repetido con acceso al driver: GTX 1650/CUDA confirmados, rechazo por RAM y falta de configuración local reproducidos; acceso a cabeceras y configuraciones, sin `load_pipeline()` ni generación.

## Riesgos y limitaciones pendientes

- Diagnóstico causal del código y preflight en la GTX 1650 confirmados; carga, generación, rendimiento y estabilidad del modelo en esa GPU pendientes.
- La configuración compartida de Forge pasó validación estructural, no una carga real de tokenizer/CLIP/UNet/VAE por Diffusers.
- Ni A ni B ofrecen todavía una garantía de RAM/VRAM o de offload al HDD. Forge también puede fallar por RAM, driver, precisión o extensiones. Sus inicializadores pueden crear cachés/estado: cualquier futura ejecución debe auditar sus rutas de escritura antes de cumplir una política estricta de instalación de solo lectura.
- Sin cuota externa efectiva y control de archivos abiertos tras desmontaje/fallo, PIZARRA no debe recibir `offload_to_disk_path` arbitrario.
- En el popover, las cifras técnicas completas son una instantánea de hasta 30 segundos; Actualizar fuerza una nueva. Disponibilidad y actividad continúan con el sondeo ligero independiente.
- No se descargaron modelos, no se copiaron checkpoints, no se modificaron archivos de Forge y no se generaron imágenes durante esta iteración.
