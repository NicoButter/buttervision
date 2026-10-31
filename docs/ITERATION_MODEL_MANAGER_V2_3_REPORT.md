# Buttervision — Model Manager v2.3

Fecha: 31 de octubre de 2026. Rama: `main`.

## Resultado y alcance

El catálogo identifica SD 1.5, SDXL y arquitectura desconocida sin inferirla del nombre ni del tamaño. Se añadió un backend SDXL separado, gestión de LoRA y controles de catálogo en Crear imagen. CyberRealistic conserva el backend SD 1.5, sus parámetros y su política de precisión anteriores. La selección de un checkpoint desconocido, no registrado o sin la configuración local SDXL requerida se rechaza antes de liberar el modelo válido.

**SDXL y Juggernaut XL v9 no se han ejecutado en la GTX 1650.** El código y las pruebas simuladas no demuestran que quepan o sean estables en 4 GB. Se requiere la validación manual descrita más abajo.

## Catálogo y rutas locales

- Los encabezados safetensors se inspeccionan con `safe_open`: claves, metadatos y formas, sin cargar todos los tensores ni usar GPU. Para checkpoints se comprueban componentes UNet, VAE y encoder de texto, la dimensión de atención cruzada y la presencia del segundo encoder. Para carpetas Diffusers se comprueban la clase y configuración del UNet y, en SDXL, los dos encoders y tokenizadores.
- Los LoRA se clasifican por metadatos de entrenamiento y/o formas de sus tensores de atención y claves del segundo encoder. Si faltan indicios suficientes o son contradictorios se muestran como desconocidos y no se pueden activar. Las firmas son conservadoras: algunos formatos o variantes válidos pueden requerir soporte futuro y la inspección no verifica la integridad de todos los pesos. Los archivos `.ckpt` se muestran como desconocidos; no se deserializan mediante pickle.
- Checkpoints predeterminados: `models/stable-diffusion`; LoRA: `models/lora` y `loras`. Las rutas adicionales las registra el operador mediante `BUTTERVISION_CHECKPOINT_DIRS` y `BUTTERVISION_LORA_DIRS`, con directorios separados por `:` en Linux (`os.pathsep`). Se exploran los checkpoints directamente contenidos en cada raíz y los LoRA recursivamente.
- Sólo se aceptan selecciones que figuren en esas raíces. Las rutas se normalizan, se comprueba que existan y se rechazan enlaces simbólicos que escapen de la raíz; no hay endpoint para añadir rutas arbitrarias remotamente. La raíz del filesystem no se acepta. Los archivos no se copian. Cambiar estas variables requiere reiniciar el servidor; el botón de actualización vuelve a inspeccionar las raíces registradas.
- El motor sólo usa fuentes locales. Se fuerza `local_files_only` en las cargas existentes relacionadas y el arranque heredado de Gradio ya no descarga automáticamente el checkpoint. Las funciones explícitas de descarga del administrador, para uso independiente, no se invocan desde la API de generación.

Se inspeccionó, sin cargar pesos, el CyberRealistic instalado: dimensión de atención 768, identificado como SD 1.5. También se inspeccionó el archivo existente `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` de Forge: dimensión 2048, segundo encoder e identificación SDXL. Se verificó que la selección de ese archivo, con el directorio de Forge y su configuración SDXL registrados, crea el manager sin cargar su pipeline. La configuración local de Forge contiene ambos tokenizadores y encoders. Esto comprueba el catálogo y la configuración, no la carga completa ni la generación.

## Motor SDXL, LoRA y concurrencia

- `StableDiffusionXLPipeline` es independiente de `StableDiffusionPipeline`. Para archivos individuales requiere una configuración SDXL local completa, obtenida de `BUTTERVISION_SDXL_CONFIG`, una carpeta junto al checkpoint o una caché SDXL existente. Se pasa como ruta local explícita para evitar el fallback de descarga de Diffusers; nunca se emplea la configuración SD 1.5 para SDXL.
- Ambos tokenizadores y encoders se usan por separado, con recorte y advertencia del límite de cada uno. El backend pasa `prompt`/`prompt_2` y sus negativos al pipeline SDXL.
- SDXL usa FP16 en CUDA y FP32 en CPU, attention slicing, VAE slicing y tiling. Respeta `config.model_config.enable_cpu_offload`: con CUDA y offload activo usa offload secuencial; con offload desactivado mueve el pipeline al dispositivo. No se modifica la precisión de CyberRealistic ni se activa xFormers.
- Antes de la primera carga SDXL se exige una reserva conservadora de RAM disponible: dos veces el tamaño de los pesos más 2 GiB en CUDA; cuatro veces más 2 GiB en CPU. Se considera el límite cgroup v2 cuando está disponible. Con offload se exigen al menos 1,5 GiB de VRAM libre; sin él se exige el tamaño de los pesos más 2 GiB. Estos umbrales son aproximaciones conservadoras, pueden rechazar equipos capaces y no garantizan evitar OOM. La comprobación impide iniciar cargas que no cumplen el presupuesto.
- El perfil con menos de 8 GiB de VRAM expone dimensiones SDXL de 512, 640 y 768 px; desde 8 GiB expone también 896 y 1024. Son límites de la implementación, no resoluciones validadas en hardware. SD 1.5 conserva sus dimensiones de 256 a 512 y batch 1. Las resoluciones SDXL solicitadas fuera del perfil se rechazan, sin convertirlas silenciosamente a 512.
- El mutex de Engine v2.2 protege cambios de modelo, activación/desactivación/peso LoRA y generación. Si está ocupado se responde HTTP 409. El lanzador principal también fija un worker. Sigue siendo necesario ejecutar una sola instancia de inferencia por GPU: el mutex no coordina procesos independientes.
- Activar un LoRA compatible carga el pipeline base si todavía no está cargado. Los adaptadores se cargan individualmente y `set_adapters` recibe siempre la lista completa y sus pesos; se comprueba que queden activos. Una modificación fallida descarta el pipeline potencialmente inconsistente, libera referencias y permite reintentar. El estado representa adaptadores realmente cargados, es volátil y se limpia al cambiar de checkpoint o de pipeline. Si un archivo LoRA cargado desaparece, continúa visible y puede desactivarse.
- El LoRA heredado `mi_cara`, cuando existe, sólo se activa automáticamente si su familia se puede verificar como SD 1.5. Su peso no se sobrescribe al activar otros adaptadores. La escala de algunos componentes no cubiertos por `set_adapters` depende de Diffusers/PEFT; debe validarse con cada LoRA. Ver [documentación oficial de adaptadores](https://huggingface.co/docs/diffusers/main/using-diffusers/loading_adapters).
- Face Reference continúa siendo SD 1.5 + IP-Adapter Face. Al seleccionar SDXL se bloquea antes de crear el trabajo, con una explicación en interfaz y API. Al usar Face Reference con SD 1.5 se libera el pipeline txt2img y se desactivan sus LoRA; la interfaz lo advierte.

## API e interfaz

`GET /api/models` incluye `family`; `GET /api/engine` expone modelo, familia, estado, resoluciones y LoRA activos; `GET /api/loras` lista compatibilidad y estado; `POST /api/loras/active` recibe `path`, `enabled` y `weight` entre 0 y 2. El selector de checkpoints, la actualización del catálogo y los controles LoRA se deshabilitan mientras el motor está ocupado. La pantalla siempre muestra modelo, familia, número y nombres de LoRA activos y estado. Los errores de validación se devuelven con HTTP 400 y los conflictos de ocupación con 409.

## Pruebas y revisión de regresiones

- `venv/bin/python -m unittest discover -s tests -v`: **60 pruebas aprobadas**. Incluye las pruebas existentes, safetensors sintéticos, identificación de familia, rutas registradas y escapes por symlink, selección y rechazo sin destruir el estado, inicio en SDXL, cambio SD 1.5 → SDXL → SD 1.5, ambos encoders, memoria insuficiente antes de cargar, exclusión, múltiples LoRA y pesos, estado durante inferencia, archivo LoRA ausente, fallos y liberación de referencias.
- `cd web && npm test -- --reporter=dot`: **16 pruebas aprobadas**, incluidos estado del motor, familia, LoRA y bloqueo de controles.
- `cd web && npm run build`: aprobado, 22 módulos transformados.
- `python -m compileall -q api core main.py tests`, `bash -n run.sh` y `git diff --check`: aprobados.

Las pruebas automatizadas usan factorías y llamadas de inferencia simuladas; bloquean Requests y la descarga del administrador. No se ejecutaron inferencias ni se descargaron o instalaron modelos. La lectura de cabeceras y la selección del Juggernaut existente fueron verificaciones locales adicionales. Las pruebas de v2.2 se adaptaron para que sus mocks suministren la familia y la validación de catálogo nuevas, conservando sus comprobaciones originales. TestClient se ejecutó fuera de la restricción del sandbox que bloquea su comunicación entre hilos.

La política de SDXL sigue siendo experimental: la RAM real disponible, otros procesos, los picos de carga, el allocator, VAE y las versiones de Diffusers/Accelerate/PEFT pueden modificar el consumo y la compatibilidad. Un SIGKILL sigue sin poder capturarse. La selección basada en estructura no garantiza que un checkpoint incompleto o dañado cargue correctamente.

## Procedimiento manual pendiente con Juggernaut XL v9 en Fedora

No se realizó generación real durante esta iteración. Los siguientes pasos son para la validación posterior del operador, usando exclusivamente los archivos ya existentes:

1. Cerrar Forge y otras aplicaciones de inferencia; mantener una sola instancia de Buttervision. Comprobar `nvidia-smi`, `free -h`, versiones de dependencias y los logs de Uvicorn. Observar también `nvidia-smi -l 1` y `vmstat 1` durante la prueba.
2. Registrar las rutas existentes, sin copiar archivos. En este equipo pueden usarse:

   ```bash
   export BUTTERVISION_CHECKPOINT_DIRS="$HOME/stable-diffusion-webui-forge/models/Stable-diffusion"
   export BUTTERVISION_LORA_DIRS="$HOME/stable-diffusion-webui-forge/models/Lora"
   export BUTTERVISION_SDXL_CONFIG="$HOME/stable-diffusion-webui-forge/backend/huggingface/stabilityai/stable-diffusion-xl-base-1.0"
   bash run.sh
   ```

   En otro equipo, ajustar las rutas a sus carpetas locales. Para varias raíces, separarlas con `:`. La configuración SDXL debe contener `model_index.json`, configuraciones de UNet, VAE, scheduler, ambos encoders y archivos de ambos tokenizadores. No se descargará si falta.
3. En Crear imagen, comprobar primero CyberRealistic / SD 1.5. En modo experto, actualizar el catálogo y seleccionar Juggernaut XL v9. Debe indicarse SDXL y debe bloquearse Face Reference. Un archivo desconocido o incompatible no debe cambiar el modelo seleccionado.
4. Consultar la RAM disponible y el mensaje del motor. Si el presupuesto conservador rechaza la carga, registrar la limitación y detener la prueba; no desactivar el offload para forzarla en la GTX 1650. Si el presupuesto la admite, ejecutar una generación de prueba autorizada con batch 1 y las dimensiones disponibles, registrando picos de RAM y VRAM, duración, errores y calidad. La admisión no garantiza éxito.
5. Si la carga funciona, probar dos LoRA locales de familia SDXL reconocida, ajustar un peso y desactivar uno. Verificar que el otro siga activo y que un LoRA SD 1.5 aparezca incompatible. Durante una generación, confirmar que cambiar checkpoint o LoRA devuelve 409 y que los controles están bloqueados.
6. Volver a CyberRealistic, comprobar liberación de SDXL y desactivación de los LoRA previos; ejecutar una prueba autorizada SD 1.5 y, si corresponde, Face Reference. Observar la memoria antes y después de cada transición. Un prompt largo debe advertir truncamiento, sin imprimir el texto descartado.
7. Ante `Killed` o fallo de memoria, registrar la hora y consultar `journalctl -k -b`, `journalctl -b -u systemd-oomd` y `oomctl`, además de los logs de la API. No atribuirlo al OOM killer sin evidencia. Registrar los resultados reales antes de declarar estable SDXL en 4 GB.

Referencias: [carga de checkpoints locales Diffusers](https://huggingface.co/docs/diffusers/using-diffusers/other-formats), [API de LoRA](https://huggingface.co/docs/diffusers/api/loaders/lora).
