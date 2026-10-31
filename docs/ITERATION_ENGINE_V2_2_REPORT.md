# Buttervision — Engine v2.2: estabilidad y memoria

Fecha: 31 de octubre de 2026. Rama de entrega: `main`.

## Diagnóstico y límites de la evidencia

El cierre con `Killed` es compatible con SIGKILL por presión de RAM, OOM del kernel, límites de cgroup o systemd-oomd. No hay registros del kernel ni del servicio que permitan atribuirlo con certeza a alguno de ellos. Una excepción CUDA OOM por sí sola tampoco demuestra por qué Linux terminó Uvicorn. No se reproduce ni se declara solucionado el OOM en esta iteración: queda pendiente una prueba real en la GTX 1650 de destino.

El código sí permitía conservar SD1.5, IP-Adapter y CLIP Vision del modo facial al cargar txt2img. CLIP-H y el offload mantienen pesos en RAM, además del uso de VRAM; la superposición y los picos de construcción son una causa probable de presión excesiva. No se dispone de mediciones del incidente para cuantificarla.

## Fallos de concurrencia encontrados

- `get_service()` podía construir más de un servicio ante solicitudes simultáneas.
- Los trabajos adquirían el bloqueo dentro del hilo, después de aceptarse y registrarse. Una segunda solicitud recibía un trabajo que luego fallaba por ocupación.
- El cambio de checkpoint usaba otro bloqueo y podía liberar o cambiar componentes durante una inferencia.
- El registro y la lectura del estado de los trabajos no proporcionaban snapshots protegidos.

## Cambios implementados

- Inicialización del servicio protegida por un mutex; un fallo de construcción permite reintentar.
- Un mismo mutex protege la admisión de trabajos, inferencia, carga, liberación y cambio de checkpoint. Se reserva antes de crear el hilo y se libera al finalizar la limpieza, incluso ante errores. También se recupera si falla la creación del trabajo o el arranque del hilo.
- Se eligió **rechazo inmediato HTTP 409**, sin cola: aplica a ambos endpoints de generación y a `POST /api/models/active`, incluso si se solicita el checkpoint actual durante un trabajo. El rechazo no crea un trabajo fallido ni modifica el trabajo existente.
- Los lanzadores de API fijan `--workers 1`, evitando multiplicación accidental por `WEB_CONCURRENCY`. Los bloqueos son de proceso: debe ejecutarse una sola instancia de inferencia por GPU, sin otro servidor API ni Gradio/Forge generando a la vez.
- Los checkpoints deben resolverse localmente; cambiar a uno inexistente no modifica la selección válida. Repetir la selección actual conserva los componentes.
- Los trabajos conservan estado terminal, resultados y advertencias para consultas posteriores. Las actualizaciones y snapshots están protegidos; se eliminan fotos y prompts del payload al terminar. Este registro permanece en memoria durante la vida del proceso; no añade persistencia de trabajos tras reiniciar. El guardado existente de resultados y metadata en disco se conserva.

## Estrategia de liberación y reutilización

1. txt2img → Face Reference: se liberan los pipelines normales antes de construir el facial.
2. Face Reference → txt2img: se libera el pipeline facial, incluyendo los pesos IP-Adapter asociados al UNet y el encoder CLIP Vision auxiliar, antes de cargar SD normal.
3. La liberación retira los hooks de offload mediante la API de Diffusers y recorre recursivamente los módulos con Accelerate. Esto cubre versiones donde `remove_all_hooks()` sólo detecta hooks en las raíces de los componentes. Se eliminan las referencias de todos los propietarios, también las de pipelines que comparten componentes, antes de `gc.collect()` y de la limpieza explícita de caché CUDA.
4. No se mueve el pipeline completo a CUDA para limpiarlo. La retirada de hooks puede restaurar pesos a su dispositivo original y sigue necesitando RAM transitoria. `empty_cache()` no libera tensores aún referenciados ni garantiza que el proceso devuelva toda su RAM al sistema.
5. Los objetos parcialmente construidos, una vez devueltos por la factoría, quedan registrados para poder descartarlos si falla el resto de la carga. Los errores de inferencia eliminan los frames del traceback antes de limpiar los managers; no se guardan excepciones con tensores en el historial.
6. El encoder facial vuelve a CPU en un `finally`, también si falla el cálculo de embeddings. Los trabajos consecutivos del mismo tipo reutilizan los modelos; un error descarta el estado posiblemente incompleto y permite una carga limpia en el siguiente intento.

Las cargas de generación usan `local_files_only=True`. Se detectó en Diffusers 0.36.0 un fallback de `from_single_file()` que puede descargar configuraciones pese a ese argumento. Ahora se resuelve primero una carpeta local de configuración SD1.5 junto al checkpoint o en las cachés existentes del proyecto/Hugging Face y se pasa explícitamente como `config`. Si falta, la carga falla sin activar ese fallback. No se instalaron ni descargaron checkpoints, LoRA, configuraciones o modelos auxiliares. Las rutas antiguas de edición/video no se amplían ni se ejecutan.

## Diagnósticos y prompts

Los eventos `engine` registran inicio/fin de carga y liberación, inicio/fin de trabajo, cambio de checkpoint, pipeline, estado, tipo de error y errores de asignación. Cuando las métricas están disponibles incluyen RAM usada/disponible, RSS del proceso, VRAM usada/disponible del dispositivo y memoria CUDA asignada/reservada, en bytes. Se toman snapshots en transiciones, sin un monitor permanente. No se registran fotos, prompts completos, texto descartado por CLIP ni mensajes crudos de excepciones de inferencia. Los errores de trabajo devuelven un mensaje genérico o uno específico de memoria insuficiente.

Se usa el tokenizer ya cargado y el límite del encoder CLIP, incluyendo tokens especiales, para comprobar prompt positivo y negativo. Los textos largos se recortan antes de invocar Diffusers, evitando su advertencia que puede imprimir el fragmento descartado. La advertencia comprensible se conserva en `warnings` del trabajo y se muestra en la pantalla existente; no se promete procesar el texto excedente. No se incorpora otro encoder.

## Pruebas ejecutadas y resultados

| Verificación | Resultado |
| --- | --- |
| `venv/bin/python -m unittest discover -s tests -v` | 37 tests aprobados: 31 del motor y 6 regresiones existentes |
| `cd web && npm test -- --reporter=dot` | 14 tests aprobados, incluyendo mensajes de ocupación y truncamiento |
| `cd web && npm run build` | Build Vite aprobado, 22 módulos transformados |
| `venv/bin/python -m compileall -q api core tests` | Aprobado |
| `bash -n run_api.sh` y `git diff --check` | Aprobados |

La suite nueva usa factorías e inferencia simuladas, bloqueos con eventos y módulos CPU diminutos para comprobar hooks y referencias débiles. Comprueba inicialización concurrente, rechazo antes de arrancar el worker, exclusión entre trabajos/checkpoints, ambas transiciones de modo, reutilización, liberación de auxiliares y componentes compartidos, fallos de carga/embeddings/inferencia, OOM simulado, recuperación, snapshots y conservación de advertencias/estados. Verifica argumentos de carga local, fallo por configuración ausente, métricas y las políticas de precisión/offload existentes. El acceso HTTP de Requests y el snapshot downloader del manager están bloqueados en los tests nuevos.

El primer intento de la suite se interrumpió porque TestClient quedó bloqueado en la comunicación entre hilos del sandbox. La suite completa pasó fuera de esa restricción. También se ajustó una prueba de referencias para distinguir la limpieza explícita del motor de las llamadas internas de Accelerate a su caché durante la retirada de hooks.

Se verificó, sin cargar pesos, que la configuración SD1.5 del checkpoint predeterminado se resuelve desde la caché local existente. No se ejecutaron los scripts raíz que realizan inferencia real. No se generaron imágenes ni videos.

## Revisión de regresiones

- Se mantienen DDIM, límites de 512 × 512 y batch 1 del perfil GTX 1650, parámetros API, formatos de resultados y privacidad del directorio de outputs.
- txt2img mantiene FP32 por defecto; Face Reference conserva el FP16 y offload que ya utilizaba. No se activó FP16 global ni xFormers, ni se añadieron optimizaciones globales.
- No se cambia Gradio ni se añaden capacidades de edición o video. La única modificación visual es mostrar advertencias de truncamiento; el cliente ya presentaba los mensajes HTTP 409.
- La necesidad de configuraciones locales es deliberada: una caché incompleta ahora produce un error en lugar de iniciar una descarga. Los checkpoints de otra arquitectura necesitan su configuración local compatible; esta validación cubre SD1.5/IP-Adapter.
- La corrección de hooks y los dtypes se comprueban con mocks y módulos CPU; no equivalen a validar el comportamiento numérico ni los máximos de memoria en CUDA real.

## Riesgos pendientes en GPUs de 4 GB

Un único trabajo puede agotar RAM o VRAM. La construcción de CLIP-H y su state dict, el checkpoint, las copias temporales y la restauración de pesos al retirar hooks siguen teniendo picos. La caché del allocator, CUDA, bibliotecas nativas y otros procesos también consumen memoria. No se implementa una reserva de memoria ni una admisión basada en una estimación no fiable de RAM libre.

SIGKILL no es interceptable por Python: no habrá transición final del trabajo ni cleanup si el kernel o systemd-oomd mata el proceso. El registro de trabajos es volátil y crece con sus metadatos, aunque ya no retiene los payloads privados. Las dependencias admiten rangos amplios; futuras versiones de Diffusers/Accelerate deben volver a validarse. No existe exclusión entre servidores independientes ni entre distintas aplicaciones.

## Procedimiento manual pendiente en Fedora

Estas acciones son una validación posterior en el equipo de destino; **no se realizaron en esta entrega**. Las generaciones de prueba sólo deben ejecutarse con autorización del operador.

1. Cerrar las otras aplicaciones de inferencia y usar los modelos/configuraciones ya instalados. Mantener una sola instancia: `bash run_api.sh`. No usar múltiples workers ni lanzar Gradio en paralelo. Para reproducibilidad, anotar versiones con `venv/bin/python -m pip show torch diffusers transformers accelerate` y consultar `nvidia-smi`.
2. Registrar una hora inicial y observar el terminal de Uvicorn. En otras terminales ejecutar `free -h`, `vmstat 1` y `nvidia-smi -l 1`. Para el RSS del PID Uvicorn, usar `ps -p <PID> -o pid,rss,vsz,cmd`. No guardar en el diagnóstico fotos ni prompts privados.
3. Con permiso para generar, hacer txt2img → txt2img → Face Reference → Face Reference → txt2img a 512 × 512, batch 1 y los defaults existentes. Repetir la alternancia varias veces. En trabajos del mismo modo no debe aparecer una nueva carga; al alternar, `release_end` debe preceder a `load_start` del otro pipeline. Medir RAM/VRAM antes, durante y después: una línea base estable importa más que alcanzar cero bytes reservados.
4. Mientras haya un trabajo `queued` o `running`, enviar desde otra terminal `curl -i -X POST http://127.0.0.1:7861/api/generate/txt2img -H 'Content-Type: application/json' -d '{"prompt":"prueba de concurrencia"}'`. Repetir para `POST /api/models/active` con `{"model_id":"<ruta de un checkpoint ya instalado>"}`. Esperar HTTP 409, sin nuevas cargas ni cambios en el trabajo activo. El bloqueo se conserva durante la limpieza final.
5. Consultar `GET /api/jobs/<job_id>` mientras corre y después de terminar. Verificar estados coherentes, resultados/advertencias conservados y ausencia de payload privado en la respuesta. Tras quedar libre, cambiar a otro checkpoint local ya instalado y comprobar que sólo se carga en el siguiente trabajo.
6. Probar un prompt de más de 77 tokens CLIP y un negativo largo. Verificar advertencia de truncamiento en la pantalla y en `warnings`; confirmar que los logs no contienen el texto descartado. Un prompt corto no debe advertir.
7. Sin provocar deliberadamente un OOM del sistema, probar un fallo recuperable con un recurso facial ausente en una instalación de prueba, restaurarlo y volver a intentar. No mover ni borrar archivos de modelos en uso. Confirmar `failed`, liberación y posterior recuperación. Un checkpoint inexistente debe rechazarse sin reemplazar la selección anterior.
8. Si vuelve a aparecer `Killed`, recoger inmediatamente `sudo journalctl -k -b --since '<hora inicial>'`, `sudo journalctl -b -u systemd-oomd --since '<hora inicial>'`, `oomctl` si está disponible y `sudo dmesg -T`. Buscar `Out of memory`, `Killed process`, `oom-kill`, restricciones de cgroup y decisiones de oomd. Si Uvicorn corre como servicio, consultar además su journal y sus límites de memoria. Sin esas evidencias, mantener la causa como probable.
9. Confirmar que no se modificaron los archivos de modelos/caché, que el PID sigue vivo durante toda la secuencia y que los picos caben con margen en RAM/VRAM. Registrar los resultados reales antes de afirmar que el incidente OOM está resuelto.

Referencias técnicas: [ciclo de vida y hooks de Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/overview), [carga de checkpoints y configuraciones locales](https://huggingface.co/docs/diffusers/using-diffusers/other-formats), [tokenización y límites](https://huggingface.co/docs/transformers/main/main_classes/tokenizer). El fallback de descarga se comprobó también en el código instalado de Diffusers 0.36.0.
