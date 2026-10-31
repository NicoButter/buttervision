# Benchmark físico PIZARRA Tensor Store v0.2

El benchmark requiere una autorización explícita en cada ejecución porque crea un workspace y guarda el tensor fuente y el resultado en el volumen habilitado. No elimina esos archivos: se conservan para inspección y el espacio de cuota debe incluir aproximadamente dos veces el tamaño lógico del tensor, además de metadatos. No se ha ejecutado en este equipo durante la iteración.

## Comando en Fedora

Use el arranque PIZARRA de la instalación para configurar `BUTTERVISION_PIZARRA_ENABLED`, raíz y UUID autorizados; la configuración predeterminada de PIZARRA permanece apagada. Con el entorno ya preparado:

```bash
venv/bin/python benchmarks/pizarra_tensor_benchmark.py --execute \
  --size-gib 1 --block-mib 8 --buffer-mib 128 --device cpu \
  --report /tmp/pizarra-tensor-1gib.json
```

El tamaño está expresado en GiB binarios. El caso de 1 GiB crea float32, lee y reduce el tensor, escala por 1.25, persiste el resultado y vuelve a leer y reducirlo. El JSON incluye referencias persistentes, exactitud, tiempos, throughput lógico, bytes contabilizados por la capa de almacenamiento, RSS inicial/final/pico muestreado, RAM de sistema y buffers administrados. El ejecutor CPU no usa CUDA; si PyTorch ya está importado y su contexto CUDA ya existe, el benchmark puede observar esos valores sin inicializarlo. En los demás casos la VRAM queda nula.

`--block-mib` no puede superar `max_fragment_bytes` derivado del presupuesto. Los bloques deben caber también en `result_partial_bytes`. `--buffer-mib` reparte el presupuesto entre hasta seis buffers, resultados parciales y reserva de seguridad. El presupuesto solo restringe buffers del ejecutor. RSS puede exceder 128 MiB por el proceso base, NumPy, Python, filesystem cache y otras asignaciones. `--device cuda` se rechaza de forma explícita: aún no hay ejecutor CUDA.

Empiece con un tamaño pequeño para revisar el almacenamiento y la cuota. El benchmark físico no es una prueba automatizada, no se ejecuta automáticamente y no debe coincidir con una inferencia que comparta el equipo, aunque este prototipo CPU no usa CUDA.
