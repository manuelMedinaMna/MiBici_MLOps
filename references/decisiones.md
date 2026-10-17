# Decisiones del proyecto

Registro de las decisiones técnicas y de análisis: qué se decidió, por qué y qué alternativas
se descartaron. Si una decisión cambia, no se borra: se marca como reemplazada y se agrega la
nueva.

## 1. Guardar los datos intermedios y procesados en Parquet, no en CSV

- **Fecha:** 2026-10-16
- **Estado:** vigente
- **Aplica a:** `data/interim/` y `data/processed/`. `data/raw/` se queda con los CSV
  originales tal como se descargaron.

### Contexto

`mibici_mlops/dataset.py` une los 142 CSV mensuales de viajes (2.9 GB) en un solo archivo
limpio con ~39 millones de filas. Había que elegir el formato de ese archivo.

### Decisión

Usar **Parquet** con compresión `zstd`.

Parquet es un formato de archivo abierto de Apache (no es de Python ni un paquete estándar).
En Python se lee y escribe con la librería `pyarrow`, que pandas usa por dentro en
`pd.read_parquet` y `df.to_parquet`. Para trabajar con él basta con:

```python
df = pd.read_parquet("data/interim/viajes.parquet")
```

y se obtiene un DataFrame normal.

### Por qué

Medido con mayo de 2024 (410,806 viajes), guardando el mismo DataFrame limpio en ambos
formatos:

| | CSV limpio | Parquet |
|---|---|---|
| Tamaño | 29.4 MB | 6.9 MB |
| Tiempo de lectura | 0.74 s | 0.15 s |
| `inicio`, `fin` al leer | texto | fecha |
| `anio_nacimiento` al leer | `float64` (`1984.0`) | entero |

Con todos los datos:

- **Espacio en disco:** el Parquet completo pesa 508 MB. En CSV serían ~2.4 GB, casi otra copia
  completa de `data/raw/`. Esto también importa en Google Drive, donde DVC guarda cada versión.
- **Tipos:** Parquet guarda el tipo de cada columna. Un CSV solo guarda texto, así que cada vez
  que se abre habría que volver a convertir las fechas, para cada integrante del equipo.
- **Velocidad:** se lee ~5 veces más rápido.

### Alternativa descartada

**CSV limpio.** Se puede abrir en Excel o en un editor de texto, pero ocupa ~5 veces más,
carga ~5 veces más lento y pierde los tipos de dato.

### Consecuencias

- El Parquet no se puede abrir con doble clic: se revisa con pandas. Para ver los datos tal
  como llegaron siguen estando los CSV en `data/raw/`.
- `pandas` y `pyarrow` son dependencias del proyecto (están en `requirements.txt`).
- Las reglas de limpieza no dependen del formato: están en `mibici_mlops/dataset.py`. Para
  cambiarlas se edita ese archivo y se vuelve a correr `python -m mibici_mlops.dataset`.
