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

## 2. Usar los viajes desde marzo de 2022

- **Fecha:** 2026-10-16
- **Estado:** vigente
- **Aplica a:** el paso `interim → processed` (`features.py`) y todos los modelos (P1, P2 y P3).
  `data/interim/` conserva la historia completa (2014-12 a 2026-09).

### Contexto

Los datos cubren de diciembre de 2014 a septiembre de 2026. La pandemia cambió por completo el
uso de MiBici, así que había que decidir desde qué mes los datos representan la operación
normal. El análisis está en `notebooks/1.0-rc-periodo-de-datos.ipynb`, que compara cada mes
contra el mismo mes de 2019 (el último año normal) para no confundir la estacionalidad con una
caída real.

### Decisión

Usar los viajes **desde el 1 de marzo de 2022** hasta el último mes disponible.

### Por qué

- **2015–2018:** el sistema todavía estaba creciendo (de ~30 mil a ~400 mil viajes al mes).
- **2019:** uso estable, ~386 mil viajes al mes en promedio.
- **2020–2021:** en abril de 2020 hubo 129,276 viajes, el 36% de abril de 2019 (361,504).
  Ningún mes de 2020 ni de 2021 alcanza el nivel de 2019.
- **Enero y febrero de 2022:** todavía bajos, 72% y 81% del mismo mes de 2019.
- **Desde marzo de 2022:** casi todos los meses quedan entre 86% y 112% del mismo mes de
  2019: el uso regresó al nivel previo a la pandemia y se mantiene.

### Alternativas descartadas

- **Desde enero de 2022:** daría años completos, pero incluye dos meses todavía afectados por
  la pandemia. Enero y febrero siguen apareciendo completos en 2023, 2024, 2025 y 2026.
- **Incluir 2019 o antes:** el nivel de uso es parecido, pero entre 2019 y 2022 hay dos años
  anormales; el modelo aprendería de una serie con un hueco de dos años.

### Consecuencias

- El periodo queda de 2022-03 a 2026-09: 55 meses.
- 2023, 2024 y 2025 están completos; 2022 empieza en marzo y 2026 llega hasta septiembre.
- La fecha de inicio debe definirse una sola vez en `mibici_mlops/config.py` y usarse desde ahí.
- **Pendiente:** septiembre de 2026 tiene 127% de los viajes de septiembre de 2019, el máximo
  de la serie. Los datos parecen correctos (mismos usuarios haciendo más viajes y más cortos
  desde mediados de mes), pero falta decidir cómo tratarlo en la partición de prueba.

## 3. P2 como clasificación de 3 clases: sin riesgo, se vacía, se llena

- **Fecha:** 2026-10-16
- **Estado:** vigente
- **Aplica a:** P2 (riesgo de quedarse sin bicicletas o sin anclajes).

### Contexto

P2 pregunta si una estación estará en riesgo de quedarse sin bicicletas o sin anclajes
disponibles. El ML Canvas (iteración 1) la planteaba como clasificación binaria: riesgo sí/no.

### Decisión

Clasificar cada estación y periodo en **3 clases**:

| Clase | Significado | Acción del equipo de rebalanceo |
|---|---|---|
| Sin riesgo | La estación opera normal | Ninguna |
| Se vacía | Salen más bicicletas de las que llegan; puede quedarse sin bicicletas | Llevar bicicletas |
| Se llena | Llegan más bicicletas de las que salen; puede quedarse sin anclajes | Retirar bicicletas |

### Por qué

- **Las acciones son opuestas.** Con la clasificación binaria el equipo sabría a qué estación
  ir, pero no si debe llegar con bicicletas o con espacio para recogerlas.
- **Los datos lo permiten.** El riesgo se aproxima con el balance neto de viajes
  (llegadas − salidas), cuyo signo indica si la estación se vacía o se llena.
- **Los modelos candidatos no cambian.** `LogisticRegression` y
  `HistGradientBoostingClassifier` de scikit-learn manejan 3 clases igual que 2 y dan
  probabilidades por clase (`predict_proba`) para ordenar la lista de estaciones.

### Alternativa descartada

**Clasificación binaria (riesgo / sin riesgo).** Es más simple, pero pierde la dirección del
riesgo, que es justo lo que el equipo de rebalanceo necesita para actuar.

### Consecuencias

- Las clases de riesgo serán minoritarias (la mayor parte del tiempo las estaciones operan
  normal). La exactitud (accuracy) no sirve como métrica: se usan recall y PR-AUC por clase, y
  pesos de clase (`class_weight`) en los modelos.
- El baseline de clase mayoritaria tendrá recall de 0 en las clases de riesgo; sirve como
  piso, no como referencia útil.
- Hay que actualizar la sección de P2 del ML Canvas (`references/MLCanvas_MiBici.pdf`).
- **Pendiente:** definir con qué umbral el balance neto frente a la capacidad cuenta como
  "se vacía" o "se llena".
