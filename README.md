# MiBici_MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Realizar tareas de manipulación, preparación, exploración y preprocesamiento de datos, gestionando grandes volúmenes de datos distribuidos y aplicando técnicas de versionado de datos para asegurar la reproducibilidad y trazabilidad en proyectos de Machine Learning.

## Datos

Los datos crudos son los [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/) (viajes mensuales 2014-12 a 2026-09 y catálogo de estaciones). No se suben a GitHub: se versionan con DVC en el Google Drive del equipo. Si no tienes acceso a ese Drive, puedes descargarlos directamente de la página oficial.

Detalles de las fuentes, cómo obtener los datos y problemas conocidos en [references/fuentes_datos.md](references/fuentes_datos.md).

Las decisiones técnicas y de análisis del proyecto (por ejemplo, por qué los datos procesados se guardan en Parquet) están en [references/decisiones.md](references/decisiones.md).

## Cómo empezar

Requiere Python 3.13. Para bajar los datos con DVC hace falta acceso a la carpeta de Google
Drive del equipo: pídele a un integrante que comparta la carpeta con tu cuenta de Google y que te
agregue como *test user* de la app OAuth en Google Cloud; sin esto, `dvc pull` muestra "access
blocked". Sin ese acceso, ver la nota debajo del bloque de comandos.

```powershell
git clone https://github.com/robertonali/MiBici_MLOps.git
cd MiBici_MLOps
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
pip install -U pyopenssl
dvc remote modify --local drive gdrive_client_secret <secreto>
dvc pull
python -m mibici_mlops.dataset
```

- **`pip install -U pyopenssl`**: pip instala un `pyOpenSSL` viejo que hace fallar a DVC con
  `module 'lib' has no attribute 'GEN_EMAIL'`. La advertencia de `pydrive2` sobre la versión se
  puede ignorar.
- **`<secreto>`**: el secreto del cliente OAuth de Google. Se comparte por un canal privado y se
  guarda en `.dvc/config.local`, que no se sube a Git. **Nunca lo pongas en `.dvc/config`.**
- **`dvc pull`**: la primera vez abre el navegador para autorizar tu cuenta de Google.
- **`python -m mibici_mlops.dataset`**: une y limpia los 142 CSV de `data/raw/viajes/` en
  `data/interim/viajes.parquet` (~39 millones de viajes; tarda entre 5 y 10 minutos). Qué
  corrige y por qué está en [references/fuentes_datos.md](references/fuentes_datos.md).

**Sin acceso al Google Drive**, omite las líneas de `pyopenssl`, `dvc remote` y `dvc pull`:
`python -m mibici_mlops.dataset` descarga de la página de MiBici los CSV que falten en
`data/raw/` (hasta `FECHA_FIN` de `config.py`, ~2.9 GB) y abril y mayo de 2023 de Internet
Archive. Si ya hiciste `dvc pull` no descarga nada; para no intentarlo, usa `--sin-descargar`.
Los archivos descargados pueden no ser idénticos a la versión de DVC si MiBici los actualiza.

### Verificar que todo quedó bien

| Después de | Revisa | Resultado esperado |
|---|---|---|
| `dvc pull` | `dvc status` | `Data and pipelines are up to date.` |
| `python -m mibici_mlops.dataset` | Las últimas líneas del log | `datos_abiertos_2026_09.csv: 758 filas duplicadas eliminadas` y `39,078,082 viajes guardados` |
| `python -m mibici_mlops.dataset` con datos descargados de la página (sin `dvc pull`) | Las últimas líneas del log | `39,078,226 viajes guardados` al 2026-10-18: MiBici corrigió algunos CSV después de que se registró la versión de DVC |
| Todo lo anterior | Correr `notebooks/1.0-rc-periodo-de-datos.ipynb` con el intérprete de `.venv` | La sección 6 muestra 0 filas con `viaje_id` repetido |

Si un resultado no coincide, avisa al equipo con el mensaje de error antes de seguir.

Al clonar, `data/` solo trae `.gitignore` y `raw.dvc`: `dvc pull` crea `data/raw/`,
`dataset.py` crea `data/interim/` y `features.py` crea `data/processed/` (ver la sección de P2).

### Versionar cambios en los datos

```powershell
dvc add data/raw
git add data/raw.dvc
git commit -m "Actualizar datos"
dvc push        # si falla con WinError 10054: dvc push -j 2
git push
```

## P2: riesgo de que una estación se vacíe o se llene

**Pregunta:** ¿una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en un
periodo futuro? La usa el equipo de rebalanceo para decidir a qué estaciones ir y si debe llevar o
retirar bicicletas.

Las decisiones y su justificación están en
[references/decisiones_p2.md](references/decisiones_p2.md), y el análisis que las respalda en
`notebooks/2.0-rc-p2-tamano-ventana.ipynb` (ventanas: secciones 1 a 10; hora de reinicio: 11;
etiqueta: 12; partición: 13).

| Aspecto | Decisión |
|---|---|
| Tarea | Clasificación en 3 clases: sin riesgo, se vacía, se llena |
| Granularidad | Estación y ventana de 2 horas (00:00, 02:00, …, 22:00) |
| Horizonte | Cada día a las 4:00 se predicen las 12 ventanas del día |
| Etiqueta | Inventario simulado: cada estación empieza el día (4:00) al 50% de su capacidad y se le suman las llegadas y restan las salidas. "Se vacía" si llega a 0 o menos; "se llena" si llega a la capacidad |
| Datos | Viajes del 2022-03-01 al 2026-09-30 y capacidad de cada estación. No usa datos de los usuarios (id, género ni año de nacimiento) |
| Partición | Por fechas: entrenamiento 2022-03 a 2025-09, validación 2025-10 a 2026-03 y prueba 2026-04 a 2026-09 |
| Métricas | Recall, precisión y PR-AUC de cada clase de riesgo en el conjunto de prueba |

### Cómo correrlo

Después de `python -m mibici_mlops.dataset`:

```powershell
python -m mibici_mlops.features p2                  # etiqueta y variables (~35 s, ~3.6 GB de RAM)
python -m mibici_mlops.modeling.train_p2            # baselines registrados en MLflow (~30 s)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # ver los runs en http://127.0.0.1:5000
```

| Comando | Resultado esperado |
|---|---|
| `features p2` | `6,979,284 ventanas de 471 estaciones` y clases `sin_riesgo: 81.39`, `se_vacia: 9.77`, `se_llena: 8.83` |
| `train_p2` | Dos runs en el experimento `mibici-equipo43`: `p2-baseline-clase-mayoritaria` y `p2-baseline-frecuencia-4-semanas` |

Los runs se guardan en `mlflow.db` y `mlruns/`, que no se suben a Git. Conviene correr `train_p2`
con todo en commit: cada run registra el commit y si había cambios sin commit.

### Resultados de los baselines (conjunto de prueba)

| Métrica | Clase mayoritaria | Frecuencia 4 semanas | Azar |
|---|---|---|---|
| PR-AUC se vacía | 0.057 | **0.368** | 0.057 |
| PR-AUC se llena | 0.072 | **0.539** | 0.072 |
| Recall se vacía | 0 | 0.470 | — |
| Precisión se vacía | 0 | 0.461 | — |
| Recall se llena | 0 | 0.623 | — |
| Precisión se llena | 0 | 0.593 | — |

- **Clase mayoritaria:** predice siempre "sin riesgo". Acierta en el 87% de las ventanas de prueba,
  pero no detecta ningún riesgo; por eso la exactitud no sirve como métrica.
- **Frecuencia de las últimas 4 semanas** (misma estación, mismo día de la semana y misma ventana):
  detecta casi la mitad de las ventanas "se vacía" y el 62% de las "se llena", con una PR-AUC
  entre 6.5 y 7.5 veces la del azar. Es la referencia que deben superar los modelos candidatos
  (`LogisticRegression` y `HistGradientBoostingClassifier` con pesos de clase).
- **Azar** es la proporción de cada clase en prueba: la PR-AUC de un modelo que no distingue nada.

Las capturas de MLflow están en `reports/figures/mlflow_p2_*.png`.

## P3: grupos de estaciones con patrones de uso parecidos

**Pregunta:** ¿qué grupos de estaciones comparten patrones de uso y dónde se ubican en la
ciudad? La usa planeación urbana para decidir dónde y de qué tipo abrir estaciones nuevas, y para
planear por tipo de estación en lugar de estación por estación.

Las decisiones y su justificación están en
[references/decisiones_p3.md](references/decisiones_p3.md), y la exploración en
`notebooks/1.0-im-initial-eda.ipynb`.

| Aspecto | Decisión |
|---|---|
| Tarea | Clustering de estaciones |
| Granularidad | Una fila por estación con su perfil semanal: fracción de sus salidas y de sus llegadas en cada día de la semana y franja del día (madrugada, mañana, tarde, noche): 56 variables |
| Horizonte | Grupos aprendidos con entrenamiento; se mide si se mantienen en los 6 meses de prueba |
| Datos | Viajes del 2022-03-01 al 2026-09-30 y ubicación de cada estación. No usa datos de los usuarios |
| Partición | Mismas fechas que P2: perfil de entrenamiento 2022-03 a 2025-09 y de prueba 2026-04 a 2026-09 |
| Métricas | Silhouette, Davies-Bouldin y Calinski-Harabasz sobre el perfil de uso (`train_*` y `test_*`), y estabilidad entre periodos (`test_ari_estabilidad`) |

### Cómo correrlo

Después de `python -m mibici_mlops.dataset`:

```powershell
python -m mibici_mlops.features p3                  # perfil de cada estación (~5 s, ~3 GB de RAM)
python -m mibici_mlops.modeling.train_p3            # baseline y KMeans registrados en MLflow (~30 s)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # ver los runs en http://127.0.0.1:5000
```

| Comando | Resultado esperado |
|---|---|
| `features p3` | `entrenamiento: 371 estaciones con perfil` y `prueba: 429 estaciones con perfil` |
| `train_p3` | Dos runs en el experimento `mibici-equipo43`: `p3-baseline-zonas-geograficas-k3` y `p3-kmeans-perfil-uso-k3` |

Cada run guarda como artefactos el mapa de los grupos (`mapa_grupos.png`), su perfil de uso
(`perfil_grupos.png`) y el grupo de cada estación (`estaciones_grupos.json`); el de KMeans,
además, la silhouette según el número de grupos.

### Resultados

| Métrica | Zonas geográficas (baseline) | KMeans sobre perfil de uso |
|---|---|---|
| Silhouette entrenamiento | 0.063 | **0.303** |
| Silhouette prueba | 0.023 | **0.246** |
| Davies-Bouldin entrenamiento (menor es mejor) | 4.059 | **1.128** |
| Davies-Bouldin prueba | 3.721 | **1.278** |
| Calinski-Harabasz entrenamiento | 36.2 | **153.1** |
| Estabilidad (ARI) | 1 (por construcción) | 0.714 |

- **Agrupar por uso da grupos mucho más coherentes que agrupar por zona**: la silhouette es
  casi 5 veces la del baseline, y la ventaja se mantiene en prueba.
- **3 grupos** es el número con mayor silhouette entre 2 y 8 (0.303; con 2 es 0.277 y con 4,
  0.209):
  - **Grupo 1, destinos de trabajo y estudio (163 estaciones):** llegadas en la mañana y salidas
    en la tarde entre semana. 104 de ellas están al poniente (longitud menor a -103.37).
  - **Grupo 2, origen residencial (185):** al revés, salidas en la mañana y llegadas en la tarde
    y la noche. 155 están en el centro y el oriente.
  - **Grupo 3, recreativo (23):** el 21% de sus salidas son en domingo (contra 8–9% de los otros
    grupos), casi la mitad el domingo en la mañana. Forman una línea de oriente a poniente que
    coincide con la Vía RecreActiva.
- **Los grupos se mantienen:** el ARI de 0.71 indica que la mayoría de las estaciones conserva su
  grupo de un periodo a otro.
- **Estaciones nuevas:** 113 estaciones del catálogo no tienen viajes antes de octubre de 2025.
  64 de ellas ya tienen perfil en prueba y el modelo les asigna un grupo (33 al 2, 20 al 1 y 11
  al 3): es lo que planeación urbana necesita para saber qué tipo de estaciones abrió.

## Project Organization

```
├── LICENSE            <- MIT (aplica al código; los datos son de MiBici)
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_mlops and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- Dependencias directas del proyecto (agregar a mano; no usar
│                         `pip freeze`, ver sección "Cómo empezar")
│
└── mibici_mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   ├── registro.py         <- Experimento, etiquetas y versión de datos de MLflow (P1, P2, P3)
    │   ├── train.py            <- Code to train models
    │   ├── train_p2.py         <- P2: baselines registrados en MLflow
    │   └── train_p3.py         <- P3: baseline y KMeans registrados en MLflow
    │
    └── plots.py                <- Code to create visualizations
```

--------

