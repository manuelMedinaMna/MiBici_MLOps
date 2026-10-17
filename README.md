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

Requiere Python 3.13 y acceso a la carpeta de Google Drive del equipo (pídela a un integrante).

```powershell
git clone https://github.com/robertonali/MiBici_MLOps.git
cd MiBici_MLOps
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
pip install -U pyopenssl
dvc remote modify --local drive gdrive_client_secret <secreto>
dvc pull
```

- **`pip install -U pyopenssl`**: pip instala un `pyOpenSSL` viejo que hace fallar a DVC con
  `module 'lib' has no attribute 'GEN_EMAIL'`. La advertencia de `pydrive2` sobre la versión se
  puede ignorar.
- **`<secreto>`**: el secreto del cliente OAuth de Google. Se comparte por un canal privado y se
  guarda en `.dvc/config.local`, que no se sube a Git. **Nunca lo pongas en `.dvc/config`.**
- **`dvc pull`**: la primera vez abre el navegador para autorizar tu cuenta de Google.

### Versionar cambios en los datos

```powershell
dvc add data/raw
git add data/raw.dvc
git commit -m "Actualizar datos"
dvc push        # si falla con WinError 10054: dvc push -j 2
git push
```

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
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

