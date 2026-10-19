from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Global seed
RANDOM_STATE = 67

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

# Archivos que usan varios módulos
VIAJES_PATH = INTERIM_DATA_DIR / "viajes.parquet"
ESTACIONES_PATH = RAW_DATA_DIR / "estaciones" / "nomenclatura_2026_09.csv"
P2_VENTANAS_PATH = PROCESSED_DATA_DIR / "p2_ventanas.parquet"
P3_CLUSTERS_PATH = PROCESSED_DATA_DIR / "p3_clusters.parquet"
P3_PERFILES_PATH = PROCESSED_DATA_DIR / "p3_perfiles.parquet"
P3_VIAJES_PATH = INTERIM_DATA_DIR / "p3_viajes.parquet"
P3_MODEL_PATH = MODELS_DIR / "p3.joblib"

# Periodo de análisis: primer día con uso recuperado tras la pandemia.
# Ver references/decisiones.md, decisión 2.
FECHA_INICIO = "2022-03-01"
# Último día con datos. Fijarlo evita que los resultados cambien al descargar meses nuevos.
FECHA_FIN = "2026-09-30"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
