from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

# Periodo de análisis: primer día con uso recuperado tras la pandemia.
# Ver references/decisiones.md, decisión 2.
FECHA_INICIO = "2022-03-01"
# Último día con datos. Fijarlo evita que los resultados cambien al descargar meses nuevos.
FECHA_FIN = "2026-09-30"

# Partición temporal por día: entrenamiento hasta antes de FECHA_INICIO_VALIDACION, validación
# hasta antes de FECHA_INICIO_PRUEBA y prueba hasta FECHA_FIN.
# Ver references/decisiones_p2.md, decisión 6.
FECHA_INICIO_VALIDACION = "2025-10-01"
FECHA_INICIO_PRUEBA = "2026-04-01"

# Registro de experimentos: un solo experimento de MLflow para P1, P2 y P3.
EXPERIMENTO_MLFLOW = "mibici-equipo43"
SEMILLA = 43

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
