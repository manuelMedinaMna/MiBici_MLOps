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

# Archivos que usan varios módulos
VIAJES_PATH = INTERIM_DATA_DIR / "viajes.parquet"
ESTACIONES_PATH = RAW_DATA_DIR / "estaciones" / "nomenclatura_2026_09.csv"
P2_VENTANAS_PATH = PROCESSED_DATA_DIR / "p2_ventanas.parquet"

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

# P2: etiqueta de riesgo y baseline de frecuencia. Ver references/decisiones_p2.md, decisiones 2
# (ventana), 3 (semanas de historia), 4 (reinicio e inventario inicial) y 5 (umbral).
# La hora de reinicio debe coincidir con el inicio de una ventana.
P2_VENTANA = "2h"
P2_HORA_REINICIO = 4
P2_INVENTARIO_INICIAL = 0.5
P2_SEMANAS_HISTORIA = 4

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
