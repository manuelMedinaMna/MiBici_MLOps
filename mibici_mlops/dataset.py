"""Une y limpia los CSV mensuales de viajes de MiBici en un solo Parquet.

Paso raw → interim: corrige las diferencias de formato entre archivos y los errores de los
datos, como filas duplicadas (ver references/fuentes_datos.md). No filtra filas por criterios de
análisis: esas decisiones, como el periodo a usar, van en features.py.

Uso:
    python -m mibici_mlops.dataset
"""

from pathlib import Path
import re

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import typer

from mibici_mlops.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Se renombra por posición porque el encabezado de la columna 4 viene dañado en varios archivos.
COLUMNAS = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio",
    "fin",
    "origen_id",
    "destino_id",
]

ESQUEMA = pa.schema(
    [
        ("viaje_id", pa.int64()),
        ("usuario_id", pa.int64()),
        ("genero", pa.string()),
        ("anio_nacimiento", pa.int16()),
        ("inicio", pa.timestamp("s")),
        ("fin", pa.timestamp("s")),
        ("origen_id", pa.int32()),
        ("destino_id", pa.int32()),
        ("origen_lon", pa.float64()),
        ("origen_lat", pa.float64()),
        ("destino_lon", pa.float64()),
        ("destino_lat", pa.float64()),
    ]
)

# Los nombres pueden traer sufijo (p. ej. datos_abiertos_2016_01-1.csv).
PATRON_ARCHIVO = re.compile(r"datos_abiertos_(\d{4})_(\d{2})")


def listar_archivos(viajes_dir: Path) -> list[Path]:
    """Regresa los CSV de viajes ordenados por año y mes."""
    archivos = []
    for ruta in viajes_dir.glob("*/*.csv"):
        coincidencia = PATRON_ARCHIVO.search(ruta.name)
        if coincidencia is None:
            logger.warning(f"Se ignora {ruta.name}: no sigue el patrón de nombre")
            continue
        archivos.append((coincidencia.groups(), ruta))
    return [ruta for _, ruta in sorted(archivos)]


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV mensual como texto, probando UTF-8 y luego Latin-1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                ruta,
                encoding=encoding,
                header=0,
                names=COLUMNAS,
                dtype=str,
                na_values=["", "NULL", "NA"],
                keep_default_na=False,
            )
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {ruta}")


def parsear_fechas(columna: pd.Series) -> pd.Series:
    """Convierte fechas ISO (YYYY-MM-DD HH:MM:SS) o DD/MM/YYYY H:MM según el archivo."""
    muestra = columna.dropna().iloc[0]
    formato = "%d/%m/%Y %H:%M" if "/" in muestra else "%Y-%m-%d %H:%M:%S"
    return pd.to_datetime(columna, format=formato, errors="coerce")


def a_entero(columna: pd.Series, tipo: str) -> pd.Series:
    """Convierte a entero nullable; acepta flotantes como '1984.0'."""
    return pd.to_numeric(columna, errors="coerce").round().astype(tipo)

def a_float(columna: pd.Series) -> pd.Series:
    """Convierte a float nullable; acepta enteros como '1984'."""
    return pd.to_numeric(columna, errors="coerce").astype(float)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica tipos a un mes de viajes. Los valores inválidos quedan como nulos."""
    return pd.DataFrame(
        {
            "viaje_id": a_entero(df["viaje_id"], "Int64"),
            "usuario_id": a_entero(df["usuario_id"], "Int64"),
            "genero": df["genero"].str.strip().str.upper(),
            "anio_nacimiento": a_entero(df["anio_nacimiento"], "Int16"),
            "inicio": parsear_fechas(df["inicio"]),
            "fin": parsear_fechas(df["fin"]),
            "origen_id": a_entero(df["origen_id"], "Int32"),
            "destino_id": a_entero(df["destino_id"], "Int32"),
        }
    )


def reportar_nulos_nuevos(crudo: pd.DataFrame, limpio: pd.DataFrame, nombre: str) -> None:
    """Avisa cuando la conversión de tipos convirtió valores no vacíos en nulos."""
    perdidos = (limpio.isna() & crudo.notna()).sum()
    perdidos = perdidos[perdidos > 0]
    if not perdidos.empty:
        logger.warning(f"{nombre}: valores no convertibles -> nulo: {perdidos.to_dict()}")


def quitar_duplicados(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Elimina filas idénticas en todas las columnas (el mismo viaje escrito dos veces)."""
    sin_duplicados = df.drop_duplicates()
    eliminadas = len(df) - len(sin_duplicados)
    if eliminadas:
        logger.warning(f"{nombre}: {eliminadas:,} filas duplicadas eliminadas")
    return sin_duplicados

def unir_estaciones(df: pd.DataFrame, estdf: pd.DataFrame) -> pd.DataFrame:
    """Une las estaciones a los viajes."""
    con_est = df.copy()
    con_est["origen_lon"] = a_float(con_est.origen_id.map(estdf.longitude))
    con_est["origen_lat"] = a_float(con_est.origen_id.map(estdf.latitude))
    con_est["destino_lon"] = a_float(con_est.destino_id.map(estdf.longitude))
    con_est["destino_lat"] = a_float(con_est.destino_id.map(estdf.latitude))
    return con_est

@app.command()
def main(
    estaciones_path: Path = RAW_DATA_DIR / "estaciones/nomenclatura_2026_09.csv",
    viajes_dir: Path = RAW_DATA_DIR / "viajes",
    output_path: Path = INTERIM_DATA_DIR / "viajes.parquet",
):
    estaciones = pd.read_csv(estaciones_path)
    estaciones.set_index("id", inplace=True)

    archivos = listar_archivos(viajes_dir)
    logger.info(f"Uniendo {len(archivos)} archivos de {viajes_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    # Se escribe mes por mes para no tener los ~39 M de filas en memoria a la vez.
    with pq.ParquetWriter(output_path, ESQUEMA, compression="zstd") as writer:
        for ruta in tqdm(archivos, desc="Meses"):
            crudo = leer_csv(ruta)
            limpio = limpiar(crudo)
            limpio = unir_estaciones(limpio, estaciones)
            reportar_nulos_nuevos(crudo, limpio, ruta.name)
            limpio = quitar_duplicados(limpio, ruta.name)
            writer.write_table(pa.Table.from_pandas(limpio, schema=ESQUEMA, preserve_index=False))
            total += len(limpio)

    logger.success(f"{total:,} viajes guardados en {output_path}")


if __name__ == "__main__":
    app()
