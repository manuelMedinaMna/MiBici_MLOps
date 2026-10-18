"""Variables de P1: demanda de viajes por estación y día, para el día siguiente.

Paso interim → processed de P1. Cada fila es una estación y un día D que se quiere predecir. La
predicción se genera una vez al día, al cierre del día D-1 (23:59), así que toda variable de
historial usa solo días anteriores a D (ver references/decisiones_p1.md).

Solo se leen las columnas ``inicio`` y ``origen_id`` de ``data/interim/viajes.parquet``: los datos
del usuario (``usuario_id``, ``genero``, ``anio_nacimiento``) no se cargan.

Uso:
    python -m mibici_mlops.features_p1
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer

from mibici_mlops import config
from mibici_mlops.config import FECHA_INICIO, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Partición temporal. Si el equipo las define en config.py, se usan desde ahí.
FECHA_FIN_ENTRENAMIENTO = getattr(config, "FECHA_FIN_ENTRENAMIENTO", "2025-12-31")
FECHA_FIN_VALIDACION = getattr(config, "FECHA_FIN_VALIDACION", "2026-04-30")

DIAS_POR_SEMANA = 7

COLUMNAS_FEATURES = [
    "dia_semana",
    "mes",
    "es_fin_de_semana",
    "es_feriado",
    "capacidad",
    "lat",
    "lon",
    "lag_7d",
    "media_7d",
]


def dias_feriados(anios: range) -> set[pd.Timestamp]:
    """Días de descanso obligatorio de la Ley Federal del Trabajo (art. 74)."""

    def lunes(anio: int, mes: int, n: int) -> pd.Timestamp:
        primero = pd.Timestamp(anio, mes, 1)
        primer_lunes = primero + pd.Timedelta(days=(7 - primero.weekday()) % 7)
        return primer_lunes + pd.Timedelta(weeks=n - 1)

    feriados = set()
    for anio in anios:
        feriados |= {
            pd.Timestamp(anio, 1, 1),
            lunes(anio, 2, 1),  # Día de la Constitución
            lunes(anio, 3, 3),  # Natalicio de Benito Juárez
            pd.Timestamp(anio, 5, 1),
            pd.Timestamp(anio, 9, 16),
            lunes(anio, 11, 3),  # Día de la Revolución
            pd.Timestamp(anio, 12, 25),
        }
        if (anio - 1994) % 6 == 0:  # transmisión del Poder Ejecutivo Federal
            feriados.add(pd.Timestamp(anio, 10, 1))
    return feriados


def es_feriado(fechas: pd.DatetimeIndex) -> np.ndarray:
    """Indica, para cada fecha, si cae en un día de descanso obligatorio."""
    feriados = dias_feriados(range(fechas.year.min(), fechas.year.max() + 1))
    return fechas.normalize().isin(feriados)


def leer_conteos(
    interim_path: Path, estaciones_ids: pd.Index, fecha_inicio: str
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Cuenta los viajes que inician en cada estación y día.

    Regresa una matriz (estaciones × días) con una malla completa: los días sin viajes quedan
    con 0. Los viajes de estaciones que no están en la nomenclatura se descartan y se reportan.
    """
    inicio_ts = pd.Timestamp(fecha_inicio)
    tabla = pq.read_table(
        interim_path,
        columns=["inicio", "origen_id"],
        filters=[("inicio", ">=", inicio_ts.to_pydatetime())],
    )
    viajes = tabla.to_pandas().dropna()
    logger.info(f"{len(viajes):,} viajes desde {fecha_inicio}")

    dia_viaje = viajes["inicio"].dt.floor("D")
    t0 = inicio_ts.normalize()
    dias = pd.date_range(t0, dia_viaje.max(), freq="D")

    t = ((dia_viaje - t0) // pd.Timedelta(days=1)).to_numpy()
    s = estaciones_ids.get_indexer(viajes["origen_id"].astype("int64"))
    conocida = s >= 0
    if not conocida.all():
        logger.warning(f"{(~conocida).sum():,} viajes con estación fuera de la nomenclatura")

    n_est, n_dias = len(estaciones_ids), len(dias)
    plano = s[conocida].astype(np.int64) * n_dias + t[conocida]
    matriz = np.bincount(plano, minlength=n_est * n_dias).reshape(n_est, n_dias)
    return matriz.astype(np.float32), dias


def calcular_historial(matriz: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula las variables de historial sin usar información del día que se predice.

    Para el día ``d``, el corte es el cierre del día ``d-1``. Todo rezago o ventana termina en
    ese día o antes. Donde no hay suficiente historial dentro de la malla, el valor es NaN.

    Parámetros:
        matriz: viajes por estación (filas) y día (columnas), sobre una malla diaria continua.
    """
    n_est, n_dias = matriz.shape

    # Mismo día de la semana anterior: d-7 (anterior al corte).
    lag_7d = np.full_like(matriz, np.nan)
    lag_7d[:, DIAS_POR_SEMANA:] = matriz[:, : n_dias - DIAS_POR_SEMANA]

    # Promedio de los 7 días anteriores (d-7 a d-1), con sumas acumuladas.
    suma = np.concatenate(
        [np.zeros((n_est, 1)), np.cumsum(matriz, axis=1, dtype=np.float64)], axis=1
    )
    media_7d = np.full_like(matriz, np.nan)
    d = np.arange(DIAS_POR_SEMANA, n_dias)
    media_7d[:, d] = (suma[:, d] - suma[:, d - DIAS_POR_SEMANA]) / DIAS_POR_SEMANA

    return {"lag_7d": lag_7d, "media_7d": media_7d}


def asignar_conjunto(fechas: pd.Series) -> pd.Categorical:
    """Etiqueta cada fila como entrenamiento, validación o prueba según su fecha."""
    fin_entrenamiento = pd.Timestamp(FECHA_FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)
    fin_validacion = pd.Timestamp(FECHA_FIN_VALIDACION) + pd.Timedelta(days=1)
    conjunto = np.where(
        fechas < fin_entrenamiento,
        "entrenamiento",
        np.where(fechas < fin_validacion, "validacion", "prueba"),
    )
    return pd.Categorical(conjunto, categories=["entrenamiento", "validacion", "prueba"])


def armar_tabla(
    matriz: np.ndarray, dias: pd.DatetimeIndex, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """Arma la tabla estación-día con la variable objetivo y las variables de P1.

    Se conservan las filas con una semana completa de historial antes del corte y al menos un
    viaje en esa semana. Las estaciones nuevas o sin actividad reciente se excluyen: no tienen
    historial con qué predecir.
    """
    historial = calcular_historial(matriz)
    elegible = np.isfinite(historial["media_7d"]) & (historial["media_7d"] > 0)
    s_idx, t_idx = np.nonzero(elegible)

    fecha = dias[t_idx]
    tabla = pd.DataFrame(
        {
            "estacion_id": estaciones.index.to_numpy()[s_idx].astype("int32"),
            "fecha": fecha,
            "viajes": matriz[s_idx, t_idx].astype("int32"),
            "dia_semana": fecha.dayofweek.astype("int8"),
            "mes": fecha.month.astype("int8"),
            "es_fin_de_semana": (fecha.dayofweek >= 5).astype("int8"),
            "es_feriado": es_feriado(dias)[t_idx].astype("int8"),
            "capacidad": estaciones["dpcapacity"].to_numpy(dtype="float32")[s_idx],
            "lat": estaciones["latitude"].to_numpy(dtype="float32")[s_idx],
            "lon": estaciones["longitude"].to_numpy(dtype="float32")[s_idx],
        }
    )
    for nombre, valores in historial.items():
        tabla[nombre] = valores[s_idx, t_idx].astype("float32")
    tabla["conjunto"] = asignar_conjunto(tabla["fecha"])
    return tabla


@app.command()
def main(
    interim_path: Path = INTERIM_DATA_DIR / "viajes.parquet",
    estaciones_path: Path = RAW_DATA_DIR / "estaciones/nomenclatura_2026_09.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_p1.parquet",
    fecha_inicio: str = FECHA_INICIO,
):
    estaciones = pd.read_csv(estaciones_path).set_index("id").sort_index()
    logger.info(f"{len(estaciones)} estaciones en la nomenclatura")

    matriz, dias = leer_conteos(interim_path, estaciones.index, fecha_inicio)
    logger.info(f"Malla: {matriz.shape[0]} estaciones × {matriz.shape[1]:,} días")

    tabla = armar_tabla(matriz, dias, estaciones)
    excluidas = matriz.size - len(tabla)
    logger.info(
        f"{len(tabla):,} filas estación-día ({excluidas:,} excluidas por falta de historial "
        "o de actividad en la semana previa)"
    )
    for conjunto, grupo in tabla.groupby("conjunto", observed=True):
        logger.info(
            f"{conjunto}: {len(grupo):,} filas, {grupo['estacion_id'].nunique()} estaciones, "
            f"{grupo['fecha'].min():%Y-%m-%d} a {grupo['fecha'].max():%Y-%m-%d}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(output_path, compression="zstd", index=False)
    logger.success(f"Variables de P1 guardadas en {output_path}")


if __name__ == "__main__":
    app()
