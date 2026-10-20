"""Entrena y evalúa un modelo de P1 (demanda de viajes por estación y día) y lo registra en MLflow.

Cada ejecución entrena un solo modelo y crea un run: el baseline (mismo día de la semana pasada)
o el gradient boosting (`hgb`) con los hiperparámetros indicados. El modelo y los hiperparámetros
por defecto están en config.py (MODELOS["p1"]). Los hiperparámetros se comparan con las métricas
de validación de cada run (decisión 7 de references/decisiones_p1.md).

Antes de correrlo, el servidor de MLflow debe estar levantado (ver la sección de MLflow del
README):
    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination mlruns --port 5000 --workers 1

Uso:
    python -m mibici_mlops.modeling.train_p1                       # modelo por defecto (hgb)
    python -m mibici_mlops.modeling.train_p1 --modelo baseline
    python -m mibici_mlops.modeling.train_p1 --hiperparametros max_iter=200,max_leaf_nodes=31
    mlflow run . -e train -P pregunta=p1 ...                       # lo mismo con el MLproject

Los runs se ven en http://127.0.0.1:5000.
"""

from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import typer

from mibici_mlops.config import (
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    FIGURES_DIR,
    P1_FEATURES_PATH,
    P1_MODEL_PATH,
    SEMILLA,
)
from mibici_mlops.features import COLUMNAS_P1
from mibici_mlops.modeling.registro import (
    etiquetas_run,
    resolver_modelo,
    usar_experimento,
    version_datos,
)

app = typer.Typer()

# Nombre del run de cada modelo en MLflow.
NOMBRES_RUN = {"baseline": "p1-baseline-semana-previa", "hgb": "p1-gradient-boosting"}


def metricas(real: pd.Series, prediccion: np.ndarray, prefijo: str) -> dict[str, float]:
    """MAE y RMSE en número de viajes por estación-día."""
    return {
        f"{prefijo}_mae": float(mean_absolute_error(real, prediccion)),
        f"{prefijo}_rmse": float(root_mean_squared_error(real, prediccion)),
    }


DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def error_por_dia_semana(
    prueba: pd.DataFrame, predicciones: dict[str, np.ndarray]
) -> pd.DataFrame:
    """MAE por día de la semana (0 = lunes) para cada modelo."""
    filas = {
        nombre: (prueba["viajes"] - pred).abs().groupby(prueba["dia_semana"]).mean()
        for nombre, pred in predicciones.items()
    }
    return pd.DataFrame(filas).rename_axis("dia_semana").reset_index()


def error_por_nivel_uso(
    entrenamiento: pd.DataFrame, prueba: pd.DataFrame, predicciones: dict[str, np.ndarray]
) -> pd.DataFrame:
    """MAE por nivel de uso de la estación (terciles de la demanda media en entrenamiento).

    Los terciles se calculan solo con el entrenamiento. Las estaciones sin historial de
    entrenamiento se reportan aparte, como "sin historial".
    """
    media = entrenamiento.groupby("estacion_id")["viajes"].mean()
    cortes = media.quantile([1 / 3, 2 / 3]).to_numpy()
    nivel = pd.cut(
        media, [-np.inf, cortes[0], cortes[1], np.inf], labels=["bajo", "medio", "alto"]
    )
    nivel_fila = prueba["estacion_id"].map(nivel).astype(object).fillna("sin historial")
    resultado = {"viajes_medios": prueba["viajes"].groupby(nivel_fila).mean()}
    for nombre, pred in predicciones.items():
        resultado[nombre] = (prueba["viajes"] - pred).abs().groupby(nivel_fila).mean()
    tabla = pd.DataFrame(resultado).rename_axis("nivel_uso").reset_index()
    tabla["filas"] = tabla["nivel_uso"].map(nivel_fila.value_counts())
    return tabla


def graficar(
    modelo: str,
    prueba: pd.DataFrame,
    predicciones: dict[str, np.ndarray],
    por_dia: pd.DataFrame,
    por_nivel: pd.DataFrame,
) -> list[Path]:
    """Genera las gráficas del run en reports/figures (`p1_<modelo>_*`) y regresa sus rutas."""
    plt.switch_backend("Agg")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rutas = []

    # 1. Demanda real y predicha, total por día en el periodo de prueba.
    diario = prueba[["fecha", "viajes"]].copy()
    for nombre, pred in predicciones.items():
        diario[nombre] = pred
    diario = diario.groupby("fecha").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(diario.index, diario["viajes"], color="black", label="Real")
    for nombre in predicciones:
        ax.plot(diario.index, diario[nombre], alpha=0.8, label=nombre)
    ax.set(title="P1: viajes por día en prueba (suma de las estaciones)", ylabel="Viajes")
    ax.legend()
    fig.autofmt_xdate()
    rutas.append(FIGURES_DIR / f"p1_{modelo}_real_vs_predicho.png")
    fig.savefig(rutas[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Error por día de la semana.
    fig, ax = plt.subplots(figsize=(8, 4))
    for nombre in predicciones:
        ax.plot(por_dia["dia_semana"], por_dia[nombre], marker="o", label=nombre)
    ax.set_xticks(range(7), DIAS)
    ax.set(title="P1: MAE por día de la semana (prueba)", ylabel="MAE (viajes)")
    ax.legend()
    rutas.append(FIGURES_DIR / f"p1_{modelo}_error_por_dia_semana.png")
    fig.savefig(rutas[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Error por nivel de uso de la estación.
    fig, ax = plt.subplots(figsize=(8, 4))
    posiciones = np.arange(len(por_nivel))
    ancho = 0.8 / len(predicciones)
    for i, nombre in enumerate(predicciones):
        ax.bar(posiciones + i * ancho, por_nivel[nombre], ancho, label=nombre)
    ax.set_xticks(posiciones + ancho * (len(predicciones) - 1) / 2, por_nivel["nivel_uso"])
    ax.set(title="P1: MAE por nivel de uso de la estación (prueba)", ylabel="MAE (viajes)")
    ax.legend()
    rutas.append(FIGURES_DIR / f"p1_{modelo}_error_por_nivel_uso.png")
    fig.savefig(rutas[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rutas


def parametros_comunes(datos: dict[str, pd.DataFrame]) -> dict[str, str | int]:
    return {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_inicio_validacion": FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": FECHA_INICIO_PRUEBA,
        "version_datos": version_datos(),
        "semilla": SEMILLA,
        "n_filas_entrenamiento": len(datos["entrenamiento"]),
        "n_filas_validacion": len(datos["validacion"]),
        "n_filas_prueba": len(datos["prueba"]),
    }


def entrenar(
    modelo: str,
    hiperparametros: dict,
    features_path: Path = P1_FEATURES_PATH,
    model_path: Path = P1_MODEL_PATH,
    muestra_entrenamiento: float = 1.0,
) -> dict[str, float]:
    """Entrena `modelo` con `hiperparametros`, lo evalúa y lo registra como un run de MLflow."""
    tabla = pd.read_parquet(features_path)
    datos = {nombre: grupo for nombre, grupo in tabla.groupby("conjunto", observed=True)}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        logger.info(f"{nombre}: {len(datos[nombre]):,} filas")
    entrenamiento, validacion, prueba = (
        datos[n] for n in ("entrenamiento", "validacion", "prueba")
    )
    if muestra_entrenamiento < 1.0:
        entrenamiento = entrenamiento.sample(frac=muestra_entrenamiento, random_state=SEMILLA)
        logger.warning(
            f"Entrenando con {len(entrenamiento):,} filas ({muestra_entrenamiento:.0%})"
        )

    usar_experimento()
    etiquetas = etiquetas_run("P1", __file__)

    # El baseline siempre se calcula: es el modelo del run o la referencia de sus gráficas.
    predicciones = {"baseline": prueba["lag_7d"].to_numpy()}
    if modelo == "baseline":
        pred_val = validacion["lag_7d"].to_numpy()
        params_modelo = {"estimador": "naive_estacional_semana_previa"}
        estimador = None
    else:
        estimador = HistGradientBoostingRegressor(
            **hiperparametros, early_stopping=False, random_state=SEMILLA
        ).fit(entrenamiento[COLUMNAS_P1], entrenamiento["viajes"])
        pred_val = np.clip(estimador.predict(validacion[COLUMNAS_P1]), 0, None)
        predicciones[modelo] = np.clip(estimador.predict(prueba[COLUMNAS_P1]), 0, None)
        params_modelo = {"estimador": "HistGradientBoostingRegressor", **hiperparametros}

    resultados = {
        **metricas(validacion["viajes"], pred_val, "val"),
        **metricas(prueba["viajes"], predicciones[modelo], "test"),
    }

    por_dia = error_por_dia_semana(prueba, predicciones)
    por_nivel = error_por_nivel_uso(entrenamiento, prueba, predicciones)
    artefactos = graficar(modelo, prueba, predicciones, por_dia, por_nivel)
    for nombre, df in {"error_por_dia_semana": por_dia, "error_por_nivel_uso": por_nivel}.items():
        artefactos.append(FIGURES_DIR / f"p1_{modelo}_{nombre}.csv")
        df.to_csv(artefactos[-1], index=False)
    if estimador is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimador, model_path)
        artefactos.append(model_path)

    with mlflow.start_run(run_name=NOMBRES_RUN[modelo], tags=etiquetas):
        mlflow.log_params({**parametros_comunes(datos), **params_modelo})
        mlflow.log_metrics(resultados)
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta))
    logger.success(f"{NOMBRES_RUN[modelo]}: {resultados}")
    return resultados


@app.command()
def main(
    modelo: str = typer.Option("config", help="baseline o hgb; config usa el de config.py."),
    hiperparametros: str = typer.Option(
        "config", help="clave=valor,clave=valor; config usa los de config.py."
    ),
    features_path: Path = P1_FEATURES_PATH,
    model_path: Path = P1_MODEL_PATH,
    muestra_entrenamiento: float = typer.Option(
        1.0, help="Fracción de filas de entrenamiento a usar (para pruebas rápidas)."
    ),
):
    modelo, valores = resolver_modelo("p1", modelo, hiperparametros)
    entrenar(modelo, valores, features_path, model_path, muestra_entrenamiento)


if __name__ == "__main__":
    app()
