"""Modelos de P2 registrados en MLflow.

Lee la tabla de `features.py` (data/processed/p2_ventanas.parquet), la parte por fechas
(decisión 6) y evalúa el modelo elegido con `--modelo` en los conjuntos de validación y prueba.
Por ahora hay dos baselines (decisiones 3 y 7):

- `mayoritaria`: clase mayoritaria (`DummyClassifier`), que siempre predice "sin riesgo";
- `frecuencia`: frecuencia de cada clase en las últimas 4 semanas (misma estación, día de la
  semana y ventana).

Cada llamada registra un run del experimento EXPERIMENTO_MLFLOW con: las fechas, la versión de los
datos, la semilla y los hiperparámetros como parámetros; recall, precisión y PR-AUC de cada clase
de riesgo como métricas `val_*` y `test_*`; el equipo, la pregunta, el commit, la familia del
modelo (`model_family`) y la etapa del trabajo (`stage_of_work`) como etiquetas, y la curva
precisión-recall de cada conjunto en `plots/`.

Uso:
    python -m mibici_mlops.modeling.train_p2 --modelo mayoritaria
    python -m mibici_mlops.modeling.train_p2 --modelo frecuencia

Antes de correrlo, el servidor de MLflow debe estar levantado (ver la sección de MLflow del
README):
    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination mlruns --port 5000 --workers 1

Los runs se ven en http://127.0.0.1:5000.
"""

from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
import typer

from mibici_mlops.config import (
    EXPERIMENTO_MLFLOW,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    MLFLOW_TRACKING_URI,
    P2_HORA_REINICIO,
    P2_INVENTARIO_INICIAL,
    P2_SEMANAS_HISTORIA,
    P2_VENTANA,
    P2_VENTANAS_PATH,
    SEMILLA,
)
from mibici_mlops.features import CLASES_P2
from mibici_mlops.modeling.registro import etiquetas_run, usar_experimento, version_datos

app = typer.Typer()

CLASES_RIESGO = ["se_vacia", "se_llena"]
NOMBRES = {"se_vacia": "Se vacía", "se_llena": "Se llena"}
# Conjuntos en que se evalúa cada modelo: prefijo de sus métricas → nombre en las gráficas.
NOMBRES_CONJUNTO = {"val": "validación", "test": "prueba"}

# Colores de la gráfica: espacios 1 y 2 de la paleta categórica de referencia (validada para
# daltonismo); el texto usa tinta neutra, nunca el color de la serie.
COLORES = {"se_vacia": "#2a78d6", "se_llena": "#eb6834"}
FONDO = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
CUADRICULA = "#e6e5e1"


def asignar_conjunto(dia: pd.Series) -> pd.Series:
    """Conjunto de cada ventana según su día: entrenamiento, validación o prueba (decisión 6)."""
    conjunto = pd.Series("entrenamiento", index=dia.index)
    conjunto[dia >= pd.Timestamp(FECHA_INICIO_VALIDACION)] = "validacion"
    conjunto[dia >= pd.Timestamp(FECHA_INICIO_PRUEBA)] = "prueba"
    return conjunto


def baseline_clase_mayoritaria(
    entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame
) -> pd.DataFrame:
    """Probabilidad de cada clase con `DummyClassifier(strategy="most_frequent")`.

    Aprende la clase más común en entrenamiento ("sin riesgo") y la predice siempre. Es el piso:
    no detecta ningún riesgo y aun así tiene exactitud alta, por eso la exactitud no sirve como
    métrica. No usa variables: recibe una columna de ceros porque scikit-learn pide una matriz.
    """
    modelo = DummyClassifier(strategy="most_frequent", random_state=SEMILLA)
    modelo.fit(np.zeros((len(entrenamiento), 1)), entrenamiento["clase"])
    probabilidades = modelo.predict_proba(np.zeros((len(evaluacion), 1)))
    return pd.DataFrame(probabilidades, columns=modelo.classes_, index=evaluacion.index)[CLASES_P2]


def baseline_frecuencia(entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada clase: su frecuencia en las últimas P2_SEMANAS_HISTORIA semanas.

    No se entrena: usa las columnas `frec_*` de features.py, que solo miran semanas anteriores.
    Recibe `entrenamiento` aunque no lo use, para que todos los modelos se llamen igual.
    """
    columnas = [f"frec_{clase}" for clase in CLASES_P2]
    return evaluacion[columnas].set_axis(CLASES_P2, axis=1)


# Modelos que se pueden correr con --modelo; cada llamada registra un run. Por modelo: nombre del
# run, familia y etapa (etiquetas model_family y stage_of_work), hiperparámetros y la función que
# da la probabilidad de cada clase a partir de (entrenamiento, conjunto a evaluar).
MODELOS = {
    "mayoritaria": {
        "run": "p2-baseline-clase-mayoritaria",
        "familia": "dummy",
        "etapa": "baseline",
        "hiperparametros": {"modelo": "DummyClassifier", "strategy": "most_frequent"},
        "probabilidades": baseline_clase_mayoritaria,
    },
    "frecuencia": {
        "run": f"p2-baseline-frecuencia-{P2_SEMANAS_HISTORIA}-semanas",
        "familia": "heuristica",
        "etapa": "baseline",
        "hiperparametros": {
            "modelo": "frecuencia_semanas_anteriores",
            "semanas": P2_SEMANAS_HISTORIA,
            "desempate": " > ".join(CLASES_P2),
        },
        "probabilidades": baseline_frecuencia,
    },
}


def predecir(probabilidades: pd.DataFrame) -> pd.Series:
    """Clase con mayor probabilidad. En empate gana la primera de CLASES_P2 (decisión 7)."""
    indice = probabilidades[CLASES_P2].to_numpy().argmax(axis=1)
    return pd.Series(np.array(CLASES_P2)[indice], index=probabilidades.index)


def calcular_metricas(
    clase: pd.Series, probabilidades: pd.DataFrame, prefijo: str
) -> dict[str, float]:
    """Recall, precisión y PR-AUC de cada clase de riesgo, con el prefijo del conjunto (val, test).

    También registra la proporción de cada clase: es la PR-AUC que tendría un modelo al azar, y
    sirve para leer las demás métricas.
    """
    prediccion = predecir(probabilidades)
    metricas = {}
    for riesgo in CLASES_RIESGO:
        real = clase == riesgo
        predicha = prediccion == riesgo
        metricas[f"{prefijo}_recall_{riesgo}"] = recall_score(real, predicha)
        metricas[f"{prefijo}_precision_{riesgo}"] = precision_score(
            real, predicha, zero_division=0
        )
        metricas[f"{prefijo}_pr_auc_{riesgo}"] = average_precision_score(
            real, probabilidades[riesgo]
        )
        metricas[f"{prefijo}_proporcion_{riesgo}"] = real.mean()
    return metricas


def graficar_curvas_pr(clase: pd.Series, probabilidades: pd.DataFrame, titulo: str) -> Figure:
    """Curva precisión-recall de cada clase de riesgo, con su proporción como referencia del azar.

    El punto marca la decisión del baseline (la clase más probable); no aparece si el baseline
    nunca predice esa clase.
    """
    prediccion = predecir(probabilidades)
    figura = Figure(figsize=(10, 4.2), facecolor=FONDO)
    ejes = figura.subplots(1, 2, sharey=True)
    for eje, riesgo in zip(ejes, CLASES_RIESGO):
        real = clase == riesgo
        precision, recall, _ = precision_recall_curve(real, probabilidades[riesgo])
        pr_auc = average_precision_score(real, probabilidades[riesgo])
        eje.step(recall, precision, where="post", color=COLORES[riesgo], linewidth=1.5)

        eje.axhline(real.mean(), color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
        eje.text(
            0.01,
            real.mean(),
            f" azar: {real.mean():.1%}",
            transform=eje.get_yaxis_transform(),
            va="bottom",
            color=TEXTO_SECUNDARIO,
            fontsize=9,
        )
        predicha = prediccion == riesgo
        if predicha.any():
            punto = (recall_score(real, predicha), precision_score(real, predicha))
            eje.plot(*punto, "o", color=COLORES[riesgo], markersize=7, markeredgecolor=FONDO)
            eje.annotate(
                "decisión del baseline",
                punto,
                xytext=(8, 6),
                textcoords="offset points",
                color=TEXTO_SECUNDARIO,
                fontsize=9,
            )

        eje.set_title(f"{NOMBRES[riesgo]} · PR-AUC {pr_auc:.3f}", loc="left", color=TEXTO)
        eje.set_xlabel("Recall", color=TEXTO_SECUNDARIO)
        eje.set_xlim(0, 1)
        eje.set_ylim(0, 1.02)
        eje.set_facecolor(FONDO)
        eje.grid(color=CUADRICULA, linewidth=0.8)
        eje.set_axisbelow(True)
        eje.tick_params(colors=TEXTO_SECUNDARIO)
        eje.spines[["top", "right"]].set_visible(False)
        eje.spines[["left", "bottom"]].set_color(CUADRICULA)
    ejes[0].set_ylabel("Precisión", color=TEXTO_SECUNDARIO)
    figura.suptitle(titulo, x=0.01, ha="left", color=TEXTO)
    figura.tight_layout()
    return figura


def registrar_run(
    nombre: str,
    hiperparametros: dict,
    conjuntos: dict[str, pd.DataFrame],
    probabilidades: dict[str, pd.DataFrame],
    parametros: dict,
    etiquetas: dict,
) -> dict[str, float]:
    """Registra un modelo como un run de MLflow: parámetros, etiquetas, métricas y gráficas.

    `conjuntos` y `probabilidades` van por prefijo (val, test): cada conjunto da sus métricas y
    su curva precisión-recall en plots/.
    """
    metricas = {}
    with mlflow.start_run(run_name=nombre, tags=etiquetas):
        mlflow.log_params({**parametros, **hiperparametros})
        for prefijo, datos in conjuntos.items():
            metricas.update(calcular_metricas(datos["clase"], probabilidades[prefijo], prefijo))
            figura = graficar_curvas_pr(
                datos["clase"],
                probabilidades[prefijo],
                f"P2 · {nombre} (conjunto de {NOMBRES_CONJUNTO[prefijo]})",
            )
            mlflow.log_figure(figura, f"plots/curva_precision_recall_{prefijo}.png")
        mlflow.log_metrics(metricas)
    resumen = {
        metrica: round(valor, 3)
        for metrica, valor in metricas.items()
        if "proporcion" not in metrica
    }
    logger.info(f"{nombre}: {resumen}")
    return metricas


@app.command()
def main(
    modelo: str = typer.Option(..., help=f"Modelo a correr: {', '.join(MODELOS)}."),
    ventanas_path: Path = P2_VENTANAS_PATH,
):
    if modelo not in MODELOS:
        raise typer.BadParameter(f"usa uno de: {', '.join(MODELOS)}", param_hint="--modelo")
    elegido = MODELOS[modelo]

    tabla = pd.read_parquet(ventanas_path)
    tabla["conjunto"] = asignar_conjunto(tabla["dia"])
    entrenamiento = tabla[tabla["conjunto"] == "entrenamiento"]
    conjuntos = {
        "val": tabla[tabla["conjunto"] == "validacion"],
        "test": tabla[tabla["conjunto"] == "prueba"],
    }
    logger.info(
        f"Ventanas: {len(entrenamiento):,} de entrenamiento, {len(conjuntos['val']):,} de "
        f"validación y {len(conjuntos['test']):,} de prueba"
    )

    etiquetas = {
        **etiquetas_run("P2", __file__),
        "model_family": elegido["familia"],
        "stage_of_work": elegido["etapa"],
    }
    parametros = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_inicio_validacion": FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": FECHA_INICIO_PRUEBA,
        "version_datos": version_datos(),
        "semilla": SEMILLA,
        "ventana": P2_VENTANA,
        "hora_reinicio": P2_HORA_REINICIO,
        "inventario_inicial": P2_INVENTARIO_INICIAL,
        "ventanas_entrenamiento": len(entrenamiento),
        "ventanas_validacion": len(conjuntos["val"]),
        "ventanas_prueba": len(conjuntos["test"]),
    }

    usar_experimento()

    probabilidades = {
        prefijo: elegido["probabilidades"](entrenamiento, datos)
        for prefijo, datos in conjuntos.items()
    }
    registrar_run(
        elegido["run"],
        elegido["hiperparametros"],
        conjuntos,
        probabilidades,
        parametros,
        etiquetas,
    )
    logger.success(
        f"Run {elegido['run']} registrado en el experimento {EXPERIMENTO_MLFLOW} "
        f"({MLFLOW_TRACKING_URI})"
    )


if __name__ == "__main__":
    app()
