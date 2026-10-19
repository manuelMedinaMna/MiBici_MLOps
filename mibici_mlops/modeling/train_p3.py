"""Baseline y modelo candidato de P3 registrados en MLflow.

Lee la tabla de `features.py` (data/processed/p3_perfiles.parquet) y agrupa las estaciones de dos
formas (decisiones 5 y 6 de references/decisiones_p3.md):

- baseline: KMeans sobre la latitud y la longitud, es decir, zonas de la ciudad que no saben
  nada de cómo se usa cada estación;
- candidato: KMeans sobre el perfil de uso (fracción de salidas y llegadas por día y franja).

Las dos se evalúan en el mismo espacio, el perfil de uso, para responder si agrupar por uso da
grupos más parecidos por dentro que agrupar por cercanía: silhouette, Davies-Bouldin y
Calinski-Harabasz con el perfil de entrenamiento (`train_*`) y con el de prueba (`test_*`), y la
estabilidad (`test_ari_estabilidad`): qué tanto conserva cada estación su grupo de un periodo a
otro.

Cada modelo es un run del experimento EXPERIMENTO_MLFLOW con las fechas, la versión de los datos,
la semilla y los hiperparámetros como parámetros; el equipo, la pregunta y el commit como
etiquetas; y como artefactos el mapa de los grupos, su perfil de uso y la tabla de estaciones.

Uso:
    python -m mibici_mlops.modeling.train_p3

Para ver los runs:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici_mlops.config import (
    ESTACIONES_PATH,
    EXPERIMENTO_MLFLOW,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    MLFLOW_TRACKING_URI,
    P3_FRANJAS,
    P3_GRUPOS,
    P3_GRUPOS_BUSQUEDA,
    P3_MIN_VIAJES,
    P3_PERFILES_PATH,
    SEMILLA,
)
from mibici_mlops.features import DIAS_SEMANA, columnas_perfil
from mibici_mlops.modeling.registro import etiquetas_run, usar_experimento, version_datos

app = typer.Typer()

COLUMNAS_SALIDAS = columnas_perfil("sal")
COLUMNAS_LLEGADAS = columnas_perfil("lle")
COLUMNAS_PERFIL = COLUMNAS_SALIDAS + COLUMNAS_LLEGADAS
COORDENADAS = ["latitude", "longitude"]
N_INIT = 10

# Colores de los grupos: paleta categórica de referencia en orden fijo. En un mapa de puntos solo
# los 3 primeros se distinguen bien con daltonismo; con más grupos habría que facetar el mapa.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FONDO = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
CUADRICULA = "#e6e5e1"


def cargar_perfiles(
    perfiles_path: Path, estaciones_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfiles de entrenamiento y de prueba, indexados por estación, con nombre y coordenadas."""
    catalogo = pd.read_csv(estaciones_path).set_index("id")[["name", *COORDENADAS]]
    tabla = pd.read_parquet(perfiles_path).join(catalogo, on="estacion").set_index("estacion")
    return (
        tabla[tabla["conjunto"] == "entrenamiento"],
        tabla[tabla["conjunto"] == "prueba"],
    )


def kmeans() -> KMeans:
    """KMeans con P3_GRUPOS grupos y la semilla del proyecto, igual para baseline y candidato."""
    return KMeans(n_clusters=P3_GRUPOS, n_init=N_INIT, random_state=SEMILLA)


def baseline_zonas(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Grupos por cercanía: KMeans sobre la latitud y la longitud estandarizadas.

    Es el agrupamiento que haría alguien sin datos de uso, dividiendo el mapa en zonas. Si el
    perfil de uso no da grupos mejores que este, no aporta nada a planeación urbana.
    """
    modelo = make_pipeline(StandardScaler(), kmeans())
    return modelo.fit_predict(entrenamiento[COORDENADAS]), modelo.predict(prueba[COORDENADAS])


def kmeans_perfil(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Grupos por patrón de uso: KMeans sobre las 56 fracciones de salidas y llegadas.

    No se estandariza: todas las columnas son fracciones en la misma escala y estandarizar
    inflaría las combinaciones con muy pocos viajes, como la madrugada (decisión 6).
    """
    modelo = kmeans().fit(entrenamiento[COLUMNAS_PERFIL])
    return modelo.labels_, modelo.predict(prueba[COLUMNAS_PERFIL])


def calcular_metricas(
    entrenamiento: pd.DataFrame,
    prueba: pd.DataFrame,
    grupos_entrenamiento: np.ndarray,
    grupos_prueba: np.ndarray,
) -> dict[str, float]:
    """Calidad de los grupos en el espacio del perfil de uso, con el prefijo del conjunto.

    - silhouette (de -1 a 1, más es mejor): qué tan cerca está cada estación de su grupo
      comparado con el grupo más cercano;
    - Davies-Bouldin (desde 0, menos es mejor): qué tan separados están los grupos respecto a lo
      dispersos que son;
    - Calinski-Harabasz (más es mejor): varianza entre grupos contra varianza dentro de ellos;
    - ARI de estabilidad (de -1 a 1, 1 = mismos grupos): compara el grupo de cada estación en
      entrenamiento y en prueba. Para el baseline es 1 por construcción, porque las estaciones no
      se mueven; sirve para ver si el candidato encuentra patrones que se mantienen.
    """
    metricas = {}
    for prefijo, tabla, grupos in [
        ("train", entrenamiento, grupos_entrenamiento),
        ("test", prueba, grupos_prueba),
    ]:
        perfil = tabla[COLUMNAS_PERFIL]
        metricas[f"{prefijo}_silhouette"] = silhouette_score(perfil, grupos)
        metricas[f"{prefijo}_davies_bouldin"] = davies_bouldin_score(perfil, grupos)
        metricas[f"{prefijo}_calinski_harabasz"] = calinski_harabasz_score(perfil, grupos)

    en_ambos = entrenamiento.index.intersection(prueba.index)
    metricas["test_ari_estabilidad"] = adjusted_rand_score(
        pd.Series(grupos_entrenamiento, index=entrenamiento.index)[en_ambos],
        pd.Series(grupos_prueba, index=prueba.index)[en_ambos],
    )
    for grupo, cantidad in enumerate(np.bincount(grupos_entrenamiento, minlength=P3_GRUPOS)):
        metricas[f"train_estaciones_grupo_{grupo + 1}"] = cantidad
    return metricas


def estilizar(eje) -> None:
    """Fondo, cuadrícula y ejes discretos, iguales a las gráficas de P2."""
    eje.set_facecolor(FONDO)
    eje.grid(color=CUADRICULA, linewidth=0.8)
    eje.set_axisbelow(True)
    eje.tick_params(colors=TEXTO_SECUNDARIO)
    eje.spines[["top", "right"]].set_visible(False)
    eje.spines[["left", "bottom"]].set_color(CUADRICULA)


def graficar_mapa(tabla: pd.DataFrame, grupos: np.ndarray, titulo: str) -> Figure:
    """Mapa de las estaciones con el color de su grupo (respuesta a "dónde se ubican")."""
    figura = Figure(figsize=(7, 7.5), facecolor=FONDO)
    eje = figura.subplots()
    for grupo in range(P3_GRUPOS):
        puntos = tabla[grupos == grupo]
        eje.scatter(
            puntos["longitude"],
            puntos["latitude"],
            s=30,
            color=COLORES[grupo],
            edgecolors=FONDO,
            linewidths=0.6,
            label=f"Grupo {grupo + 1} ({len(puntos)})",
        )
    # Un grado de longitud mide menos que uno de latitud a esta altura del planeta.
    eje.set_aspect(1 / np.cos(np.radians(tabla["latitude"].mean())))
    eje.set_xlabel("Longitud", color=TEXTO_SECUNDARIO)
    eje.set_ylabel("Latitud", color=TEXTO_SECUNDARIO)
    estilizar(eje)
    eje.set_title(f"{titulo} · estaciones por grupo", loc="left", color=TEXTO)
    # La leyenda va debajo del mapa para no tapar estaciones.
    figura.legend(frameon=False, labelcolor=TEXTO, loc="lower center", ncols=P3_GRUPOS)
    figura.tight_layout(rect=(0, 0.04, 1, 1))
    return figura


def graficar_perfiles(tabla: pd.DataFrame, grupos: np.ndarray, titulo: str) -> Figure:
    """Perfil promedio de cada grupo: qué fracción de sus salidas y llegadas cae en cada franja.

    Es lo que dice "qué patrón comparten": por ejemplo, un grupo con salidas en la mañana y
    llegadas en la tarde entre semana.
    """
    figura = Figure(figsize=(12, 6.5), facecolor=FONDO)
    ejes = figura.subplots(2, 1, sharex=True, sharey=True)
    posiciones = np.arange(len(COLUMNAS_SALIDAS))
    for eje, columnas, nombre in [
        (ejes[0], COLUMNAS_SALIDAS, "Salidas"),
        (ejes[1], COLUMNAS_LLEGADAS, "Llegadas"),
    ]:
        promedio = tabla[columnas].groupby(grupos).mean()
        for grupo, fila in promedio.iterrows():
            eje.plot(
                posiciones,
                fila.to_numpy(),
                color=COLORES[grupo],
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"Grupo {grupo + 1}",
            )
        # Una línea vertical separa cada día; las franjas van de madrugada a noche dentro de él.
        for limite in range(len(P3_FRANJAS), len(posiciones), len(P3_FRANJAS)):
            eje.axvline(limite - 0.5, color=CUADRICULA, linewidth=1)
        eje.set_title(nombre, loc="left", color=TEXTO)
        eje.set_ylabel("Fracción de los viajes", color=TEXTO_SECUNDARIO)
        estilizar(eje)
        eje.grid(axis="x", visible=False)
    ejes[0].legend(frameon=False, labelcolor=TEXTO, ncols=P3_GRUPOS, loc="upper right")
    centros = np.arange(len(DIAS_SEMANA)) * len(P3_FRANJAS) + (len(P3_FRANJAS) - 1) / 2
    ejes[1].set_xticks(centros, DIAS_SEMANA)
    ejes[1].set_xlabel(
        f"Día de la semana; dentro de cada día, franjas: {', '.join(P3_FRANJAS)}",
        color=TEXTO_SECUNDARIO,
    )
    figura.suptitle(titulo, x=0.01, ha="left", color=TEXTO)
    figura.tight_layout()
    return figura


def graficar_busqueda(silhouettes: dict[int, float]) -> Figure:
    """Silhouette de entrenamiento según el número de grupos, con el elegido marcado."""
    figura = Figure(figsize=(7, 4), facecolor=FONDO)
    eje = figura.subplots()
    eje.plot(list(silhouettes), list(silhouettes.values()), color=COLORES[0], linewidth=2)
    eje.plot(list(silhouettes), list(silhouettes.values()), "o", color=COLORES[0], markersize=7)
    eje.annotate(
        f"elegido: {P3_GRUPOS} grupos",
        (P3_GRUPOS, silhouettes[P3_GRUPOS]),
        xytext=(10, -4),
        textcoords="offset points",
        color=TEXTO_SECUNDARIO,
        fontsize=9,
    )
    eje.set_xlabel("Número de grupos", color=TEXTO_SECUNDARIO)
    eje.set_ylabel("Silhouette (entrenamiento)", color=TEXTO_SECUNDARIO)
    estilizar(eje)
    eje.set_title("P3 · KMeans sobre el perfil de uso", loc="left", color=TEXTO)
    figura.tight_layout()
    return figura


def buscar_grupos(entrenamiento: pd.DataFrame) -> dict[int, float]:
    """Silhouette de entrenamiento para cada número de grupos de P3_GRUPOS_BUSQUEDA (decisión 5)."""
    perfil = entrenamiento[COLUMNAS_PERFIL]
    return {
        grupos: silhouette_score(
            perfil,
            KMeans(n_clusters=grupos, n_init=N_INIT, random_state=SEMILLA).fit_predict(perfil),
        )
        for grupos in P3_GRUPOS_BUSQUEDA
    }


def tabla_estaciones(
    entrenamiento: pd.DataFrame,
    prueba: pd.DataFrame,
    grupos_entrenamiento: np.ndarray,
    grupos_prueba: np.ndarray,
) -> pd.DataFrame:
    """Grupo de cada estación en cada periodo, con su ubicación y uso. Solo datos públicos."""
    tabla = entrenamiento[["name", *COORDENADAS, "salidas_dia", "llegadas_dia", "balance"]].copy()
    tabla["grupo_entrenamiento"] = grupos_entrenamiento + 1
    tabla["grupo_prueba"] = pd.Series(grupos_prueba + 1, index=prueba.index).astype("Int64")
    return tabla.round(4).reset_index()


def registrar_run(
    nombre: str,
    hiperparametros: dict,
    entrenamiento: pd.DataFrame,
    prueba: pd.DataFrame,
    grupos: tuple[np.ndarray, np.ndarray],
    parametros: dict,
    etiquetas: dict,
    busqueda: dict[int, float] | None = None,
) -> dict[str, float]:
    """Registra un agrupamiento como un run de MLflow: parámetros, etiquetas, métricas y gráficas."""
    grupos_entrenamiento, grupos_prueba = grupos
    with mlflow.start_run(run_name=nombre, tags=etiquetas):
        mlflow.log_params({**parametros, **hiperparametros})
        metricas = calcular_metricas(entrenamiento, prueba, grupos_entrenamiento, grupos_prueba)
        mlflow.log_metrics(metricas)
        mlflow.log_figure(
            graficar_mapa(entrenamiento, grupos_entrenamiento, f"P3 · {nombre}"),
            "mapa_grupos.png",
        )
        mlflow.log_figure(
            graficar_perfiles(
                entrenamiento, grupos_entrenamiento, f"P3 · {nombre} · perfil de uso por grupo"
            ),
            "perfil_grupos.png",
        )
        mlflow.log_table(
            tabla_estaciones(entrenamiento, prueba, grupos_entrenamiento, grupos_prueba),
            "estaciones_grupos.json",
        )
        if busqueda is not None:
            for grupos_k, valor in busqueda.items():
                mlflow.log_metric("train_silhouette_busqueda", valor, step=grupos_k)
            mlflow.log_figure(graficar_busqueda(busqueda), "silhouette_por_numero_de_grupos.png")
    resumen = {
        metrica: round(valor, 3)
        for metrica, valor in metricas.items()
        if "estaciones_grupo" not in metrica
    }
    logger.info(f"{nombre}: {resumen}")
    return metricas


@app.command()
def main(perfiles_path: Path = P3_PERFILES_PATH, estaciones_path: Path = ESTACIONES_PATH):
    entrenamiento, prueba = cargar_perfiles(perfiles_path, estaciones_path)
    logger.info(f"Estaciones: {len(entrenamiento)} de entrenamiento y {len(prueba)} de prueba")

    etiquetas = etiquetas_run("P3", __file__)
    parametros = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_inicio_validacion": FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": FECHA_INICIO_PRUEBA,
        "version_datos": version_datos(),
        "semilla": SEMILLA,
        "franjas": ", ".join(f"{franja}={hora}h" for franja, hora in P3_FRANJAS.items()),
        "min_viajes": P3_MIN_VIAJES,
        "estaciones_entrenamiento": len(entrenamiento),
        "estaciones_prueba": len(prueba),
    }
    hiperparametros = {"modelo": "KMeans", "n_clusters": P3_GRUPOS, "n_init": N_INIT}

    usar_experimento()

    registrar_run(
        f"p3-baseline-zonas-geograficas-k{P3_GRUPOS}",
        {**hiperparametros, "variables": "latitud y longitud estandarizadas"},
        entrenamiento,
        prueba,
        baseline_zonas(entrenamiento, prueba),
        parametros,
        etiquetas,
    )
    registrar_run(
        f"p3-kmeans-perfil-uso-k{P3_GRUPOS}",
        {**hiperparametros, "variables": f"perfil de uso ({len(COLUMNAS_PERFIL)} fracciones)"},
        entrenamiento,
        prueba,
        kmeans_perfil(entrenamiento, prueba),
        parametros,
        etiquetas,
        busqueda=buscar_grupos(entrenamiento),
    )
    logger.success(
        f"Runs registrados en el experimento {EXPERIMENTO_MLFLOW} ({MLFLOW_TRACKING_URI})"
    )


if __name__ == "__main__":
    app()
