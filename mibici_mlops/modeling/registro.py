"""Funciones comunes para registrar los runs de P1, P2 y P3 en MLflow.

Todas las preguntas registran sus runs en el mismo experimento, con la misma versión de los
datos y las mismas etiquetas de equipo, pregunta y commit. Tenerlas en un solo lugar evita que
cada script lo haga distinto. También resuelve qué modelo e hiperparámetros entrenar, con los
valores por defecto de config.py (MODELOS y MODELO_POR_DEFECTO).
"""

import os
from pathlib import Path
import subprocess
import urllib.request

from loguru import logger
import mlflow

from mibici_mlops.config import (
    COMANDO_SERVIDOR_MLFLOW,
    DATA_DIR,
    EXPERIMENTO_MLFLOW,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    MODELO_POR_DEFECTO,
    MODELOS,
    PROJ_ROOT,
)

EQUIPO = "43"


def version_datos(ruta: Path = DATA_DIR / "raw.dvc") -> str:
    """md5 que DVC calculó para data/raw (decisión 7 de P2): cambia si cambia cualquier CSV."""
    for linea in ruta.read_text().splitlines():
        if "md5:" in linea:
            return linea.split("md5:")[1].strip()
    raise ValueError(f"No se encontró el md5 en {ruta}")


def estado_git() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin commit: liga cada run con el código que lo produjo."""

    def git(*argumentos: str) -> str:
        resultado = subprocess.run(
            ["git", *argumentos], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        )
        return resultado.stdout.strip()

    return git("rev-parse", "HEAD"), git("status", "--porcelain") != ""


def etiquetas_run(pregunta: str, script: str) -> dict[str, str]:
    """Etiquetas de equipo, pregunta y commit para los runs de `script` (su `__file__`).

    mlflow.user y mlflow.source.name se reemplazan para no registrar el usuario de la computadora
    ni la ruta de su carpeta personal (datos personales).
    """
    commit, cambios_sin_commit = estado_git()
    if cambios_sin_commit:
        logger.warning(
            "Hay cambios sin commit: el commit registrado no es todo el código que corrió"
        )
    return {
        "equipo": EQUIPO,
        "pregunta": pregunta,
        "commit": commit,
        "cambios_sin_commit": str(cambios_sin_commit).lower(),
        "mlflow.user": f"equipo{EQUIPO}",
        "mlflow.source.name": Path(script).resolve().relative_to(PROJ_ROOT).as_posix(),
    }


def verificar_servidor(uri: str = MLFLOW_TRACKING_URI) -> None:
    """Falla de inmediato si el servidor de MLflow no responde.

    Sin esto, MLflow reintenta la conexión durante varios minutos antes de dar un error.
    """
    if not uri.startswith("http"):
        return
    try:
        urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=5)
    except OSError as error:
        raise RuntimeError(
            f"No hay un servidor de MLflow en {uri}. Levántalo en otra terminal, desde la raíz "
            f"del proyecto, con:\n    {COMANDO_SERVIDOR_MLFLOW}"
        ) from error


def verificar_mlflow_run() -> None:
    """Si el script lo lanzó `mlflow run`, revisa que su run use el servidor y el experimento.

    `mlflow run` crea el run antes de correr el script y le pasa su id (MLFLOW_RUN_ID) y su
    servidor (MLFLOW_TRACKING_URI). Si se lanzó sin esas opciones, el run quedaría fuera del
    servidor o en otro experimento; es mejor fallar con las instrucciones.
    """
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id is None:
        return
    if not MLFLOW_TRACKING_URI.startswith("http"):
        raise RuntimeError(
            f"`mlflow run` está guardando el run en {MLFLOW_TRACKING_URI}, no en el servidor. "
            'Antes de correrlo: $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"'
        )
    experimento = mlflow.get_experiment(mlflow.get_run(run_id).info.experiment_id).name
    if experimento != EXPERIMENTO_MLFLOW:
        raise RuntimeError(
            f"`mlflow run` creó el run en el experimento '{experimento}'. Córrelo con "
            f"--experiment-name {EXPERIMENTO_MLFLOW}"
        )


def usar_experimento() -> None:
    """Apunta MLflow al servidor de tracking y al experimento del equipo; lo crea si no existe."""
    verificar_servidor()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    verificar_mlflow_run()
    if mlflow.get_experiment_by_name(EXPERIMENTO_MLFLOW) is None:
        mlflow.create_experiment(
            EXPERIMENTO_MLFLOW, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)


def resolver_modelo(
    pregunta: str, modelo: str = "config", hiperparametros: str = "config"
) -> tuple[str, dict]:
    """Modelo e hiperparámetros que se van a entrenar: los de config.py, con los cambios pedidos.

    `modelo="config"` toma el de MODELO_POR_DEFECTO. `hiperparametros="config"` (o vacío) usa los
    de MODELOS; `"clave=valor,clave=valor"` cambia solo esas claves. Ese formato no necesita
    comillas, así que pasa igual por `mlflow run` en cmd de Windows y en bash.
    """
    if pregunta not in MODELOS:
        raise ValueError(f"Pregunta '{pregunta}' desconocida. Opciones: {', '.join(MODELOS)}")
    modelos = MODELOS[pregunta]
    if modelo == "config":
        modelo = MODELO_POR_DEFECTO[pregunta]
    if modelo not in modelos:
        raise ValueError(
            f"{pregunta} no tiene el modelo '{modelo}'. Opciones: {', '.join(modelos)}"
        )

    valores = dict(modelos[modelo])
    if hiperparametros in ("", "config"):
        return modelo, valores
    for par in hiperparametros.split(","):
        clave, _, texto = par.partition("=")
        if clave not in valores:
            permitidos = ", ".join(valores) or "ninguno"
            raise ValueError(
                f"'{clave}' no es un hiperparámetro de {pregunta}/{modelo}. Opciones: {permitidos}"
            )
        # Se convierte al tipo del valor por defecto: "300" -> 300, "0.05" -> 0.05.
        valores[clave] = type(valores[clave])(texto)
    return modelo, valores
