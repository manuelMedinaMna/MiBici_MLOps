"""Recupera del Model Registry el modelo que sirve la API.

El modelo se identifica por nombre y alias (SERVICIO_MODELO_URI, el `champion` de P1). Hay dos
formas de llegar a él, y la API usa la primera que aplique:

1. Si la variable de entorno MODELO_URI está definida, se carga del registry en el momento de
   arrancar (p. ej. MODELO_URI=models:/mibici-equipo43-p1-demanda@champion con
   MLFLOW_TRACKING_URI=http://host.containers.internal:5000). Requiere el servidor de MLflow.
2. Si no, se carga la copia que `exportar` descargó del registry en SERVICIO_MODELO_DIR y que se
   empaqueta en la imagen. Es la opción elegida (ver la sección de despliegue del README): el
   contenedor no depende de que el servidor de MLflow esté corriendo.

Junto al modelo, `exportar` guarda version.json con el nombre, alias, versión y run de origen,
para que la API diga qué versión del registry está sirviendo.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.api.modelo exportar
    python -m mibici_mlops.api.modelo exportar --uri models:/mibici-equipo43-p1-demanda@challenger
    python -m mibici_mlops.api.modelo probar
"""

import json
import os
from pathlib import Path
import shutil

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from mibici_mlops.config import P1_FEATURES_PATH, SERVICIO_MODELO_DIR, SERVICIO_MODELO_URI
from mibici_mlops.features import COLUMNAS_P1

app = typer.Typer()


def separar_uri(uri: str) -> tuple[str, str]:
    """Nombre y alias de un URI models:/<nombre>@<alias>."""
    if not uri.startswith("models:/") or "@" not in uri:
        raise ValueError(f"Se esperaba models:/<nombre>@<alias>, no {uri}")
    nombre, alias = uri.removeprefix("models:/").split("@", 1)
    return nombre, alias


def metadatos(uri: str) -> dict:
    """Versión y run de origen a los que apunta el alias en el registry."""
    nombre, alias = separar_uri(uri)
    version = MlflowClient().get_model_version_by_alias(nombre, alias)
    return {
        "nombre": nombre,
        "alias": alias,
        "version": version.version,
        "run_id": version.run_id,
    }


def cargar_modelo() -> tuple[mlflow.pyfunc.PyFuncModel, dict]:
    """El modelo que sirve la API y sus metadatos (nombre, alias, versión, run de origen)."""
    uri = os.getenv("MODELO_URI")
    if uri:
        logger.info(f"Cargando {uri} desde el registry en {mlflow.get_tracking_uri()}")
        return mlflow.pyfunc.load_model(uri), {**metadatos(uri), "origen": "registry"}

    if not (SERVICIO_MODELO_DIR / "MLmodel").exists():
        raise FileNotFoundError(
            f"No hay modelo en {SERVICIO_MODELO_DIR}: corre `python -m mibici_mlops.api.modelo "
            "exportar` con el servidor de MLflow levantado, o define MODELO_URI"
        )
    info = json.loads((SERVICIO_MODELO_DIR / "version.json").read_text(encoding="utf-8"))
    logger.info(
        f"Cargando la versión {info['version']} de {info['nombre']} empaquetada en la imagen"
    )
    return mlflow.pyfunc.load_model(str(SERVICIO_MODELO_DIR)), {**info, "origen": "imagen"}


@app.command()
def exportar(
    uri: str = typer.Option(
        SERVICIO_MODELO_URI, help="Modelo del registry: models:/<nombre>@<alias>."
    ),
    destino: Path = SERVICIO_MODELO_DIR,
):
    """Descarga del registry el modelo que se empaqueta en la imagen."""
    info = metadatos(uri)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    # Se descarga con la versión fija y no con el alias: si alguien mueve el alias mientras
    # tanto, el modelo y version.json siguen siendo la misma versión.
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{info['nombre']}/{info['version']}", dst_path=str(destino)
    )
    (destino / "version.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.success(
        f"{uri} (versión {info['version']}, run {info['run_id']}) guardado en {destino}"
    )


@app.command()
def probar():
    """Carga el modelo como lo hará la API y predice unas filas reales."""
    modelo, info = cargar_modelo()
    ejemplo = pd.read_parquet(P1_FEATURES_PATH, columns=[*COLUMNAS_P1, "viajes"]).tail(3)
    ejemplo["prediccion"] = modelo.predict(ejemplo[COLUMNAS_P1])
    logger.success(f"{info}\n{ejemplo[['viajes', 'prediccion']].round(2).to_string()}")


if __name__ == "__main__":
    app()
