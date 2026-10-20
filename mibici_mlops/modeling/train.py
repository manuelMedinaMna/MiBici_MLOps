"""Punto de entrada `train` del MLproject: entrena un modelo de P1, P2 o P3.

Recibe la pregunta, el modelo y sus hiperparámetros. Con "config", el modelo y los
hiperparámetros son los de config.py (MODELO_POR_DEFECTO y MODELOS); las rutas, fechas y semillas
las toma cada script de config.py. Cada pregunta tiene su propio script en modeling/; este solo
elige cuál correr.

Uso (con el servidor de MLflow levantado, ver la sección de MLflow del README):
    mlflow run . -e train -P pregunta=p1 --experiment-name mibici-equipo43 --env-manager local
    mlflow run . -e train -P pregunta=p1 -P hiperparametros=max_iter=200,max_leaf_nodes=31 ...
    python -m mibici_mlops.modeling.train --pregunta p2 --modelo mayoritaria
"""

from loguru import logger
import typer

from mibici_mlops.modeling import train_p1, train_p2, train_p3
from mibici_mlops.modeling.registro import resolver_modelo

app = typer.Typer()


@app.command()
def main(
    pregunta: str = typer.Option(..., help="p1, p2 o p3."),
    modelo: str = typer.Option("config", help="Modelo de la pregunta; config usa config.py."),
    hiperparametros: str = typer.Option(
        "config", help="clave=valor,clave=valor; config usa los de config.py."
    ),
):
    modelo, valores = resolver_modelo(pregunta, modelo, hiperparametros)
    logger.info(f"{pregunta.upper()}: modelo {modelo} con hiperparámetros {valores}")
    if pregunta == "p1":
        train_p1.entrenar(modelo, valores)
    elif pregunta == "p2":
        train_p2.entrenar(modelo, valores)
    else:
        # Pendiente: la versión actual de train_p3.py no usa registro.py ni la partición común
        # (ver la nota de P3 en la sección del MLproject del README).
        train_p3.p3train(n_clusters=valores["n_clusters"], n_init=valores["n_init"])


if __name__ == "__main__":
    app()
