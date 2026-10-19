from ast import Dict
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from loguru import logger
from pathlib import Path
import pandas as pd
import joblib
import mlflow
import typer

from mibici_mlops.config import P3_CLUSTERS_PATH, P3_MODEL_PATH, P3_PERFILES_PATH, RANDOM_STATE

app = typer.Typer()

class P3Trainer:
    METADATA_COLUMNS = [
        "origen_id", "destino_id", "origen_lat", "origen_lon", "destino_lat", "destino_lon",
    ]

    def __init__(self, df, parametros):
        self.df = df.copy()
        self.parametros = dict(parametros)

        if "origen_id" in self.df.columns:
            self.station_ids = self.df["origen_id"].to_numpy()
        else:
            self.station_ids = self.df.index.to_numpy()
        self.X = self.df.drop(columns=self.METADATA_COLUMNS, errors="ignore").select_dtypes(include=['number'])
        self.modelo = KMeans(**self.parametros)

    def fit_predict(self):
        self.etiquetas = self.modelo.fit_predict(self.X)

        self.asignaciones = pd.DataFrame({
            "origen_id": self.station_ids,
            "cluster": self.etiquetas,
        })

        # Añade coordenadas a las asignaciones si están en el archivo.
        for columna in ("origen_lat", "origen_lon"):
            if columna in self.df.columns:
                self.asignaciones[columna] = self.df[columna].to_numpy()

        n_clusters = len(set(self.etiquetas))
        if not 1 < n_clusters < len(self.X):
            raise ValueError(
                "Silhouette requiere al menos 2 clusters y menos clusters que filas."
            )

        self.metricas = {
            "silhouette": float(
                silhouette_score(self.X, self.etiquetas)
            ),
            "inertia": float(self.modelo.inertia_),
        }

        return self.asignaciones, self.metricas

    def save(self, model_path, output_path):
        joblib.dump(self.modelo, model_path)
        self.asignaciones.to_parquet(
            output_path,
            index=False,
        )

@app.command()
def p3train(
    input_path : Path = P3_PERFILES_PATH,
    output_path : Path = P3_CLUSTERS_PATH,
    model_path : Path = P3_MODEL_PATH,
    random_state : int = RANDOM_STATE,
    n_clusters : int = 3,
    n_init : int = 10
):
    df = pd.read_parquet(input_path)
    parametros = {
        'n_clusters': n_clusters,
        'n_init': n_init,
        'random_state': random_state
    }
    p3 = P3Trainer(df, parametros)
    p3.fit_predict()
    p3.save(model_path, output_path)


    logger.success(f"Modelo entrenado, datos guardados en {output_path}")


    with mlflow.start_run() as run:
        mlflow.log_params(p3.parametros)
        mlflow.log_metrics(p3.metricas)
        mlflow.sklearn.log_model(
            p3.modelo,
            name="nombre_modelo",
            input_example=p3.X.head(3),
            serialization_format="cloudpickle"
        )


if __name__ == "__main__":
    app()
