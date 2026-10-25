from sklearn.metrics import adjusted_rand_score, silhouette_score
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from loguru import logger
from pathlib import Path
from ast import Dict
import pandas as pd
import joblib
import mlflow
import typer

from mibici_mlops.config import ESTACIONES_PATH, P3_CLUSTERS_PATH, P3_MODEL_PATH, P3_PERFILES_PATH, RANDOM_STATE

app = typer.Typer()

class P3Trainer:
    def __init__(self, perfiles, estaciones, parametros):
        self.perfiles = perfiles.copy()
        self.estaciones = estaciones[["id", "latitude", "longitude"]].rename(columns={
            "id": "estacion",
            "latitude": "estacion_lat",
            "longitude": "estacion_lon",
        })
        self.parametros = dict(parametros)

        self.ids_estaciones_train = self.perfiles[self.perfiles.conjunto=='entrenamiento'].estacion.to_numpy()
        self.ids_estaciones_test = self.perfiles[self.perfiles.conjunto=='prueba'].estacion.to_numpy()

        conjunto = self.perfiles.conjunto
        columnas_perfil = [c for c in self.perfiles.columns if c.startswith(('sal_', 'lle_'))]
        X_perfiles = self.perfiles[columnas_perfil].astype('float64')

        self.X = X_perfiles.loc[conjunto == 'entrenamiento']
        self.X_test = X_perfiles.loc[conjunto == 'prueba']

        self.modelo = KMeans(**self.parametros)
        self.modelo_test = KMeans(**self.parametros)

    def fit_predict(self):
        self.etiquetas = self.modelo.fit_predict(self.X)
        self.etiquetas_test = self.modelo_test.fit_predict(self.X_test)

        self.asignaciones = pd.concat(
            [
                pd.DataFrame({
                    "estacion": self.ids_estaciones_train,
                    "conjunto": "entrenamiento",
                    "cluster": self.etiquetas,
                }),
                pd.DataFrame({
                    "estacion": self.ids_estaciones_test,
                    "conjunto": "prueba",
                    "cluster": self.etiquetas_test,
                }),
            ],
            ignore_index=True,
        )

        # Añadir coordenadas a las asignaciones
        self.asignaciones = self.asignaciones.merge(
            self.estaciones,
            on="estacion",
            how="left",
            validate="many_to_one",
        )

        n_clusters = len(set(self.etiquetas))
        if not 1 < n_clusters < len(self.X):
            raise ValueError(
                'Silhouette requiere al menos 2 clusters y menos clusters que filas.'
            )

        grupos_train = pd.Series(
            self.etiquetas,
            index=self.ids_estaciones_train,
        )
        grupos_test = pd.Series(
            self.etiquetas_test,
            index=self.ids_estaciones_test,
        )

        estaciones_comunes = grupos_train.index.intersection(grupos_test.index)
        if len(estaciones_comunes) < 2:
            raise ValueError("Se necesitan al menos dos estaciones comunes para calcular ARI.")
        ari =  adjusted_rand_score(
            grupos_train.loc[estaciones_comunes],
            grupos_test.loc[estaciones_comunes],
        )

        self.metricas = {
            'silhouette': float(silhouette_score(self.X, self.etiquetas)),
            'inertia': float(self.modelo.inertia_),
            'ari_estabilidad': float(ari)
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
    stations_path : Path = ESTACIONES_PATH,
    output_path : Path = P3_CLUSTERS_PATH,
    model_path : Path = P3_MODEL_PATH,
    random_state : int = RANDOM_STATE,
    n_clusters : int = 3,
    n_init : int = 10,
    run_id : str = None
):
    perfiles = pd.read_parquet(input_path)
    estaciones = pd.read_csv(stations_path)
    parametros = {
        'n_clusters': n_clusters,
        'n_init': n_init,
        'random_state': random_state
    }
    p3 = P3Trainer(perfiles, estaciones, parametros)
    p3.fit_predict()

    p3.save(model_path, output_path)
    logger.success(f'Modelo entrenado, datos guardados en {output_path}')
    logger.info('Métricas: {}', p3.metricas)

    with mlflow.start_run(run_id=run_id) as run:
        logger.info("MLflow run ID: {}", run.info.run_id)
        mlflow.log_params(p3.parametros)
        mlflow.log_metrics(p3.metricas)
        mlflow.sklearn.log_model(
            p3.modelo,
            name='p3_kmeans',
            input_example=p3.X.head(3),
            signature=infer_signature(p3.X, p3.etiquetas),
            serialization_format='skops'
        )
    return run.info.run_id


if __name__ == '__main__':
    app()
