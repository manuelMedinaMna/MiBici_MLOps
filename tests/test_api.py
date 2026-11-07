"""Pruebas de la API: validación de entradas, formato de la respuesta y recorte de negativos.

Usan un modelo falso en lugar del exportado del registry, así que corren sin servidor de MLflow
ni models/servicio/p1/.
"""

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici_mlops.api.app import app, obtener_modelo
from mibici_mlops.features import COLUMNAS_P1

INFO = {
    "nombre": "mibici-equipo43-p1-demanda",
    "alias": "champion",
    "version": "3",
    "run_id": "abc123",
    "origen": "imagen",
}


class ModeloFalso:
    """Predice media_7d - 10, para que haya negativos que recortar. Recuerda lo que recibió."""

    metadata = None

    def predict(self, tabla: pd.DataFrame) -> np.ndarray:
        self.recibido = tabla
        return tabla["media_7d"].to_numpy() - 10


@pytest.fixture
def modelo():
    return ModeloFalso()


@pytest.fixture
def cliente(modelo):
    # Sin `with`, TestClient no corre el arranque, que cargaría el modelo real.
    app.dependency_overrides[obtener_modelo] = lambda: (modelo, INFO)
    yield TestClient(app)
    app.dependency_overrides.clear()


def fila(**cambios) -> dict:
    base = {
        "estacion_id": 2,
        "dia_semana": 2,
        "mes": 10,
        "es_fin_de_semana": 0,
        "es_feriado": 0,
        "capacidad": 23,
        "lat": 20.67,
        "lon": -103.37,
        "lag_7d": 41,
        "media_7d": 37.4,
    }
    return {**base, **cambios}


def test_salud_y_modelo(cliente):
    assert cliente.get("/salud").json() == {"estado": "ok", "version_modelo": "3"}
    assert cliente.get("/modelo").json() == INFO


def test_prediccion_devuelve_una_por_fila_y_recorta_negativos(cliente, modelo):
    filas = [fila(estacion_id=1, media_7d=37.4), fila(estacion_id=7, media_7d=4.0)]
    respuesta = cliente.post("/prediccion", json={"filas": filas})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo"] == INFO
    assert cuerpo["predicciones"] == [
        {"estacion_id": 1, "viajes": 27.4},
        {"estacion_id": 7, "viajes": 0.0},
    ]
    # El modelo recibe exactamente sus columnas, en orden y sin estacion_id.
    assert list(modelo.recibido.columns) == COLUMNAS_P1


def test_estacion_id_es_opcional(cliente):
    sin_id = {k: v for k, v in fila().items() if k != "estacion_id"}
    respuesta = cliente.post("/prediccion", json={"filas": [sin_id]})
    assert respuesta.status_code == 200
    assert respuesta.json()["predicciones"][0]["estacion_id"] is None


@pytest.mark.parametrize(
    "cambios",
    [
        {"dia_semana": 7},
        {"mes": 0},
        {"es_feriado": 2},
        {"capacidad": 0},
        {"lat": 19.4},  # fuera de Guadalajara
        {"lag_7d": -1},
        {"media_7d": "muchos"},
        {"dia_semana": 5, "es_fin_de_semana": 0},  # sábado marcado como entre semana
        {"columna_extra": 1},
    ],
)
def test_rechaza_filas_invalidas(cliente, cambios):
    respuesta = cliente.post("/prediccion", json={"filas": [fila(**cambios)]})
    assert respuesta.status_code == 422


def test_rechaza_filas_incompletas_o_vacias(cliente):
    incompleta = {k: v for k, v in fila().items() if k != "lag_7d"}
    assert cliente.post("/prediccion", json={"filas": [incompleta]}).status_code == 422
    assert cliente.post("/prediccion", json={"filas": []}).status_code == 422
