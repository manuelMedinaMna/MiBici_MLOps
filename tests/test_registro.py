"""Pruebas de cómo se resuelven el modelo y los hiperparámetros del punto de entrada `train`."""

import pytest

from mibici_mlops.config import MODELO_POR_DEFECTO, MODELOS
from mibici_mlops.modeling.registro import resolver_modelo


def test_config_usa_los_valores_de_config_py():
    for pregunta in MODELOS:
        modelo, valores = resolver_modelo(pregunta)
        assert modelo == MODELO_POR_DEFECTO[pregunta]
        assert valores == MODELOS[pregunta][modelo]


def test_cambia_solo_los_hiperparametros_indicados_y_respeta_su_tipo():
    _, valores = resolver_modelo("p1", "hgb", "max_iter=200,learning_rate=0.05")
    assert valores["max_iter"] == 200
    assert valores["learning_rate"] == 0.05
    assert valores["loss"] == MODELOS["p1"]["hgb"]["loss"]


def test_no_modifica_los_valores_de_config_py():
    original = dict(MODELOS["p1"]["hgb"])
    resolver_modelo("p1", "hgb", "max_iter=1")
    assert MODELOS["p1"]["hgb"] == original


@pytest.mark.parametrize(
    ("pregunta", "modelo", "hiperparametros"),
    [
        ("p4", "config", "config"),  # pregunta que no existe
        ("p1", "rf", "config"),  # modelo que no existe
        ("p1", "hgb", "profundidad=3"),  # hiperparámetro que no tiene el modelo
        ("p1", "baseline", "max_iter=3"),  # el baseline no tiene hiperparámetros
        ("p1", "hgb", "max_iter=abc"),  # valor que no se puede convertir
    ],
)
def test_valores_invalidos_fallan(pregunta, modelo, hiperparametros):
    with pytest.raises(ValueError):
        resolver_modelo(pregunta, modelo, hiperparametros)
