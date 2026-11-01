"""Validación de la tabla de perfiles de P3 con DeepChecks.

Se corre con el entorno aparte de DeepChecks (ver requirements-deepchecks.txt), no con .venv:

    .venv-deepchecks/bin/python scripts/validar_datos_p3.py

Corre dos suites y guarda un reporte HTML de cada una en reports/deepchecks/:
- data_integrity: tipos, valores faltantes, duplicados, columnas constantes, outliers, etc.,
  sobre los perfiles de entrenamiento.
- train_test_validation: drift de cada variable y multivariado entre entrenamiento y prueba.

Termina con código 1 si alguna condición falla, para que `make` o CI lo detecten.
"""

from pathlib import Path
import sys
import warnings

from itertools import combinations

from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import FeatureFeatureCorrelation
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd

warnings.filterwarnings("ignore")

PROJ_ROOT = Path(__file__).resolve().parents[1]
PERFILES_PATH = PROJ_ROOT / "data" / "processed" / "p3_perfiles.parquet"
REPORTES_DIR = PROJ_ROOT / "reports" / "deepchecks"

UMBRAL_CORRELACION = 0.9
DIAS_ENTRE_SEMANA = {"lun", "mar", "mie", "jue", "vie"}


def correlacion_esperada(a: str, b: str) -> bool:
    """Pares que se correlacionan por cómo se construye el perfil, no por un error en los datos.

    - La misma franja en dos días entre semana (p. ej. sal_lun_manana y sal_mar_manana): una
      estación que se usa para ir al trabajo lo hace igual de lunes a viernes.
    - salidas_dia y llegadas_dia: en una estación casi todas las bicicletas que salen regresan.
    """
    if {a, b} == {"salidas_dia", "llegadas_dia"}:
        return True
    partes_a, partes_b = a.split("_"), b.split("_")
    if len(partes_a) != 3 or len(partes_b) != 3:
        return False
    (sentido_a, dia_a, franja_a), (sentido_b, dia_b, franja_b) = partes_a, partes_b
    return (
        sentido_a == sentido_b
        and franja_a == franja_b
        and {dia_a, dia_b} <= DIAS_ENTRE_SEMANA
    )


def sin_correlaciones_inesperadas(correlaciones: pd.DataFrame) -> ConditionResult:
    """Falla si hay un par con correlación > UMBRAL que correlacion_esperada no explica."""
    altas = [
        (a, b)
        for a, b in combinations(correlaciones.columns, 2)
        if abs(correlaciones.loc[a, b]) > UMBRAL_CORRELACION
    ]
    inesperadas = [par for par in altas if not correlacion_esperada(*par)]
    if inesperadas:
        return ConditionResult(ConditionCategory.FAIL, f"Pares inesperados: {inesperadas}")
    return ConditionResult(
        ConditionCategory.PASS, f"{len(altas)} pares > {UMBRAL_CORRELACION}, todos esperados"
    )


def suite_integridad() -> Suite:
    """data_integrity, con la condición de correlación ajustada al perfil de uso de P3.

    La condición por defecto (ningún par > 0.9) siempre falla con el perfil de P3 por diseño
    (decisión 2). En su lugar, solo falla con pares que no se explican por el diseño.
    """
    suite = data_integrity()
    for indice, chequeo in list(suite.checks.items()):
        if isinstance(chequeo, FeatureFeatureCorrelation):
            suite.remove(indice)
    suite.add(
        FeatureFeatureCorrelation().add_condition(
            f"Sin pares inesperados con correlación > {UMBRAL_CORRELACION}",
            sin_correlaciones_inesperadas,
        )
    )
    return suite


def cargar_conjuntos() -> tuple[Dataset, Dataset]:
    """Datasets de DeepChecks de entrenamiento y prueba con las variables del perfil de uso.

    P3 es agrupamiento: no hay etiqueta. `estacion` no se pasa como índice porque las mismas
    estaciones aparecen a propósito en cada periodo (decisión 4 de P3); DeepChecks lo
    reportaría como fuga de índice entre entrenamiento y prueba.
    """
    perfiles = pd.read_parquet(PERFILES_PATH)
    variables = [c for c in perfiles.columns if c.startswith(("sal_", "lle_"))]
    variables += ["salidas_dia", "llegadas_dia", "balance"]
    conjuntos = {
        nombre: Dataset(
            perfiles.loc[perfiles["conjunto"] == nombre, variables],
            cat_features=[],
        )
        for nombre in ["entrenamiento", "prueba"]
    }
    return conjuntos["entrenamiento"], conjuntos["prueba"]


def main():
    entrenamiento, prueba = cargar_conjuntos()
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    resultados = {
        "p3_integridad": data_integrity().run(entrenamiento),
        "p3_entrenamiento_vs_prueba": train_test_validation().run(entrenamiento, prueba),
    }
    todo_bien = True
    for nombre, resultado in resultados.items():
        ruta = REPORTES_DIR / f"{nombre}.html"
        # HTML estático: el modo widget (por defecto) carga desde unpkg un html-manager de
        # ipywidgets 7 que no entiende el estado de ipywidgets 8, y el reporte se ve en blanco.
        # DeepChecks no sobrescribe: si el archivo existe, guarda una copia "nombre (1).html".
        ruta.unlink(missing_ok=True)
        resultado.save_as_html(str(ruta), as_widget=False)
        fallidas = resultado.get_not_passed_checks()
        print(f"\n{nombre}: {len(fallidas)} chequeos con condiciones que no se cumplen ({ruta})")
        for chequeo in fallidas:
            for condicion in chequeo.conditions_results:
                if not condicion.is_pass():
                    print(f"  - {chequeo.get_header()}: {condicion.name} -> {condicion.details}")
        todo_bien = todo_bien and resultado.passed()
    return 0 if todo_bien else 1


if __name__ == "__main__":
    main()
