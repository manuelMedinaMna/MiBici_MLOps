"""Variables y etiquetas para los modelos (paso interim → processed).

Lee `data/interim/viajes.parquet` y el catálogo de estaciones, y guarda en `data/processed/` una
tabla por pregunta. Ninguna variable usa información posterior al momento de la predicción.

P2 (riesgo de que una estación se vacíe o se llene): una fila por estación y ventana de 2 horas,
con la etiqueta y la frecuencia de cada clase en las semanas anteriores. Las reglas están en
references/decisiones_p2.md y el análisis, en las secciones 12 y 13 de
notebooks/2.0-rc-p2-tamano-ventana.ipynb.

P3 (grupos de estaciones con patrones de uso parecidos): una fila por estación y conjunto
(entrenamiento o prueba), con la fracción de sus salidas y llegadas en cada día de la semana y
franja del día. Las reglas están en references/decisiones_p3.md y la exploración, en
notebooks/1.0-im-initial-eda.ipynb.

Uso:
    python -m mibici_mlops.features p2
    python -m mibici_mlops.features p3
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mibici_mlops.config import (
    ESTACIONES_PATH,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    P2_HORA_REINICIO,
    P2_INVENTARIO_INICIAL,
    P2_SEMANAS_HISTORIA,
    P2_VENTANA,
    P2_VENTANAS_PATH,
    P3_FRANJAS,
    P3_MIN_VIAJES,
    P3_PERFILES_PATH,
    VIAJES_PATH,
)

app = typer.Typer()

# Clases de P2, en el orden en que ganan un empate en el baseline de frecuencia (decisión 7).
CLASES_P2 = ["se_vacia", "se_llena", "sin_riesgo"]

# Columnas que se guardan: la etiqueta y variables conocidas al momento de predecir. El inventario
# simulado (mínimo, máximo, final) no se guarda porque describe la misma ventana que se predice.
COLUMNAS_P2 = [
    "estacion",
    "ventana",
    "dia",
    "capacidad",
    "clase",
    "frec_se_vacia",
    "frec_se_llena",
    "frec_sin_riesgo",
]


def dia_de_simulacion(momentos: pd.Series) -> pd.Series:
    """Día de la simulación de cada momento: de las P2_HORA_REINICIO a la misma hora del siguiente.

    Por qué para P2: el inventario se reinicia a esa hora, cuando casi no hay viajes (decisión 4);
    lo que pasa de madrugada cuenta en el día anterior.
    """
    return (momentos - pd.Timedelta(hours=P2_HORA_REINICIO)).dt.normalize()


def construir_movimientos(viajes: pd.DataFrame, estaciones: np.ndarray) -> pd.DataFrame:
    """Convierte cada viaje en una salida (-1) en su origen y una llegada (+1) en su destino.

    Por qué para P2: MiBici no publica cuántas bicicletas hay en cada estación, así que el
    inventario se aproxima con estos movimientos. Se filtra por estación al final para que una
    llegada desde una estación fuera del catálogo sí cuente en su destino.
    """
    salidas = pd.DataFrame(
        {"estacion": viajes["origen_id"], "momento": viajes["inicio"], "cambio": -1}
    )
    llegadas = pd.DataFrame(
        {"estacion": viajes["destino_id"], "momento": viajes["fin"], "cambio": 1}
    )
    movimientos = pd.concat([salidas, llegadas], ignore_index=True)
    return movimientos[movimientos["estacion"].isin(estaciones)]


def simular_inventario(movimientos: pd.DataFrame, capacidad: pd.Series) -> pd.DataFrame:
    """Agrega el día de la simulación y el inventario estimado después de cada movimiento.

    Cada estación empieza el día con P2_INVENTARIO_INICIAL de su capacidad, sin redondear, y se le
    suman las llegadas y se le restan las salidas. Los movimientos del mismo segundo conservan su
    orden (primero las salidas) para que el resultado no dependa de la versión de pandas.

    Por qué para P2: el riesgo de una ventana depende de lo que pasó antes ese mismo día, no solo
    de los viajes de la ventana (decisión 4).
    """
    movimientos = movimientos.sort_values("momento", kind="stable", ignore_index=True)
    movimientos["dia"] = dia_de_simulacion(movimientos["momento"])
    movimientos["inventario"] = (
        P2_INVENTARIO_INICIAL * movimientos["estacion"].map(capacidad)
        + movimientos.groupby(["estacion", "dia"])["cambio"].cumsum()
    )
    return movimientos


def resumir_ventanas(movimientos: pd.DataFrame) -> pd.DataFrame:
    """Resume cada estación y ventana: inventario mínimo, máximo y final, y balance neto.

    Por qué para P2: la predicción es por ventana de 2 horas (decisión 2), y el mínimo y el máximo
    dicen si en algún momento de la ventana el inventario cruzó un umbral (decisión 5).
    """
    ventana = movimientos["momento"].dt.floor(P2_VENTANA).rename("ventana")
    return (
        movimientos.groupby(["estacion", ventana])
        .agg(
            minimo=("inventario", "min"),
            maximo=("inventario", "max"),
            final=("inventario", "last"),
            neto=("cambio", "sum"),
        )
        .reset_index()
    )


def completar_ventanas(por_ventana: pd.DataFrame, capacidad: pd.Series) -> pd.DataFrame:
    """Agrega las ventanas sin movimientos y completa el inventario de todas.

    Cada estación tiene todas sus ventanas desde su primer hasta su último día con movimientos. Una
    ventana sin viajes conserva el inventario con el que terminó la anterior (`ffill` dentro de
    cada estación y día), y si el día empieza sin viajes tiene el inventario inicial. El mínimo y
    el máximo también cuentan el inventario con el que empieza cada ventana. Se descarta el día
    incompleto anterior a FECHA_INICIO.

    Por qué para P2: si una estación quedó vacía y nadie llegó después, sigue vacía; sin estas
    ventanas se perdería ~20% de las ventanas "se vacía". Empezar en el primer día con viajes evita
    contar como "sin riesgo" los días en que la estación todavía no existía.
    """
    por_ventana = por_ventana.assign(dia=dia_de_simulacion(por_ventana["ventana"]))
    por_ventana = por_ventana[por_ventana["dia"] >= pd.Timestamp(FECHA_INICIO)]

    activa = por_ventana.groupby("estacion")["dia"].agg(["min", "max"])
    activa["desde"] = activa["min"] + pd.Timedelta(hours=P2_HORA_REINICIO)
    activa["hasta"] = (
        activa["max"] + pd.Timedelta(days=1, hours=P2_HORA_REINICIO) - pd.Timedelta(P2_VENTANA)
    )
    todas = pd.DataFrame(
        {"ventana": pd.date_range(activa["desde"].min(), activa["hasta"].max(), freq=P2_VENTANA)}
    )

    ventanas = activa[["desde", "hasta"]].reset_index().merge(todas, how="cross")
    ventanas = ventanas[ventanas["ventana"].between(ventanas["desde"], ventanas["hasta"])]
    ventanas = ventanas[["estacion", "ventana"]].merge(
        por_ventana.drop(columns="dia"), on=["estacion", "ventana"], how="left"
    )
    ventanas["dia"] = dia_de_simulacion(ventanas["ventana"])
    ventanas["capacidad"] = ventanas["estacion"].map(capacidad)

    ventanas["final"] = ventanas.groupby(["estacion", "dia"])["final"].ffill()
    ventanas["final"] = ventanas["final"].fillna(P2_INVENTARIO_INICIAL * ventanas["capacidad"])
    ventanas["neto"] = ventanas["neto"].fillna(0)
    ventanas["inicio"] = ventanas["final"] - ventanas["neto"]
    ventanas["minimo"] = ventanas[["minimo", "inicio"]].min(axis=1)
    ventanas["maximo"] = ventanas[["maximo", "inicio"]].max(axis=1)
    return ventanas


def etiquetar(ventanas: pd.DataFrame) -> pd.Series:
    """Clase de cada ventana: "se_vacia", "se_llena" o "sin_riesgo".

    "se_vacia" si el inventario llega a 0 o menos; "se_llena" si llega a la capacidad o más. Si
    pasan las dos cosas, gana el umbral más cercano al inventario con que empieza la ventana.

    Por qué para P2: es la variable objetivo (decisiones 1 y 5). Las dos clases de riesgo piden
    acciones opuestas al equipo de rebalanceo: llevar bicicletas o retirarlas.
    """
    se_vacia = ventanas["minimo"] <= 0
    se_llena = ventanas["maximo"] >= ventanas["capacidad"]
    empieza_abajo = ventanas["inicio"] <= ventanas["capacidad"] / 2

    clase = pd.Series("sin_riesgo", index=ventanas.index)
    clase[se_vacia] = "se_vacia"
    clase[se_llena] = "se_llena"
    clase[se_vacia & se_llena & empieza_abajo] = "se_vacia"
    return clase


def frecuencia_semanas_anteriores(
    ventanas: pd.DataFrame, semanas: int = P2_SEMANAS_HISTORIA
) -> pd.DataFrame:
    """Fracción de las semanas anteriores en que la misma estación y ventana tuvo cada clase.

    Compara con el mismo día de la semana y la misma hora: dentro de cada grupo (estación, día de
    la semana, hora), `shift(1)` es la misma ventana una semana antes. Supone que `ventanas` tiene
    todas las ventanas de cada estación en orden, como las deja `completar_ventanas`. Las semanas
    en que la estación todavía no existía cuentan como "sin riesgo".

    Por qué para P2: es el baseline de frecuencia (decisiones 3 y 7) y una variable para los
    modelos candidatos. Solo mira semanas que ya pasaron, así que no usa información posterior al
    momento de la predicción.
    """
    grupos = [ventanas["estacion"], ventanas["ventana"].dt.dayofweek, ventanas["ventana"].dt.hour]
    frecuencias = pd.DataFrame(index=ventanas.index)
    for clase in ["se_vacia", "se_llena"]:
        por_grupo = (ventanas["clase"] == clase).astype(float).groupby(grupos)
        anteriores = [por_grupo.shift(k).fillna(0) for k in range(1, semanas + 1)]
        frecuencias[f"frec_{clase}"] = sum(anteriores) / semanas
    frecuencias["frec_sin_riesgo"] = (
        1 - frecuencias["frec_se_vacia"] - frecuencias["frec_se_llena"]
    )
    return frecuencias


# Con un callback, cada pregunta es un subcomando (`features p2`) y P1 o P3 pueden agregar el suyo.
@app.callback()
def main():
    """Construye las tablas de variables y etiquetas en data/processed."""


@app.command()
def p2(
    viajes_path: Path = VIAJES_PATH,
    estaciones_path: Path = ESTACIONES_PATH,
    output_path: Path = P2_VENTANAS_PATH,
    grupos: int = 8,
):
    """P2: etiqueta de riesgo y frecuencia de cada clase por estación y ventana de 2 horas."""
    if pd.Timedelta(hours=P2_HORA_REINICIO) % pd.Timedelta(P2_VENTANA):
        raise ValueError("P2_HORA_REINICIO debe coincidir con el inicio de una ventana")

    estaciones = pd.read_csv(estaciones_path)
    capacidad = estaciones[estaciones["dpcapacity"] > 0].set_index("id")["dpcapacity"]
    viajes = pd.read_parquet(
        viajes_path,
        columns=["inicio", "fin", "origen_id", "destino_id"],
        filters=[
            ("inicio", ">=", pd.Timestamp(FECHA_INICIO)),
            ("inicio", "<", pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)),
        ],
    )
    logger.info(f"{len(viajes):,} viajes de {FECHA_INICIO} a {FECHA_FIN}")

    # Por grupos de estaciones para usar menos memoria. El resultado es el mismo, porque el
    # inventario de cada estación solo depende de sus propios movimientos.
    partes = []
    for grupo in tqdm(np.array_split(np.sort(capacidad.index), grupos), desc="Grupos"):
        en_grupo = viajes["origen_id"].isin(grupo) | viajes["destino_id"].isin(grupo)
        movimientos = simular_inventario(construir_movimientos(viajes[en_grupo], grupo), capacidad)
        ventanas = completar_ventanas(resumir_ventanas(movimientos), capacidad)
        ventanas["clase"] = etiquetar(ventanas)
        ventanas = ventanas.join(frecuencia_semanas_anteriores(ventanas))
        partes.append(ventanas[COLUMNAS_P2])

    tabla = pd.concat(partes, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(output_path, index=False, compression="zstd")

    porcentajes = (tabla["clase"].value_counts(normalize=True) * 100).round(2).to_dict()
    logger.success(
        f"{len(tabla):,} ventanas de {tabla['estacion'].nunique()} estaciones guardadas en "
        f"{output_path}. Clases (%): {porcentajes}"
    )


# Días de la semana en el orden de `dt.dayofweek` (0 = lunes).
DIAS_SEMANA = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]


def columnas_perfil(prefijo: str) -> list[str]:
    """Nombres de las columnas del perfil de P3, p. ej. `sal_lun_manana` (decisión 2 de P3)."""
    return [f"{prefijo}_{dia}_{franja}" for dia in DIAS_SEMANA for franja in P3_FRANJAS]


def franja_del_dia(momentos: pd.Series) -> np.ndarray:
    """Número de la franja del día de cada momento: 0 = madrugada, 1 = mañana, etc.

    Las franjas empiezan a las horas de P3_FRANJAS. Por qué para P3: cuatro franjas bastan para
    distinguir los usos que importan a planeación urbana (ir al trabajo o la escuela en la
    mañana, regresar en la tarde, salir de noche) sin repartir los viajes de cada estación en
    tantas columnas que el perfil quede con ruido (decisión 2).
    """
    return np.searchsorted(list(P3_FRANJAS.values()), momentos.dt.hour, side="right") - 1


def perfil_de_uso(estacion: pd.Series, momentos: pd.Series, prefijo: str) -> pd.DataFrame:
    """Fracción de los viajes de cada estación en cada día de la semana y franja del día.

    Una fila por estación y 28 columnas (7 días × 4 franjas) que suman 1. Por qué para P3: al
    dividir entre el total de la estación se comparan sus patrones de uso, no su volumen; si no,
    los grupos solo separarían estaciones muy usadas de poco usadas (decisión 2).
    """
    codigo = momentos.dt.dayofweek.to_numpy() * len(P3_FRANJAS) + franja_del_dia(momentos)
    conteo = (
        pd.DataFrame({"estacion": estacion.to_numpy(), "codigo": codigo})
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=range(len(DIAS_SEMANA) * len(P3_FRANJAS)), fill_value=0)
    )
    conteo.columns = columnas_perfil(prefijo)
    return conteo.div(conteo.sum(axis=1), axis=0)


def perfiles_estaciones(
    viajes: pd.DataFrame,
    estaciones: np.ndarray,
    desde: str,
    hasta: str,
    min_viajes: int = P3_MIN_VIAJES,
) -> pd.DataFrame:
    """Perfil de salidas y llegadas de cada estación con los viajes que inician en [desde, hasta).

    Agrega el promedio de salidas y llegadas por día y el balance (llegadas - salidas) / total,
    que no se usan para agrupar sino para describir cada grupo. Deja fuera las estaciones que
    no están en el catálogo y las que tienen menos de `min_viajes` salidas o llegadas.

    Por qué para P3: las salidas dicen a qué hora la gente toma bicicletas en la estación y las
    llegadas a qué hora las deja; juntas distinguen, por ejemplo, una estación de vivienda (sale
    en la mañana, llega en la tarde) de una de trabajo (al revés). Con pocos viajes las
    fracciones son ruido (decisión 3). Recibir el periodo permite calcular el perfil solo con
    los datos de entrenamiento y, por separado, con los de prueba (decisión 4).
    """
    periodo = viajes[viajes["inicio"].between(desde, hasta, inclusive="left")]
    salidas = periodo["origen_id"].value_counts()
    llegadas = periodo["destino_id"].value_counts()
    dias = (pd.Timestamp(hasta) - pd.Timestamp(desde)).days

    perfil = perfil_de_uso(periodo["origen_id"], periodo["inicio"], "sal").join(
        perfil_de_uso(periodo["destino_id"], periodo["fin"], "lle"), how="inner"
    )
    perfil["salidas_dia"] = salidas / dias
    perfil["llegadas_dia"] = llegadas / dias
    perfil["balance"] = (llegadas - salidas) / (llegadas + salidas)

    suficientes = (salidas.reindex(perfil.index) >= min_viajes) & (
        llegadas.reindex(perfil.index) >= min_viajes
    )
    return perfil[perfil.index.isin(estaciones) & suficientes]


@app.command()
def p3(
    viajes_path: Path = VIAJES_PATH,
    estaciones_path: Path = ESTACIONES_PATH,
    output_path: Path = P3_PERFILES_PATH,
):
    """P3: perfil de uso de cada estación en los periodos de entrenamiento y de prueba."""
    estaciones = pd.read_csv(estaciones_path)["id"].to_numpy()
    viajes = pd.read_parquet(
        viajes_path,
        columns=["inicio", "fin", "origen_id", "destino_id"],
        filters=[
            ("inicio", ">=", pd.Timestamp(FECHA_INICIO)),
            ("inicio", "<", pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)),
        ],
    )
    logger.info(f"{len(viajes):,} viajes de {FECHA_INICIO} a {FECHA_FIN}")

    # La validación no se usa: P3 no ajusta hiperparámetros con ella (decisión 4).
    fin = (pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    periodos = {
        "entrenamiento": (FECHA_INICIO, FECHA_INICIO_VALIDACION),
        "prueba": (FECHA_INICIO_PRUEBA, fin),
    }
    partes = []
    for conjunto, (desde, hasta) in periodos.items():
        perfil = perfiles_estaciones(viajes, estaciones, desde, hasta)
        logger.info(f"{conjunto}: {len(perfil)} estaciones con perfil ({desde} a {hasta})")
        partes.append(perfil.reset_index().assign(conjunto=conjunto))

    tabla = pd.concat(partes, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(output_path, index=False, compression="zstd")
    logger.success(f"{len(tabla):,} perfiles guardados en {output_path}")


if __name__ == "__main__":
    app()
