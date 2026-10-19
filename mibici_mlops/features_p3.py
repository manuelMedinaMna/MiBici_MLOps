from loguru import logger
from pathlib import Path

import pandas as pd

import typer

from mibici_mlops.config import FECHA_FIN, FECHA_INICIO, P3_PERFILES_PATH, P3_VIAJES_PATH, VIAJES_PATH

app = typer.Typer()

class P3:
    columnas_de_interes = ['inicio', 'origen_id', 'destino_id', 'origen', 'destino', 'origen_lon', 'origen_lat', 'destino_lon', 'destino_lat']

    def __init__(self, viajes):
        self.viajes = viajes
        self._agregar_caracteristicas()
        self._crear_perfiles_estacion()

    def save(self, output_path):
        self.viajes_loc.to_parquet(output_path, index=False, compression="zstd")

    def _agregar_caracteristicas(self):
        # Creamos el dataset que se usuará para este modelo
        self.viajes_loc = viajes_loc = self.viajes[P3.columnas_de_interes].copy()
        viajes_loc['inicio'] = pd.to_datetime(viajes_loc['inicio'])


        # Agregamos características adicionales: dia, semana y día de la semana
        viajes_loc['dia'] = viajes_loc['inicio'].dt.date
        self.num_viajes_por_dia = (
            viajes_loc.groupby('dia')
            .size()
            .reset_index(name='num_viajes')
        )

        viajes_loc['semana'] = viajes_loc['inicio'].dt.to_period('W')
        self.num_viajes_por_semana = (
            viajes_loc.groupby('semana')
            .size()
            .reset_index(name='num_viajes')
        )

        viajes_loc['dia_de_semana'] = viajes_loc['inicio'].dt.dayofweek.map(
            {0: "lun", 1: "mar", 2: "mie", 3: "jue",
            4: "vie", 5: "sab", 6: "dom"}
        )
        self.num_viajes_por_dia_de_semana = (
            viajes_loc.groupby('dia_de_semana')
            .size()
            .reset_index(name='num_viajes')
        )

    def _crear_perfiles_estacion(self):
        df = self.viajes_loc.dropna(subset=["origen_id", "inicio"]).copy()

        dias = {
            0: "lun", 1: "mar", 2: "mie", 3: "jue",
            4: "vie", 5: "sab", 6: "dom",
        }
        df["dia_semana"] = df["inicio"].dt.dayofweek.map(dias)
        df["franja"] = pd.cut(
            df["inicio"].dt.hour,
            bins=[-1, 5, 11, 17, 23],
            labels=["madrugada", "manana", "tarde", "noche"],
        )
        df["patron"] = df["dia_semana"] + "_" + df["franja"].astype(str)

        # Una fila por estación; cada valor es la proporción de viajes de esa estación en un patrón temporal.
        perfiles = pd.crosstab(
            df["origen_id"],
            df["patron"],
            normalize="index",
        )

        coords_origen = df[
            ["origen_id", "origen_lat", "origen_lon"]
        ].rename(columns={
            "origen_id": "estacion_id",
            "origen_lat": "lat",
            "origen_lon": "lon",
        })

        coords_destino = df[
            ["destino_id", "destino_lat", "destino_lon"]
        ].rename(columns={
            "destino_id": "estacion_id",
            "destino_lat": "lat",
            "destino_lon": "lon",
        })

        coords = (
            pd.concat([coords_origen, coords_destino], ignore_index=True)
            .dropna(subset=["estacion_id", "lat", "lon"])
            .groupby("estacion_id")[["lat", "lon"]]
            .median()
            .rename(columns={
                "lat": "origen_lat",
                "lon": "origen_lon",
            })
        )
        coords.index.name = "origen_id"

        self.perfiles = perfiles.join(coords, how="left")
        self.perfiles.index.name = "origen_id"
        self.prefiles = self.perfiles.rename_axis("origen_id").reset_index()

    def save(self, ruta_perfiles, ruta_viajes):
        # Opcionalmente, se podrían guardar num_viajes_*
        perfiles = self.perfiles.rename_axis("origen_id").reset_index()

        perfiles.to_parquet(
            ruta_perfiles,
            index=False,
            compression="zstd",
        )
        self.viajes_loc.to_parquet(
            ruta_viajes,
            index=False,
            compression="zstd",
        )

@app.command()
def p3features(
    viajes_path: Path = VIAJES_PATH,
    output_path: Path = P3_PERFILES_PATH,
    interim_path: Path = P3_VIAJES_PATH,
    fecha_inicio: str = FECHA_INICIO,
    fecha_fin: str = FECHA_FIN
):
    
    viajes = pd.read_parquet(
        viajes_path,
        columns=P3.columnas_de_interes,
        filters=[
            ("inicio", ">=", pd.Timestamp(fecha_inicio)),
            ("inicio", "<", pd.Timestamp(fecha_fin) + pd.Timedelta(days=1)),
        ],
    ).dropna()
    logger.info(f"Conjunto de datos con {len(viajes):,} viajes de {fecha_inicio} a {fecha_fin} con {viajes.shape[1]} características")

    p3feat = P3(viajes)
    p3feat.save(output_path, interim_path)

    logger.success(
        f"Archivo procesado con {p3feat.viajes_loc.shape[1]} características guardado en {output_path}"
    )


if __name__ == "__main__":
    app()
