from loguru import logger
from pathlib import Path

import pandas as pd

import folium
import typer

from mibici_mlops.config import FIGURES_DIR, P3_CLUSTERS_PATH, P3_PERFILES_PATH, P3_VIAJES_PATH

app = typer.Typer()

from pathlib import Path


class P3Plotter:
    COLUMNAS = ['origen_id', 'origen_lat', 'origen_lon', 'destino_lat', 'destino_lon']
    COLORES = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "gray", "pink", "black"]
    COLUMNAS_ESTACIONES = ["origen_id", "origen_lat", "origen_lon"]

    def __init__(self, perfiles, clusters, viajes):
        perfiles = perfiles.copy()
        if "origen_id" not in perfiles.columns:
            perfiles = perfiles.rename_axis("origen_id").reset_index()

        self.perfiles = perfiles
        self.clusters = clusters
        self.viajes = viajes.copy()

        self.estaciones = perfiles[P3Plotter.COLUMNAS_ESTACIONES].merge(
            clusters[["origen_id", "cluster"]],
            on="origen_id",
            how="inner",
            validate="one_to_one",
        )

        self.viajes_destino = self.viajes.merge(
            clusters[["origen_id", "cluster"]].rename(
                columns={"cluster": "cluster_origen"}
            ),
            on="origen_id",
            how="left",
            validate="many_to_one",
        )
        self.mapa_origen = self._generar_mapa('origen')
        self.mapa_destino = self._generar_mapa('destino')

    def _generar_mapa(self, tipo):
        if tipo == "origen":
            datos = self.estaciones.copy()
            lat_col, lon_col = "origen_lat", "origen_lon"
            cluster_col, id_col = "cluster", "origen_id"

        elif tipo == "destino":
            datos = (
                self.viajes_destino
                .dropna(subset=["destino_id", "destino_lat", "destino_lon", "cluster_origen"])
                .groupby(["destino_id", "cluster_origen"], as_index=False)
                .agg(
                    destino_lat=("destino_lat", "median"),
                    destino_lon=("destino_lon", "median"),
                    num_viajes=("destino_id", "size"),
                )
            )
            lat_col, lon_col = "destino_lat", "destino_lon"
            cluster_col, id_col = "cluster_origen", "destino_id"
        else:
            raise ValueError("tipo debe ser 'origen' o 'destino'")

        datos = datos.dropna(subset=[lat_col, lon_col, cluster_col])
        if datos.empty:
            raise ValueError(f"No hay datos para generar el mapa de {tipo}")

        mapa = folium.Map(
            location=[datos[lat_col].mean(), datos[lon_col].mean()],
            zoom_start=14,
            tiles="OpenStreetMap",
        )

        grupos = sorted(self.clusters["cluster"].dropna().unique())
        colores = {
            grupo: P3Plotter.COLORES[i % len(P3Plotter.COLORES)]
            for i, grupo in enumerate(grupos)
        }

        for grupo in grupos:
            puntos = datos[datos[cluster_col] == grupo]
            capa = folium.FeatureGroup(name=f"Grupo {grupo}")

            for _, fila in puntos.iterrows():
                tooltip = f"Estación {fila[id_col]} · Grupo {grupo}"
                if tipo == "destino":
                    tooltip += f" · Viajes: {fila['num_viajes']}"

                folium.CircleMarker(
                    location=[fila[lat_col], fila[lon_col]],
                    radius=5,
                    color=colores[grupo],
                    fill=True,
                    fill_color=P3Plotter.COLORES[grupo],
                fill_opacity=0.8,
                tooltip=tooltip,
            ).add_to(capa)

        capa.add_to(mapa)
        folium.LayerControl().add_to(mapa)
        return mapa

    def _generar_mapa_origen(self):
        # Crear y devolver un mapa de Folium
        ...

    def _generar_mapa_destinos(self):
        # Crear y devolver un mapa de Folium
        ...

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.mapa_origen.save(str(output_dir / "clusters_origen.html"))
        self.mapa_destino.save(str(output_dir / "clusters_destinos.html"))

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    profiles_path: Path = P3_PERFILES_PATH,
    clusters_path: Path = P3_CLUSTERS_PATH,
    viajes_path: Path = P3_VIAJES_PATH,
    output_dir: Path = FIGURES_DIR / "p3"
    # -----------------------------------------
):
    profiles = pd.read_parquet(profiles_path)
    clusters = pd.read_parquet(clusters_path)
    viajes = pd.read_parquet(viajes_path)
    
    logger.info("Generando gráficas y mapas de los clusters de P3...")
    p3plt = P3Plotter(profiles, clusters, viajes)
    p3plt.save(output_dir)
    logger.success("Gráficas y mapas de los clusters de P3 generadas.")


if __name__ == "__main__":
    app()
