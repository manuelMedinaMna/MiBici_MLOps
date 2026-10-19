from loguru import logger
from pathlib import Path

import pandas as pd

import folium
import typer

from mibici_mlops.config import ESTACIONES_PATH, FIGURES_DIR, P3_CLUSTERS_PATH, P3_PERFILES_PATH, P3_VIAJES_PATH

app = typer.Typer()

from pathlib import Path


class P3Plotter:
    COLUMNAS = ['origen_id', 'origen_lat', 'origen_lon', 'destino_lat', 'destino_lon']
    COLORES = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "gray", "pink", "black"]
    COLUMNAS_ESTACIONES = ["origen_id", "origen_lat", "origen_lon"]

    def __init__(self, perfiles, clusters, viajes, estaciones):
        perfiles = perfiles.copy()
        if "origen_id" not in perfiles.columns:
            perfiles = perfiles.rename_axis("origen_id").reset_index()

        self.perfiles = perfiles
        self.clusters = clusters
        self.viajes = viajes.copy()

        catalogo = estaciones.rename(columns={
            "id": "origen_id",
            "latitude": "origen_lat",
            "longitude": "origen_lon",
        })

        self.estaciones = clusters[["origen_id", "cluster"]].merge(
            catalogo[["origen_id", "origen_lat", "origen_lon"]],
            on="origen_id",
            how="left",
            validate="one_to_one",
        )

        # Alias para que _generar_mapa(tipo) acepte "destino".
        self.estaciones["destino_id"] = self.estaciones["origen_id"]
        self.estaciones["destino_lat"] = self.estaciones["origen_lat"]
        self.estaciones["destino_lon"] = self.estaciones["origen_lon"]

        self.mapa_origen = self._generar_mapa("origen")
        self.mapa_destino = self._generar_mapa("destino")

    def _generar_mapa(self, tipo):
        if tipo not in {"origen", "destino"}:
            raise ValueError("tipo debe ser 'origen' o 'destino'")

        id_col = f"{tipo}_id"
        lat_col = f"{tipo}_lat"
        lon_col = f"{tipo}_lon"

        datos = self.estaciones.dropna(subset=[lat_col, lon_col])
        if datos.empty:
            raise ValueError(f"No hay coordenadas para el mapa de {tipo}")

        mapa = folium.Map(
            location=[datos[lat_col].mean(), datos[lon_col].mean()],
            zoom_start=14,
            tiles="OpenStreetMap",
        )

        grupos = sorted(datos["cluster"].dropna().unique())

        for i, grupo in enumerate(grupos):
            color = P3Plotter.COLORES[i % len(P3Plotter.COLORES)]
            capa = folium.FeatureGroup(name=f"Grupo {grupo}")
            puntos = datos[datos["cluster"] == grupo]

            for _, fila in puntos.iterrows():
                folium.CircleMarker(
                    location=[fila[lat_col], fila[lon_col]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    tooltip=f"Estación {fila[id_col]} · Grupo {grupo}",
                ).add_to(capa)

            capa.add_to(mapa)

        folium.LayerControl().add_to(mapa)
        return mapa       
        

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
    estaciones_path: Path = ESTACIONES_PATH,
    output_dir: Path = FIGURES_DIR / "p3"
    # -----------------------------------------
):
    profiles = pd.read_parquet(profiles_path)
    clusters = pd.read_parquet(clusters_path)
    viajes = pd.read_parquet(viajes_path)
    estaciones = pd.read_csv(estaciones_path)

    logger.info("Generando gráficas y mapas de los clusters de P3...")
    p3plt = P3Plotter(profiles, clusters, viajes, estaciones)
    p3plt.save(output_dir)

    logger.success("Gráficas y mapas de los clusters de P3 generadas.")

if __name__ == "__main__":
    app()
