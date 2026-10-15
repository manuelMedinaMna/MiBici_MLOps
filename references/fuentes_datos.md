# Fuentes de datos y problemas conocidos

Notas sobre el origen de los datos crudos en `data/raw/` y las particularidades que hay que
manejar al procesarlos. La carpeta `data/` está en `.gitignore`, así que este archivo es la
referencia versionada de cómo se obtuvieron.

## Fuente oficial

- **Sitio:** <https://mibici.net/es/datos-abiertos/>
- **Descarga:** octubre de 2026
- **Cobertura:** viajes mensuales de 2014-12 a 2026-09 (142 archivos), más el catálogo de
  estaciones.
- **Nota:** el servidor bloquea peticiones sin un `User-Agent` de navegador (responde
  `406 Not Acceptable`), así que las descargas con `curl`/`requests` necesitan ese encabezado.

## Estructura en `data/raw/`

```
data/raw/
├── viajes/
│   └── <año>/datos_abiertos_<año>_<mes>.csv
└── estaciones/
    └── nomenclatura_2026_09.csv     # id, name, latitude, longitude, dpcapacity
```

Los archivos conservan el nombre exacto con el que se descargaron. **No renombrarlos.**

## Archivos con nombre irregular

Así vienen del sitio oficial. Su contenido corresponde al mes indicado (verificado por fechas).

| Archivo | Contenido |
|---|---|
| `2016/datos_abiertos_2016_01-1.csv` | 2016-01-01 → 2016-01-31 |
| `2017/datos_abiertos_2017_04-1.csv` | 2017-04-01 → 2017-04-30 |

Al extraer año y mes del nombre, usar una regex que ignore el sufijo, p. ej.
`datos_abiertos_(\d{4})_(\d{2})`.

## Archivos recuperados de Internet Archive

En la página oficial, **abril y mayo de 2023** tienen el enlace vacío (`<a href="">`) y las URLs
originales ya responden 404. Se descargaron de copias del Wayback Machine:

| Archivo | URL de descarga |
|---|---|
| `2023/datos_abiertos_2023_04.csv` | <https://web.archive.org/web/20261009131614id_/https://www.mibici.net/site/assets/files/46758/datos_abiertos_2023_04.csv> |
| `2023/datos_abiertos_2023_05.csv` | <https://web.archive.org/web/20240617215841id_/https://www.mibici.net/site/assets/files/46759/datos_abiertos_2023_05.csv> |

Validación:

- Las fechas de inicio cubren el mes completo (abril: 333,093 viajes; mayo: 364,158).
- Los `Viaje_Id` son continuos con los meses vecinos: marzo termina en `28080052`, abril va de
  `28080053` a `28467097` y mayo empieza en `28467098`.

## Diccionario de columnas (viajes)

Todas las versiones tienen las mismas 8 columnas en el mismo orden.

| # | Columna | Descripción |
|---|---|---|
| 1 | `Viaje_Id` | Identificador del viaje (consecutivo) |
| 2 | `Usuario_Id` | Identificador del usuario |
| 3 | `Genero` | `M` / `F`; ver valores faltantes abajo |
| 4 | `Año_de_nacimiento` | Año de nacimiento del usuario |
| 5 | `Inicio_del_viaje` | Fecha y hora de inicio (`YYYY-MM-DD HH:MM:SS`) |
| 6 | `Fin_del_viaje` | Fecha y hora de fin |
| 7 | `Origen_Id` | Estación de origen (→ `id` en nomenclatura) |
| 8 | `Destino_Id` | Estación de destino (→ `id` en nomenclatura) |

## Diferencias de formato entre archivos

El formato cambia a lo largo del tiempo y no siempre de forma consistente dentro de un año:

- **Comillas:** hasta 2021-05 los valores van sin comillas; desde 2021-06 todos van entre
  comillas, excepto `2023_11`, `2024_09` y `2024_10`.
- **Codificación:** mezcla de UTF-8 e ISO-8859-1 (Latin-1). Están en Latin-1:
  2021-06 a 2023-12 (salvo 2022-12), 2024 (salvo 2024-07), 2025-01, 2025-02, 2025-07 y 2026-01.
  El resto es UTF-8. Conviene detectar la codificación por archivo o leer con `encoding_errors`.
- **Encabezado de la columna 4 dañado:** según el archivo aparece como `Año_de_nacimiento`,
  `A�o_de_nacimiento` o `A}�e_nacimiento` (este último en `2023_04`, `2023_05`, `2024_02` y
  `2024_05`). **Renombrar las columnas por posición**, no por nombre.
- **Año de nacimiento:** en los primeros años viene como flotante (`1984.0`).
- **Género faltante:** en los primeros años aparece como campo vacío; desde 2022 aparece como
  `NULL` sin comillas (aunque el resto de la fila sí las tenga). Tratar ambos como nulo
  (`na_values=["NULL"]`).
- **Viajes que cruzan de mes:** un viaje iniciado el último día del mes puede terminar el mes
  siguiente; se asigna al archivo del mes en que inicia.
