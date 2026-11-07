"""
Preparación de datos — Zona Metropolitana del Valle de México (ZMVM)
====================================================================

Une las dos fuentes de AMAI y genera la base que usan los agentes:

  * VALLE_DE_MEXICO.shp  -> 5,875 AGEB urbanas de la ZMVM con su polígono y
                            nivel socioeconómico (NSE) predominante.
  * NSE_por_AGEB_AMAI.xlsx -> número de viviendas por nivel (AB, C+, C, C-,
                            D+, D, E) en cada AGEB del país.

La unión se hace por la clave geoestadística de 13 caracteres
(entidad 2 + municipio 3 + localidad 4 + AGEB 4).

Salida: datos/agebs_zmvm.csv  (una fila por AGEB, con centroide en lat/lon)

Uso:
    pip install pandas openpyxl pyshp pyproj shapely
    python preparar_datos.py --shp ruta/VALLE_DE_MEXICO.shp --xlsx ruta/NSE_por_AGEB_AMAI.xlsx
"""

import argparse
from pathlib import Path

import pandas as pd
import shapefile                      # pyshp
from pyproj import CRS, Transformer
from shapely.geometry import shape

NIVELES = ["AB", "C+", "C", "C-", "D+", "D", "E"]
CONECTORES = {"De", "Del", "La", "Las", "Los", "Y"}


def nombre_propio(texto: str) -> str:
    """'GUSTAVO A. MADERO' -> 'Gustavo A. Madero'; 'CIUDAD DE MÉXICO' -> 'Ciudad de México'."""
    palabras = texto.title().split()
    return " ".join(w.lower() if i > 0 and w in CONECTORES else w for i, w in enumerate(palabras))


def leer_excel(ruta_xlsx: str) -> pd.DataFrame:
    """Lee la tabla nacional de AMAI (encabezado en 2 filas) y deja solo la ZMVM."""
    df = pd.read_excel(ruta_xlsx, header=None, skiprows=2, dtype={6: str})
    df.columns = ["ent", "nom_ent", "mun", "nom_mun", "loc", "nom_loc", "ageb",
                  *NIVELES, "nse_predominante_xlsx", "viviendas", "tam_localidad"]
    df = df[df["ent"].isin([9, 13, 15])].copy()          # CDMX, Hidalgo, Edomex
    df["cve_ageb"] = (df["ent"].astype(int).map("{:02d}".format)
                      + df["mun"].astype(int).map("{:03d}".format)
                      + df["loc"].astype(int).map("{:04d}".format)
                      + df["ageb"].astype(str).str.zfill(4))
    for n in NIVELES:   # 'N/D' (dato no disponible por confidencialidad) -> 0
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0).astype(int)
    df["viviendas"] = pd.to_numeric(df["viviendas"], errors="coerce").fillna(0).astype(int)
    return df[["cve_ageb", *NIVELES, "viviendas"]]


def leer_shapefile(ruta_shp: str) -> pd.DataFrame:
    """Lee las AGEB de la ZMVM y calcula su centroide en coordenadas geográficas."""
    sf = shapefile.Reader(ruta_shp, encoding="latin1")
    crs = CRS.from_wkt(Path(ruta_shp).with_suffix(".prj").read_text())
    a_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    filas = []
    for sr in sf.iterShapeRecords():
        r = sr.record.as_dict()
        geom = shape(sr.shape.__geo_interface__)
        c = geom.centroid
        lon, lat = a_wgs84.transform(c.x, c.y)
        filas.append({
            "cve_ageb": r["CVE_AGEBC"],
            "entidad": nombre_propio(r["NOM_ENT"]),
            "cve_mun": r["CVE_MUNC"],
            "municipio": nombre_propio(r["NOM_MUN"]),
            "nse_predominante": r["NIVEL"],
            "area_km2": round(geom.area / 1e6, 4),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
        })
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", required=True)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--salida", default="datos/agebs_zmvm.csv")
    a = ap.parse_args()

    geo = leer_shapefile(a.shp)
    nse = leer_excel(a.xlsx)
    df = geo.merge(nse, on="cve_ageb", how="left")
    sin_dato = df["viviendas"].isna().sum()
    df[NIVELES + ["viviendas"]] = df[NIVELES + ["viviendas"]].fillna(0).astype(int)

    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.salida, index=False, encoding="utf-8")
    print(f"AGEB en el shapefile: {len(geo):,}")
    print(f"AGEB sin conteo de viviendas en el Excel: {sin_dato:,}")
    print(f"Municipios/alcaldías: {df['cve_mun'].nunique()}")
    print(f"Viviendas totales: {df['viviendas'].sum():,}")
    print(f"Archivo generado: {a.salida}")


if __name__ == "__main__":
    main()
