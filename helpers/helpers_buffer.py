import os
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.windows import from_bounds, Window
from shapely.geometry import box, mapping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import ndimage
import fiona

import matplotlib.pyplot as plt
import sys
from pathlib import Path

from shapely.ops import unary_union
from pathlib import Path
from typing import List, Optional, Union, Tuple
from shapely.geometry import Polygon, MultiPolygon



def fill_holes_in_geometry(geom):
    """
    Return geometry with interior holes removed.
    - If geom is Polygon: returns Polygon(shell) (no interiors).
    - If geom is MultiPolygon: returns MultiPolygon([Polygon(shell) ...]).
    - Otherwise returns geom unchanged.
    """
    if geom is None or geom.is_empty:
        return geom
    # Single polygon
    if isinstance(geom, Polygon):
        try:
            return Polygon(geom.exterior)
        except Exception:
            return geom
    # MultiPolygon
    if isinstance(geom, MultiPolygon):
        new_parts = []
        for part in geom.geoms:
            try:
                new_parts.append(Polygon(part.exterior))
            except Exception:
                # fallback to original part
                new_parts.append(part)
        return MultiPolygon(new_parts)
    # other geometry types: return as-is
    return geom

def buffer_and_dissolve(
    input_shp: str,
    buffer_m: float,
    out_buffered: str,
    out_dissolved: str,
    out_filled_dissolved: str = None,
    min_area_remove: float = 0.0
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Buffer features, dissolve them, and produce a version with holes filled (no interiors).

    Returns:
      gdf_buffered, dissolved_gdf, filled_dissolved_gdf
    """
    shp = Path(input_shp)
    if not shp.exists():
        raise FileNotFoundError(f"Input file not found: {shp}")

    gdf = gpd.read_file(str(shp))
    print("Loaded:", shp)
    print("Original CRS:", gdf.crs)

    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS. Assign a CRS (e.g., EPSG:4326) and re-run.")


    # Fix invalid geometries
    gdf['geometry'] = gdf.geometry.buffer(0)

    # Buffer with sharp corners (mitre)
    join_style = 2  # mitre
    cap_style = 1   # round (for lines only)
    resolution = 16

    buffered_geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            buffered_geoms.append(None)
            continue
        b = geom.buffer(buffer_m, resolution=resolution,
                        join_style=join_style, cap_style=cap_style)
        buffered_geoms.append(b)

    gdf_buffered = gdf.copy()
    gdf_buffered['geometry'] = buffered_geoms
    gdf_buffered = gdf_buffered[~(gdf_buffered.geometry.is_empty | gdf_buffered.geometry.isnull())].reset_index(drop=True)
    print(f"Created {len(gdf_buffered)} buffered geometries (buffer={buffer_m} m).")

    # Remove tiny features
    if min_area_remove > 0:
        before = len(gdf_buffered)
        gdf_buffered['area_m2'] = gdf_buffered.geometry.area
        gdf_buffered = gdf_buffered[gdf_buffered['area_m2'] >= min_area_remove].copy()
        gdf_buffered.drop(columns=['area_m2'], inplace=True)
        after = len(gdf_buffered)
        print(f"Removed {before-after} features smaller than {min_area_remove} m²")

    # Dissolve into single geometry
    dissolved_geom = unary_union(gdf_buffered.geometry.values)
    dissolved_gdf = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=gdf_buffered.crs)
    print("Dissolved all buffered geometries into one.")

    # Fill holes: remove interior rings
    filled_geom = fill_holes_in_geometry(dissolved_geom)
    # As a safety step, union again to merge adjacent parts
    try:
        filled_geom = unary_union(filled_geom)
    except Exception:
        pass
    filled_dissolved_gdf = gpd.GeoDataFrame(geometry=[filled_geom], crs=gdf_buffered.crs)
    print("Filled holes inside dissolved geometry (interiors removed).")

    # Save outputs
    Path(out_buffered).parent.mkdir(parents=True, exist_ok=True)
    Path(out_dissolved).parent.mkdir(parents=True, exist_ok=True)
    gdf_buffered.to_file(out_buffered)
    dissolved_gdf.to_file(out_dissolved)
    print("Saved buffered:", out_buffered)
    print("Saved dissolved:", out_dissolved)

    if out_filled_dissolved:
        Path(out_filled_dissolved).parent.mkdir(parents=True, exist_ok=True)
        filled_dissolved_gdf.to_file(out_filled_dissolved)
        print("Saved filled dissolved:", out_filled_dissolved)

    return gdf_buffered, dissolved_gdf, filled_dissolved_gdf