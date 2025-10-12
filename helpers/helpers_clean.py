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




## separate all the cropped features and name them and remove small features from it 
def explode_and_filter_features(input_shp: str, 
                                exploded_shp: str, 
                                filtered_shp: str, 
                                min_area_sqm: float) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Explode multipart polygons, calculate area, assign class="settlement",
    keep only two attributes (class, area_sqm), save exploded + filtered shapefiles.
    """

    # Read input
    gdf = gpd.read_file(input_shp)
    if gdf.empty:
        raise ValueError("Input shapefile is empty.")

    # Check CRS
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS defined. Assign a projected CRS in meters.")

  
    # Fix invalid geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)

    # Explode into singlepart polygons
    try:
        exploded = gdf.explode(index_parts=False).reset_index(drop=True)
    except TypeError:
        exploded = gdf.explode().reset_index(drop=True)

    # Calculate area
    exploded["area_sqm"] = exploded.geometry.area

    # Keep only class + area
    exploded = exploded[["geometry", "area_sqm"]].copy()
    exploded["class"] = "Settlement"

    # Reorder columns → class, area, geometry
    exploded = exploded[["class", "area_sqm", "geometry"]]

    # Save exploded shapefile
    Path(exploded_shp).parent.mkdir(parents=True, exist_ok=True)
    exploded.to_file(exploded_shp)

    # Filter by area
    filtered = exploded[exploded["area_sqm"] >= min_area_sqm].copy().reset_index(drop=True)

    # Save filtered shapefile
    Path(filtered_shp).parent.mkdir(parents=True, exist_ok=True)
    filtered.to_file(filtered_shp)

    return exploded, filtered



def clean_attributes_with_area(input_shp: str, output_shp: str) -> gpd.GeoDataFrame:
    
    shp = Path(input_shp)
    if not shp.exists():
        raise FileNotFoundError(f"Input file not found: {shp}")

    gdf = gpd.read_file(str(shp))
    if "class" not in [c.lower() for c in gdf.columns]:
        raise ValueError("'class' attribute not found in input shapefile.")

    # normalize to lowercase 'class' if needed
    for col in gdf.columns:
        if col.lower() == "class" and col != "class":
            gdf = gdf.rename(columns={col: "class"})

    # compute area in square meters
    gdf["Area(sq.m)"] = gdf.geometry.area.apply(lambda a: f"{a:.4f}")

    # keep only required columns
    gdf = gdf[["class", "Area(sq.m)", "geometry"]]

    # save cleaned shapefile
    Path(output_shp).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_shp)

    print(f"Saved cleaned shapefile to {output_shp} with {len(gdf)} features.")

    return gdf