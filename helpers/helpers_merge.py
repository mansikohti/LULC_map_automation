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




## merged all classes in one shape file except settlement 
def merge_shapefiles_with_class(
    input_shp_and_class: List[Tuple[Union[str, Path], str]],
    output_shp: Union[str, Path],
    dissolve: bool = False
) -> gpd.GeoDataFrame:
    
    pairs = [(Path(p), str(c)) for p, c in input_shp_and_class]
    if len(pairs) == 0:
        raise ValueError("No shapefile/class pairs provided.")

    # check files exist
    for p, c in pairs:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # Read first to get target CRS
    first_path = pairs[0][0]
    gdf0 = gpd.read_file(str(first_path))
    target_crs = gdf0.crs

    out_gdfs = []
    for shp_path, class_name in pairs:
        gdf = gpd.read_file(str(shp_path))

        # Reproject to target CRS if necessary
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # assign class column (string)
        gdf["class"] = class_name

        out_gdfs.append(gdf)

    # concatenate
    merged = gpd.GeoDataFrame(pd.concat(out_gdfs, ignore_index=True), crs=target_crs)

    if dissolve:
        # union all geometries into a single geometry and create single-row GeoDataFrame
        union_geom = unary_union(merged.geometry.values)
        out_gdf = gpd.GeoDataFrame(geometry=[union_geom], crs=target_crs)
        # If you want a meaningful class name for the dissolved output, use "Merged"
        out_gdf["class"] = "Merged"
        out_gdf.to_file(str(output_shp))
        return out_gdf
    else:
        # ensure output folder exists
        Path(output_shp).parent.mkdir(parents=True, exist_ok=True)
        merged.to_file(str(output_shp))
        return merged



def merge_two_shapefiles_keep_class(shp1, shp2, out_shp):

    p1 = Path(shp1)
    p2 = Path(shp2)
    if not p1.exists():
        raise FileNotFoundError(f"Input file not found: {p1}")
    if not p2.exists():
        raise FileNotFoundError(f"Input file not found: {p2}")

    gdf1 = gpd.read_file(str(p1))
    gdf2 = gpd.read_file(str(p2))

    if gdf1.empty:
        raise ValueError(f"{p1} contains no features.")
    if gdf2.empty:
        raise ValueError(f"{p2} contains no features.")

    # Helper: find 'class' column case-insensitively and standardize to 'class'
    def standardize_class_col(gdf):
        for col in gdf.columns:
            if col.lower() == "class":
                if col != "class":
                    gdf = gdf.rename(columns={col: "class"})
                return gdf
        raise ValueError("Missing 'class' attribute (case-insensitive).")

    # Check and standardize
    try:
        gdf1 = standardize_class_col(gdf1)
    except ValueError:
        raise ValueError(f"Shapefile {p1} does not contain a 'class' attribute.")
    try:
        gdf2 = standardize_class_col(gdf2)
    except ValueError:
        raise ValueError(f"Shapefile {p2} does not contain a 'class' attribute.")

    # Ensure 'class' is string type
    gdf1["class"] = gdf1["class"].astype(str)
    gdf2["class"] = gdf2["class"].astype(str)

    # Reproject gdf2 to gdf1 CRS if needed
    if gdf1.crs != gdf2.crs:
        if gdf1.crs is None:
            raise ValueError(f"CRS of {p1} is undefined. Define CRS before merging.")
        gdf2 = gdf2.to_crs(gdf1.crs)

    # Concatenate (preserve all attributes; missing columns will be NaN)
    merged = gpd.GeoDataFrame(
        gpd.pd.concat([gdf1, gdf2], ignore_index=True),
        crs=gdf1.crs
    )

    # Write out
    Path(out_shp).parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(str(out_shp))

    print(f"Merged {len(gdf1)} + {len(gdf2)} features -> {len(merged)} written to: {out_shp}")
    return merged




