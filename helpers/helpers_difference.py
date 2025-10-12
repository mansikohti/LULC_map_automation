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
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import polygonize, linemerge
import shapely



## subtract the classes layer except settlement from filled_dissolved_buffered_settlement 
def subtract_shapefiles(dissolved_shp: str, merged_shp: str, output_shp: str) -> gpd.GeoDataFrame:

    # read inputs
    gdf_d = gpd.read_file(dissolved_shp)
    gdf_m = gpd.read_file(merged_shp)

    if gdf_d.empty:
        raise ValueError("dissolved_shp contains no features.")
    if gdf_m.empty:
        # nothing to subtract; just save dissolved as output
        print("merged_shp is empty — copying dissolved_shp to output.")
        out = gdf_d.copy()
        Path(output_shp).parent.mkdir(parents=True, exist_ok=True)
        out.to_file(output_shp)
        return out

    # ensure a common CRS (use dissolved CRS as target)
    target_crs = gdf_d.crs
    if gdf_m.crs != target_crs:
        gdf_m = gdf_m.to_crs(target_crs)

    # fix invalid geometries where possible
    gdf_d['geometry'] = gdf_d.geometry.buffer(0)
    gdf_m['geometry'] = gdf_m.geometry.buffer(0)

    # union geometries to single geometry each side
    union_d = unary_union(gdf_d.geometry.values)
    union_m = unary_union(gdf_m.geometry.values)

    # compute difference
    diff_geom = union_d.difference(union_m)

    # If result is empty, return empty GeoDataFrame
    if diff_geom.is_empty:
        print("Resulting difference is empty (no area remains after subtraction).")
        out_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=target_crs)
    else:
        out_gdf = gpd.GeoDataFrame(geometry=[diff_geom], crs=target_crs)

    # add optional attribute (e.g., source info)
    out_gdf['source'] = f"({Path(dissolved_shp).stem}) - ({Path(merged_shp).stem})"

    # save output
    Path(output_shp).parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(output_shp)
    print(f"Saved difference shapefile to: {output_shp}")

    # print summary areas (in CRS units — if projected units are meters, area in m^2)
    try:
        area_out = out_gdf.geometry.area.sum()
        area_d = gdf_d.geometry.area.sum()
        area_m = gdf_m.geometry.area.sum()
        print(f"Area (dissolved): {area_d:.2f}")
        print(f"Area (merged to subtract): {area_m:.2f}")
        print(f"Area (difference/out): {area_out:.2f}")
    except Exception:
        pass

    return out_gdf



## subtract merged features from AOI
def subtract_using_aoi_inset_and_write(
    aoi_shp: str,
    classes_shp: str,
    unclassified_shp: str,
    merged_shp: str,
    inset_m: float = 1.0,
    reproject_classes_to_aoi: bool = True,
    explode_result: bool = True,
    min_area_sqm: float = 0.0,
    dissolve_by_class: bool = False
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
   
    # basic checks
    aoi_path = Path(aoi_shp)
    classes_path = Path(classes_shp)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")

    # read inputs
    aoi = gpd.read_file(str(aoi_path))
    classes = gpd.read_file(str(classes_path))

    if aoi.empty:
        raise ValueError("AOI shapefile is empty.")

    # Ensure both have CRS defined
    if aoi.crs is None:
        raise ValueError("AOI shapefile has no CRS defined. Define CRS and re-run.")
    if classes.crs is None:
        raise ValueError("Classes shapefile has no CRS defined. Define CRS and re-run.")


    # Optionally reproject classes to AOI (projected) CRS for correct geometry math
    if reproject_classes_to_aoi and (classes.crs != aoi.crs):
        classes = classes.to_crs(aoi.crs)

    # Clean geometries
    aoi['geometry'] = aoi.geometry.buffer(0)
    classes['geometry'] = classes.geometry.buffer(0) if not classes.empty else classes.geometry

    # Compute AOI_outer_union (outer boundary area) BEFORE inset — we'll use this to clip final merged features
    # Use unary_union of AOI polygons (and buffer(0) already applied); results in a Polygon/MultiPolygon
    aoi_outer_union = None
    try:
        aoi_outer_union = unary_union(aoi.geometry.values)
    except Exception:
        # fallback: union piecewise
        aoi_outer_union = unary_union([g for g in aoi.geometry.values if g is not None])

    # Create the inward inset AOI (buffer negative); handle possible empty geometries
    inset_distance = -abs(inset_m)  # negative for inward buffer
    aoi_inset_list = []
    for idx, row in aoi.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            inset_geom = geom.buffer(inset_distance, join_style=2)  # mitre join to preserve shape
        except Exception:
            inset_geom = geom.buffer(inset_distance)  # fallback
        if inset_geom is None or inset_geom.is_empty:
            # if inset wiped out the AOI feature, skip
            continue
        rec = row.copy()
        rec.geometry = inset_geom
        aoi_inset_list.append(rec)

    if not aoi_inset_list:
        # nothing left after inset -> no unclassified area
        print(f"AOI inset by {inset_m} m produced no geometry (all AOI features disappeared).")
        unclassified_gdf = gpd.GeoDataFrame(columns=aoi.columns, geometry='geometry', crs=aoi.crs)
        merged_all_gdf = gpd.GeoDataFrame(columns=['class','geometry'], geometry='geometry', crs=aoi.crs)
        # save empty outputs
        Path(unclassified_shp).parent.mkdir(parents=True, exist_ok=True)
        Path(merged_shp).parent.mkdir(parents=True, exist_ok=True)
        unclassified_gdf.to_file(unclassified_shp)
        merged_all_gdf.to_file(merged_shp)
        return unclassified_gdf, merged_all_gdf

    aoi_inset_gdf = gpd.GeoDataFrame(aoi_inset_list, columns=aoi.columns, crs=aoi.crs)

    # Union class geometries (if classes empty, set None)
    classes_union = None
    if not classes.empty:
        try:
            classes_union = unary_union(classes.geometry.values)
        except Exception:
            classes_union = None

    # Subtract classes_union from each AOI_inset geometry -> build unclassified records
    unclassified_records = []
    for idx, row in aoi_inset_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if classes_union is None:
            diff = geom
        else:
            diff = geom.difference(classes_union)
        if diff is None or diff.is_empty:
            continue
        rec = row.copy()
        rec.geometry = diff
        unclassified_records.append(rec)

    # Create unclassified GeoDataFrame
    if not unclassified_records:
        unclassified_gdf = gpd.GeoDataFrame(columns=aoi_inset_gdf.columns, geometry='geometry', crs=aoi_inset_gdf.crs)
    else:
        unclassified_gdf = gpd.GeoDataFrame(unclassified_records, columns=aoi_inset_gdf.columns, crs=aoi_inset_gdf.crs)

    # Explode multipart -> singlepart if requested
    if explode_result and not unclassified_gdf.empty:
        unclassified_gdf = unclassified_gdf.explode(index_parts=False).reset_index(drop=True)

    # Ensure 'class' column exists and set to 'unclassified'
    if 'class' in unclassified_gdf.columns:
        unclassified_gdf['class'] = 'Unclassified'
    else:
        unclassified_gdf = unclassified_gdf.copy()
        unclassified_gdf['class'] = 'Unclassified'

    # Remove tiny polygons if requested (area in CRS units, typically m^2)
    if min_area_sqm and min_area_sqm > 0 and not unclassified_gdf.empty:
        unclassified_gdf['area_sqm'] = unclassified_gdf.geometry.area
        unclassified_gdf = unclassified_gdf[unclassified_gdf['area_sqm'] >= min_area_sqm].drop(columns=['area_sqm']).reset_index(drop=True)

    # Prepare classes_gdf and ensure it has a 'class' column (case-insensitive)
    def ensure_class_col(gdf, fallback_name):
        for col in gdf.columns:
            if col.lower() == 'class':
                if col != 'class':
                    gdf = gdf.rename(columns={col: 'class'})
                gdf['class'] = gdf['class'].astype(str)
                return gdf
        gdf = gdf.copy()
        gdf['class'] = fallback_name
        return gdf

    classes_gdf = classes.copy()
    if not classes_gdf.empty:
        classes_gdf = ensure_class_col(classes_gdf, fallback_name=classes_path.stem)

    # Concatenate classes + unclassified
    to_concat = []
    if not classes_gdf.empty:
        to_concat.append(classes_gdf)
    if not unclassified_gdf.empty:
        to_concat.append(unclassified_gdf)

    if not to_concat:
        merged_all_gdf = gpd.GeoDataFrame(geometry=[], crs=aoi_inset_gdf.crs)
    else:
        merged_all_gdf = gpd.GeoDataFrame(pd.concat(to_concat, ignore_index=True), crs=aoi_inset_gdf.crs)

    # Optionally dissolve by class (one geometry per unique class)
    if dissolve_by_class and not merged_all_gdf.empty:
        dissolved_list = []
        for cls_val, grp in merged_all_gdf.groupby('class'):
            geom = unary_union(grp.geometry.values)
            dissolved_list.append({'class': cls_val, 'geometry': geom})
        merged_all_gdf = gpd.GeoDataFrame(dissolved_list, columns=['class', 'geometry'], crs=merged_all_gdf.crs)

    # --- NEW: Clip merged_all_gdf to AOI outer union (remove parts outside AOI) ---
    if aoi_outer_union is not None and not merged_all_gdf.empty:
        clipped_records = []
        for idx, row in merged_all_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                clipped = geom.intersection(aoi_outer_union)
            except Exception:
                # if intersection fails, skip this feature
                clipped = None
            if clipped is None or clipped.is_empty:
                # feature lies completely outside AOI outer boundary -> drop it
                continue
            rec = row.copy()
            rec.geometry = clipped
            clipped_records.append(rec)
        if clipped_records:
            merged_all_gdf = gpd.GeoDataFrame(clipped_records, columns=merged_all_gdf.columns, crs=merged_all_gdf.crs)
        else:
            # nothing remains after clipping
            merged_all_gdf = gpd.GeoDataFrame(geometry=[], crs=aoi_inset_gdf.crs)

    # Final cleanup: remove empty geometries
    unclassified_gdf = unclassified_gdf[~(unclassified_gdf.geometry.is_empty | unclassified_gdf.geometry.isnull())].reset_index(drop=True)
    merged_all_gdf = merged_all_gdf[~(merged_all_gdf.geometry.is_empty | merged_all_gdf.geometry.isnull())].reset_index(drop=True)

    # Write outputs (in the projected CRS used)
    Path(unclassified_shp).parent.mkdir(parents=True, exist_ok=True)
    Path(merged_shp).parent.mkdir(parents=True, exist_ok=True)


    ## clean the files
    def try_make_polygon(geom):
        """Return Polygon/MultiPolygon if possible, else None."""
        if geom is None or geom.is_empty:
            return None

        gtype = geom.geom_type
        if gtype in ("Polygon", "MultiPolygon"):
            return geom

        # If it's a LineString / MultiLineString, try to polygonize
        if gtype in ("LineString", "MultiLineString"):
            try:
                merged = linemerge(geom)
                polys = list(polygonize(merged))
                if not polys:
                    return None
                if len(polys) == 1:
                    return polys[0]
                return MultiPolygon(polys)
            except Exception:
                return None

        # GeometryCollection: extract polys or polygonize lines inside
        if gtype == "GeometryCollection":
            polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
            if lines:
                try:
                    merged = linemerge(lines)
                    polys += list(polygonize(merged))
                except Exception:
                    pass
            if not polys:
                return None
            # ensure MultiPolygon if multiple
            if len(polys) == 1:
                return polys[0]
            return MultiPolygon(polys)

        # fallback: try buffer(0) (sometimes fixes invalid or converts)
        try:
            buf = geom.buffer(0)
            if buf and buf.geom_type in ("Polygon", "MultiPolygon"):
                return buf
        except Exception:
            pass

        return None

    # --- debug: show problematic geometry types before cleaning ---
    if not merged_all_gdf.empty:
        geom_types = merged_all_gdf.geometry.geom_type.value_counts()
        print("Geometry types before cleaning:\n", geom_types)

    # Attempt conversion where feasible, otherwise drop non-polygons
    cleaned_records = []
    for idx, row in merged_all_gdf.iterrows():
        new_geom = try_make_polygon(row.geometry)
        if new_geom is None:
            # skip features we can't convert to polygon (alternatively collect them to a separate file)
            continue
        rec = row.copy()
        rec.geometry = new_geom
        cleaned_records.append(rec)

    if cleaned_records:
        merged_all_gdf = gpd.GeoDataFrame(pd.concat(cleaned_records, axis=1).T, columns=merged_all_gdf.columns, crs=merged_all_gdf.crs)
    else:
        merged_all_gdf = gpd.GeoDataFrame(geometry=[], crs=merged_all_gdf.crs)

    # debug: show geometry types after cleaning
    if not merged_all_gdf.empty:
        print("Geometry types after cleaning:\n", merged_all_gdf.geometry.geom_type.value_counts())



    unclassified_gdf.to_file(unclassified_shp)
    merged_all_gdf.to_file(merged_shp)

    # Print quick summary
    try:
        print(f"AOI inset by {inset_m} m: features {len(aoi_inset_gdf)}")
        print(f"Unclassified features: {len(unclassified_gdf)}, total area (units²): {unclassified_gdf.geometry.area.sum():.2f}")
        print(f"Merged (clipped to AOI) features: {len(merged_all_gdf)}, total area (units²): {merged_all_gdf.geometry.area.sum():.2f}")
    except Exception:
        pass

    print(f"Saved unclassified -> {unclassified_shp}")
    print(f"Saved merged (classes + unclassified, clipped to AOI) -> {merged_shp}")

    return unclassified_gdf, merged_all_gdf
