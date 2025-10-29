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


#### operation 1: 
# get the settlement file --> 1. Add buffer 2. dissolve the shapes 3. fill the gaps 

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

    # Reproject if CRS is geographic (degrees)
    if gdf.crs.is_geographic:
        try:
            utm_crs = gdf.estimate_utm_crs()
            gdf = gdf.to_crs(utm_crs)
            print("Reprojected to metric CRS:", utm_crs)
        except Exception:
            gdf = gdf.to_crs("EPSG:3857")
            print("Falling back to EPSG:3857 (Web Mercator)")

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



### operation 2: 

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
        raise ValueError("Input shapefile has no CRS defined. Assign a projected CRS in meters (e.g., EPSG:32643).")

    # Reproject if geographic
    if gdf.crs.is_geographic:
        print(f"Input CRS {gdf.crs} is geographic (degrees). Reprojecting to UTM (metric)...")
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
        print("Reprojected to:", gdf.crs)

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




######## Operation 3:

## merge the all features into single shape file (all features + filtered settlement)
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

    # If AOI CRS is geographic, reproject AOI to estimated UTM (metric) for buffering
    aoi_original_crs = aoi.crs
    if aoi.crs.is_geographic:
        try:
            utm_crs = aoi.estimate_utm_crs()
        except Exception:
            utm_crs = "EPSG:3857"
        aoi = aoi.to_crs(utm_crs)
        print(f"Reprojected AOI from {aoi_original_crs} -> {utm_crs} for metric buffering.")
    else:
        utm_crs = aoi.crs

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
        unclassified_gdf['class'] = 'unclassified'
    else:
        unclassified_gdf = unclassified_gdf.copy()
        unclassified_gdf['class'] = 'unclassified'

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