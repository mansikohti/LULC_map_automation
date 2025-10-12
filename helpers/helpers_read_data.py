import os
import geopandas as gpd
from pathlib import Path
from typing import List, Tuple, Optional

from pathlib import Path
from typing import List, Tuple, Optional

# def fetch_shapefile_paths(folder: str) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
   
#     folder_path = Path(folder)
#     if not folder_path.exists():
#         raise FileNotFoundError(f"Folder not found: {folder}")

#     # collect all .shp files in folder
#     shp_files = [f for f in folder_path.glob("*.shp")]

#     aoi_path = None
#     settlement_path = None
#     other_paths = []

#     for shp in shp_files:
#         name = shp.stem.lower()  # case-insensitive check
#         if "aoi" in name:
#             aoi_path = str(shp)
#         elif "settlement" in name:
#             settlement_path = str(shp)
#         else:
#             other_paths.append(str(shp))

#     return aoi_path, settlement_path, other_paths



def fetch_shapefile_paths(folder: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # collect all .shp files in folder
    shp_files = list(folder_path.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No shapefiles (.shp) found in the given folder.")

    aoi_path = None
    settlement_path = None
    other_paths = []

    # --- Check for CRS consistency ---
    crs_set = set()
    for shp in shp_files:
        gdf = gpd.read_file(shp)
        if gdf.crs is not None:
            crs_set.add(gdf.crs.to_string())
        else:
            raise ValueError(f"Shapefile {shp} has no CRS defined.")

    if len(crs_set) > 1:
        raise ValueError("All input data is not in the same coordinate system.")

    # --- Assign shapefile paths ---
    for shp in shp_files:
        name = shp.stem.lower()
        if "aoi" in name:
            aoi_path = str(shp)
        elif "settlement" in name:
            settlement_path = str(shp)
        else:
            other_paths.append(str(shp))

    return aoi_path, settlement_path, other_paths


def make_shapefile_pairs(shapefiles: List[str]) -> List[Tuple[str, str]]:
    """
    Convert a list of shapefile paths into pairs: (path, name).
    The name is derived from filename (before .shp), with underscores replaced by spaces.
    """
    pairs = []
    for shp in shapefiles:
        stem = Path(shp).stem  # file name without extension
        name = stem.replace("_", " ").title()  # format: underscores -> spaces, Title Case
        pairs.append((str(shp), name))
    return pairs
