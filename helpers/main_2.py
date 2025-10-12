import os

import rasterio
from rasterio.transform import from_origin
from rasterio import _io, sample, vrt, env, crs, errors, warp, transform, coords, windows, dtypes, enums, plot, features, mask, merge

from helpers_clean import *
from helpers_difference import *
from helpers_buffer import *
from helpers_merge import *
from helpers_read_data import *


def processing(input_folder_path, 
               output_folder_path,
               buffer,
               min_area_removed,
               inset_buffer_for_AOI):

    ## fetch the data and check the CRS (preprocessing)
    aoi_path, settlement_path, other_shps = fetch_shapefile_paths(input_folder_path)
    pairs = make_shapefile_pairs(other_shps)
    print(pairs)


    ## operation 1 
    settlement_folder_path = os.path.join(output_folder_path, "related_files_settlement")
    os.makedirs(settlement_folder_path, exist_ok=True)

    buffered_settlement_path = os.path.join(settlement_folder_path, "buffered_settlement_shp.shp")
    dissolved_settlement_path = os.path.join(settlement_folder_path, "dissolved_settlement_shp.shp")
    dissolved_and_gap_filled_settlement_path = os.path.join(settlement_folder_path, "dissoved_and_gap_filled_settlement_shp.shp")


    # Example usage:
    gdf_buf, gdf_diss, gdf_diss_filled = buffer_and_dissolve(
        input_shp=settlement_path,
        buffer_m=buffer,
        out_buffered=buffered_settlement_path,
        out_dissolved=dissolved_settlement_path,
        out_filled_dissolved=dissolved_and_gap_filled_settlement_path ,
        min_area_remove=0
    )


    ## operation 2
    # merge all classes without dissolved settlment

    classes_folder_path = os.path.join(output_folder_path, "related_files_classes")
    os.makedirs(classes_folder_path, exist_ok=True)

    merged_classes_without_settlement_path = os.path.join(classes_folder_path, "merged_classes_except_settlement_shp.shp")

    merged_classes_without_settlement_gdf = merge_shapefiles_with_class(pairs, merged_classes_without_settlement_path, dissolve=False)



    ## subtract all features from dissolved settlement 

    cropped_dissolved_settlement_path = os.path.join(settlement_folder_path, "cropped_dissolved_settlement_shp.shp")

    cropped_dissolved_settlement_gdf = subtract_shapefiles(dissolved_and_gap_filled_settlement_path, 
                                                            merged_classes_without_settlement_path, 
                                                            cropped_dissolved_settlement_path)



    exploded_path = os.path.join(settlement_folder_path, "exploded_settlement_shp.shp")
    filtered_settlement_path = os.path.join(settlement_folder_path, "final_cropped_settlement_shp.shp")

    exploded, filtered = explode_and_filter_features(
        cropped_dissolved_settlement_path,
        exploded_path,
        filtered_settlement_path,
        min_area_sqm=min_area_removed
    )


    ## operation 3
    merged_classes_path = os.path.join(classes_folder_path, "merged_classes_except_unclassified.shp")
    merged_gdf = merge_two_shapefiles_keep_class(filtered_settlement_path, 
                                                merged_classes_without_settlement_path, 
                                                merged_classes_path)


    unclassified_path = os.path.join(output_folder_path, "unclassified_shp.shp")
    merged_with_unclassified_path = os.path.join(classes_folder_path, "all_classes_including_unclassified_shp.shp")
    unclassified_gdf, merged_gdf = subtract_using_aoi_inset_and_write(
        aoi_shp=aoi_path,
        classes_shp=merged_classes_path,
        unclassified_shp=unclassified_path,
        merged_shp=merged_with_unclassified_path,
        inset_m=inset_buffer_for_AOI,
        reproject_classes_to_aoi=True,
        explode_result=True,
        min_area_sqm=0.1,
        dissolve_by_class=False
    )


    output_path = os.path.join(output_folder_path, "final_classes_shp.shp")

    cleaned_gdf = clean_attributes_with_area(
        input_shp=merged_with_unclassified_path ,
        output_shp=output_path
    )


    return None 


input_folder_path = "D:/2_Analytics/9_LULC_classification/automation/data_4/Input"
output_folder_path = "D:/2_Analytics/9_LULC_classification/automation/data_4/output"
buffer = 10
min_area_removed = 25
inset_buffer = 0.2


# processing( input_folder_path = input_folder_path, 
#                         output_folder_path = output_folder_path,
#                         buffer= buffer,
#                         min_area_removed = min_area_removed,
#                         inset_buffer_for_AOI= inset_buffer)