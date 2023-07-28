# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:47:06 2023

@author: djc40
"""

#####
#3
#####

#Take k-means classified images and 'polygonise' these so they are in a .shp file format. 
#This is done to calculate the portion of each cluster type within given boundaries

#Flow
from osgeo import ogr
from osgeo import gdal
import os
import time


path_with_files = 'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test2'
input_files = os.listdir(path_with_files)

start = time.time() 

for j in range(len(input_files)): #len(input_files)
    clusterised_path = path_with_files + '/' + input_files[j] + '/clustered.tiff' 
    
    
    
    # get raster data source
    open_image = gdal.Open(clusterised_path)
    input_band = open_image.GetRasterBand(1)
    
    output_path = path_with_files + '/' + input_files[j] + '/clustered_poly.shp'
    
    # create output data source
    output_shp = output_path
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    # create output file name
    output_shapefile = shp_driver.CreateDataSource(output_shp)
    new_shapefile = output_shapefile.CreateLayer(output_shp, srs = None )
    
    ##
    newField = ogr.FieldDefn('MYFLD', ogr.OFTInteger)
    new_shapefile.CreateField(newField)
    
    
    ##
    
    
    gdal.Polygonize(input_band, None, new_shapefile, 0, [], callback=None)
    new_shapefile.SyncToDisk()



    end = time.time()
    
    print(end-start)
    
    
    
    
    
#####################################################
#FIN (FLOW)    
#####################################################    
    
    
    
    


######REAL




from osgeo import ogr
from osgeo import gdal
import time

# get raster data source
open_image = gdal.Open("D:/Dev/Python/Bushfire/Unsupervised/NAIP_final.tif")
input_band = open_image.GetRasterBand(1)

# create output data source
output_shp = "D:/Dev/Python/Bushfire/Unsupervised/NAIP_final_rasTOvec_3.shp"
shp_driver = ogr.GetDriverByName("ESRI Shapefile")
# create output file name
output_shapefile = shp_driver.CreateDataSource(output_shp)
new_shapefile = output_shapefile.CreateLayer(output_shp, srs = None )

##
newField = ogr.FieldDefn('MYFLD', ogr.OFTInteger)
new_shapefile.CreateField(newField)


##


gdal.Polygonize(input_band, None, new_shapefile, 0, [], callback=None)
new_shapefile.SyncToDisk()

######YES
#30/01/2023 - DONE - WE HAVE THE SHAPEFILE FOR EACH OF THE CLUSTERS










from osgeo import ogr
from osgeo import gdal
import time





#Test for KC Output
start = time.time()


# get raster data source
open_image = gdal.Open("D:/Dev/Python/Bushfire/Unsupervised/NAIP_final.tif")
input_band = open_image.GetRasterBand(1)

# create output data source
output_shp = "D:/Dev/Python/Bushfire/Unsupervised/NAIP_final_rasTOvec.shp"
shp_driver = ogr.GetDriverByName("ESRI Shapefile")
# create output file name
output_shapefile = shp_driver.CreateDataSource(output_shp)
new_shapefile = output_shapefile.CreateLayer(output_shp, srs = None )

gdal.Polygonize(input_band, None, new_shapefile, -1, [], callback=None)
new_shapefile.SyncToDisk()







end = time.time()
print(end - start)
'''
# Emulates GDAL's gdal_polygonize.py

import argparse
import logging
import subprocess
import sys

import fiona
import numpy as np
import rasterio
from rasterio.features import shapes


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger('rasterio_polygonize')


def main(raster_file, vector_file, driver, mask_value):
    
    with rasterio.drivers():
        
        with rasterio.open(raster_file) as src:
            image = src.read(1)
        
        if mask_value is not None:
            mask = image == mask_value
        else:
            mask = None
        
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                shapes(image, mask=mask, transform=src.affine)))

        with fiona.open(
                vector_file, 'w', 
                driver=driver,
                crs=src.crs,
                schema={'properties': [('raster_val', 'int')],
                        'geometry': 'Polygon'}) as dst:
            dst.writerecords(results)
    
    return dst.name

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Writes shapes of raster features to a vector file")
    parser.add_argument(
        'input', 
        metavar='INPUT', 
        help="Input file name")
    parser.add_argument(
        'output', 
        metavar='OUTPUT',
        help="Output file name")
    parser.add_argument(
        '--output-driver',
        metavar='OUTPUT DRIVER',
        help="Output vector driver name")
    parser.add_argument(
        '--mask-value',
        default=None,
        type=int,
        metavar='MASK VALUE',
        help="Value to mask")
    args = parser.parse_args()

    name = main(args.input, args.output, args.output_driver, args.mask_value)
    
    print subprocess.check_output(
            ['ogrinfo', '-so', args.output, name])

''' and None