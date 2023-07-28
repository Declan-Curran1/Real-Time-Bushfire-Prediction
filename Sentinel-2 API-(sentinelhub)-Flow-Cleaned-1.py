# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:11:18 2023

@author: djc40
"""

#####
#1
#####

###
#Using the 'sentinelhub' api provider which provides mosiacs of sentinel 2 10m resolution satellite imagery based on latitude/longditude coordinates + timeframe
#https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html

#NOTE: You can use any satellite imagery provider. Just save it in a time-series format as done below with each image named as its date

#Configuration of account instructions are here: https://sentinelhub-py.readthedocs.io/en/latest/configure.html

#Login here and follow instructions"
#https://services.sentinel-hub.com/oauth/auth?client_id=30cf1d69-af7e-4f3a-997d-0643d660a478&redirect_uri=https%3A%2F%2Fapps.sentinel-hub.com%2Fdashboard%2FoauthCallback.html&scope=&response_type=token&state=%252Faccount%252Fsettings


###@@@###
#Query API for images
###@@@###

from sentinelhub import SHConfig

config = SHConfig()
#My sentinelhub api query details used here but replace with your own
config.instance_id = 'd087a269-218e-4aa8-9f59-f743c2d898d9' #'<your instance id>'
config.sh_client_id = '8a9d5c18-f68f-4d56-a6da-26b5bf2f9f01'
config.sh_client_secret = '(L8ix@g}<X9|BZvMo:xHd/Gc;[)DO%~5Q3@822|v'


import datetime
from datetime import timedelta
import time
import os

import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


# The following is not a package. It is a file utils.py which should be in the same folder as this notebook.
os.chdir("D:\Dev\Python\Bushfire\Sentinel-2 API")
from utils import plot_image
import .utils

Sydney_coords_wgs84 = [150.095215,-34.352507,151.369629,-33.146750]

resolution = 100
Sydney_bbox = BBox(bbox=Sydney_coords_wgs84, crs=CRS.WGS84)
Sydney_size = bbox_to_dimensions(Sydney_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {Sydney_size} pixels")

#Can retrieve true colour from link above

#SENTINEL2_L2A DOES NOT HAVE A BAND 10

evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
                units: "DN"
            }],
            output: {
                bands: 13,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01,
                sample.B02,
                sample.B03,
                sample.B04,
                sample.B05,
                sample.B06,
                sample.B07,
                sample.B08,
                sample.B8A,
                sample.B09,
                
                sample.B11,
                sample.B12];
    }
"""

#####
#Iterate this part x no. times (2 - 3 days seems to be the sweet spot to get reliable image data)

#for i in range(100):
    

#

request_all_bands = SentinelHubRequest(
    evalscript=evalscript_all_bands,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A, #SENTINEL2_L1C
            time_interval=("2019-12-15", "2019-12-18"), #Choose time interval
            mosaicking_order=MosaickingOrder.LEAST_CC,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=Sydney_bbox,
    size=Sydney_size,
    config=config,
)

#####

all_bands_response = request_all_bands.get_data()

request_all_bands.data_folder='D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test'
print(request_all_bands.data_folder)

import time

start = time.time()

request_all_bands.save_data()

end = time.time()
print(end-start)



###@@@###
#Change folder name to the date of the image
###@@@###


#Try renaming all files in a certain folder iteratively
#Get list of file names in a directory
import os
import datetime
import json


path_to_convert = 'D:/path' #'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test5(10m-res-Mallacoota)'
old_names = os.listdir(path_to_convert)

#Define function to add a '0' digit when either day or month is a single digit
def construct_path(year, month, day):
    return f'{year}{month:02d}{day:02d}'



for i in range(len(old_names)):
    
    old_path = path_to_convert + '/' + old_names[i]
    
    #Read in the .json file from the old path, and use this to define the date
    #which will become the new name
    
    with open(old_path + '/request.json', 'r') as f:
      json_temp = json.load(f)
    
    date_temp = datetime.datetime.strptime(json_temp['request']['payload']['input']['data'][0]['dataFilter']['timeRange']['to'],'%Y-%m-%dT%H:%M:%SZ')
    
    #Old method to aggregate names - didn't have a 0 in front of single digits
    #new_name = str(date_temp.year) + str(date_temp.month) + str(date_temp.day) #+ str(date_temp.hour) #could also do 'minute' + 'second'
    
    #New method - uses a function to add the digit
    

    new_name = construct_path(date_temp.year,date_temp.month,date_temp.day)
    
    
    new_path = path_to_convert + '/' + new_name
    
    
    os.rename(old_path,new_path)

###@@@###
#Plot band 12 (Short wave infrared)
###@@@###

# Image showing the SWIR band B12
# Factor 1/1e4 due to the DN band values in the range 0-10000
# Factor 3.5 to increase the brightness
plot_image(all_bands_response[0][:, :, 12], factor=3.5 / 1e4, vmax=1)





###@@@###
#Query sentinelhub about the cloud cover for each image
###@@@###

#More information here:
#https://sentinelhub-py.readthedocs.io/en/latest/examples/aws_request.html
#ctrl f 'cloud' to get relevant section

from statistics import mean

from sentinelhub import CRS, BBox, DataCollection, SHConfig, WebFeatureService

#config = SHConfig()

search_bbox = Sydney_bbox #BBox(bbox=(46.16, -16.15, 46.51, -15.58), crs=CRS.WGS84)
search_time_interval = ("2019-06-04", "2019-06-05") #("2019-06-02", "2019-06-03") 


wfs_iterator = WebFeatureService(
    search_bbox, search_time_interval, data_collection=DataCollection.SENTINEL2_L2A, maxcc=1.0, config=config #data_collection=DataCollection.SENTINEL2_L1C
)

temp = 0
length = 0 

for tile_info in wfs_iterator:
    temp = temp + tile_info['properties']['cloudCoverPercentage']
    length = length + 1 
    print(tile_info['properties']['cloudCoverPercentage']) #Use '.keys()' part of __dict__ object to see parts
    #print(mean(tile_info['properties']['cloudCoverPercentage']))

#AVG CLOUD COVER
avgcloudcover = temp/length
temp/length

if length == 0:
    #You can insert code here - NO images found! -



#Honestly may be best to read in all images, and just use this to identify
#-(a) images with high cloud cover
#-(b) images that had no data for that date
    
    
    
    
    