# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 19:31:51 2022

@author: djc40
"""


#####
#2
#####


#Train k-means clustering model on our saved time-series satellite imagery. 
#Then, apply this clustering to our images to group images into eight groups; seven of these are clusters based on 
#k-means groupings of the 12 satellite bands (RGB,Infrared etc.) and the eighth is the 12th band (Short wave infrared)
#which is a good proxy for the heat of an object - I am using this as whether a 'fire' is present in a given 10m pixel
#and will be our output variable.

#Import packages

from sklearn.cluster import KMeans
import gdal
import numpy as np
from osgeo import gdal #idk why this works 
import time
import pickle
import os


######################################
#TRAIN MODEL
######################################
##Rudimentary for now##


#Read in image to classify with gdal
#For now, use images from 19th Dec 2019, and 22nd Dec 2019 as training data for the model

#naip_fn = 'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test/c363e19c3be538caada57694e5f7b249/response.tiff'
naip_fn = 'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test/2019121923/response.tiff'
naip_fn = 'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test/2019122223/response.tiff'



driverTiff = gdal.GetDriverByName('GTiff')

naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount

# create an empty array, each column of the empty array will hold one band of data from the image
# loop through each band in the image nad add to the data array
data = np.empty((naip_ds.RasterXSize*naip_ds.RasterYSize, nbands))
for i in range(1, nbands+1): #range(1, nbands+1)
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()

# set up the kmeans classification, fit, and predict
start = time.time()
km = KMeans(n_clusters=7)
km.fit(data)
#km.predict(data)
end = time.time()

print(end-start)


#####
#Can check the cluster centers:
t = km.cluster_centers_


#####
#You can your KMeans file. Try using this example to do this: https://stackoverflow.com/questions/54879434/how-to-use-pickle-to-save-sklearn-model

#Save

with open("D:/Dev/Python/Bushfire/Unsupervised/Flow-test.pkl", "wb") as f:
    pickle.dump(km, f)
    
#Load
with open("D:/Dev/Python/Bushfire/Unsupervised/Flow-test.pkl", "rb") as f:
    model = pickle.load(f)





###########################################
#Trained model is saved in object 'km'


###########################################
#APPLY MODEL

#Begin iteratively going through each image and apply trained algorithm
###########################################

#INPUTS
#-The trained 'km' object above
#-The .tiff image read in below


#Loop through all the files, apply the model, and save the output .tif to an image in the same folder


path_with_files = 'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test2'
input_files = os.listdir(path_with_files)

start = time.time() 

for j in range(len(input_files)):

    # read in image to classify with gdal
    image_path = path_with_files + '/' + input_files[j] + '/response.tiff' 
    #naip_fn_1 = 'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test/c363e19c3be538caada57694e5f7b249/response.tiff'
    
    
    driverTiff = gdal.GetDriverByName('GTiff')
    
    image = gdal.Open(image_path)
    nbands = image.RasterCount
    
    #naip_ds_1 = gdal.Open(naip_fn_1)
    #nbands = naip_ds_1.RasterCount
    
    # create an empty array, each column of the empty array will hold one band of data from the image
    # loop through each band in the image nad add to the data array
    data_loop = np.empty((image.RasterXSize*image.RasterYSize, nbands))
    for i in range(1, nbands+1): #range(1, nbands+1)
        band = image.GetRasterBand(i).ReadAsArray()
        data_loop[:, i-1] = band.flatten()
    
    #Take the already trained model 'km' and apply this to the image that has been read in:
    km.predict(data)
    
    
    
    
    # format the predicted classes to the shape of the original image
    out_dat = km.labels_.reshape((image.RasterYSize, image.RasterXSize))
    
    
    output_path = path_with_files + '/' + input_files[j] + '/clustered.tiff'
    
    
    # save the original image with gdal
    clfds = driverTiff.Create(output_path, image.RasterXSize, image.RasterYSize, 1, gdal.GDT_Float32) #Add _1 for new sentinelhub api data
    clfds.SetGeoTransform(image.GetGeoTransform()) #naip_ds_1.GetGeoTransform()
    clfds.SetProjection(image.GetProjection())
    clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
    clfds.GetRasterBand(1).WriteArray(out_dat)
    clfds = None


    end = time.time()
    
    print(end-start)


######################################
#Fin
######################################



