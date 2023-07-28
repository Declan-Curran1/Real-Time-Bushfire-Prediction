# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 19:07:54 2023

@author: djc40
"""

#####
#4
#####


#Credit to 'Land Use Analyser' function 
#https://github.com/kowalski93/Land_Use_Analyzer/blob/main/land_use_mix/Land%20Use%20Analyzer.ipynb

#Takes our classified k-means clustering and produces a 550m x 550m grid of each of our images. Determines
#the proportion of each type of cluster within each grid




import geopandas as gpd
import pandas as pd
import os
from numpy import log,array,unique
import numpy as np
import time 


lu=gpd.read_file(r"D:\Dev\Python\Bushfire\Land Use Analyser\SLUP2019.shp")
grid=gpd.read_file(r"D:\Dev\Python\Bushfire\Land Use Analyser\Neighborhood_Boundaries.shp")
uid_lc="SLUP_LATES"
uid_grid="NHD_NUM_ST"
temp_path=r"D:\Dev\Python\Bushfire\Land Use Analyser"

####
#Replicate the above exactly, with n 
lu=gpd.read_file(r"D:\Dev\Python\Bushfire\Unsupervised\NAIP_final_rasTOvec_3.shp")
grid=gpd.read_file(r"D:\Dev\Python\Bushfire\Unsupervised\KC_GRID_test.shp")
uid_lc="MYFLD"
uid_grid="id"
temp_path=r"D:\Dev\Python\Bushfire\Land Use Analyser"


####

 

class lum:
    def __init__(self,lu,grid,uid_lc,uid_grid,temp_path):
        self.lu = lu
        self.grid = grid
        self.uid_lc = uid_lc
        self.uid_grid = uid_grid
        self.temp_path = temp_path
    
    def intermediate(self):
        intersection = gpd.overlay(self.lu,self.grid,how = 'intersection')
        dissolved = intersection.dissolve(by = [self.uid_lc,self.uid_grid])
    
        dissolved['poly_area'] = dissolved.area
        dissolved.to_file(os.path.join(self.temp_path,"dissolved.shp"))
    
        dissolved_new = gpd.read_file(os.path.join(self.temp_path,"dissolved.shp"))
        area_sum = dissolved_new[[self.uid_grid, "poly_area"]].groupby(self.uid_grid).sum()
    
        dissolved_new = dissolved_new.merge(area_sum, on = self.uid_grid)
        dissolved_new.rename(columns = {'poly_area_y':'total_area_cell'}, inplace = True) 
    
        ratios = (dissolved_new["poly_area_x"]/dissolved_new["total_area_cell"])
        
        num_classes_per_grid_feature = dissolved_new[[self.uid_grid, uid_lc]].groupby(self.uid_grid).count()
        
        return dissolved_new,ratios,num_classes_per_grid_feature,intersection
    
    def entropy(self):
        dataset = self.intermediate()[0]
        ratios_c = self.intermediate()[1]
        nuclapegrif=self.intermediate()[2]
        
        log_ratios = log(ratios_c)
        dataset['area_perc_log'] = log_ratios*ratios_c
        
        ln_num_classes_per_grid_feature = log(nuclapegrif)
        
        sum_logs = dataset[[self.uid_grid, "area_perc_log"]].groupby(self.uid_grid).sum()
        
        sum_logs_merged_ln_num_classes = sum_logs.merge(ln_num_classes_per_grid_feature,on = self.uid_grid)
        
        sum_logs_merged_ln_num_classes['ENTROPY'] = -1*(sum_logs_merged_ln_num_classes['area_perc_log']/sum_logs_merged_ln_num_classes[self.uid_lc])
        
        grid_final_entropy = (self.grid).merge(sum_logs_merged_ln_num_classes,on = self.uid_grid)
        
        return grid_final_entropy
    
    def hhi(self):
        dataset = self.intermediate()[0]
        ratios_c = self.intermediate()[1]
        hhi = ((ratios_c))*2
        
        dataset['HHI'] = hhi
        
        sum_squared_ratios = dataset[[self.uid_grid, "HHI"]].groupby(self.uid_grid).sum()
        grid_final_hhi = (self.grid).merge(sum_squared_ratios,on = self.uid_grid)
        
        return grid_final_hhi
    
    
        
#l=lum(lu,grid,uid_lc,uid_grid,temp_path)


#shape_hhi=l.hhi()

###shape_hhi.to_file('SF_land_use_hhi.shp')


#shape_entropy=l.entropy()

###shape_entropy.to_file('SF_land_use_entropy.shp')

#shape_entropy.plot(column="ENTROPY",cmap='Reds')






class stats(lum):
    
    def count(self):
        dataset = self.intermediate()[0]
        nuclapegrif=self.intermediate()[2]
        
        grid_count=(self.grid).merge(nuclapegrif,on = self.uid_grid)
        grid_count.rename(columns = {self.uid_lc:'COUNT'}, inplace = True)
        
        return grid_count
    
    def mode_count(self):
        dataset=self.intermediate()[3]
        dataset=dataset[[self.uid_lc,self.uid_grid]]
        
        mode_count=dataset.groupby([self.uid_grid]).agg(lambda x:x.value_counts().index[0])
        
        grid_mode_count=(self.grid).merge(mode_count,on = self.uid_grid)
        grid_mode_count.rename(columns = {self.uid_lc:'MODE_COUNT'}, inplace = True) 
        
        return grid_mode_count
    
    def mode_area(self):
        dataset=self.intermediate()[0][[self.uid_grid,self.uid_lc, "poly_area_x"]]
        mode_area=dataset.sort_values('poly_area_x',ascending=False).groupby(self.uid_grid).first()
        grid_mode_area=grid.merge(mode_area,on=uid_grid)
        grid_mode_area.rename(columns = {'poly_area_x':'AREA',self.uid_lc:'MODE_AREA'}, inplace = True)
        
        return grid_mode_area
    
    def report(self):
        dataset=self.lu
        dataset['AREA_sq_m']=dataset.area
        table=lu.groupby(self.uid_lc).describe()['AREA_sq_m']
        table=table.drop(columns=['25%','50%','75%'])
        
        return table
    
    def ratios(self):
        dataset=self.intermediate()[0]

        dataset['ratios']=(100*round(dataset["poly_area_x"]/dataset["total_area_cell"],3))

        cols_lc=np.unique(dataset[self.uid_lc]).tolist()
        total_lc_types=len(cols_lc)
        
        grid_reind=self.grid.reindex(columns=grid.columns.tolist()+cols_lc)
        grid_reind_sorted=grid_reind.sort_values(by=[self.uid_grid])
        
        #Error here - change sorted_new to grid_reind to get around error, rather than writing it
        #grid_reind_sorted.to_file(os.path.join(self.temp_path,"sorted.shp"))
        #sorted_new=gpd.read_file(os.path.join(self.temp_path,"sorted.shp"))
        sorted_new = grid_reind_sorted
        
        
        for column in cols_lc:
            sorted_new=sorted_new.astype({column:float})

        uid_grid_unique=np.unique(dataset[self.uid_grid])

        list_ratios=[]
        list_lcs=[]
        for grid_feature in uid_grid_unique:
            feature_ratios=np.array(dataset[dataset[self.uid_grid]==grid_feature]['ratios']).tolist()
            feature_lcs=np.array(dataset[dataset[self.uid_grid]==grid_feature][self.uid_lc]).tolist()
            
            list_ratios.append(feature_ratios)
            list_lcs.append(feature_lcs)

        for i in range(len(sorted_new)):
            sorted_new.loc[i,list_lcs[i]]=list_ratios[i]
            
        sorted_final=sorted_new.fillna(0.0)
        
        return sorted_final

#l=stats(lu,grid,uid_lc,uid_grid,temp_path)

###layer_count=l.mode_count()
###layer_count.to_file('SF_Neighborhoods_mode_count.shp')
###layer_count

#Find the land use type with the highest area:
###layer_area=l.mode_area()
###layer_area.to_file('SF_Neighborhoods_mode_count.shp')
###layer_area.plot()


    
################
#Generate grid 
#(Done with degrees lat/lon = 0.005 ~ 555 meters roughly) | Calculated w/ 1 degree = 111km


#Did not work for now. Try troubleshooting installation of the 'qgis' package in python when you have time 
#For now, just use manual grids via QGIS
################


path_with_files = 'D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test2'
input_files = os.listdir(path_with_files)

nboxes = 61710 #Currently 61710 boxes from the grid in the Sydney example
ntime = len(input_files) #228 is the number of files, with one file per sat image
nclusters = 7
#Get the final output format
Ratios = np.zeros((nboxes,nclusters+2,ntime)) #Add one to nclusters, to account for the unique grid identifier & fire classifier

#Define path to grid
grid = gpd.read_file(r"D:\Dev\Python\Bushfire\Flow\Grid-Flow-Test.shp")

start = time.time()

for j in range(163,len(input_files)): #len(input_files) 
    
    #clusterised_path = path_with_files + '/' + input_files[j] + '/clustered.tiff' 
    input_clustered = path_with_files + '/' + input_files[j] + '/clustered_poly.shp'
    lu= gpd.read_file(input_clustered)
    uid_lc="MYFLD"
    uid_grid="id"
    temp_path=r"D:/Dev/Python/Bushfire/Sentinel-2 API/Download-API-Data/Test2"

    #lu=gpd.read_file(r"D:\Dev\Python\Bushfire\Unsupervised\NAIP_final_rasTOvec_3.shp")
    #grid=gpd.read_file(r"D:\Dev\Python\Bushfire\Unsupervised\KC_GRID_test.shp")
    #uid_lc="MYFLD"
    #uid_grid="id"
    #temp_path=r"D:\Dev\Python\Bushfire\Land Use Analyser"
    
    l=stats(lu,grid,uid_lc,uid_grid,temp_path)
    
    #Find the land use type and area for all land use types.0

    layer_ratio = np.array(l.ratios()) #AN error here - maybe get the grid to overlap exactly with land use types?
    #layer_ratio[:,:,np.newaxis]
    
    #Write a piece of code here -> If there are less than 13 columns then fill then add the missing columns with a 
    #value of 0 (In the future, you can come back and make these observations equal to land areas from the previous
    #time period)
    
    #Manual dirty solution for when there are images without one of the clusters
    #(Just adds 0s for the remaining areas)
    
    if layer_ratio.shape[1] < nclusters+7: #If there are less than expected no. dimensions for 'layer_ratio'
        lay_rat_dummy = pd.DataFrame(np.zeros((nboxes,nclusters+7-layer_ratio.shape[1])))
        layer_ratio = pd.DataFrame(layer_ratio)
        layer_ratio = pd.concat([layer_ratio, lay_rat_dummy], axis=1).to_numpy() #.reset_index(drop=True)
    
    
    #ttt = layer_ratio[:,:, np.newaxis]
    
    Ratios[:,:,j] = layer_ratio[:,[0,6,7,8,9,10,11,12,13]]
    
    #Remove the existing file before writing the new one
    os.remove('D:/Dev/Python/Bushfire/Flow/Ratios_test2.npy')
    
    #Save results so far
    np.save('D:/Dev/Python/Bushfire/Flow/Ratios_test2.npy',Ratios)

    end = time.time()
    print(end-start)
    

#Save initial
#Save results so far
np.save('D:/Dev/Python/Bushfire/Flow/Ratios_test2_initial2.npy',Ratios)
tttt = np.load('D:/Dev/Python/Bushfire/Flow/Ratios_test2.npy')


#Save portion 
#np.save('D:/Dev/Python/Bushfire/Flow/Ratios_test2_initial23_47.npy',tttt)
#np.save('D:/Dev/Python/Bushfire/Flow/Ratios_test2_initial47_127.npy',Ratios)
#np.save('D:/Dev/Python/Bushfire/Flow/Ratios_test2_initial147_162.npy',tttt)
#np.save('D:/Dev/Python/Bushfire/Flow/Ratios_test2_initial162_168.npy',Ratios)

tttt = np.load('D:/Dev/Python/Bushfire/Flow/Ratios_test2_initial47_127.npy')

layer_ratio.plot()


ttt = np.array([2,0,1,8])
tt = [2,0,1,8]


q = ttt[:,np.newaxis]
