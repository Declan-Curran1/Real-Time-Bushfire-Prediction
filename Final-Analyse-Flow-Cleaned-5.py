# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 00:21:57 2023

@author: djc40
"""


#####
#5
#####

#Use regression on our time series of how clusters are distributed to see if any of them have a 
#causal effect on our fire variable. Several cluster types from the model saved in this file have 
#a statistically significant effect on predicting our fire outcome variable. This result could be 
#different each time the model is trained due to the nature of k-means clustering. 


import numpy as np
import statsmodels.api as sm


#You are going to have the final 3 dimension array with grids x clusters x time periods
tttt = np.load('D:/Dev/Python/Bushfire/Flow/Ratios_test2_initial47_127.npy')
Ratios = tttt

#So, take the finalised array 

Ratiost1 = Ratios #Initialise it at the correct size 

Ratiost1[:,:,0] =  float("inf") #The first time period has no t-1

Ratiost1[:,:,1:len(Ratiost1)] = Ratios[:,:,0:Ratios.shape[2]-1]

#Is it Ratios.shape[1] OR Ratios.dimensions[1]


#Add a version for the (2 days prior) observation
#Ratiost2 = Ratios #Initialise it at the correct size 

#Ratiost2[:,:,0:2] = float("inf") #The first time period has no t-1

#Ratiost2[:,:,2:len(Ratiost2)-2] = Ratios[:,:,0:Ratios.shape[2]-2]



#Add a version for 3 months
Ratiost45 = Ratios #Initialise it at the correct size 

Ratiost45[:,:,0:44] = float("inf") #The first time period has no t-1

Ratiost45[:,:,45:len(Ratiost45)-45] = Ratios[:,:,0:Ratios.shape[2]-45]



#Implement Linear Regression:

    
####Tutorial using sklearn:
####https://realpython.com/linear-regression-in-python/#multiple-linear-regression-with-scikit-learn
###import numpy as np
###from sklearn.linear_model import LinearRegression


#Try and see if any of the 7 land use types from 3 months prior can predict the 8th
#Land use type (Fire)

##
#Do this for three months prior
#1st Land-use type today
LUT1 = Ratios[:,6,:].flatten() #CHECK 6 IS THE RIGHT INDEX
#1st land-use type 3 mnths prior
LUT1t45 = Ratiost45[:,6,:].flatten()0
#Remove infinity values
LUT1 = LUT1[LUT1t45!=float("inf")]
LUT1t45 = LUT1t45[LUT1t45!=float("inf")]
##

##
#Do this for the day prior
#1st Land-use type today
LUT1 = Ratios[:,8,:].flatten() #CHECK 6 IS THE RIGHT INDEX
#1st land-use type day prior
LUT1t1 = Ratiost1[:,8,:].flatten()
#Remove infinity values
LUT1 = LUT1[LUT1t1!=float("inf")]
LUT1t1 = LUT1t1[LUT1t1!=float("inf")]
##


#1st land-use type on day prior
#LUT1t1 = Ratiost1[:,6,:].flatten()



#LUT1 = LUT1[(not LUT1t45==float("inf")).all(),:] #OR .all()
#LUT1t45 = LUT1t45[(not LUT1t45==float("inf")).any,:]


#If python doesn't handle 'inf' well in OLS, then get all 'inf' values in
#Ratiost45 and remove them + their equivalent index in Ratios


#Tutorial using statsmodels (much easier to view results):
#  https://www.statsmodels.org/stable/regression.html


mod = sm.OLS(LUT1, LUT1t45) #sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())


mod = sm.OLS(LUT1, LUT1t1) #sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())




