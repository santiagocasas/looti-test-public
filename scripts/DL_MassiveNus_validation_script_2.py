#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:05:22 2020

@author: raphaelbaena
"""

import numpy as np



from looti import datahandle as dhl
from looti import automatic_validation as av
#sns.set()

datafile_ext = "Massive_Nus"
datafile_LCDM = "Massive_Nus_LCDM"
outputfolder = "./data/"

emulation_data = dhl.DataHandle( datafile_ext, outputfolder,datafile_LCDM, 
                                  num_parameters=3)
 

emulation_data.read_csv_pandas(verbosity=2)


emulation_data.calculate_ratio_by_redshifts(emulation_data.z_vals, normalize=True)

linkgrid = emulation_data.lin_k_grid
mask = [k for k in np.where(linkgrid <10)[0] if k in np.where(linkgrid >0.1)[0]]
GLOBAL_applymask = True


thinning = 1
min_ntrain = 70
max_ntrain = 71
wanted_ntest = 30
PCA_dict_cross,PCA_dict_cros_all = av.cross_validation(emulation_data=emulation_data, wanted_ntest=wanted_ntest, n_vali=1,
                                 operator="DL", max_train_size = max_ntrain ,min_train_size=min_ntrain,interp_type="GP",number_of_splits=2)

PCA_dict_cross.to_csv("./DL_MassiveNus_2.csv")
