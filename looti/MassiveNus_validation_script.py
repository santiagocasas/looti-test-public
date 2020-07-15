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
outputfolder = "../data/"

emulation_data = dhl.DataHandle( datafile_ext, outputfolder,datafile_LCDM, 
                                  param_names_dict={'parameter_1':'mnv','parameter_2':'om','parameter_3':'As'},
                               multindex_cols_ext=[0,1,2,3,4,5,6,7],multindex_cols_ref=[0,1])

emulation_data.read_csv_pandas(verbosity=2)

redshift=[k for k in range(0,60)]
emulation_data.calculate_ratio_by_redshifts(redshift)

linkgrid = emulation_data.lin_k_grid
mask = [k for k in np.where(linkgrid <8e-4)[0] if k in np.where(linkgrid >1e-4)[0]]
GLOBAL_applymask = True
thinning = 1
max_ntrain = 5
PCA_dict_cross,PCA_dict_cros_all = av.cross_validation(emulation_data=emulation_data, wanted_ntest=10, n_vali= 10,
                                 operator="PCA", max_train_size=6,interp_type="GP")
PCA_cross_validation_grouped =av.op_crossval_df_dict_mingroup_function(emulation_data=emulation_data, op_crossval_df_dict=PCA_dict_cross)
datatest_df_dict=av.RMSE_dictionary(emulation_data,PCA_crossval_df_dict_mingroup=PCA_cross_validation_grouped,wanted_ntest=10,max_ntrain=max_ntrain,turnoff_PCA=False)
print(datatest_df_dict['theo'].groupby(['n_train']).mean())
datatest_df_dict["theo"].to_csv('result_cross_validation.py')