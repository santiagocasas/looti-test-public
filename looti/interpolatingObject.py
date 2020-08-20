#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:47:54 2020

@author: raphaelbaena
"""
import numpy as np
from scipy import interpolate
from looti import dictlearn as dcl
from looti import datahandle as dhl
from looti import PlottingModule as pm

from looti import tools as too
from looti import PlottingModule as pm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:47:54 2020

@author: raphaelbaena
"""

class Interpolating_function :
    def __init__(self):
        self.redshift_available = []
        self.list_interpolation_function = []

    def predict(self,z,k,parameters):

        parameters = [z] + parameters
        parameters = np.atleast_2d(parameters)
        ind_z_min,ind_z_max= self.extrema(z)
        z_min, z_max = self.redshift_available[ind_z_min],self.redshift_available[ind_z_max]

        f_min = self.list_interpolation_function[ind_z_min].predict(k,parameters)
        f_max = self.list_interpolation_function[ind_z_max].predict(k,parameters)

        interpolation_function_at_z = interpolate.interp1d([z_min, z_max ],[f_min,f_max])
        return float(interpolation_function_at_z (z))

    def extrema(self,z):
        ind_argmin = np.abs(np.array(self.redshift_available) - z).argmin()
        if self.redshift_available[ind_argmin]>z:
            return ind_argmin-1,ind_argmin
        else :
            return ind_argmin,ind_argmin+1


class Interpolating_function_redshift :
    def __init__(self,emulation_data,interpolation_function,redshift,normalize = False):
        self.k_grid = np.power(10,emulation_data.masked_k_grid)
        self.redshift =  redshift
        self.spectra_ref = emulation_data.df_ref.loc[emulation_data.level_of_noise,
                                                     self.redshift].values.flatten()[emulation_data.mask_true]
        self.interpolation_function_factor = dcl.Interpolate_over_factor(emulation_data,
                                                                         pos_norm = emulation_data.pos_norm)
        self.interpolation_function = interpolation_function
        self.normalize = normalize
        self.parameters_predicted=[] ##talk with Santiago about it !
        self.saved_interpolation ={}
    def predict_each_k(self,parameters):
        ratio = np.array(list(self.interpolation_function.predict(parameters).values())).flatten()
        spectra = ratio
        if self.normalize == True :
            F_norm = self.interpolation_function_factor.predict(parameters)
        else:
            F_norm = 1
            print("f",F_norm)
        spectra_normalised = spectra* F_norm

        return spectra_normalised
    def predict(self,k,parameters,linear_grid=False):
        parameters = np.atleast_2d(parameters)
        tupled_param = tuple(parameters.flatten())

        if tupled_param  not in self.parameters_predicted:
            self.interpolation_function_each_k = interpolate.interp1d(self.k_grid,self.predict_each_k(parameters))
            self.parameters_predicted.append(tupled_param)

            self.saved_interpolation[tupled_param ]=self.interpolation_function_each_k
        return float(self.saved_interpolation[tupled_param ](k))
