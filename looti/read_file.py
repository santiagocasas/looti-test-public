#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:39:07 2020

@author: raphaelbaena
"""
import numpy as np
from scipy import interpolate
import os
import pandas as pd
import fnmatch
import yaml

def transform_index(data_frame,index_pos, function):
    val = function(list(data_frame.index.levels[index_pos]))
    data_frame.index = data_frame.index.set_levels(val,level=index_pos)
    return  data_frame


class Frame_Constructor:
    def __init__(self,path_config_file = "../config_read.yaml") :
       self.__read__config(path_config_file)
       self.noise_level = ["theo"]
       
    def __read__config(self,path_config_file):
        """Read the yaml file
        """
        
        with open(path_config_file,'r') as file:
            Param_list = yaml.load(file, Loader=yaml.FullLoader)
            
        self.main_folder = Param_list["datafolder"] ###the folder where the data are stored
        self.feature_filename = Param_list["feature_filename"] ###name of data files e.G power_spec
        self.filename_format = Param_list["filename_format"] ###format of the data e.g txt
        self.LCDM_mode = Param_list["LCDM_mode"] 
        self.LCDM_folder = Param_list["LCDM_folder"]
        self.grid_index = Param_list["grid_pos"] ### the column of the grid 
        self.pk_index = Param_list["feature_pos"]### the column of the features
        self.param_str_array = Param_list["param_str_array"] ###List of param e.g As, mnv, Om
        
        self.sep_param = Param_list["sep_param"] ###seperator between the parameter 
        self.sep_param_value = Param_list["sep_param_value"]  ###seperator between the parameter name/value 
        self.redshift_in_header = Param_list["redshift_in_header"] ###should we read the redshift in the header ?
        self.redshift_digit = Param_list["redshift_digit"] ###number of digit in the header
        self.starting_row = Param_list["starting_row"] 
        self.end_row = Param_list["end_row"]
        
    def create_DataFrame(self):
        """Create a frame EITHER for LAMBDACDM/ref or for the extended model
        """
        List_of_param_str_array = self.param_str_array     
    #multiIndex =get_multiIndex(z_str_array, List_of_param_str_array, Dic_param_values)
        folder_list=read_folder(self.main_folder)
        ###Here we check if we want to create a frame for LAMBDACDM only
        if self.LCDM_mode == False :
            if self.LCDM_folder in folder_list:
    
                del folder_list[folder_list.index(self.LCDM_folder)]
        else :
            folder_list=[self.LCDM_folder]
    
    
        ###Get the size of the grid, and a standard grid that will be use=d for the data.
        ## => each spectra will be expressed on the same grid 
        k_grid_size,k_grid,interpolation=self.get_max_grill_size(folder_list)
    
    
        #data_frame=pd.DataFrame( index=multiIndex, columns=np.arange(0,k_grid_size) )
        dic_index_values={}
        self.folder_list = folder_list
        ### we go through each folder/paramters
        for folder in folder_list :
            try:
                    files_list = [self.main_folder+"/"+folder+"/"+ff for ff in read_file_list(self.main_folder+"/"+folder, f_pattern=self.feature_filename, ext=self.filename_format)]
    
            except Exception as e:
                print(folder,type(e))
                files_list = []

            for file in files_list:
                try:
                    ###read the parameters values
                    Param_values=self.read_parameter(folder)
                    ###read the redshift, grid and spectrum
                    k,pk,redshift=self.read_values(file)
                    
                    ###if each example has a different grid we do an interpolation to use the same grid.
                    if interpolation == True :
                       Ratio_intpd=interpolate.interp1d(k, pk)
                       pk = Ratio_intpd(k_grid)
                       
                    ###/!\ must be update /!\
                    ## read the redshift within the file 
                    if self.redshift_in_header  == False:
                        f1_ = open(file)
                        lines_1=f1_.readlines()
                        a = float(lines_1[1].split('=', 1)[-1])
                        index=self.get_index(1/a-1, Param_values)
                    index=self.get_index(redshift, Param_values)
    
                    dic_index_values[index] = np.array([k_grid,pk])
                except :
                    print(file,"can't be  read")
                    
        ###tuples in order to construct a multindex by tupple
        tuples = [tt for tt in dic_index_values.keys() ]
        
        ### LCDM_mode we do'nt need to store the parameter
        if self.LCDM_mode==True:
            List_of_param_str_array = []
        ### Create the multindex
        multiIndex =  self.get_multiIndex(tuples,List_of_param_str_array)
        data_frame = pd.DataFrame( index=multiIndex, columns=np.arange(0,k_grid_size) )
        for data in( dic_index_values.keys()):
            M = dic_index_values[data]
            data_frame.loc[data,:] = M[1]
        data_frame.loc["k_grid",:] = k_grid
        return data_frame
    
    def get_multiIndex(self,tuples,List_of_param_str_array):
        """Create the multindex :noise,redshift, parameters names & values
        
        """
        names=["noise_model","redshift"]
        if self.LCDM_mode == False :
            for i,pp in enumerate(List_of_param_str_array):
                names.append("parameter_"+str(i+1))
                names.append("parameter_"+str(i+1)+"_value")
        multiIndex= pd.MultiIndex.from_tuples(tuples, names=names)
        return multiIndex#multiIndex

    def get_index(self,redshift, Param_values):
        index=[self.noise_level[0], redshift]
        if self.LCDM_mode == False:
            for pp in Param_values:
                index.append(pp)
                index.append(Param_values[pp])
        return tuple(index)


    def read_parameter(self,foldername):
        """Read the parameter within the folder
        Args:
            foldername: folder's name that should contain the simulation for some given parameters
                        parameters MUST be indicate within the name
        """
        splitted_filename = foldername.split(self.sep_param)
        Param_values={}
        for x in splitted_filename:
            for p in self.param_str_array :
                if p in x :
                    param_values_p=float(x.strip(p))
                    if self.sep_param_value != None :
                         param_values_p=param_values_p.strip( self.sep_param_value)
                    Param_values[p]=param_values_p
    
        return Param_values

    def read_values(self,file_name):
        """Read the grid, spectrum and redshift
        Args:
            file_name: name of the simulation can contain the redshift within it
        """
        file = np.loadtxt(file_name)
        k_grid = file[self.starting_row:self.end_row, self.grid_index]
        pk = file[self.starting_row:self.end_row, self.pk_index]
    
        redshift= self.get_redshift(file_name)
        return k_grid,pk,redshift


    def get_redshift(self,file_name):
        """Read redshift from the file's name
        Args:
            file_name: name of the simulation can contain the redshift within it
        """
        redshift =  file_name[-len(self.filename_format)-self.redshift_digit:-len(self.filename_format)-1]
    
        return redshift
    
    
    def get_max_grill_size(self,folder_names):
        """Get the size of the grid that should be used. Return a common grid.
        Args:
            folder_names: list of folders which contains simulations.
        Returns:
            min_size: the size of the grid that should be used.
            k_grid_all: the grid that should be used for every simulation
            interpolation: say if interpolation will be needed. False if each simulation has the same grid.
        
        """

        max_size =0
        min_size = float('inf')
        k_grid_max = None
        k_grid_min = None
        k_grid_all = None
        k_grid_size = 0
        files=[""]
        for folder in folder_names:
    
            try :
                files = read_folder(self.main_folder+"/"+folder )
    
            except :
                 pass
            try:
                for file in files:
                    data=np.loadtxt(self.main_folder+"/"+folder+"/"+file)
    
                    k_grid = data[self.starting_row:self.end_row,self.grid_index]
                    k_grid_size = len(k_grid)
    
                    k_grid_all = k_grid
                    ###Look for the smallest/biggest grid
                    if k_grid_size >max_size:
                        max_size = k_grid_size
                        k_grid_max = k_grid
    
                    if  k_grid_size < min_size   :
                        min_size = k_grid_size
                        k_grid_min = k_grid
    
    
            except:
                pass
        ### find k_min and k_max that work for each simulation
        if max_size>min_size :
            interpolation = True
    
            min_k = max(k_grid_max[0],k_grid_min[0])
            max_k = min(k_grid_max[-1],k_grid_min[-1])
    
            ###Create the grid
            k_grid_all = np.linspace(min_k ,max_k,min_size)
        else :
            interpolation = False
        if max_size<min_size:
            min_size = max_size
        return min_size,k_grid_all,interpolation



def read_file_list(folder, f_pattern="", ext='txt'):
    fls =fnmatch.filter(os.listdir(folder), f_pattern+"*")
    return fls

def read_folder(folder):
    return os.listdir(folder)

def split(filename,split):
    splitted_file = filename.split(split)
    return splitted_file

