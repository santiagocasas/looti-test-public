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
        with open(path_config_file,'r') as file:
            Param_list = yaml.load(file, Loader=yaml.FullLoader)
            
        self.main_folder = Param_list["datafolder"]
        self.feature_filename = Param_list["feature_filename"]
        self.filename_format = Param_list["filename_format"]
        self.LCDM_mode = Param_list["LCDM_mode"]
        self.LCDM_folder = Param_list["LCDM_folder"]
        self.grid_index = Param_list["grid_pos"]
        self.pk_index = Param_list["feature_pos"]
        self.param_str_array = Param_list["param_str_array"]
        
        self.sep_param = Param_list["sep_param"]
        self.sep_param_value = Param_list["sep_param_value"]
        self.redshift_in_header = Param_list["redshift_in_header"]
        self.redshift_digit = Param_list["redshift_digit"]
        self.starting_row = Param_list["starting_row"]
        self.end_row = Param_list["end_row"]
        
    def create_DataFrame(self):

        List_of_param_str_array = self.param_str_array     
    #multiIndex =get_multiIndex(z_str_array, List_of_param_str_array, Dic_param_values)
        folder_list=read_folder(self.main_folder)
        if self.LCDM_mode == False :
            if self.LCDM_folder in folder_list:
    
                del folder_list[folder_list.index(self.LCDM_folder)]
        else :
            folder_list=[self.LCDM_folder]
    
        k_grid_size,k_grid,interpolation=self.get_max_grill_size(folder_list)
    
    
        #data_frame=pd.DataFrame( index=multiIndex, columns=np.arange(0,k_grid_size) )
        dic_index_values={}
        self.folder_list = folder_list
        for folder in folder_list :
            try:
                    files_list = [self.main_folder+"/"+folder+"/"+ff for ff in read_file_list(self.main_folder+"/"+folder, f_pattern=self.feature_filename, ext=self.filename_format)]
    
            except Exception as e:
                print(folder,type(e))
                files_list = []

            for file in files_list:
                try:
    
                    Param_values=self.read_parameter(folder)
                    
                    k,pk,redshift=self.read_values(file)
                    if interpolation == True :
                       Ratio_intpd=interpolate.interp1d(k, pk)
                       pk = Ratio_intpd(k_grid)
                    if self.redshift_in_header  == False:
                        f1_ = open(file)
                        lines_1=f1_.readlines()
                        a = float(lines_1[1].split('=', 1)[-1])
                        index=self.get_index(1/a-1, Param_values)
                    index=self.get_index(redshift, Param_values)
    
                    dic_index_values[index] = np.array([k_grid,pk])
                except :
                    print(file,"can't be  read")
        tuples = [tt for tt in dic_index_values.keys() ]
        if self.LCDM_mode==True:
            List_of_param_str_array = []
        multiIndex =  self.get_multiIndex(tuples,List_of_param_str_array)
        data_frame = pd.DataFrame( index=multiIndex, columns=np.arange(0,k_grid_size) )
        for data in( dic_index_values.keys()):
            M = dic_index_values[data]
            data_frame.loc[data,:] = M[1]
        data_frame.loc["k_grid",:] = k_grid
        return data_frame
    
    def get_multiIndex(self,tuples,List_of_param_str_array):
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
        file = np.loadtxt(file_name)
        k_grid = file[self.starting_row:self.end_row, self.grid_index]
        pk = file[self.starting_row:self.end_row, self.pk_index]
    
        redshift= self.get_redshift(file_name)
        return k_grid,pk,redshift


    def get_redshift(self,file_name):
    
        redshift =  file_name[-len(self.filename_format)-self.redshift_digit:-len(self.filename_format)-1]
    
        return redshift
    
    
    def get_max_grill_size(self,folder_names):
        
        """ We assume that grid of same size are the same"""
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
    
                    if k_grid_size >max_size:
                        max_size = k_grid_size
                        k_grid_max = k_grid
    
                    if  k_grid_size < min_size   :
                        min_size = k_grid_size
                        k_grid_min = k_grid
    
    
            except:
                pass
        if max_size>min_size :
            interpolation = True
    
            min_k = max(k_grid_max[0],k_grid_min[0])
            max_k = min(k_grid_max[-1],k_grid_min[-1])
    
    
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

def ask_param(arr):
    print("Please provide the name of parameters which are indicated in this file and separates them with a comma e.g :  mnv,om,As")
    print(" Name of the first folder in main directory : "+ arr[0] )
    param_str_array = input( " => give the list of parameters of this file, separated by comma, no spaces : ")
    param_str_array=param_str_array.split(",")
    sep_param_value_bool = input("Is there a separation between the parameter name and value ? (y/n) : " )
    if sep_param_value_bool =='y':
         sep_param_value = input("Please provide the separator character bewteen the parameter name and value, e.g. (-, _, --): " )
    else :
        sep_param_value  =None
    sep_param=input("Provide the separation between each (parameter,value) pair, e.g. (-, _, --) : " )
    return param_str_array,sep_param,sep_param_value

"""
def create_DataFrame(main_folder=datafolder, param_str_array=param_str_array, sep_param= sep_param, sep_param_value=sep_param_value,
                     feature_filename=feature_filename, filename_format=filename_format,
                     grid_pos=grid_pos, feature_pos=feature_pos, LCDM_mode=LCDM_mode,LCDM_folder = LCDM_folder, end_row=None):

    List_of_param_str_array =  param_str_array

    #multiIndex =get_multiIndex(z_str_array, List_of_param_str_array, Dic_param_values)
    folder_list=read_folder(main_folder)
    if LCDM_mode == False :
        if LCDM_folder in folder_list:

            del folder_list[folder_list.index(LCDM_folder)]
    else :
        print(folder_list)
        folder_list=[LCDM_folder]

    k_grid_size,k_grid,interpolation=get_max_grill_size(main_folder,folder_list,grid_pos,end_row=end_row)


    #data_frame=pd.DataFrame( index=multiIndex, columns=np.arange(0,k_grid_size) )
    dic_index_values={}
    for folder in folder_list :
        try:
                files_list = [main_folder+"/"+folder+"/"+ff for ff in read_file_list(main_folder+"/"+folder, f_pattern=feature_filename, ext=filename_format)]

        except Exception as e:
            print(type(e))
            files_list = []
        for file in files_list:

            try:

                Param_values=read_parameter(folder, List_of_param_str_array, sep_param, sep_param_value)
                k,pk,redshift=read_values(file, 3, filename_format,
                                          grid_pos, feature_pos,end_row=end_row)
                
                #if interpolation == True :
                  #  Ratio_intpd=interpolate.interp1d(k, pk)
                  #  pk = Ratio_intpd(k_grid)
                f1_ = open(file)
                lines_1=f1_.readlines()
                a = float(lines_1[1].split('=', 1)[-1])
                index=get_index(1/a-1, Param_values,LCDM_mode)

                dic_index_values[index] = np.array([k_grid,pk])
            except :
                print(file,"can't be  read")
    tuples = [tt for tt in dic_index_values.keys() ]
    if LCDM_mode==True:
        List_of_param_str_array = None
    multiIndex =  get_multiIndex(tuples,List_of_param_str_array,LCDM_mode)
    data_frame = pd.DataFrame( index=multiIndex, columns=np.arange(0,k_grid_size) )
    for data in( dic_index_values.keys()):
        M = dic_index_values[data]
        try:
            data_frame.loc[data,:] = M[1]
        except Exception as e:
            print(type(e))
    data_frame.loc["k_grid",:] = k_grid
    return data_frame

"""





# Converts the string into a integer. If you need
# to convert the user input into decimal format,
# the float() function is used instead of int()
