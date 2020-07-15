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

with open(r"../config_read.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    Param_list = yaml.load(file, Loader=yaml.FullLoader)

datafolder = Param_list["datafolder"]
feature_filename = Param_list["feature_filename"]
filename_format = Param_list["filename_format"]
LCDM_mode = Param_list["LCDM_mode"]
LCDM_folder = Param_list["LCDM_folder"]
grid_pos = Param_list["grid_pos"]
feature_pos = Param_list["feature_pos"]
param_str_array = Param_list["param_str_array"]
sep_param = Param_list["sep_param"]
sep_param_value = Param_list["sep_param_value"]


# # TODO:
###  class readModel:
###  def __init__(self, model_name, datafolder, feature_filename):
###      self.model_name = model_name
###
###
##### in the notebook
########   fofR_readfiles = readModel("fofR", "../fofR/")
########   codecs_readfiles = readModel("EXP", "../Codec/")
########### in DataHandle
#############  DataHandle(fofR_readfiles.data_frame)


def read_file_list(folder, f_pattern="", ext='txt'):
    fls =os.listdir(folder)
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


def read_ext(main_folder = datafolder, feature_filename=feature_filename, filename_format=filename_format):
    folder_list=read_folder(main_folder)
    param_str_array, sep_param, sep_param_value=ask_param(folder_list)
    Dic_param_values={}
    z_str_array=[]
    ####GET PARAM LIST
    for p in param_str_array:
        Dic_param_values[p]=[]
    for folder in folder_list :
            Param_values=read_parameter(folder,param_str_array,sep_param,sep_param_value)
            try:
                for p in param_str_array:
                    Dic_param_values[p].append(Param_values[p])
            except:
                print(ValueError)

    for i,pp in enumerate(param_str_array):
        Dic_param_values[pp]=np.unique(Dic_param_values[pp])

    ####GET redshift_list
    files_list = read_file_list(main_folder+"/"+folder_list[0], f_pattern=feature_filename, ext=filename_format)
    for file in files_list:
         try:
             z_str_array.append(get_redshift(file, feature_filename, filename_format))
         except :
             print(file," can't be read")
    z_str_array=np.unique(z_str_array)

    return (z_str_array, Dic_param_values, sep_param, sep_param_value)

def create_DataFrame(main_folder=datafolder, param_str_array=param_str_array, sep_param= sep_param, sep_param_value=sep_param_value,
                     feature_filename=feature_filename, filename_format=filename_format,
                     grid_pos=grid_pos, feature_pos=feature_pos, LCDM_mode=LCDM_mode,LCDM_folder = LCDM_folder ):

    List_of_param_str_array =  param_str_array

    #multiIndex =get_multiIndex(z_str_array, List_of_param_str_array, Dic_param_values)
    folder_list=read_folder(main_folder)
    if LCDM_mode == False :
        if LCDM_folder in folder_list:

            del folder_list[folder_list.index(LCDM_folder)]
    else :
        folder_list=[LCDM_folder]

    k_grid_size,k_grid,interpolation=get_max_grill_size(main_folder,folder_list,grid_pos)


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
                                          grid_pos, feature_pos)
                if interpolation == True :
                    Ratio_intpd=interpolate.interp1d(k, pk)
                    pk = Ratio_intpd(k_grid)

                index=get_index(redshift, Param_values,LCDM_mode)

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

def transform_index(data_frame,index_pos, function):
    val = function(list(data_frame.index.levels[index_pos]))
    data_frame.index = data_frame.index.set_levels(val,level=index_pos)
    return  data_frame

def get_index(redshift, Param_values,LCDM_mode =False, noise_level=["theo"]):
    index=[noise_level[0], redshift]
    if LCDM_mode == False:

        for pp in Param_values:
            index.append(pp)
            index.append(Param_values[pp])
    return tuple(index)

def get_multiIndex(tuples,List_of_param_str_array,LCDM_mode=False):
    names=["noise_model","redshift"]
    if LCDM_mode == False :
        for i,pp in enumerate(List_of_param_str_array):
            names.append("parameter_"+str(i+1))
            names.append("parameter_"+str(i+1)+"_value")
    multiIndex= pd.MultiIndex.from_tuples(tuples, names=names)
    return multiIndex#multiIndex


def get_redshift(file_name,redshift_digit,filextension):

    redshift =  file_name[-len(filextension)-redshift_digit:-len(filextension)-1]


    #redshift= '{:.6f}'.format(redshift).replace('.','p')
    return redshift
## # TODO: change grill to grid
def get_max_grill_size(main_folder,folder_names,grid_index,starting_row=0,end_row=None):
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
            files = read_folder(main_folder+"/"+folder )

        except :
             print(ValueError)
        try:
            for file in files:
                data=np.loadtxt(main_folder+"/"+folder+"/"+file)

                k_grid = data[starting_row:end_row,grid_index]
                k_grid_size = len(k_grid)

                k_grid_all = k_grid

                if k_grid_size >max_size:
                    max_size = k_grid_size
                    k_grid_max = k_grid

                if  k_grid_size < min_size   :
                    min_size = k_grid_size
                    k_grid_min = k_grid


        except:
            print(ValueError)
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


def read_values(file_name,redshift_digit,filextension,
                  grid_index=0, pk_index=1,
                  starting_row=0, end_row=None):
    file = np.loadtxt(file_name)
    k_grid = file[starting_row:end_row, grid_index]
    pk = file[starting_row:end_row, pk_index]

    redshift= get_redshift(file_name,redshift_digit, filextension)
    return k_grid,pk,redshift
   # except :
   #     print("Warning this file "+  file_name+ " can't be read")
   # return None

def read_parameter(foldername, param_str_array,sep_param,sep_param_value=None):
    splitted_filename = foldername.split(sep_param)
    Param_values={}
    for x in splitted_filename:
        for p in param_str_array :
            if p in x :
                param_values_p=float(x.strip(p))
                if sep_param_value != None :
                     param_values_p=param_values_p.strip( sep_param_value)
                Param_values[p]=param_values_p

    return Param_values


# Converts the string into a integer. If you need
# to convert the user input into decimal format,
# the float() function is used instead of int()
