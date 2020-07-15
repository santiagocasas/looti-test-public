#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:01:43 2020

@author: raphaelbaena
"""
import numpy as np
class loctr:
    
    def __init__(self,train_data,train_samples) : 
        
        globchu=self.calc_globalchunks(train_samples)
        indlist = globchu[1]
        local_ind = np.array(indlist)
        extr_loc_ind  = local_ind[[0,-1]]
        local_train_space = train_samples[local_ind]
        extr_loc_space = train_samples[extr_loc_ind]

        local_matrixdata = train_data[local_ind];
        local_noiseless_matrixdata = train_data[local_ind];
        extr_loc_matrixdata = train_data [extr_loc_ind];
        
    
        self.local_ind = local_ind 
        self.lextr_loc_ind = extr_loc_ind 
        self.local_train_space = local_train_space 

        self.extr_loc_space = extr_loc_space
        self.local_matrixdata = local_matrixdata
        self.local_noiseless_matrixdata  = local_noiseless_matrixdata
        self.extr_loc_matrixdata = extr_loc_matrixdata
        
    def local_vectors(self,samples):

        data_space = samples
        data_loc_space = data_space[(data_space>=  self.local_train_space [0]) & (data_space <=  self.local_train_space [-1])]
        data_loctrain_space = np.sort(np.unique(np.concatenate((data_loc_space,self.extr_loc_space,self.local_train_space))))
    
        self.data_loc_space = data_loc_space # 
        self.data_loctrain_space = data_loctrain_space #


    def calc_globalchunks(self,train_samples):
         indices_list=list(range(len(train_samples)))
         chunk_indy=3
         lst = indices_list
         if chunk_indy>=1 and chunk_indy<=len(lst)+1: #for cc in range(1,len(lst)+1):
             size=chunk_indy   #cc
             chunks=[]
             for i in range(size):
                 if i==0:
                     mini = (i*len(lst))//size
                     maxi = ((i+1)*len(lst))//size
                     chu=lst[mini:maxi]
                     #print("mini", mini, "maxi", maxi)
                 elif i>0:
                     mini = (i*len(lst))//size - 1
                     maxi = ((i+1)*len(lst))//size
                     chu=lst[mini:maxi]
                 if len(chu)>1:
                     chunks.append(chu)
         return chunks#globalchunks