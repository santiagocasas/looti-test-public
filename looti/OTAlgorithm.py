#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:06:21 2020

@author: raphaelbaena
"""

import copy
import time
from collections import OrderedDict
import pandas as pd
import numpy as np

import looti.OTdatasplit as OTdst
import looti.tools as too



class OT_Algorithm:

    def __init__(self,**kwargs):
        self.nsteps=kwargs.get('nsteps')
        self.gamma=kwargs.get('gamma')
        self.num_iter=kwargs.get('num_iter')


    def OT_Algorithm(self,samples,xgrid,mode=None,data=None):

        if mode=='train':
            weights=None
            self.loctr=OTdst.loctr(data,samples) # compute the chunk and the local vectors for training
        else:
            weights=self.predi_weights_arr
           # self.loctr.local_vectors(samples) # local vector any test/validation
#
        trainingmodes=self.loctr.extr_loc_matrixdata


        #otweights, otbary =
        otweights, otbary=self.calcOT(xgrid,models_in=trainingmodes,weights=weights,nsteps=self.nsteps)
        modelSeries=[]
        outDict = OrderedDict()

        #for orig in trainingmodes:
        #    modelSeries.append(pd.Series(orig, index=xgrid))
        for ubary in otbary:
            modelSeries.append(pd.Series(ubary, index=xgrid))
        # first and last weight correspond to initial training models
        if len(otweights)!=len(modelSeries):
            print("Warning: modelSeries and extended weights do not have the same length")

        for ww,mm in zip(otweights, modelSeries):
            outDict[ww]=mm

        outDF = pd.DataFrame(outDict)

        if mode=='train':
                self.trainedOT_df= outDF
                self.computeWeightsFuncAlt()
        else:
                self.reconst_OT= outDF
        return outDF

    def computeWeightsFuncAlt(self):
        weigths_arra=self.trainedOT_df.columns.values
        middlemodels=self.loctr.local_matrixdata
        middlepars=self.loctr.local_train_space
        otmodels=self.trainedOT_df.T.values
        middlepars=middlepars
        norm_dic=OrderedDict()
        self.weights_dic=OrderedDict()
        index_dic=OrderedDict()
        self.recoOTdata_dic=OrderedDict()
        for jj, bb in enumerate(middlepars):
            normax=100
            for ii,recons in enumerate(otmodels):
                norma=too.root_mean_sq_err(recons,middlemodels[jj])
                if norma < normax:
                    normax=norma
                    cc=ii #get index of weight closest to the wanted model
            norm_dic[jj]=normax
            self.weights_dic[bb]=weigths_arra[cc]
            index_dic[bb]=cc
            self.recoOTdata_dic[bb]=otmodels[index_dic[bb]]


    def interpolateWeights(self,interpolator_func):
        recospace=self.loctr.data_loc_space
        wex = np.array(list(self.weights_dic.keys()))
        wey = np.array(list(self.weights_dic.values()))
        intpWeights =  interpolator_func (wex, [wey])  #list of y values since function accepts list of y lists

        interp_space = np.sort(np.unique(np.concatenate((wex,recospace ))))
        self.recoweights = intpWeights[0](interp_space)
        self.recoweights_dict = OrderedDict(zip(interp_space,self.recoweights))
        self.predi_weights_arr = np.array(list((self.recoweights_dict.values())))
        return self.predi_weights_arr




    def calcOT(self,xgrid,models_in=[],weights=None,nsteps=30):
      # run OT algorithm
      print("Running OT algorithm...")
      trainingmodes = copy.deepcopy(models_in)
      dasminimum = np.min(trainingmodes)
      #print(dasminimum)
      iota=0.1  # buffer  to add to the minimum of the model
      jc=np.abs(dasminimum) + iota
      print("minimum model value", jc)
      trainingmodes += jc
      normaliz = np.sum(trainingmodes,axis=1).reshape(-1,1)
      normalmodes = trainingmodes/normaliz

      otweights,otbary= self.wasserOT(xgrid,normalmodes=normalmodes, normaliz=normaliz,
                                 minc=jc,weights=weights,nsteps=nsteps)
      trainingmodes -= jc
      print("OT algorithm completed...")
      return otweights, otbary



    def wasserOT(self,xgrid,normalmodes=[], normaliz=[], minc=0.,gamma_reg=1e-7, num_iter=500,weights=None,nsteps=30):
        xgrid=copy.deepcopy(xgrid)
        # Wasserstein params and ground metric

        start = time.time()


        import logOT_bary as ot

        print('Theano compilation done in {}s.'.format(time.time()-start))

        gamma = self.gamma
        n_iter_sink = self.num_iter
        C = ot.simpleCost(xgrid)


        # barycenters for pre-given weights
        if weights is not None:
            print("performing OT for pre-given weights")
            nb_steps = len(weights)
            weights = np.array([weights, 1-weights])
            # barycenters for number of interpolation steps
        elif weights==None:
            nb_steps = nsteps
            interp_steps = np.linspace(1e-10,1.-1e-10,nb_steps)
            weights = np.array([interp_steps, 1-interp_steps])


        weightsT = weights.T

        print('OT interpolation started, number of steps: ', nb_steps)
        start = time.time()
        #num_cores = multiprocessing.cpu_count()
        #interp_models = Parallel(n_jobs=num_cores)(delayed(ot.log_Theano_wass_bary)(normalmodes.T, lbda, gamma, C, n_iter_sink) for lbda in weightsT )
        interp_models = [ot.log_Theano_wass_bary(normalmodes.T, lbda, gamma, C, n_iter_sink) for lbda in weightsT ]
        interp_models = np.array(interp_models)
        print('OT interpolation done in {}s ({}s per step).'.format(time.time()-start, (time.time()-start)/nb_steps))

        unnorm_w = weights * normaliz
        unnormbary = [bary * uw - minc for bary, uw in zip(interp_models, np.sum(unnorm_w, axis=0)) ]

        return weights[0],unnormbary
