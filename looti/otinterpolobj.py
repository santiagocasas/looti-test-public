#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:13:59 2020

@author: raphaelbaena
"""
import numpy as np
import copy
import time
from collections import OrderedDict
import pandas as pd
import sys

import looti.tools as too
import looti.interpolators as itp


class OPinterporl:
    """blabla"""

    def __init__(self, **kwargs):
        self.set_defaults()
        self.set_hyperpars(**kwargs)
    def set_defaults(self):
        self._def_epsstr='beta'
        self._def_weights=None
    def set_hyperpars(self,**kwargs):
        self.epsstr=kwargs.get('epsstr=', self._def_epsstr)
        self.weights=kwargs.get('weights=', self._def_weights)


    def calc_globalchunks(self):
        train_samples=self.emulation_data.train_samples
        indices_list=list(range(len(train_samples)))
        chunk_indy=3
        lst = indices_list
        print("list indices: ", lst)
        if chunk_indy>=1 and chunk_indy<=len(lst)+1: #for cc in range(1,len(lst)+1):
            size=chunk_indy   #cc
            print("len list: ", len(lst))
            print("size: ", size)
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
                    #print("mini", mini, "maxi", maxi)
                if len(chu)>1:
                    chunks.append(chu)
                    #if len(chu)<=5:
            #globalchunks.append(chunks)
            print("chunks: ", chunks)
            print("********")
        print("#######********")
        return chunks#globalchunks

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


    def trainLoadOT(self,zchoice,split_run_ind ,emulation_data,mode,check_file=False,nsteps=30):
        self.noise='noise_100'
        self.thin=str(5)
        self.zsrt = '{:.6f}'.format(zchoice).replace('.','p')
        self.emulation_data=emulation_data
        self.train_size=str(emulation_data.train_size)
        self.globchu=self.calc_globalchunks()
        self.loc1 = self.globchu[1]
        self.splstr='split_'+"{:02}".format(split_run_ind)
        self.loctr=loctr(self.loc1,self.emulation_data, self.noise)
        self.OTfilename_specs = 'testa-'+self.noise+'-z_'+str(self.zsrt)+'-thin_'+str(self.thin)+'-'+self.splstr+'-trainsize_'+str(self.train_size)+'-localind_'+('-'.join([str(l) for l in self.loctr.local_ind]))+'-barysteps_'+str(nsteps)
        self.check_file=check_file
        if mode=='train':
            prefix='pandas_OT_interpolation-'
            traineps=self.loctr.extr_loc_space
            weights=None
        elif mode=='valid':
            prefix='pandas_OT_reconstruction-'
            traineps=self.loctr.vali_loctrain_space
            weights=self.predi_weights_arr


        outfileprefix=prefix+self.OTfilename_specs
        trainingmodes=self.loctr.extr_loc_matrixdata
        self.xgrid=self.emulation_data.masked_k_grid
        traineps=self.loctr.extr_loc_space

        outputfolder='./codecs_fits/'

        too.mkdirp(outputfolder)
        sep="__"
        joinreds=sep.join(str('{:.4f}'.format(tt)) for tt in traineps[0:1])
        joinreds=joinreds.replace('.','p')
        #outputtable=outfileprefix+'--'+epsstr+"_"+joinreds+'.txt'
        outputpickle=outputfolder+outfileprefix+'--'+self.epsstr+"_"+joinreds+'.pkl'
        if self.check_file==False:
            compute_always=True
        elif self.check_file==True:
            compute_always= not too.fileexists(outputpickle)
        if compute_always==True:

            #otweights, otbary =
            otweights, otbary=self.calcOT(models_in=trainingmodes,weights=weights,nsteps=nsteps)
            modelSeries=[]
            outDict = OrderedDict()

            #for orig in trainingmodes:
            #    modelSeries.append(pd.Series(orig, index=xgrid))
            for ubary in otbary:
                modelSeries.append(pd.Series(ubary, index=self.xgrid))
            # first and last weight correspond to initial training models
            if len(otweights)!=len(modelSeries):
                print("Warning: modelSeries and extended weights do not have the same length")

            for ww,mm in zip(otweights, modelSeries):
                outDict[ww]=mm

            outDF = pd.DataFrame(outDict)
            outDF.to_pickle(outputpickle)
            print(("Exported OT interpolations and weights to: "+outputpickle))
        else:
            print("OT interpolation pandas dataframe pickle file exists, importing file...")
        if too.fileexists(outputpickle):  #import always file, also after new creation
            if mode=='train':
                self.trainedOT_df= pd.read_pickle(outputpickle)
                print("File "+outputpickle+"  -> imported successfully.")
            elif mode=='valid':
                self.reconstOT_df= pd.read_pickle(outputpickle)


    def calcOT(self,models_in=[],weights=None,nsteps=30):
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

      otweights,otbary= self.wasserOT(normalmodes=normalmodes, normaliz=normaliz,
                                 minc=jc,weights=weights,nsteps=nsteps)
      trainingmodes -= jc
      print("OT algorithm completed...")
      return otweights, otbary



    def wasserOT(self,normalmodes=[], normaliz=[], minc=0.,gamma_reg=1e-7, num_iter=500,weights=None,nsteps=30):
        xgrid=copy.deepcopy(self.xgrid)
        # Wasserstein params and ground metric

        start = time.time()


        import logOT_bary as ot

        print('Theano compilation done in {}s.'.format(time.time()-start))

        gamma = gamma_reg
        n_iter_sink = num_iter
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


    def interpolateWeights(self, interp_method='int1d',
                        interp_dim=1, interp_opts=dict()):
        recospace=self.loctr.vali_loc_space
        wex = np.array(list(self.weights_dic.keys()))
        wey = np.array(list(self.weights_dic.values()))
        interpolator_func = itp.Interpolators(interp_method, interp_dim, interp_opts)
        intpWeights = interpolator_func(wex, [wey])  #list of y values since function accepts list of y lists
        interp_space = np.sort(np.unique(np.concatenate((wex,recospace ))))
        self.recoweights = intpWeights[0](interp_space)
        self.recoweights_dict = OrderedDict(zip(interp_space,self.recoweights))
        self.predi_weights_arr = np.array(list((self.recoweights_dict.values())))





    def dic_from_daf(self):
        data_daf=self.reconstOT_df
        r_space=self.loctr.vali_loctrain_space
        pred_ws=self.predi_weights_arr
        if len(r_space)!=len(pred_ws):
            print("Warning: recospace and pred weights are not of same size")
        self.data_dic = dict()
        for ii,vv in enumerate(r_space):
            self.data_dic[r_space[ii]] = data_daf[pred_ws[ii]].values










class loctr:

    def __init__(self,indlist,data_object,noise_id)       :
        local_ind = np.array(indlist)
        extr_loc_ind  = local_ind[[0,-1]]
        print("local train: ",local_ind)
        print("extrema: ",extr_loc_ind)
        local_train_space = data_object.train_samples[local_ind]
        extr_loc_space = data_object.train_samples[extr_loc_ind]
        valispace = data_object.vali_samples
        vali_loc_space = valispace[(valispace >= local_train_space[0]) & (valispace <= local_train_space[-1])]
        vali_loctrain_space = np.sort(np.unique(np.concatenate((vali_loc_space,extr_loc_space,local_train_space))))

        local_matrixdata = data_object.matrix_datalearn_dict[noise_id]['train'][local_ind];
        local_noiseless_matrixdata = data_object.matrix_datalearn_dict['theo']['train'][local_ind];
        extr_loc_matrixdata = data_object.matrix_datalearn_dict[noise_id]['train'][extr_loc_ind];


        self.local_ind = local_ind
        self.lextr_loc_ind = extr_loc_ind
        self.local_train_space = local_train_space
        self.extr_loc_space = extr_loc_space
        self.vali_loc_space = vali_loc_space
        self.vali_loctrain_space = vali_loctrain_space
        self.local_matrixdata = local_matrixdata
        self.local_noiseless_matrixdata  = local_noiseless_matrixdata
        self.extr_loc_matrixdata = extr_loc_matrixdata
