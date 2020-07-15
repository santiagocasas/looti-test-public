import numpy as np
import copy
from collections import OrderedDict
import time
import os
import sys
from future.utils import viewitems
import pandas as pd
import pickle
import emulatorpaths as emup
sys.path.append('./utilities/')
import tools as too
import interpolators as itp

#import dill as pickle

cachedir = './ot_cache/'
too.mkdirp(cachedir)
from joblib import Memory
memory = Memory(cachedir, verbose=0)

from joblib import Parallel, delayed
import multiprocessing

OT_outputdir = emup.OT_trainval_dir
#_daf subscript for all pandas dataframes
#_dic subscript for all dictionaries

#@memory.cache  ##turn-off for tests
def wasserOT(normalmodes=[], normaliz=[], xgrid=[], nsteps=30, minc=0., weights=None,
              gamma_reg=1e-7, num_iter=500):
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

def calcOT(models_in=[], xgrid=[], nsteps=30, weights=None, **kwargs):
    # run OT algorithm
    print("Running OT algorithm...")
    trainingmodes = copy.deepcopy(models_in)
    dasminimum = np.min(trainingmodes)
    lentm = trainingmodes.shape[1]
    #print(dasminimum)
    iota=0.1  # buffer  to add to the minimum of the model
    jc=np.abs(dasminimum) + iota
    print("minimum model value", jc)
    trainingmodes += jc
    normaliz = np.sum(trainingmodes,axis=1).reshape(-1,1)
    normalmodes = trainingmodes/normaliz

    otweights,otbary=wasserOT(normalmodes=normalmodes, normaliz=normaliz, xgrid=xgrid, nsteps=nsteps,
                                   minc=jc, weights=weights, **kwargs)
    trainingmodes -= jc
    print("OT algorithm completed...")
    return otweights, otbary

def trainLoadOT(trainingmodes=[], xgrid=[], nsteps=30, weights=None, traineps=[],
                epsstr='', outfileprefix='', outputfolder=OT_outputdir, check_file=False,
                **kwargs):
    if not traineps.any():
        return
    too.mkdirp(outputfolder)
    sep="__"
    joinreds=sep.join(str('{:.4f}'.format(tt)) for tt in traineps[0:1])
    joinreds=joinreds.replace('.','p')
    #outputtable=outfileprefix+'--'+epsstr+"_"+joinreds+'.txt'
    outputpickle=outputfolder+outfileprefix+'--'+epsstr+"_"+joinreds+'.pkl'
    if check_file==False:
        compute_always=True
    elif check_file==True:
        compute_always= not too.fileexists(outputpickle)
    if compute_always==True:

        otweights, otbary = calcOT(models_in=trainingmodes, xgrid=xgrid, nsteps=nsteps, weights=weights, **kwargs)

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
        outDF.to_pickle(outputpickle)
        print(("Exported OT interpolations and weights to: "+outputpickle))
    else:
        print("OT interpolation pandas dataframe pickle file exists, importing file...")
    if too.fileexists(outputpickle):  #import always file, also after new creation
        outDF = pd.read_pickle(outputpickle)
        print("File "+outputpickle+"  -> imported successfully.")

        return (outDF)

#_daf subscript for all pandas dataframes
#_dic subscript for all dictionaries
#_arr subscript for some numpy arrays

def computeWeightsFunc(otmodels_daf, middlemodels=None, middlepars=[]):
    weigths_arra=otmodels_daf.columns.values
    otmodels=otmodels_daf.T.values
    middlepars=middlepars
    norm_dic=OrderedDict()
    weight_dic=OrderedDict()
    index_dic=OrderedDict()
    recoOTdata_dic=OrderedDict()
    allones=np.ones(otmodels.shape[1])
    for jj, bb in enumerate(middlepars):
        normax=1000
        for ii,recons in enumerate(otmodels):
            ratio_arr=recons/middlemodels[jj]
            diff_arr=ratio_arr-allones
            norma=np.linalg.norm(diff_arr)
            if norma < normax:
                normax=norma
                cc=ii #get index of weight closest to the wanted model
        norm_dic[jj]=normax
        weight_dic[bb]=weigths_arra[cc]
        index_dic[bb]=cc
        recoOTdata_dic[bb]=otmodels[index_dic[bb]]
    return weight_dic, recoOTdata_dic

def interpolateWeights(weights_dic, recospace=np.array([]), interp_method='int1d',
                        interp_dim=1, interp_opts=dict()):
    wex = np.array(list(weights_dic.keys()))
    wey = np.array(list(weights_dic.values()))
    interpolator_func = itp.Interpolators(interp_method, interp_dim, interp_opts)
    intpWeights = interpolator_func(wex, [wey])  #list of y values since function accepts list of y lists
    interp_space = np.sort(np.unique(np.concatenate((wex,recospace ))))
    recoweights = intpWeights[0](interp_space)
    recoweights_dict = OrderedDict(zip(interp_space,recoweights))
    return recoweights_dict

def dic_from_daf(data_daf, r_space, pred_ws):
    if len(r_space)!=len(pred_ws):
        print("Warning: recospace and pred weights are not of same size")
    data_dic = dict()
    for ii,vv in enumerate(r_space):
        data_dic[r_space[ii]] = data_daf[pred_ws[ii]].values
    return data_dic

def minMaxErrorReco(valispace,datareco_dic,datafull_dic):
    ratiodata_dict={}
    minerr=10000
    maxerr=0.
    for pv in valispace:
        ratiodata_dict[pv] = (datareco_dic[pv]-datafull_dic[pv])/datafull_dic[pv]
        minerror = np.abs(np.min(ratiodata_dict[pv]))
        maxerror = np.abs(np.max(ratiodata_dict[pv]))
        if minerror < minerr:
            minerr=minerror
        if maxerror > maxerr:
            maxerr=maxerror
    return minerr,maxerr

def interpolOTtraining(xgrid, trainspace, trainingdata_dict, valispace, validata_dict, extremaspace, extremadata_dict, paramchosen='par', returndata=False):
    models=np.array(list(extremadata_dict.values()))
    trainedOT_df=trainLoadOT(trainingmodes=models,
                        xgrid=xgrid, nsteps=30, traineps=extremaspace, epsstr=paramchosen,
                        outfileprefix='pandas_OT_interpolation')

    weights_dict, recondata_dict = computeWeightsFunc(trainedOT_df, middlemodels_dic=trainingdata_dict)
    reconswhts=interpolateWeights(weights_dict, recospace=valispace)

    reconstOT_df=trainLoadOT(trainingmodes=models,
                         xgrid=xgrid, weights=reconswhts, traineps=extremaspace, epsstr=paramchosen,
                        outfileprefix='pandas_OT_reconst')

    reconstOT_dict = dic_from_daf(reconstOT_df, valispace)

    minerrOT, maxerrOT = minMaxErrorReco(valispace,reconstOT_dict,validata_dict)
    if returndata==False:
        return minerrOT, maxerrOT
