import os
import sys
import numpy as np
import errno
import copy

from future.utils import viewitems
import pandas as pd
import pickle
#from random import randint, randrange
import random as rn

from sklearn.decomposition import PCA, FactorAnalysis, DictionaryLearning
from sklearn.decomposition import SparseCoder

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.decomposition import KernelPCA


import time
import looti.tools as too
import looti.interpolators as itp
import looti.dataplotters as dtp
import looti.OTAlgorithm as otAlg

class LearningOperator:
    """ Representation Learning Operator Interpolator:
    Class that performs interpolations on data, using a linear operator
    Operators available: LIN (linear), PCA (Principal Component Analysis), DL (Dictionary Learning)
    """
    def __init__(self, method, **kwargs):
        self.method = method
        self.set_defaults()
        self.set_hyperpars(**kwargs)

    def set_defaults(self):
        self._def_interp_type = 'int1d'
        self._def_interp_dim = 1
        self._def_interp_opts = {'kind':'linear'}
        self._def_ncomp = 1
        self._def_dl_alpha = 0.001
        self._def_dl_tralgo = 'lasso_lars'
        self._def_dl_fitalgo = 'lars'
        self._def_dl_maxiter = 2000
        self._def_gp_const = 10.0
        self._def_gp_length = 5.0
        self._def_gp_n_rsts = 1
        self._def_ot_gamma=1e-7
        self._def_ot_num_iter=500
        self._def_ot_nsteps=32
        self._verbosity = 1

    def set_hyperpars(self, **kwargs):
        self.interp_type = kwargs.get('interp_type', self._def_interp_type)
        self.interp_dim = kwargs.get('interp_dim', self._def_interp_dim)
        self.interp_opts = kwargs.get('interp_opts', self._def_interp_opts)
        self.ncomp = kwargs.get('ncomp', self._def_ncomp)
        self.gp_n_rsts = kwargs.get('gp_n_rsts', self._def_gp_n_rsts)
        self.gp_const =  kwargs.get('gp_alpha', self._def_gp_const)
        self.gp_length =  kwargs.get('gp_length', self._def_gp_const)
        if(self.method == "DL"):
            self.dl_alpha =  kwargs.get('dl_alpha', self._def_dl_alpha)
            self.dl_tralgo = kwargs.get('transform_algorithm', self._def_dl_tralgo)
            self.dl_fitalgo = kwargs.get('fit_algorithm', self._def_dl_fitalgo)
            self.dl_maxiter = kwargs.get('max_iter', self._def_dl_maxiter)
        if(self.method == 'GP'):
            self.gp_const =  kwargs.get('gp_alpha', self._def_gp_const)
            self.gp_length =  kwargs.get('gp_length', self._def_gp_const)
        if(self.method=='OT'):
            self.ot_nsteps=kwargs.get('nsteps', self._def_ot_nsteps)
            self.ot_gamma=kwargs.get('gamma', self._def_ot_gamma)
            self.ot_num_iter=kwargs.get('num_iter', self._def_ot_num_iter)
            self.ot_xgrids=kwargs.get('xgrids')
        self.verbosity = kwargs.get('verbosity', self._verbosity)


class LearnData:
    def __init__(self, operator_obj):
        self.operator    = operator_obj
        self.method      = self.operator.method
        self.interp_type = self.operator.interp_type
        self.interp_dim  = self.operator.interp_dim
        self.interp_opts  = self.operator.interp_opts
        self.ncomp       = self.operator.ncomp
        self.gp_n_rsts = self.operator.gp_n_rsts 
        self.gp_const =  self.operator.gp_const
        self.gp_length =  self.operator.gp_length
        
        if(self.method=='DL'):
            self.dl_alpha    = self.operator.dl_alpha
            self.dl_tralgo   = self.operator.dl_tralgo
            self.dl_fitalgo  = self.operator.dl_fitalgo
            self.dl_maxiter  = self.operator.dl_maxiter
        if(self.method == 'GP'):
            self.gp_const =  self.operator.gp_const
            self.gp_length =  self.operator.gp_length
            self.gp_n_rsts = self.operator.gp_n_rsts 
            
        self.verbosity = self.operator.verbosity
        return None

    def interpolate(self, train_data=[], train_samples=[], train_noise=[False]):
        self.train_noise = train_noise
        if self.interp_type == "GP":
            self.CreateGP()
        else :
            self.interpolator_func = itp.Interpolators(self.interp_type, self.interp_dim, interp_opts=self.interp_opts)
        self.trainspace_mat = train_data
        self.trainspace =  train_samples
        
        if self.method == "LIN":
            self.LINtraining()
        elif self.method == "PCA":
            self.PCAtraining()
        elif self.method == "DL":
            self.DLtraining()
        elif self.method == "GP":
            self.GPtraining()
        elif self.method == "OT":
            self.OTtraining()
        else:
            raise ValueError("Error: Method inexistent or not yet implemented.")



    def LINtraining(self):
        too.condprint("Shape of data matrix: "+str(self.trainspace_mat.shape), level=2, verbosity=self.verbosity)
        trainingdataT=np.transpose(self.trainspace_mat)  ##the interpolator, interpolates sampling space, feature by feature.
        interpolFuncsLin_matrix=self.interpolator_func(self.trainspace, trainingdataT)
        self.interpol_matrix = interpolFuncsLin_matrix
        return self.interpol_matrix  ## matrix of interpolating functions at each feature

    def PCAtraining(self):
        pca=PCA(n_components=self.ncomp)   
        self.pca=pca## take n principal components
        matPCA=pca.fit(self.trainspace_mat).transform(self.trainspace_mat)
        ncomp=pca.n_components_
        Vpca=pca.components_
        meanvec=pca.mean_
        self.pca_mean = meanvec
        too.condprint("Shape of data matrix: "+str(self.trainspace_mat.shape), level=2, verbosity=self.verbosity)
        too.condprint("Shape of PCA matrix: "+str(matPCA.shape), level=1, verbosity=self.verbosity)
        too.condprint("Number of PCA components: "+str(ncomp), level=1, verbosity=self.verbosity)
        too.condprint("Shape of PCA coefficients: "+str(Vpca.shape), level=2, verbosity=self.verbosity)
        self.dictionary = Vpca
        self.representation = matPCA
        coeffsPCA=np.transpose(matPCA)
        if self.interp_type == "GP":
            self.gp_regressor.fit(self.trainspace,matPCA)
        else :
            interpolFuncsPCA_matrix=self.interpolator_func(self.trainspace, coeffsPCA)
            self.interpol_matrix = interpolFuncsPCA_matrix
            return self.interpol_matrix ## matrix of interpolating functions at each feature


    def DLtraining(self):
        DL=DictionaryLearning(n_components=self.ncomp,alpha=self.dl_alpha, max_iter=self.dl_maxiter, fit_algorithm=self.dl_fitalgo,
                          transform_algorithm=self.dl_tralgo, transform_alpha=self.dl_alpha)
        fitDL=DL.fit(self.trainspace_mat)
        matDL=fitDL.transform(self.trainspace_mat)
        VDL=fitDL.components_
        too.condprint("Shape of data matrix: "+str(self.trainspace_mat.shape), level=2, verbosity=self.verbosity)
        too.condprint("Shape of DL matrix: "+str(matDL.shape), level=1, verbosity=self.verbosity)
        too.condprint("Number of DL components: "+str(len(VDL)), level=1, verbosity=self.verbosity)
        too.condprint("Shape of DL coefficients: "+str(VDL.shape), level=2, verbosity=self.verbosity)
        self.dictionary = VDL
        self.representation = matDL
        coeffsDL=np.transpose(matDL)
        interpolFuncsDL_matrix=self.interpolator_func(self.trainspace, coeffsDL)
        self.interpol_matrix = interpolFuncsDL_matrix
        return self.interpol_matrix  ## matrix of interpolating functions at each feature


    def CreateGP(self):
        if np.all(self.train_noise) == False:
            Y_noise = 1e-10
        else:
            Y_noise = self.train_noise
        self.gp_noise = Y_noise

        n_rsts = self.gp_n_rsts
        kernel = C(self.gp_const, (1e-3,1e3)) * RBF(self.gp_length, (1e-2,1e2))
        self.gp_regressor = GaussianProcessRegressor(kernel=kernel ,
                   n_restarts_optimizer=n_rsts, alpha=Y_noise**2)


    def GPtraining(self):
        X_train = self.trainspace
        Y_train = self.trainspace_mat
        self.CreateGP()
        self.gp_regressor.fit(X_train, Y_train)


    def OTtraining(self):
        """Input
        Emulation Data to create loctr and perform training
        Noise Level to create loctr
        """
        #_mat : data / betas
        self.OT=otAlg.OT_Algorithm(nsteps=self.operator.ot_nsteps,gamma= self.operator.ot_gamma,num_iter=self.operator.ot_num_iter)  # OT object
        self.OT.OT_Algorithm(self.trainspace,self.operator.ot_xgrids,mode='train',data=self.trainspace_mat,)


    def predict(self, predict_space):



        #if isinstance(predict_space, (np.ndarray)) == False:
            #predict_space = np.array([valispace]).flatten()

        self.predict_space = np.copy(predict_space)
        self.predict_mat = self.reconstruct_data(self.predict_space)
        self.predict_mat_dict = dict()
        if self.method=="OT":
            pred_ws=self.predi_weights_arr
            recospace=self.OT.loctr.data_loctrain_space
            if len(recospace)!=len(pred_ws):
                print("Warning: recospace and pred weights are not of same size")

            for ii,vv in enumerate(recospace):

                self.predict_mat_dict[recospace[ii]] = self.predict_mat[pred_ws[ii]].values


        else:
            self.predict_mat_dict = self.matrixdata_to_dict(self.predict_mat, self.predict_space)
        return self.predict_mat_dict

    def interpolated_atoms(self, parspace):
        arra=[]
        for intp_atoms in self.interpol_matrix:
            if  self.interp_dim ==1:
                arra.append([intp_atoms(val) for val in parspace])
            if  self.interp_dim ==2:
                arra.append([intp_atoms(val[0],val[1]) for val in parspace])
        arraT=np.transpose(np.array(arra))
        return arraT



    def reconstruct_data(self, parspace):
        if self.method=="PCA":
            if self.interp_type == "GP":    
                interp_atoms= self.gp_regressor.predict(parspace)
    
            else :
                interp_atoms = self.interpolated_atoms(parspace)
            
            reco = np.dot(interp_atoms,self.dictionary)+self.pca_mean
        elif self.method=="DL":
            interp_atoms = self.interpolated_atoms(parspace)
            reco=np.dot(interp_atoms,self.dictionary)
        elif self.method=="LIN":
            interp_atoms = self.interpolated_atoms(parspace)
            reco = interp_atoms
        elif self.method=="GP":
            xpred_2d = parspace
            reco = self.gp_regressor.predict(xpred_2d)
        elif self.method=="OT":
            self.OT.loctr.local_vectors(parspace)
            self.predi_weights_arr=self.OT.interpolateWeights(self.interpolator_func)
            reco=self.OT.OT_Algorithm(parspace,self.operator.ot_xgrids)
        return reco

    @staticmethod
    def matrixdata_to_dict(data_mat, data_space):
        matdata_dict = {}
        for vv,dd in zip(data_space, data_mat):
            if type(vv) == np.ndarray:
                matdata_dict[tuple(vv)] = dd
            else :
                matdata_dict[vv] = dd
        return matdata_dict




    def calc_statistics(self, testvali_data, testvali_space):
        testvali_data_dict = self.matrixdata_to_dict(testvali_data, testvali_space)
        ratiodata_dict={}
        val_len = len(testvali_space)
        if (np.array_equal(testvali_space, self.predict_space) == False):
            print("WARNING: Calculating statistics on predicted data which does not match test/validation space")
        minerr_dict={}
        maxerr_dict={}
        rmse_dict={}
        for pv in self.predict_space:
            pv=tuple(pv)
            ratiodata_dict[pv] = (self.predict_mat_dict[pv]-testvali_data_dict[pv])/testvali_data_dict[pv]
            minerr_dict[pv], maxerr_dict[pv] = too.minmax_abs_err(self.predict_mat_dict[pv],testvali_data_dict[pv], percentage=False)
            rmse_dict[pv] = too.root_mean_sq_err(self.predict_mat_dict[pv],testvali_data_dict[pv])
        self.global_max_maxerr = np.max(np.array(list(maxerr_dict.values())))
        self.global_mean_maxerr = np.mean(np.array(list(maxerr_dict.values())))
        self.global_min_minerr = np.min(np.array(list(minerr_dict.values())))
        self.global_mean_minerr = np.mean(np.array(list(minerr_dict.values())))
        self.global_max_rmse = np.max(np.array(list(rmse_dict.values())))
        self.global_mean_rmse = np.mean(np.array(list(rmse_dict.values())))
        self.global_var_rmse = np.var(np.array(list(rmse_dict.values())))

        self.ratiodata_dict = ratiodata_dict
        self.minerr_dict = minerr_dict
        self.maxerr_dict = maxerr_dict
        self.rmse_dict = rmse_dict


    def print_statistics(self):
        print(" global mean min error: ", self.global_mean_minerr,
              "\n global mean max error: ", self.global_mean_maxerr,
              "\n global max rmse: ", self.global_max_rmse,
              "\n global mean rmse: ", self.global_mean_rmse,
              "\n global var rmse: ", self.global_var_rmse)
        return None

    def fill_stats_dict(self, cols, failed=False):
        stat_dict = {}
        for cc in cols:
            try:
                if failed==False:   ##values did not fail outside of method
                    stat_dict[cc] = self.__dict__[cc]
                elif failed==True:
                    stat_dict[cc] = np.nan
            except KeyError:
                pass
        return stat_dict

    @staticmethod
    def dataframe_group(df, groupby='', sortby='', filename='', savedir='./savedir/'):
        if type(df) != pd.core.frame.DataFrame:
            print("object passed is not a pandas DataFrame")
            return None
        #print("min value of sortby: ")
        print(df[sortby].min() )
        df_grouped = df.loc[df.groupby(groupby)[sortby].idxmin()]
        df_grouped[['n_train']] = df_grouped[['n_train']].astype(int)
        df_grouped.set_index(['n_train'], inplace=True)
        if filename!='':
            too.mkdirp(savedir)
            df_grouped.to_csv(savedir+filename+'.csv')
            print("DataFrame saved to: "+savedir+filename+'.csv')
        return df_grouped
