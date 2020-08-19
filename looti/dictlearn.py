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
from sklearn.gaussian_process.kernels import RBF,  WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.decomposition import KernelPCA
from sklearn.utils import shuffle

import time
import looti.tools as too
import looti.interpolators as itp
import looti.dataplotters as dtp
import looti.OTAlgorithm as otAlg

class LearningOperator:
    """LearningOperator:
    Class that define a learning object to construct an interpolation of the 
    """
    def __init__(self, method, **kwargs):
        self.method = method
        self.set_defaults()
        self.set_hyperpars(**kwargs)

    def set_defaults(self):
        self._def_interp_type = 'int1d'
        self._def_interp_dim = 2
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
    """ LearnData:
    Class that performs interpolations on data, using a LearningOperator object 
    Operators available: LIN (linear), PCA (Principal Component Analysis), DL (Dictionary Learning)
    """
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
        """Construct the interpolation between the paremeters and the spectra of the training set.
        Args:
            train_data: spectra 
            train_samples: parameters, may includes redshift
            train_noise: Noise level for GP inteprolator
        """
        
        
        
        self.train_noise = train_noise
        if self.interp_type == "GP":
            self.CreateGP()
            self.interpolator_func =  self.gp_regressor.fit
        else :
            self.interpolator_func = itp.Interpolators(self.interp_type, self.interp_dim, interp_opts=self.interp_opts)
        self.trainspace_mat = train_data
        self.trainspace =  train_samples
        if self.interp_dim <2:
             self.trainspace =  self.trainspace[:,-1].flatten()


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
        """Contruct the interpolation function according to a simple spline"""
        
        too.condprint("Shape of data matrix: "+str(self.trainspace_mat.shape), level=2, verbosity=self.verbosity)
        trainingdataT=np.transpose(self.trainspace_mat)  ##the interpolator, interpolates sampling space, feature by feature.
        interpolFuncsLin_matrix=self.interpolator_func(self.trainspace, trainingdataT)
        self.interpol_matrix = interpolFuncsLin_matrix
        return self.interpol_matrix  ## matrix of interpolating functions at each feature

    def PCAtraining(self):
        """Computing the PCA representation and contruct the interpolation over the PCA components"""
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
        """Computing the sparse DL representation and contruct the interpolation over the dictionary components"""
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
        if self.interp_type == "GP":
            self.gp_regressor.fit(self.trainspace,matDL)
        else:
            interpolFuncsDL_matrix=self.interpolator_func(self.trainspace, coeffsDL)
            self.interpol_matrix = interpolFuncsDL_matrix
            return self.interpol_matrix  ## matrix of interpolating functions at each feature


    def CreateGP(self):
        """Create the GP interpolator object"""
        self.interp_dim = 2
        if np.all(self.train_noise) == False:
            Y_noise = 1e-10
        else:
            Y_noise = self.train_noise
        self.gp_noise = Y_noise

        n_rsts = self.gp_n_rsts
        kernel =  C(self.gp_const, (1e-3,1e3)) + C(self.gp_const, (1e-3,1e3)) * RBF(self.gp_length, (1e-2,1e2))+WhiteKernel()
        self.gp_regressor = GaussianProcessRegressor(kernel=kernel ,
                   n_restarts_optimizer=n_rsts, alpha=Y_noise**2)


    def GPtraining(self):
        """Contruct the interpolation function according to a Gaussian Process"""
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
            #self.trainspace =  self.trainspace[:,-1].flatten()
        self.OT=otAlg.OT_Algorithm(nsteps=self.operator.ot_nsteps,gamma= self.operator.ot_gamma,num_iter=self.operator.ot_num_iter)  # OT object
        self.OT.OT_Algorithm(self.trainspace,self.operator.ot_xgrids,mode='train',data=self.trainspace_mat,)


    def predict(self, predict_space):
        """Predict the spectra for a given set of paremeters.
        Args:
            predict_space: the paremeters of the spectra to predict.
        """


        #if isinstance(predict_space, (np.ndarray)) == False:
            #predict_space = np.array([valispace]).flatten()

        self.predict_space = np.copy(predict_space)
        if self.interp_dim<2:
            self.predict_space = self.predict_space[:,-1].flatten()
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
            self.predict_mat_dict = self.matrixdata_to_dict(self.predict_mat, predict_space)
        return self.predict_mat_dict

    def interpolated_atoms(self, parspace):
        """Predict the atoms of the representation.
        Args:
            predict_space: the paremeters of the representation to predict.
        """
        arra=[]
        for intp_atoms in self.interpol_matrix:
            if  self.interp_dim ==1:
                arra.append([intp_atoms(val) for val in parspace])
            if  self.interp_dim ==2:
                arra.append([intp_atoms(val[0],val[1]) for val in parspace])
        arraT=np.transpose(np.array(arra))
        return arraT



    def reconstruct_data(self, parspace):
        """Predict the spectra for a given set of paremeters.
        Args:
            predict_space: paremeters of the spectra to predict.
        Returns:
            reco: spectra predicted
        """
        if self.method=="PCA":
            if self.interp_type == "GP":
                interp_atoms= self.gp_regressor.predict(parspace)

            else :
                interp_atoms = self.interpolated_atoms(parspace)

            reco = np.dot(interp_atoms,self.dictionary)+self.pca_mean
        elif self.method=="DL":
            if self.interp_type == "GP":
                interp_atoms= self.gp_regressor.predict(parspace)
            else :
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
        """Construct a dictionary parameters -> spectra
        Args:
            data_mat: spectra
            data_space: parameters 
        Returns:
            matdata_dict: dictionary parmeters -> spectra"""
        matdata_dict = {}
        for vv,dd in zip(data_space, data_mat):
            matdata_dict[tuple(np.array([vv]).flatten())] = dd
        return matdata_dict




    def calc_statistics(self, testvali_data, testvali_space,emulation_data = []):
        """Compute standars stastistics : min,max, mean, var of RMSE, min, max, mean err
        Args:
           testvali_data: references'spectra
           testvali_space: references'parameters 
            """
        
        testvali_data_dict = self.matrixdata_to_dict(testvali_data, testvali_space)
        ratiodata_dict={}
        val_len = len(testvali_space)
        if (np.array_equal(testvali_space, self.predict_space) == False):
            print("WARNING: Calculating statistics on predicted data which does not match test/validation space")
        minerr_dict={}
        maxerr_dict={}
        maxerr_spectra_dict={}
        
        rmse_dict={}
        rmse_spectra_dict = {}
        if emulation_data != []:
            spectra = reconstruct_spectra(self.predict_mat_dict,emulation_data)
            
        for pv in self.predict_space:
            pv=tuple(pv)
            ratiodata_dict[pv] = (self.predict_mat_dict[pv]-testvali_data_dict[pv])/testvali_data_dict[pv]
            minerr_dict[pv], maxerr_dict[pv] = too.minmax_abs_err(self.predict_mat_dict[pv],testvali_data_dict[pv], percentage=False)
            rmse_dict[pv] = too.root_mean_sq_err(self.predict_mat_dict[pv],testvali_data_dict[pv])
           
            if emulation_data != []:
                index = emulation_data.get_index_param(list(pv ),multiple_redshift=emulation_data.multiple_z)
                truth =  emulation_data.df_ext.loc[index].values.flatten()[emulation_data.mask_true]
                rmse_spectra_dict[pv] = too.root_mean_sq_err(spectra[pv],truth )
        
                __, maxerr_spectra_dict[pv] =    too.minmax_abs_err(spectra[pv],truth, percentage=False)    
            
        self.global_max_maxerr = np.max(np.array(list(maxerr_dict.values())))
        self.global_mean_maxerr = np.mean(np.array(list(maxerr_dict.values())))
        self.global_min_minerr = np.min(np.array(list(minerr_dict.values())))
        self.global_mean_minerr = np.mean(np.array(list(minerr_dict.values())))
        self.global_max_rmse = np.max(np.array(list(rmse_dict.values())))
        self.global_mean_rmse = np.mean(np.array(list(rmse_dict.values())))
        self.global_var_rmse = np.var(np.array(list(rmse_dict.values())))
        
        if emulation_data != []:
            self.global_spectra_max_maxerr = np.max(np.array(list(maxerr_spectra_dict.values())))
            self.global_spectra_mean_maxerr = np.mean(np.array(list(maxerr_spectra_dict.values())))
            self.global_spectra_max_rmse = np.max(np.array(list(rmse_spectra_dict.values())))
            self.global_spectra_mean_rmse = np.mean(np.array(list(rmse_spectra_dict.values())))
            self.global_spectra_var_rmse = np.var(np.array(list(rmse_spectra_dict.values())))
            self.rmse_spectra_dict =  rmse_spectra_dict
            self.maxerr_spectra_dict = maxerr_spectra_dict
        


        self.ratiodata_dict = ratiodata_dict
        self.minerr_dict = minerr_dict
        self.maxerr_dict = maxerr_dict
        self.rmse_dict = rmse_dict
        



    def print_statistics(self):
        """Print the statistics previsouly computed by the function calc_statistics"""
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


def Predict_ratio(emulation_data,
              Operator = "GP",
              ncomp = 1,
              train_noise = 1e-10,
              gp_n_rsts = 10,
              gp_const = 1,
              gp_length = 10 ,
              interp_type='GP',
              n_train=None,
              n_test=None,
              n_splits=1,
              split = 0,
              test_indices=[1],
                  train_redshift_indices = [0],
                  test_redshift_indices = [0],
              return_interpolator = False,
              thinning = 1, min_k= None,max_k =None,mask=None,interp_dim=2):
    """Construct the interpolation of a set of training vectors and return the prediction over the set of test parameters.
     The user can previously split the data into train/vali/test or provide the number of training and test vectors wanted.
     In the last case, the function perfoms the split automatically.
     The user can also pass a mask, or choose the extrema of the grid
     Args:
         Operator: operator used for interpolation or the determine a representation of the data
         train_noise = 1e-10: the level of noise used for the Gaussian Process 
         gp_n_rsts = 10: the number of restars authorised for the Gaussian Process. Higher values slow the process
         gp_const = 1: 
         gp_length = 10 : the length used for the matrix of the kernel 
         interp_type='GP': operator used to perfom the interpolation
         n_train=None: number of training vectors If not provided, it is assumed that the data are already splited
         n_test=None: number of test vectors. If not provided, it is assumed that the data are already splited
         n_splits=1: number of splits.
         split = 0: number of the split, the user wants to consider
         test_indices: list of list of test indices
         train_redshift_indices: list of indices of redshift used for the training vectors
         test_redshift_indices: list of indices of redshift used for the test vectors
         thinning: thinning of the mask. 
         min_k: min k to used for the mask. Lower values would not be considered.
         max_k: max k to used for the mask. Higher values would not be considered.
         mask:
    Returns: 
        spectra: spectra/ratios of test vectors predicted"""
   

    if min_k is not None and max_k is not None :
        mask = [k for k in np.where(emulation_data.lin_k_grid <max_k)[0] if k in np.where(emulation_data.lin_k_grid  >min_k     )[0]]
        GLOBAL_apply_mask = True
    elif  mask is not None:
         GLOBAL_apply_mask = True
    else :
        GLOBAL_apply_mask = False


    noise_case= emulation_data.level_of_noise
    if (n_train is not None and n_test is not None):
        emulation_data.calculate_data_split(n_train = n_train,n_vali=1, n_test=n_test,
                                    n_splits=n_splits, verbosity=0,manual_split=True,test_indices=test_indices,
                                    train_redshift_indices =  train_redshift_indices,
                                    test_redshift_indices = test_redshift_indices)
        
    elif (emulation_data.train_redshift !=emulation_data.z_requested[train_redshift_indices] or emulation_data.test_redshift !=emulation_data.z_requested[test_redshift_indices] ):
        emulation_data.calculate_data_split(n_train = emulation_data.train_size,n_vali=1, n_test=emulation_data.test_size,
                                    n_splits=n_splits, verbosity=0,manual_split=True,test_indices = emulation_data.test_indices,
                                    train_indices = emulation_data.train_indices,
                                    train_redshift_indices =  train_redshift_indices,
                                    test_redshift_indices = test_redshift_indices)
    emulation_data.data_split(split,thinning=thinning, mask=mask,
                                          apply_mask = GLOBAL_apply_mask, verbosity=0)
    PCAop = LearningOperator(Operator,ncomp=ncomp,gp_n_rsts =gp_n_rsts,gp_const=gp_const,gp_length = gp_length,interp_type=   interp_type,interp_dim=interp_dim)
    intobj = LearnData(PCAop)
###Perfoming the PCA reduction and interpolation
    intobj.interpolate(train_data=emulation_data.matrix_datalearn_dict[noise_case]['train'],
                       train_samples=emulation_data.train_samples,train_noise = train_noise )
    ratios_predicted = intobj.predict(emulation_data.test_samples)

    if return_interpolator == True:
        return  ratios_predicted,emulation_data,intobj 


    return  ratios_predicted,emulation_data


def reconstruct_spectra(ratios_predicted,
                        emulation_data,
                        normalization = False, pos_norm = 2):
    """Reconstruct the spectra from ratios
     Args:
         ratios_predicted: a dictionary parameters -> ratios
         emulation_data: datahandle frame used
         normalization: if True would assume that a normalization has been carried out and that a denormalization
         is required to reconstruct the spectra. To do so, it would interpolate the p_k(k =pos_norm) of train vectors
         pos_norm: the position where the normalisation factor is expected to have been computed(see datahandle)
    Returns:
        spectra: spectra reconstructed"""
     
     
    spectra={}
    if normalization  == True:
        Interpolatation_of_f= Interpolate_over_factor(emulation_data)
    for parameters in list(ratios_predicted.keys()):
       # ind = emulation_data.get_index_param(parameters,multiple_redshift=emulation_data.multiple_z)
        if emulation_data.multiple_z == True:
            LCDM_ref = emulation_data.df_ref.loc[emulation_data.level_of_noise,parameters[0]].values.flatten()
        else:
           LCDM_ref = emulation_data.df_ref.loc[(emulation_data.level_of_noise),:].values.flatten()
        if normalization  == False:
            F = 1
        else:

            F = Interpolatation_of_f.predict(np.atleast_2d(parameters),emulation_data.pos_norm)#/LCDM_ref[emulation_data.pos_norm]
            F=F[0]
            ind = emulation_data.get_index_param(parameters,emulation_data.multiple_z)
            y = emulation_data.df_ext.loc[ind].values.flatten()[emulation_data.pos_norm]

            F_true = np.atleast_1d(y/LCDM_ref[emulation_data.pos_norm])
            print( "RMSE of normalisation factor",
                  too.root_mean_sq_err(np.atleast_1d(F), np.atleast_1d(F_true)))

        spectrum = ratios_predicted[parameters] * LCDM_ref [emulation_data.mask_true]*F
        spectra[parameters] =  spectrum
    return spectra

def RMSE_parameters(emulation_data,
              Operator = "GP",
              ncomp = 1,
              train_noise = 1e-10,
              gp_n_rsts = 10,
              gp_const = 1,
              gp_length = 10 ,
              interp_type='GP',
              n_splits=1,
              n_train = 2,
              split = 0,
              test_indices=[1],
              train_redshift_indices = [0],
              test_redshift_indices = [0],
              thinning = 1, min_k= None,max_k =None,mask=None,interp_dim=2):
    
    
    n_test = len(emulation_data.matrix_ratios_dict["theo"])/len(emulation_data.z_requested)
    Params = []
    RMSE_array = []
    for i in range(n_test):
        ratios_predicted = Predict_ratio(Operator, ncomp,train_noise, gp_n_rsts, gp_const, gp_length, interp_type,
                  n_train,
                  n_test,
                  n_splits,
                  split,
                  train_redshift_indices,
                  test_redshift_indices,
                  thinning, min_k,max_k,mask,interp_dim,test_indices = [i])
    Params.append([list(rr) for rr in ratios_predicted.keys()])
    Ratios_truth = emulation_data.matrix_datalearn_dict["theo"]["test"]
    Ratios = list (ratios_predicted.values())
    RMSE_array.append(too.root_mean_sq_err(Ratios_truth[0],Ratios[0]))   
    return Params,RMSE_array


def Interpolate_over_parameter_for_any_redshift(emulation_data,test_indices,n_train = 10,n_param_test = 5, n_test_redshift=10,Y_noise = 1e-4):

    D_redshift = []
    for i,z_vals in enumerate(emulation_data.z_requested):
        emulation_data.calculate_data_split(n_train = n_train,n_vali = 1, n_test = n_param_test,
                            n_splits=1, verbosity=0,manual_split=True,test_indices=test_indices,
                                    train_redshift_indices = [i],
                                    test_redshift_indices = [i])
        emulation_data.data_split(0,thinning=1, mask=None,
                                      apply_mask = False, verbosity=0)

        PCAop = LearningOperator("PCA",ncomp=n_train,gp_n_rsts =10,gp_const=1,gp_length = 0.5,interp_type="GP")
        intobj = LearnData(PCAop)
        intobj.interpolate(train_data=emulation_data.matrix_datalearn_dict["theo"]['train'],
                   train_samples=emulation_data.train_samples,train_noise= Y_noise )
        ratios_predicted=np.array([ii for ii in (intobj.predict(emulation_data.test_samples).values())])
        D_redshift.append(ratios_predicted.reshape(n_param_test,len(emulation_data.k_grid)))


    return np.array(D_redshift)

def RMSE_Interpolate_over_redshift(emulation_data,D_redshifts,test_indices,min_n_train,max_n_train,n_test,Y_noise=1e-4):
    redshift_indices = list(range(len(emulation_data.z_requested)))
    redshift_indices = shuffle(redshift_indices )
    test_redshift_indices = redshift_indices[:n_test]
    n_param = int(len(emulation_data.matrix_ratios_dict["theo"])/len(emulation_data.z_requested))
    RMSE_list=[]
    for n_train in range(min_n_train,max_n_train):
        train_redshift_indices = redshift_indices[n_test:n_test+n_train]
        n_param_test = len(test_indices[0])

        rmse = 0
        for j,indices_param in enumerate(test_indices[0]):
            PCAop = LearningOperator("PCA",ncomp=len(train_redshift_indices),gp_n_rsts =10,gp_const=1,gp_length = 0.5,interp_type="GP")
            intobj = LearnData(PCAop)
            intobj.interpolate(train_samples = np.array(train_redshift_indices).reshape((len(train_redshift_indices),emulation_data.num_parameters)), train_data = D_redshifts[train_redshift_indices][:,j],train_noise= Y_noise )
            ratios_predicted =  intobj.predict(np.array([test_redshift_indices]).reshape((len(test_redshift_indices),emulation_data.num_parameters)))
            prediction=  np.array([rr for rr in ratios_predicted.values()]).reshape(len(test_redshift_indices),len(emulation_data.k_grid))


            mask = [indices_param +n_param*redshift for redshift in test_redshift_indices ]
            ref = emulation_data.matrix_ratios_dict["theo"][mask]
            rmse+=too.root_mean_sq_err(prediction,ref)
        rmse/=( n_param_test*len(test_redshift_indices))
        RMSE_list.append(rmse)
    return RMSE_list

def Interpolate_over_factor(emulation_data,
                            normalization = False, pos_norm = 2):
    """Interpolate the normalisation factor
     Args:
         emulation_data: datahandle object used 
         pos_norm:tion has been carried out and that a denormalization
         is required to reconstruct the spectra. To do so, it would interpolate the p_k(k =pos_norm) of train vectors
         pos_norm: the position where the normalisation factor is expected to have been computed(see datahandle)
    Returns:
         gp_regressor: an inteporlation parameters -> p_k(k=pos_norm)
     """
    X = emulation_data.train_samples
    Y = []
    for x in X:
        ind = emulation_data.get_index_param(x,emulation_data.multiple_z)
        y = emulation_data.df_ext.loc[ind].values.flatten()[pos_norm]
        if emulation_data.multiple_z == True:
            LCDM_ref = emulation_data.df_ref.loc[emulation_data.level_of_noise,x[0]].values.flatten()[pos_norm]
        else:
           LCDM_ref = emulation_data.df_ref.loc[(emulation_data.level_of_noise),:].values.flatten()[pos_norm]
        
        Y.append(y/LCDM_ref)
    Y = np.array(Y)
    kernel =  C()+ C() * RBF()
    gp_regressor = GaussianProcessRegressor(kernel=kernel ,
                   n_restarts_optimizer=10, alpha=10e-3**2)
    gp_regressor.fit(X,Y)
    return gp_regressor

        



def Interpolate_over_parameter_and_redshift(emulation_data,test_indices,min_n_train,max_n_train,n_test,Y_noise = 1e-4, min_k = None ,max_k = None, mask =None):
    redshift_indices = list(range(len(emulation_data.z_requested)))
    redshift_indices = shuffle(redshift_indices)
    test_redshift_indices = redshift_indices[:n_test]


    if min_k is not None and max_k is not None :
        mask = [k for k in np.where(emulation_data.lin_k_grid <max_k)[0] if k in np.where(emulation_data.lin_k_grid  >1e-4)[0]]
        GLOBAL_apply_mask = True
    elif  mask is not None:
         GLOBAL_apply_mask = True
    else :
        GLOBAL_apply_mask = False


    RMSE_list=[]
    for n_train in range(min_n_train,max_n_train):
        train_redshift_indices = redshift_indices[n_test:n_test+n_train]
        emulation_data.calculate_data_split(n_train = 50,n_vali = 1, n_test = n_test,
                            n_splits=1, verbosity=0,manual_split=True,test_indices=test_indices,
                                    train_redshift_indices = train_redshift_indices,
                                    test_redshift_indices = test_redshift_indices)

        emulation_data.data_split(0,thinning=1, mask=mask,
                                      apply_mask =GLOBAL_apply_mask, verbosity=0)

        PCAop = LearningOperator("PCA",ncomp=n_train,gp_n_rsts =10,gp_const=1,gp_length = 0.5,interp_type="GP")
        intobj = LearnData(PCAop)
        intobj.interpolate(train_data=emulation_data.matrix_datalearn_dict["theo"]['train'],
                   train_samples=emulation_data.train_samples,train_noise= Y_noise )
        ratios_predicted=np.array([ii for ii in (intobj.predict(emulation_data.test_samples).values())])
        rmse=too.root_mean_sq_err(ratios_predicted,emulation_data.matrix_datalearn_dict["theo"]['test'])

        RMSE_list.append(rmse)
    return RMSE_list
