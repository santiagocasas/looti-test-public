import os
import sys
import numpy as np
import pandas as pd
import time

from looti import dictlearn as dcl
from looti import datahandle as dhl
from looti import PlottingModule as pm

from looti import tools as too
from looti import PlottingModule as pm
from looti import interpolatingObject as ito

import pickle
import joblib

###data_folder = '../../SimulationData/CDE_fitting_formulae/'
###results_folder = './results/'
###
######Name of the file for the external input data, without the extension
###datafile_ext = 'EXP_Pk_32_betas_60_redzs'
######Name of the file for the LCDM input data
###datafile_ref = 'LCDM_Pk_60_redzs'
###
###emulation_data = dhl.DataHandle( datafile_ext, data_folder, datafile_ref, num_parameters=1) 
###emulation_data.read_csv_pandas() 
###
#### Available redshifts
###emulation_data.z_vals
###
##### Set normalize=False, since Fitting Formulae are already normalized
##### First argument contains all the redshifts at which simulations are available
###emulation_data.calculate_ratio_by_redshifts(emulation_data.z_vals,normalize=False)

n_train = 30 # Number of training vectors without taking acount the extrema 

npca = 30

filename='../interpolating_objects/CDEfittings-interpObj-new.sav'

if not too.fileexists(filename):
    Interpolation = ito.Interpolating_function()

    for i,redshift in enumerate(emulation_data.z_requested):

        ratios_predicted , emulation_data,interpolation_function = dcl.Predict_ratio(emulation_data,Operator = "PCA",
                                                              train_noise = 1e-3, ##noise for the GP's kernel
                                                              gp_n_rsts = 10,##times allowed to restart the optimiser
                                                              ncomp=npca , ##number of components
                                                              gp_const = 1, ##Constant for the RBF kernel
                                                              gp_length = 1 , ## Length for  GP 
                                                              interp_type='GP', ##kind of interpolator,e.g int1d or GP 
                                                              n_splits = 1, ##number of splits
                                                              n_train=n_train, 
                                                               n_test=0,
                                                             train_redshift_indices = [i],
                                                             test_redshift_indices = [i],##indices of the test vectors
                                                             min_k =1e-2,max_k=10e1,return_interpolator=True)
        function = ito.Interpolating_function_redshift (emulation_data,interpolation_function,redshift,normalize=True)
        Interpolation.redshift_available.append(redshift)
        Interpolation.list_interpolation_function.append(function)
    with open(filename, 'wb') as f:
        joblib.dump(Interpolation, f)
else:
    print("File with interpolation object exists: ", filename)
    
print("Loading file from joblib")
with open(filename, 'rb') as f:
    Interpolation_loaded = joblib.load(f)
    
    
import emcee
import pyccl as ccl
from multiprocessing import Pool

truth_omegam=0.32
truth_beta=0.1


def power_CDE(k_array, power_ref=None, extra_param=None, redshift=0.):
    Rofbeta = np.array([Interpolation_loaded.predict(redshift,k,[extra_param]) for k in k_array ])
    pnonlin_beta = Rofbeta*power_ref
    return pnonlin_beta


def fiducial_power(cosmo_param):
    zfix=0.5
    afix = 1/(1+zfix)
    Omega_m, beta = cosmo_param
    Omega_b = 0.048
    ns =0.95
    bias =2.
    h=0.70
    As109=2.2
    sigma8=0.8
    
    kbins=np.logspace(np.log10(0.012), np.log10(5.0), 31)
    kspace=0.5*(kbins[1:] + kbins[:-1])
    
    cosmo = ccl.Cosmology(Omega_c=Omega_m-Omega_b,
                       Omega_b=Omega_b,
                       h=h,
                       #A_s=As109*10**(-9),
                       sigma8=sigma8,
                       n_s=ns,
                       transfer_function='eisenstein_hu',
                       matter_power_spectrum='halofit')
    
    
    pknonlin_ccl=np.array([ccl.power.nonlin_matter_power(cosmo, kk, afix) for kk in kspace])
    pknonlin_beta=power_CDE(kspace,power_ref=pknonlin_ccl,extra_param=beta,
                        redshift=zfix)
    
    return pknonlin_beta, kbins, kspace


pknlfid, kbins, kspace =  fiducial_power([truth_omegam,truth_beta])

np.savetxt("./results/pk_CDE_FittingFormula-observed-truth.txt", 
           np.column_stack([kspace,pknlfid]), 
           header='z=0.5,  beta=0.1, pk_nonlin=halofit, [k]=Mpc^-1')

arr=np.loadtxt("./results/pk_CDE_FittingFormula-observed-truth.txt")

def get_bounds():
    """
    Theoretical or numerical bounds on parameters
    """
    bounds = [
         (0.1, 0.5),     # Omega_m
         (0.05, 0.15)      # beta
    ]
    return np.array(bounds)

def log_prior(cosmo_param):
    """ 
    Sets the prior functions
    """

    Omega_m, beta = cosmo_param

    # Get bounds
    bounds = get_bounds()

    if bounds[0][0] < Omega_m < bounds[0][1] and bounds[1][0] < beta < bounds[1][1]:
        return 0.0 
    return -np.inf

def log_likelihood(cosmo_param, pk_obs, inv_cov):
    """
    defines the log likelihood
    """
    pknlfid, kbins, kspace =  fiducial_power(cosmo_param)
    
    x = pk_obs - pknlfid
    return -0.5* (x.T @ inv_cov @ x)


def log_probability(cosmo_param, pk_obs, inv_cov):
    """
    """

    lp = log_prior(cosmo_param)

    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(cosmo_param, pk_obs, inv_cov)

def get_cov(cosmo_param):
    Volume = 50*(1000)**3
    ngal = 10**(-3)
    factor = 4*(np.pi**2)
    pnonlinfid, kbins, kspace = fiducial_power(cosmo_param)
    deltak=np.diff(kbins)
    cc1=Volume*kspace**2*deltak
    cc2=factor*(pnonlinfid+(1/ngal))**2
    cov=np.diag(cc2/cc1)
    return cov

def run_emcee_mp(nsample=1000 ):#, pool=None):
    """
    run MCMC for the parameters
    """
    param_dict ={}
   
    obs_vec = np.loadtxt("./results/pk_CDE_FittingFormula-observed-truth.txt")
    obs_vec=obs_vec[:,1]
    
    cov_simple = get_cov([0.32,0.1])
    print(np.linalg.cond(cov_simple))
    
    inv_cov = np.linalg.inv(cov_simple)
    previous_best = np.array([0.3, 0.09])
    
    ndim = len(previous_best)

    pos = previous_best + 1e-3 * np.random.randn(16, ndim)
    nwalkers, ndim = pos.shape

    #with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(obs_vec, inv_cov))#, pool=pool)
    sampler.run_mcmc(pos, nsample, progress=True)

    return sampler