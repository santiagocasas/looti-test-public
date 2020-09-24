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

import emcee
import pyccl as ccl
from multiprocessing import set_start_method, Pool

#set_start_method('spawn')

truth_ratio_data_path = './obs_data/ratio-Codecs-website-truth-norm-beta0p1-z0.txt'#"./obs_data/Pk-Codecs-website-z0-beta0p10.txt"
truth_omegam=0.32
truth_beta=0.1
truth_z=0.
truth_param_arr = np.array([truth_omegam, truth_beta])

truth_data_path = './obs_data/Pk_ratioTimesCCL-Codecs-website-truth-norm-beta0p1-z0.txt'

filename_codecs='../interpolating_objects/CDE_ratio-Codecs_Website-z0-interpObj.sav'
print("Loading file from joblib")
with open(filename_codecs, 'rb') as f:
    Interpolation_Codecs_loaded = joblib.load(f)


filename_fits='../interpolating_objects/CDE_ratio-CDE_Fits-z0-interpObj.sav'
print("Loading file from joblib")
with open(filename_fits, 'rb') as f:
    Interpolation_Fits_loaded = joblib.load(f)


def center_to_bins(center):
    arr_mean =  0.5*(center[1:] + center[:-1])
    arr_start = center[0] - (arr_mean[0]-center[0])
    arr_end = center[-1] + (center[-1]-arr_mean[-1])
    return np.concatenate(([arr_start],arr_mean,[arr_end]))

def bins_to_center(bins):
    center = 0.5*(bins[1:] + bins[:-1])
    return center

def get_obs_datavector(data_path=truth_data_path, grid_min=0.2, grid_max=2.0):

    obs_arr_full = np.loadtxt(truth_data_path)
    obs_grid_full = obs_arr_full[:,0]

    mask = ((obs_grid_full>grid_min) & (obs_grid_full<grid_max))

    obs_arr = np.copy(obs_arr_full)
    obs_arr = obs_arr[mask]
    obs_vec = obs_arr[:,1]
    obs_grid = obs_arr[:,0]
    return obs_grid, obs_vec

def power_CDE(k_array, power_ref, extra_param=None, redshift=0.,
             interpolating_Func=None):
    if interpolating_Func is not None:
        Rofbeta = np.array([interpolating_Func.predict(k,extra_param) for k in k_array ])
    else:
        Rofbeta = 1
    pnonlin_beta = Rofbeta*power_ref
    return pnonlin_beta


def fiducial_power(cosmo_param, kspace=None, interpolating_Func=None):
    zfix=0.
    afix = 1/(1+zfix)
    Omega_m, beta = cosmo_param
    Omega_b = 0.048
    ns =0.95
    bias =2.
    h=0.70
    As109=2.2
    sigma8=0.95

      #np.logspace(np.log10(0.012), np.log10(5.0), 31)
    if kspace is None:
        kspace = np.logspace(np.log10(0.012), np.log10(5.0), 32)

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
                        redshift=zfix, interpolating_Func=interpolating_Func)

    return pknonlin_beta


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

def log_likelihood(cosmo_param, pk_obs, kspace, inv_cov, interp_model):
    """
    defines the log likelihood
    """

    pknlfid =  fiducial_power(cosmo_param, kspace, interp_model)

    x = pk_obs - pknlfid
    return -0.5* (x.T @ inv_cov @ x)


def log_probability(cosmo_param, pk_obs, kspace, inv_cov, interp_model):
    """
    """

    lp = log_prior(cosmo_param)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(cosmo_param, pk_obs, kspace, inv_cov, interp_model)

def get_cov(cosmo_param, kspace, interp_model=None):
    Volume = 50*(1000)**3
    ngal = 10**(-3)
    factor = 4*(np.pi**2)
    kbins = center_to_bins(kspace)
    pnonlinfid = fiducial_power(cosmo_param, kspace, interp_model)
    deltak=np.diff(kbins)
    cc1=Volume*kspace**2*deltak
    cc2=factor*(pnonlinfid+(1/ngal))**2
    cov=np.diag(cc2/cc1)
    return cov

def run_emcee_mp(nsample=1000, interp_model=None):#, pool=None):
    """
    run MCMC for the parameters
    """
    param_dict ={}

    grid_min=0.2
    grid_max=2.0
    obs_grid, obs_vec = get_obs_datavector(data_path=truth_data_path, grid_min=grid_min, grid_max=grid_max)



    cov_simple = get_cov(truth_param_arr, obs_grid, interp_model=interp_model)
    #print(np.linalg.cond(cov_simple))

    inv_cov = np.linalg.inv(cov_simple)

    previous_best = np.array(truth_param_arr)
    ndim = len(previous_best)
    pos = previous_best + 1e-3 * np.random.randn(16, ndim)
    nwalkers, ndim = pos.shape

    with Pool(processes=8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(obs_vec,
                                                                           obs_grid, inv_cov, interp_model) #)
              , pool=pool)
        sampler.run_mcmc(pos, nsample, progress=True)

    return sampler

print("starting sampler...")
start = time.time()
sampler = run_emcee_mp(5000, interp_model=Interpolation_Codecs_loaded)
flat_samples = sampler.get_chain( flat=True)
np.savetxt('results/flat_chain_5000_intp_codecs.txt', flat_samples)
end = time.time()
mcmc_time = end - start
print("Multiprocessing MCMC took {0:.1f} seconds".format(mcmc_time))
print("Calculation finished")

