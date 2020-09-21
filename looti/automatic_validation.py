#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:51:25 2020

@author: raphaelbaena
"""
import pandas as pd
import looti.dictlearn as dcl
import numpy as np
from sklearn.utils import shuffle


columns_tuple = ('noise_case', 'paramid','parameter','parameter_value','split','n_train',
                 'interp_type','ncomp','gp_noise', 'gp_length', 'gp_const','dl_alpha','rmse','maxerr','global_mean_minerr',
                 'global_mean_maxerr', 'global_mean_rmse',
                 'global_max_rmse', 'global_var_rmse',)



def _configuration_parameters(**kwargs):

        _alpha_tests = np.array([0.0008])
        _ll_tests = np.arange(1.0,11.0,3.0)
        #_cc_tests = np.linspace(0.1,10.0,3)
        _noise_tests = np.logspace(-5.0, -1, num=3)
        alpha_tests = kwargs.get('alpha_tests', _alpha_tests)
        ll_tests = kwargs.get('ll_tests', _ll_tests)
        #cc_tests = kwargs.get('cc_tests', _cc_tests)
        noise_tests =kwargs.get('noise_tests', _noise_tests)
        return (alpha_tests, ll_tests,noise_tests)



def cross_validation(emulation_data,n_vali,wanted_ntest,operator,max_train_size,
                    min_train_size =1,number_of_splits=1, thinning = 1,
                    mask=None,split_run_ind = 0,GLOBAL_applymask = False, interp_type = 'GP',eps=1e-2,**kwargs):
    """Performs validation to determine the optimal hyperparemeters.
    Args:
        emulation_data: frame used for the crossvalidation
        n_vali:
        wanted_ntest: number of test vector on which the validation is averaged
        operator: operator used for the representation or interpolation 
        max_train_size: maximum number of training vectors 
        min_train_size: mimimum number of training vectors 
        number_of_splits: number of split (should be one principle)
        thinning: thinning of the grid
        mask: mask of the grid
        split_run_ind: index of the split chosen
        GLOBAL_applymask : True if the mask should be applied
        interp_type: type of the interpolator 
    Returns:
        op_crossval_df: frame average over the test vectors : containing number of train, parameters,
        and stastical results
        op_crossval_df_dict_all: frame containing number of train, parameters,
        and stastical results for each test vectors 
    """
    
    
    alpha_tests, ll_tests, noise_tests = _configuration_parameters(**kwargs)
    op_crossval_df_dict = {} #Dictionnary of cross validation for the given operator
    op_crossval_df_dict_all ={}
    for noi in (emulation_data.noise_names):
        op_crossval_df_dict[noi] =  pd.DataFrame(columns=columns_tuple)
        op_crossval_df_dict_all [noi] = pd.DataFrame(columns=columns_tuple)
    #numtrain=max_train_size+1
    intobj_all_dict={}
   # 
    for numtr in range(min_train_size,max_train_size,2):
        emulation_data.calculate_data_split(n_train=numtr, n_test=wanted_ntest,n_vali=n_vali,
                                            n_splits=number_of_splits, verbosity=0,manual_split = True)
        for nsplit in range(number_of_splits):
            emulation_data.data_split(split_index=nsplit, thinning=thinning, mask=mask,
                                      apply_mask=GLOBAL_applymask, verbosity=0)

            if operator == "LIN":
                for intpmeth in ['int1d']: #, 'spl1d']:  #hyperparam
                    Op = dcl.LearningOperator('LIN', interp_type=intpmeth,interp_dim =1)
                    op_crossval_df_dict, op_crossval_df_dict_all,global_mean_rmse=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict,split =nsplit)
            elif operator =="PCA":
                minncp = numtr//2
                maxncp = numtr
                rmse=float('inf')
                for ncp in range(minncp,maxncp+1):
                    if interp_type =='GP':
                        for ll in ll_tests:
                            for Y_noise in noise_tests : 
                                print("====== CPA Noise parameter: ",Y_noise)
                                Op = dcl.LearningOperator('PCA', ncomp=ncp, interp_type=interp_type)
                                op_crossval_df_dict, op_crossval_df_dict_all,global_mean_rmse=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,
                                                                                              op_crossval_df_dict_all,intobj_all_dict, Y_noise,split =nsplit)
   
                    if np.abs(rmse-global_mean_rmse)<eps:
                        break
                    rmse = global_mean_rmse
                                 #print('Warning was raised as an exception!')

            elif operator == "GP":
                print("ll_tests:" , ll_tests)
                for ll in ll_tests:  #hyperparam  GP
                    Op = dcl.LearningOperator('GP', gp_length=ll)
                    for Y_noise in noise_tests : 
                        print("====== GP Noise parameter: ",Y_noise)
                        try:
                            op_crossval_df_dict, op_crossval_df_dict_all,global_mean_rmse=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict,Y_noise,split =nsplit)
                        except:
                            print('Warning was raised as an exception!')
            elif operator =="DL":
                print("alpha tests: ", alpha_tests)
                for aa in alpha_tests:
                        print("====== DL alpha parameter: ", aa)
                        ncp_min = numtr#//2
                        ncp_max = numtr +1#+ 5  # :
                        rmse=float('inf')
                        for ncp in range(ncp_min, ncp_max):
                            for Y_noise in noise_tests :
                                print("====== DL Noise parameter: ",Y_noise)
                                try:
                                    Op=dcl.LearningOperator('DL', ncomp=ncp, dl_alpha=aa, interp_type=interp_type)
                                    op_crossval_df_dict, op_crossval_df_dict_all,global_mean_rmse=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict,Y_noise,split =nsplit)
                                except:
                                    print('Warning was raised as an exception!')
                            if np.abs(rmse-global_mean_rmse)<eps:
                                break
                            rmse = global_mean_rmse
        op_noise_dicts = [op_crossval_df_dict[noi] for noi in (emulation_data.noise_names)]
        op_crossval_df = pd.concat(op_noise_dicts)
    return op_crossval_df,op_crossval_df_dict_all

def cross_leave_one_out(emulation_data,n_train,operator,
                    number_of_splits=1, thinning = 1,
                    mask=None,split_run_ind = 0,GLOBAL_applymask = False, interp_type = 'GP',**kwargs):
    """Performs validation to determine the optimal hyperparemeters.
    Args:
        emulation_data: frame used for the crossvalidation
        n_vali:
        wanted_ntest: number of test vector on which the validation is averaged
        operator: operator used for the representation or interpolation 
        max_train_size: maximum number of training vectors 
        min_train_size: mimimum number of training vectors 
        number_of_splits: number of split (should be one principle)
        thinning: thinning of the grid
        mask: mask of the grid
        split_run_ind: index of the split chosen
        GLOBAL_applymask : True if the mask should be applied
        interp_type: type of the interpolator 
    Returns:
        op_crossval_df: frame average over the test vectors : containing number of train, parameters,
        and stastical results
        op_crossval_df_dict_all: frame containing number of train, parameters,
        and stastical results for each test vectors 
    """
    

    alpha_tests, ll_tests, noise_tests = _configuration_parameters(**kwargs)
    op_crossval_df_dict = {} #Dictionnary of cross validation for the given operator
    op_crossval_df_dict_all ={}
    for noi in (emulation_data.noise_names):
        op_crossval_df_dict[noi] =  pd.DataFrame(columns=columns_tuple)
        op_crossval_df_dict_all [noi] = pd.DataFrame(columns=columns_tuple)
    #numtrain=max_train_size+1
    intobj_all_dict={}

    
    for i in range(len(emulation_data.extparam_vals)):

        emulation_data.calculate_data_split(n_train=n_train, n_test=1,n_vali=1,
                                            n_splits=number_of_splits, verbosity=0,manual_split = True,test_indices=[i])
        emulation_data.data_split(split_index=split_run_ind, thinning=thinning, mask=mask,
                                  apply_mask=GLOBAL_applymask, verbosity=0)

        if operator == "LIN":
            for intpmeth in ['int1d']: #, 'spl1d']:  #hyperparam
                Op = dcl.LearningOperator('LIN', interp_type=intpmeth,interp_dim =1)
                op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict)
        elif operator =="PCA":
            minncp = n_train - 5
            if minncp<1:
                minncp = 2
            maxncp = n_train
            for ncp in range(1,emulation_data.train_size+1):
                print(emulation_data.train_samples)
                if interp_type =='GP':
                    for ll in ll_tests:
                        for Y_noise in noise_tests : 
                            print("====== CPA Noise parameter: ",Y_noise)
                            try:
                                Op = dcl.LearningOperator('PCA', ncomp=ncp, interp_type=interp_type)
                                op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,
                                                                                                        op_crossval_df_dict_all,intobj_all_dict, Y_noise)
                            except:
                                 print('Warning was raised as an exception!')
                break
        elif operator == "GP":
            print("ll_tests:" , ll_tests)
            for ll in ll_tests:  #hyperparam  GP
                Op = dcl.LearningOperator('GP', gp_length=ll)
                for Y_noise in noise_tests : 
                    print("====== GP Noise parameter: ",Y_noise)
                    try:
                        op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict,Y_noise)
                    except:
                        print('Warning was raised as an exception!')
        elif operator =="DL":
            print("alpha tests: ", alpha_tests)
            for aa in alpha_tests:
                    print("====== DL alpha parameter: ", aa)
                    ncp_min = n_train-1
                    ncp_max = n_train + 1  # :
                    for ncp in range(ncp_min, ncp_max):
                        for Y_noise in noise_tests :
                            print("====== DL Noise parameter: ",Y_noise)
                            try:
                                Op=dcl.LearningOperator('DL', ncomp=ncp, dl_alpha=aa, interp_type=interp_type)
                                op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict,Y_noise)
                            except:
                                print('Warning was raised as an exception!')
    op_noise_dicts = [op_crossval_df_dict[noi] for noi in (emulation_data.noise_names)]
    op_crossval_df = pd.concat(op_noise_dicts)
    return op_crossval_df,op_crossval_df_dict_all







    
def validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict,Y_noise = None,split =1):
    """ Compute the interpolation and the statistics
    Args:
        Op:
        emulation_data: dahahandle object
        op_crossval_df: frame average over the test vectors : containing number of train, parameters,
        and stastical results
        op_crossval_df_dict_all: frame containing number of train, parameters,
        and stastical results for each test vectors 
        intobj_all_dict: interpolator object
        Y_noise: noise level for the GP interpolator 
    Returns:
        op_crossval_df: updated frame average over the test vectors : containing number of train, parameters,
        and stastical results
        op_crossval_df_dict_all: updated frame containing number of train, parameters,
        and stastical results for each test vectors 
    """
    
    for noi in (emulation_data.noise_names):

        print("case : "+noi)
        intobj_all_dict[noi] = dcl.LearnData(Op)

        train_data=emulation_data.matrix_datalearn_dict[noi]['train']

        intobj_all_dict[noi].interpolate(train_data=train_data,
                   train_samples=emulation_data.train_samples,train_noise = Y_noise)

        intobj_all_dict[noi].predict(emulation_data.test_samples)

        intobj_all_dict[noi].calc_statistics(emulation_data.matrix_datalearn_dict.theo['test'],
                                    emulation_data.test_samples)

        intobj_all_dict[noi].print_statistics()

        
        app_dict = fill_stats_dict(intobj_all_dict[noi], columns_tuple)
        app_dict['noise_case'] = noi
        app_dict['n_train'] = emulation_data.train_size
        app_dict['split'] = split
        
        
        op_crossval_df_dict[noi] = op_crossval_df_dict[noi].append(app_dict, ignore_index=True)
        for param in emulation_data.test_samples:
            param = tuple(param)
            app_dict['parameter'] = "parameters"
            app_dict['parameter_value'] = param
            app_dict['rmse']= intobj_all_dict[noi].rmse_dict[param]
            app_dict['maxerr']= intobj_all_dict[noi].maxerr_dict[param]

            op_crossval_df_dict_all[noi] = op_crossval_df_dict_all[noi].append(app_dict, ignore_index=True)

    return op_crossval_df_dict,op_crossval_df_dict_all,app_dict['global_mean_rmse']

def op_crossval_df_dict_mingroup_function(emulation_data,op_crossval_df_dict):
    """ Find the optimal parameters which minimize the global mean rmse
    Args:
        Op:
        emulation_data: dahahandle object
        op_crossval_df: frame average over the test vectors : containing number of train, parameters,
        and stastical results
    Returns:
        op_crossval_df_dict_mingroup : frame containing the optimal hyperparameter
    """
    op_crossval_df_dict_mingroup={}
    for noi in (emulation_data.noise_names):
            op_crossval_df_dict_mingroup[noi] = dataframe_group(op_crossval_df_dict[op_crossval_df_dict["noise_case"]==noi],
                                                                    groupby='n_train', sortby='global_mean_rmse')
    return op_crossval_df_dict_mingroup



def RMSE_dictionary(emulation_data,wanted_ntest,max_ntrain,min_ntrain=1,
                    redshift_index = [0], lin_crossval_df_dict_mingroup=None,
                    DL_crossval_df_dict_mingroup=None,GP_crossval_df_dict_mingroup=None,
                    PCA_crossval_df_dict_mingroup=None,turnoff_PCA = True,
                    turnoff_GP = True,turnoff_DL = True, turnoff_LIN = True,
                    dictparam = dict(),
                    number_of_splits=1,
                    thinning=1,mask=None,split_run_ind = 0,GLOBAL_applymask = False,step=1):
    """ Compute the RMSE as a function of training vector using provided hyperparameters
    Args:
        emulation_data:
        wanted_ntest:
        max_ntrain:
        min_ntrain:
        redshift_index: redshift used to compute the RMSE
        lin_crossval_df_dict_mingroup: hyperparameters frames of LIN
        DL_crossval_df_dict_mingroup: hyperparameters frames of DL
        GP_crossval_df_dict_mingroup: hyperparameters frames of GP
        PCA_crossval_df_dict_mingroup: hyperparameters frames of PCA
        turnoff_PCA: 
        turnoff_GP:
        turnoff_DL: 
        turnoff_PCA: 
        turnoff_LIN: 
        dictparam: dictionnary parameters -> names (optional)
        number_of_splits: 
        thinning: thinning of the grid
        mask: mask used
        split_run_ind: index of the split used
        GLOBAL_applymask: True if the mask is applied
Returns:
        index_sort(datatest_df_dict,emulation_data): frame containing the statistics 
    """
  #  %%time




    columns_tuple = ('method','noise_case','n_train', 'parameter', 'parameter_value',
                         'single_rmse', 'single_maxerr','single_minerr',
                         'ncomp', 'dl_alpha', 'global_mean_minerr',
                         'global_mean_maxerr', 'global_mean_rmse','global_spectra_mean_maxerr',
                         'global_spectra_mean_rmse',
                         'global_max_rmse', 'global_var_rmse')



    datatest_df_dict = {}


     ## these are actually the training vectors minus the extrema vectors, added automatically later

    for noi in (["theo"]):
        datatest_df_dict[noi] =  pd.DataFrame(columns=columns_tuple)

    emulation_data.calculate_data_split(n_train=1, n_test=1, n_splits=1,
                                            verbosity=0,manual_split = True,test_indices=[0])
    

    for numtr in range(min_ntrain,max_ntrain,step):  #+1 for range
    

        emulation_data.calculate_data_split(n_train=numtr, n_test=wanted_ntest, n_splits=number_of_splits,
                                            verbosity=1,manual_split = True,train_redshift_indices = redshift_index,test_redshift_indices= redshift_index)
        emulation_data.data_split(split_index=split_run_ind, thinning=thinning, mask=mask,
                                  apply_mask=GLOBAL_applymask, verbosity=0)

        tr_sz = emulation_data.train_size
        print("Train Size",tr_sz)
        print("numtr",numtr)
        

        for noi in (["theo"]):
            if turnoff_LIN ==False :
 
                if lin_crossval_df_dict_mingroup is not None :
                    LIN_intptype = lin_crossval_df_dict_mingroup[noi].loc[tr_sz]['interp_type']
                else:
                    LIN_intptype = dictparam['interp_type']
                    
                print("++++"+noi)
                print("LIN :", LIN_intptype)
                intobj_all = test_return_obj(emulation_data,
                             method='LIN', interp_type=LIN_intptype, interp_dim =1,noisecase=noi)
            
                for ppi in emulation_data.test_samples:
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                         columns_tuple=columns_tuple)
            
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)

            if turnoff_PCA==False:
                if PCA_crossval_df_dict_mingroup is not None :
                    PCA_ncp = int(PCA_crossval_df_dict_mingroup[noi].loc[PCA_crossval_df_dict_mingroup[noi]["n_train"]==numtr]['ncomp'])
                    Y_noise = int(PCA_crossval_df_dict_mingroup[noi].loc[PCA_crossval_df_dict_mingroup[noi]["n_train"]==numtr]['gp_noise'])
                else:
                    PCA_ncp = dictparam["ncomp"]
                    Y_noise = dictparam["gp_noise"]
                
                print("++++"+noi)
                print("PCA : ", PCA_ncp)

                intobj_all = test_return_obj(emulation_data,
                             method='PCA', ncomp=PCA_ncp, dl_alpha=None, interp_type='GP', noisecase=noi,Y_noise=Y_noise)
                
                for ppi in emulation_data.test_samples:
                    ppi=tuple(ppi)
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                         columns_tuple=columns_tuple)
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)
                    
                

            if turnoff_DL==False:
                
                if DL_crossval_df_dict_mingroup is not None :
                    DL_ncp = int(DL_crossval_df_dict_mingroup[noi].loc[DL_crossval_df_dict_mingroup[noi]["n_train"]==numtr]['ncomp'])
                    DL_alpha = int(DL_crossval_df_dict_mingroup[noi].loc[DL_crossval_df_dict_mingroup[noi]["n_train"]==numtr]['dl_alpha']) 
                    Y_noise = int(DL_crossval_df_dict_mingroup[noi].loc[DL_crossval_df_dict_mingroup[noi]["n_train"]==numtr]['gp_noise'])
                else:
                    DL_ncp = dictparam["ncomp"]
                    DL_alpha = dictparam["dl_alpha"]
                    
                print("DL : ", DL_ncp)
                intobj_all = test_return_obj(emulation_data,
                             method='DL', ncomp=DL_ncp, dl_alpha=DL_alpha, interp_type='GP',
                                            noisecase=noi,Y_noise=Y_noise)

                for ppi in emulation_data.test_samples:
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                         columns_tuple=columns_tuple)
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)



            if turnoff_GP==False:
                
                if GP_crossval_df_dict_mingroup is not None:
                    GP_length = int(GP_crossval_df_dict_mingroup[noi].loc[GP_crossval_df_dict_mingroup[noi]["n_train"]==numtr]['gp_length'])
                    GP_const = int(GP_crossval_df_dict_mingroup[noi].loc[GP_crossval_df_dict_mingroup[noi]["n_train"]==numtr]['gp_const'])
                else :
                    GP_length = dictparam["gp_length"]
                    GP_const = dictparam["gp_const"]
     
                intobj_all = test_return_obj(emulation_data,
                             method='GP',gp_length=GP_length,gp_const=GP_const )
#
                for ppi in emulation_data.test_samples:
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                     columns_tuple=columns_tuple)
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)
   
                

    return  index_sort(datatest_df_dict,emulation_data)



def RMSE_dictionary_redshift(emulation_data,max_redshift,min_redshift=1,test_redshift_indices= [1],
                             ntrain = 2 ,lin_crossval_df_dict_mingroup=None,
                    DL_crossval_df_dict_mingroup=None,GP_crossval_df_dict_mingroup=None,
                    PCA_crossval_df_dict_mingroup=None,turnoff_PCA = True,
                    turnoff_GP = True,turnoff_DL = True, turnoff_LIN = True,
                    dictparam = dict(),
                    number_of_splits=1,
                    thinning=1,mask=None,split_run_ind = 0,GLOBAL_applymask = False):
    
    

  #  %%time




    columns_tuple = ('method','noise_case','n_train', 'parameter', 'parameter_value',
                         'single_rmse', 'single_maxerr','single_minerr','single_spectra_rmse',
                         'single_spectra_maxerr',
                         'ncomp', 'dl_alpha', 'global_mean_minerr',
                         'global_mean_maxerr', 'global_mean_rmse','global_spectra_mean_maxerr',
                         'global_spectra_mean_rmse',
                         'global_max_rmse', 'global_var_rmse')



    datatest_df_dict = {}


     ## these are actually the training vectors minus the extrema vectors, added automatically later

    for noi in (emulation_data.noise_names):
        datatest_df_dict[noi] =  pd.DataFrame(columns=columns_tuple)

    emulation_data.calculate_data_split(n_train=1, n_test=1, n_splits=1,
                                            verbosity=1,manual_split = True,test_indices=[0])
    
    nb_param = int(len(emulation_data.fullspace)/len(emulation_data.z_requested))
    test = [ii for ii in range(0,nb_param)  ]
        
    L_train_redshift = [ii for  ii in range(len(emulation_data.z_requested)) if ii not in test_redshift_indices]
                           
    L_train_redshift = shuffle( L_train_redshift )
    for nredshift in range(min_redshift,max_redshift):  #+1 for range
        train_redshift=L_train_redshift[:nredshift]
        
        
        
        
        
        emulation_data.calculate_data_split(n_train=len(test), n_test=len(test), n_splits=number_of_splits,
                                            
                                            verbosity=0,manual_split = True,test_indices=[test],
                                            train_redshift_indices = train_redshift,test_redshift_indices= test_redshift_indices,
                                            interpolate_over_redshift_only = True)
        emulation_data.data_split(split_index=split_run_ind, thinning=thinning, mask=mask,
                                  apply_mask=GLOBAL_applymask, verbosity=0)

        tr_sz = emulation_data.train_size

        for noi in (["theo"]):
            try:
                if turnoff_LIN ==False :
                    
                    if lin_crossval_df_dict_mingroup is not None :
                        LIN_intptype = lin_crossval_df_dict_mingroup[noi].loc[tr_sz]['interp_type']
                    else:
                        LIN_intptype = dictparam['interp_type']
                        
                    print("++++"+noi)
                    print("LIN :", LIN_intptype)
                    intobj_all = test_return_obj(emulation_data,
                                 method='LIN', interp_type=LIN_intptype, noisecase=noi)
                    
                    for ppi in emulation_data.test_samples:
                        app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = nredshift , noisecase=noi,
                                             columns_tuple=columns_tuple)
                
                        datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)
            except:
                    ass

            if turnoff_PCA==False:
                
                if PCA_crossval_df_dict_mingroup is not None :
                    PCA_ncp = int(PCA_crossval_df_dict_mingroup[noi].loc[tr_sz]['ncomp'])
                    Y_noise = int(PCA_crossval_df_dict_mingroup[noi].loc[tr_sz]['gp_noise'])
                else:
                    PCA_ncp = dictparam["ncomp"]
                    Y_noise = dictparam["gp_noise"]
                
                print("++++"+noi)
                print("PCA : ", PCA_ncp)

                intobj_all = test_return_obj(emulation_data,
                             method='PCA', ncomp=PCA_ncp, dl_alpha=None, interp_type='GP', noisecase=noi,Y_noise=Y_noise)
                
                for ppi in emulation_data.test_samples:
                    ppi=tuple(ppi)
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = nredshift , noisecase=noi,
                                         columns_tuple=columns_tuple)
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)

            if turnoff_DL==False:
                
                if DL_crossval_df_dict_mingroup is not None :
                    DL_ncp = int(DL_crossval_df_dict_mingroup[noi].loc[tr_sz]['ncomp'])
                    DL_alpha = DL_crossval_df_dict_mingroup[noi].loc[tr_sz]['dl_alpha']
                else:
                    DL_ncp = dictparam["ncomp"]
                    DL_alpha = dictparam["dl_alpha"]
                    
                print("DL : ", DL_ncp)
                intobj_all = test_return_obj(emulation_data,
                             method='DL', ncomp=DL_ncp, dl_alpha=DL_alpha, interp_type='int1d',
                                            noisecase=noi)

                for ppi in emulation_data.test_samples:
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = nredshift , noisecase=noi,
                                         columns_tuple=columns_tuple)
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)



            if turnoff_GP==False:
                
                if GP_crossval_df_dict_mingroup is not None:
                    GP_length = int(GP_crossval_df_dict_mingroup[noi].loc[tr_sz]['gp_length'])
                    GP_const = int(GP_crossval_df_dict_mingroup[noi].loc[tr_sz]['gp_const'])
                else :
                    GP_length = dictparam["gp_length"]
                    GP_const = dictparam["gp_const"]
                try:
                    intobj_all = test_return_obj(emulation_data,
                                 method='GP',gp_length=GP_length,gp_const=GP_const )
#
                    for ppi in emulation_data.test_samples:
                        app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = nredshift , noisecase=noi,
                                         columns_tuple=columns_tuple)
                        datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)
                except :
                    print("print_error_GP")

    return  index_sort(datatest_df_dict,emulation_data)
















def index_sort(datatest_df_dict,emulation_data):
    for noi in (["theo"]):
        datatest_df_dict[noi].set_index(['method', 'n_train'], inplace=True)
    for noi in (["theo"]):
        datatest_df_dict[noi].sort_index(inplace=True)
    return datatest_df_dict


def test_return_obj(emulation_data, method='PCA', ncomp=None, dl_alpha=None, interp_type='int1d', noisecase='theo',gp_const=10, gp_length=5,Y_noise=None,interp_dim = 2):

    Dop = dcl.LearningOperator(method, ncomp=ncomp, dl_alpha=dl_alpha, interp_type=interp_type,gp_const=gp_const,gp_length=gp_length,interp_dim =interp_dim )
    if noisecase!='theo' and method=='GP':
        y_noi = noise_per_sample(emulation_data.matrix_datalearn_dict[noisecase]['train'],
            emulation_data.matrix_datalearn_dict['theo']['train'])
    elif noisecase=='theo' and method=='GP':
        y_noi = 1e-10 ##the default of GP


    intobj_all_dict = dcl.LearnData(Dop)
    if method=='GP':
        intobj_all_dict.interpolate(train_data=emulation_data.matrix_datalearn_dict[noisecase]['train'],
               train_samples=emulation_data.train_samples,train_noise=y_noi)
    else:
        intobj_all_dict.interpolate(train_data=emulation_data.matrix_datalearn_dict[noisecase]['train'],
               train_samples=emulation_data.train_samples,train_noise=Y_noise)

    intobj_all_dict.predict(emulation_data.test_samples)

    intobj_all_dict.calc_statistics(emulation_data.matrix_datalearn_dict.theo['test'],
                                emulation_data.test_samples,emulation_data)



    return intobj_all_dict


def noise_per_sample(noisy_obs, theo_obs):
    features_noise = noisy_obs-theo_obs
    noise_samp = np.array([])
    for nn in features_noise:
        noise_samp = np.append(noise_samp, [np.std(nn)])
    return noise_samp


def fill_app_dict(intobj, param_val=0.05, param_name='beta', n_train = 1, noisecase='theo', 
                  columns_tuple=['']):
    
    app_dict = intobj.fill_stats_dict(columns_tuple)
    param_val=tuple(param_val)
    app_dict['noise_case'] = noisecase
    app_dict['n_train'] = n_train
    app_dict['parameter'] = param_name
    app_dict['parameter_value'] = param_val
    app_dict['single_rmse'] = intobj.rmse_dict[param_val]
    app_dict['single_maxerr'] = intobj.maxerr_dict[param_val]
    app_dict['single_minerr'] = intobj.minerr_dict[param_val]
    
    app_dict['single_spectra_rmse'] = intobj.rmse_spectra_dict[param_val]
    app_dict['single_spectra_maxerr'] = intobj.maxerr_spectra_dict[param_val]
    
    return app_dict


def fill_stats_dict(object, cols, failed=False):
    stat_dict = {}
    for cc in cols:
        try:
            stat_dict[cc] = object.__dict__[cc]
        except KeyError:
            stat_dict[cc] = np.nan
            pass
    return stat_dict





def dataframe_group(df, groupby='', sortby='', filename='', savedir='./savedir/'):
    if type(df) != pd.core.frame.DataFrame:
        print("object passed is not a pandas DataFrame")
        return None
    #print("min value of sortby: ")
    df_grouped = df.loc[df.groupby(groupby)[sortby].idxmin()]
    df_grouped[['n_train']] = df_grouped[['n_train']].astype(int)
    df_grouped.set_index(['n_train'], inplace=True)

    return df_grouped
