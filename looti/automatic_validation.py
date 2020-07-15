#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:51:25 2020

@author: raphaelbaena
"""
import pandas as pd
import looti.dictlearn as dcl
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
columns_tuple = ('noise_case', 'parameter','parameter_value','n_train',
                 'interp_type','ncomp','gp_noise', 'gp_length', 'gp_const','dl_alpha','rmse','maxerr','global_mean_minerr',
                 'global_mean_maxerr', 'global_mean_rmse',
                 'global_max_rmse', 'global_var_rmse',)


def _configuration_parameters(**kwargs):

        _alpha_tests = np.array([0.0008, 0.001, 0.002, 0.005, 0.01])
        _ll_tests = np.arange(1.0,11.0,3.0)
        _cc_tests = np.linspace(0.1,10.0,3)
        _noise_tests = np.logspace(-5.0, -1, num=10)
        alpha_tests = kwargs.get('alpha_tests', _alpha_tests)
        ll_tests = kwargs.get('ll_tests', _ll_tests)
        cc_tests = kwargs.get('cc_tests', _cc_tests)
        noise_tests =kwargs.get('noise_tests', _noise_tests)
        return (alpha_tests, ll_tests, cc_tests,noise_tests)



def cross_validation(emulation_data,n_vali,wanted_ntest,operator,max_train_size,
                    min_train_size =1,number_of_splits=1, thinning = 1,
                    mask=None,split_run_ind = 0,GLOBAL_applymask = False, interp_type = 'in1td',**kwargs):

    alpha_tests, ll_tests, cc_tests ,noise_tests = _configuration_parameters(**kwargs)
    op_crossval_df_dict = {} #Dictionnary of cross validation for the given operator
    op_crossval_df_dict_all ={}
    for noi in (emulation_data.noise_names):
        op_crossval_df_dict[noi] =  pd.DataFrame(columns=columns_tuple)
        op_crossval_df_dict_all [noi] = pd.DataFrame(columns=columns_tuple)
    #numtrain=max_train_size+1
    intobj_all_dict={}

    for numtr in range(min_train_size,max_train_size):
        emulation_data.calculate_data_split(n_train=numtr, n_test=wanted_ntest,n_vali=n_vali,
                                            n_splits=number_of_splits, verbosity=1,manual_split = True,test_indices=[0])
        emulation_data.data_split(split_index=split_run_ind, thinning=thinning, mask=mask,
                                  apply_mask=GLOBAL_applymask, verbosity=2)

        if operator == "LIN":
            for intpmeth in ['int1d']: #, 'spl1d']:  #hyperparam
                Op = dcl.LearningOperator('LIN', interp_type=intpmeth)
                op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict)
        elif operator =="PCA":
            for ncp in range(1,emulation_data.train_size+1):
                if interp_type =='GP':
                    for ll in ll_tests:
                        for Y_noise in noise_tests : 
                            Op = dcl.LearningOperator('PCA', ncomp=ncp, interp_type=interp_type)
                            op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,
                                                                                                     op_crossval_df_dict_all,intobj_all_dict, Y_noise)

        elif operator == "GP":
            print("ll_tests:" , ll_tests)
            print("cc_tests:" , cc_tests)
            for ll in ll_tests:  #hyperparam  GP
                for cc in cc_tests:  #hyperparam  GP
                    Op = dcl.LearningOperator('GP', gp_const=cc, gp_length=ll)
                    try:
                        op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict)
                    except:
                        print('Warning was raised as an exception!')
        elif operator =="DL":
            print("alpha tests: ", alpha_tests)
            for aa in alpha_tests:
                    print("====== DL alpha parameter: ", aa)
                    ncp_min = 1
                    ncp_max = emulation_data.train_size + 2  # :
                    for ncp in range(ncp_min, ncp_max + 1):
                        Op=dcl.LearningOperator('DL', ncomp=ncp, dl_alpha=aa, interp_type=interp_type)
                        op_crossval_df_dict, op_crossval_df_dict_all=validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict)
    op_noise_dicts = [op_crossval_df_dict[noi] for noi in (emulation_data.noise_names)]
    op_crossval_df = pd.concat(op_noise_dicts)
    return op_crossval_df,op_crossval_df_dict_all


def validation_over_noise_level(Op,emulation_data,op_crossval_df_dict,op_crossval_df_dict_all,intobj_all_dict,Y_noise = None):
    
    for noi in (emulation_data.noise_names):

        print("case : "+noi)
        intobj_all_dict[noi] = dcl.LearnData(Op)

        train_data=emulation_data.matrix_datalearn_dict[noi]['train']

        intobj_all_dict[noi].interpolate(train_data=train_data,
                   train_samples=emulation_data.train_samples,train_noise = Y_noise)

        intobj_all_dict[noi].predict(emulation_data.vali_samples)

        intobj_all_dict[noi].calc_statistics(emulation_data.matrix_datalearn_dict.theo['vali'],
                                    emulation_data.vali_samples)

        intobj_all_dict[noi].print_statistics()
        
        
        
        app_dict = fill_stats_dict(intobj_all_dict[noi], columns_tuple)
        app_dict['noise_case'] = noi
        app_dict['n_train'] = emulation_data.train_size
        
        op_crossval_df_dict[noi] = op_crossval_df_dict[noi].append(app_dict, ignore_index=True)
        for param in emulation_data.vali_samples:
            param = tuple(param)
            app_dict['parameter'] = emulation_data.extparam1_name
            app_dict['parameter_value'] = param
            app_dict['rmse']= intobj_all_dict[noi].rmse_dict[param]
            app_dict['maxerr']= intobj_all_dict[noi].maxerr_dict[param]

            op_crossval_df_dict_all[noi] = op_crossval_df_dict_all[noi].append(app_dict, ignore_index=True)
        
            
        
        
            
    return op_crossval_df_dict,op_crossval_df_dict_all

def op_crossval_df_dict_mingroup_function(emulation_data,op_crossval_df_dict):
    op_crossval_df_dict_mingroup={}
    for noi in (emulation_data.noise_names):
            op_crossval_df_dict_mingroup[noi] = dataframe_group(op_crossval_df_dict[op_crossval_df_dict["noise_case"]==noi],
                                                                    groupby='n_train', sortby='global_mean_rmse')
    return op_crossval_df_dict_mingroup



def RMSE_dictionary(emulation_data,wanted_ntest,max_ntrain,min_ntrain=1,redshift_index = 0, lin_crossval_df_dict_mingroup=None,DL_crossval_df_dict_mingroup=None,GP_crossval_df_dict_mingroup=None,PCA_crossval_df_dict_mingroup=None,turnoff_PCA = True,turnoff_GP = True,turnoff_DL = True, turnoff_LIN = True,number_of_splits=1,thinning=1,mask=None,split_run_ind = 0,GLOBAL_applymask = False):

  #  %%time




    columns_tuple = ('method','noise_case','n_train', 'parameter', 'parameter_value',
                         'single_rmse', 'single_maxerr','single_minerr',
                         'ncomp', 'dl_alpha', 'global_mean_minerr',
                         'global_mean_maxerr', 'global_mean_rmse',
                         'global_max_rmse', 'global_var_rmse')



    datatest_df_dict = {}


    min_ntrain = 1  ## these are actually the training vectors minus the extrema vectors, added automatically later

    for noi in (emulation_data.noise_names):
        datatest_df_dict[noi] =  pd.DataFrame(columns=columns_tuple)



    for numtr in range(min_ntrain,max_ntrain):  #+1 for range
        nb_param = int(len(emulation_data.fullspace)/len(emulation_data.z_vals))
        
        test = [ii for ii in range(0,nb_param)  ]
        test=test[::wanted_ntest]
        emulation_data.calculate_data_split(n_train=numtr, n_test=wanted_ntest, n_splits=number_of_splits,
                                            verbosity=1,manual_split = True,test_indices=[test])
        emulation_data.data_split(split_index=split_run_ind, thinning=thinning, mask=mask,
                                  apply_mask=GLOBAL_applymask, verbosity=2)

        tr_sz = emulation_data.train_size
        emulation_data.test_samples.sort()
        beta = emulation_data.extparam1_name

        for noi in (emulation_data.noise_names):
            print("turn",turnoff_PCA==False)
            if turnoff_LIN ==False :
                LIN_intptype = lin_crossval_df_dict_mingroup[noi].loc[tr_sz]['interp_type']
                print("++++"+noi)
                print("LIN :", LIN_intptype)
                intobj_all = test_return_obj(emulation_data,
                             method='LIN', interp_type=LIN_intptype, noisecase=noi)
            
                for ppi in emulation_data.test_samples:
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                         columns_tuple=columns_tuple, param_name=beta)
            
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)


            if turnoff_PCA==False:
                PCA_ncp = int(PCA_crossval_df_dict_mingroup[noi].loc[tr_sz]['ncomp'])
                Y_noise = int(PCA_crossval_df_dict_mingroup[noi].loc[tr_sz]['gp_noise'])
                print("++++"+noi)
                print("PCA : ", PCA_ncp)
                intobj_all = test_return_obj(emulation_data,
                             method='PCA', ncomp=PCA_ncp, dl_alpha=None, interp_type='GP', noisecase=noi,Y_noise=Y_noise)
                
                for ppi in emulation_data.test_samples:
                    ppi=tuple(ppi)
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                         columns_tuple=columns_tuple, param_name=beta)
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)

            if turnoff_DL==False:
                DL_ncp = int(DL_crossval_df_dict_mingroup[noi].loc[tr_sz]['ncomp'])
                DL_alpha = DL_crossval_df_dict_mingroup[noi].loc[tr_sz]['dl_alpha']
                print("DL : ", DL_ncp)
                intobj_all = test_return_obj(emulation_data,
                             method='DL', ncomp=DL_ncp, dl_alpha=DL_alpha, interp_type='int1d',
                                            noisecase=noi)

                for ppi in emulation_data.test_samples:
                    app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                         columns_tuple=columns_tuple, param_name=beta)
                    datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)



            if turnoff_GP==False:

                GP_length = int(GP_crossval_df_dict_mingroup[noi].loc[tr_sz]['gp_length'])
                GP_const = int(GP_crossval_df_dict_mingroup[noi].loc[tr_sz]['gp_const'])
                try:
                    intobj_all = test_return_obj(emulation_data,
                                 method='GP',gp_length=GP_length,gp_const=GP_const )
#
                    for ppi in emulation_data.test_samples:
                        app_dict = fill_app_dict(intobj_all, param_val=ppi, n_train = tr_sz, noisecase=noi,
                                         columns_tuple=columns_tuple, param_name=beta)
                        datatest_df_dict[noi] = datatest_df_dict[noi].append(app_dict, ignore_index=True)
                except :
                    "print_error_GP"
    return  index_sort(datatest_df_dict,emulation_data)




def plot_RMSE(zchoice,datatest_df_dict,noi='noise_100',turnoff_LIN=True,turnoff_PCA=True,turnoff_GP=True,turnoff_DL=True):
    fig=plt.figure(1, figsize=(20,12), dpi=80,facecolor='w')

    #simulation_box_size = 3000   #the larger the size, the smaller sampling errors at large scales

    ax = fig.subplots(2, sharex=True)

    zval = zchoice
    zst= '{:.2f}'.format(zval)

    #xlim=(0.01,None)

    color_list = plt.cm.cool( np.linspace(0,1,4 ) )

    
    
    #noi='theo'

    #c=next(color_iter)
    #pk_vec = pkresult['P_nonlin'][zlabel_to_key[zst]]
    #lab0='$\\Lambda$CDM '+"z="+str(zzi)
    lab_dl=noi+' DL '
    lab_pca=noi+' PCA '
    lab_lin=noi+' LIN '

    dat_df = datatest_df_dict[noi];

    ax[0].text(0.5,0.95, '$z=$'+zst, fontsize=12, ha='center', transform=ax[1].transAxes)
    ax[1].text(0.5,0.95, '$z=$'+zst, fontsize=12, ha='center', transform=ax[1].transAxes)

    ylabdict = dict(zip(['global_mean_rmse','global_mean_maxerr'], 
                                     ['RMSE','Max. relative error']))

    for xx, (sn,gl) in enumerate(zip(['single_rmse','single_maxerr'], 
                                     ['global_mean_rmse','global_mean_maxerr'])):
        ### *** PCA 
        print('xx', xx)
        print('gl',gl)

        color_iter=iter(color_list)
        c=next(color_iter)
        
        pca_col = c
        pca_namedcol = c
        
        if turnoff_PCA==False:

    
            xvals = np.unique(dat_df.loc['PCA'].index.values)
            yvals = np.array(dat_df.loc['PCA'][gl].groupby('n_train').mean())
            #[dat_df.loc['PCA',ii][gl].values[0] for ii in xvals]
    
            ax[xx].plot(xvals,yvals, '--', color=pca_namedcol, lw=3, ms=5,  alpha=0.6, label=lab_pca  )
            
        
            parvals = np.unique(dat_df.loc['PCA']['parameter_value'].values)
            ppvals=parvals
            print("len parvals", len(parvals))
    
            for jj,pp in enumerate(parvals):
    
                xy = get_param_array(dat_df.loc['PCA'], pp, sn);
    
                xscat = xy[:,0]###dat_df.loc['PCA'].index.values
                yscat = xy[:,1]  ##dat_df.loc['PCA']['single_rmse'].values
                alpha_pp = pp/np.max(ppvals)
                print(yscat)
                ax[xx].plot(xscat,yscat, 's', color=pca_col, lw=1.9, ms=5,  alpha=alpha_pp,   label=lab_pca)
    
                yinterv = (yvals[0]-yvals.min())/len(parvals)
           # ax[xx].annotate('{:.3f}'.format(pp), xy=(3, yscat[0]),  xycoords='data',
            #    xytext=(2, yvals.min()+yinterv*jj ), textcoords='data', ha='center',
             #    arrowprops=dict(arrowstyle="->",
                 #                   connectionstyle="arc,angleA=0,armA=30,rad=10") ,
                 #          bbox=dict(pad=-0.5, facecolor="none", edgecolor="none"))#dict(arrowstyle="-"))
            #ax[xx].annotate('{:.3f}'.format(pp), ((2.4+jj*0.4, yscat[0])))






        ### *** LIN 
        c=next(color_iter)

        lin_col = c
        lin_namedcol= c
        
        if turnoff_LIN==False:


            for pp in np.unique(dat_df.loc['LIN']['parameter_value'].values):
        
                xy = get_param_array(dat_df.loc['LIN'], pp, sn);
        
                xscat = xy[:,0]+0.1###dat_df.loc['PCA'].index.values
                yscat = xy[:,1]  ##dat_df.loc['PCA']['single_rmse'].values
                alpha_pp = pp/np.max(ppvals)
                ax[xx].plot(xscat,yscat, '^', color=lin_col, lw=1.9, ms=5,  alpha=alpha_pp  )
        
        
        
            xvals = np.unique(dat_df.loc['LIN'].index.values)
            yvals = dat_df.loc['LIN'][gl].groupby('n_train').mean()
            #[dat_df.loc['LIN',ii][gl].values[0] for ii in xvals]
            print(len(xvals),len(yvals))
        
            ax[xx].plot(xvals,yvals, '-.', color=lin_namedcol, lw=3, ms=5,  alpha=0.6, label=lab_lin  )






        c=next(color_iter)

        dl_col = c
        dl_namedcol= c

        ### *** DL 
        if turnoff_DL ==False:


            for pp in np.unique(dat_df.loc['DL']['parameter_value'].values):

                xy = get_param_array(dat_df.loc['DL'], pp, sn);

                xscat = xy[:,0]+0.1###dat_df.loc['PCA'].index.values
                yscat = xy[:,1]  ##dat_df.loc['PCA']['single_rmse'].values
                alpha_pp = pp/np.max(ppvals)
                ax[xx].plot(xscat,yscat, 'o', color=dl_col, lw=1.9, ms=5,  alpha=alpha_pp  )



            xvals = np.unique(dat_df.loc['DL'].index.values)
            yvals = dat_df.loc['DL'][gl].groupby('n_train').mean()
            #[dat_df.loc['LIN',ii][gl].values[0] for ii in xvals]

            ax[xx].plot(xvals,yvals, ':', color=dl_namedcol, lw=3, ms=5,  alpha=0.6, label=lab_lin  )

        ### *** GP
        c=next(color_iter)

        gp_col = c
        gp_namedcol= c
        if turnoff_GP==False:



            for pp in np.unique(dat_df.loc['GP']['parameter_value'].values):

                xy = get_param_array(dat_df.loc['GP'], pp, sn);

                xscat = xy[:,0]+0.1###dat_df.loc['PCA'].index.values
                yscat = xy[:,1]  ##dat_df.loc['PCA']['single_rmse'].values
                alpha_pp = pp/np.max(ppvals)
                ax[xx].plot(xscat,yscat, 'v', color=gp_col, lw=1.9, ms=5,  alpha=alpha_pp  )


    
            xvals = np.unique(dat_df.loc['GP'].index.values)
            yvals = dat_df.loc['GP'][gl].groupby('n_train').mean()
            #[dat_df.loc['LIN',ii][gl].values[0] for ii in xvals]
            print(len(xvals),len(yvals))
            ax[xx].plot(xvals,yvals, ':', color=gp_namedcol, lw=3, ms=5,  alpha=0.6, label=lab_lin  )





        ax[xx].set_ylabel(ylabdict[gl], fontsize=18)
        #ax[xx].set_xlabel('# training vectors', fontsize=18)
        #ax[xx].legend(loc='lower right', fontsize=15)
        ax[xx].set_xticks(xvals[::])
        ax[xx].tick_params(axis='both', which='both', labelsize=18, length=5)
        #ax[xx].set_ylim((1e-2,2e-2))
        ax[xx].set_xlim((1,22))


    ########################################################
    #ax[0].set_xlabel('# training vectors', fontsize=18)
    ax[1].set_xlabel('# training vectors', fontsize=18)
    #legend_elements = [mpatches.Patch(facecolor=lin_col, edgecolor='w',
    #                             label='LIN'),
    #                           mpatches.Patch(facecolor=pca_col, edgecolor='w',
    #                             label='PCA') ]

    l1 = Line2D([0,0.2], [0,0],  color=pca_col, linestyle='', linewidth=1.5, marker='s', markersize=10, alpha=0.8)
    l2 = Line2D([0,0.2], [0,0],  color=lin_col, linestyle='', linewidth=1.5, marker='^', markersize=10,  alpha=0.8)
    l3 = Line2D([0,0.2], [0,0],  color=dl_col, linestyle='', linewidth=1.5, marker='o', markersize=10,  alpha=0.8)
    l4 = Line2D([0,0.2], [0,0],  color=gp_col, linestyle='', linewidth=1.5, marker='v', markersize=10,  alpha=0.8)
    l0labs = ["single PCA",'single LIN', 'single DL','single GP']

    leg0 = ax[0].legend([l1,l2,l3,l4], l0labs, loc='center right', fontsize=18, frameon=False)
    leg1 = ax[1].legend([l1,l2,l3,l4], l0labs, loc='center right', fontsize=18, frameon=False)

    ax[0].add_artist(leg0)
    ax[1].add_artist(leg1)

    #lines = ax[0].get_lines()
    #print(len(lines))
    #legend1 = ax[0].legend([lines[::5]], ["mean PCA", "mean LIN"], loc='upper left', fontsize=16,
    #                    markerscale=1.2, frameon = False)
    h1 = Line2D([0,0.4], [0,0],  color=pca_namedcol, linestyle='--', linewidth=3,  alpha=0.6 )
    h2 = Line2D([0,0.4], [0,0],  color=lin_namedcol, linestyle='-.', linewidth=3, alpha=0.6  )
    h3 = Line2D([0,0.4], [0,0],  color=dl_namedcol, linestyle=':', linewidth=3, alpha=0.6  )
    h4 = Line2D([0,0.4], [0,0],  color=gp_namedcol, linestyle=':', linewidth=3, alpha=0.6  )
    mlabs = ["mean PCA",'mean LIN', 'mean DL','mean GP']

    legend01 = ax[1].legend([h1,h2,h3,h4], mlabs, 
                           loc='upper right', fontsize=18, handlelength=3.0, frameon = False)
    ax[1].add_artist(legend01)

    legend02 = ax[0].legend([h1,h2,h3,h4],mlabs, 
                           loc='upper right', fontsize=18, handlelength=3.0, frameon = False)

    ax[0].add_artist(legend02)

    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    #ax[0].add_artist(legend1)

    return plt.show()















def index_sort(datatest_df_dict,emulation_data):
    for noi in (emulation_data.noise_names):
        datatest_df_dict[noi].set_index(['method', 'n_train'], inplace=True)
    for noi in (emulation_data.noise_names):
        datatest_df_dict[noi].sort_index(inplace=True)
    return datatest_df_dict


def test_return_obj(emulation_data, method='PCA', ncomp=None, dl_alpha=None, interp_type='int1d', noisecase='theo',gp_const=10, gp_length=5,Y_noise=None):

    Dop = dcl.LearningOperator(method, ncomp=ncomp, dl_alpha=dl_alpha, interp_type=interp_type,gp_const=gp_const,gp_length=gp_length)
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
                                emulation_data.test_samples)

    intobj_all_dict.print_statistics()

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
    print(intobj.rmse_dict)
    app_dict['single_rmse'] = intobj.rmse_dict[param_val]
    app_dict['single_maxerr'] = intobj.maxerr_dict[param_val]
    app_dict['single_minerr'] = intobj.minerr_dict[param_val]
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

def get_param_array(data_df, parval, quanty):
    resuarr = [[ii, data_df.loc[ii][data_df.loc[ii].parameter_value==parval][quanty].values[0]] \
               for ii in np.unique(data_df.index.values)]
    return np.array(resuarr)



def dataframe_group(df, groupby='', sortby='', filename='', savedir='./savedir/'):
    if type(df) != pd.core.frame.DataFrame:
        print("object passed is not a pandas DataFrame")
        return None
    #print("min value of sortby: ")
    print(df[sortby].min() )
    df_grouped = df.loc[df.groupby(groupby)[sortby].idxmin()]
    df_grouped[['n_train']] = df_grouped[['n_train']].astype(int)
    df_grouped.set_index(['n_train'], inplace=True)

    return df_grouped
