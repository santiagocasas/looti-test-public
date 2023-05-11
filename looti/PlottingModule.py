#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:33:52 2020

@author: raphaelbaena
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib.cm as cm
def Plot_prediction_One_parameter(emulation_data, paramindex = None, predictions = None,
                                   xlabel='k in 1/Mpc',Factor = 1,ratio_mode = True,y_scale_log = False,
                                   plot_training_vectors = False,save_path=None,name_of_plot="pred_temp" ):
    """Plot the prediction for one single parameter 

        Args:
            emulation_data:
            paramvalue_predicted: dictionary of prediction : parameters -> ratios/spectra 
            xlabel: label for y axis
            ratio_mode: if True it plots directly the ratios, otherwise it perfoms a reconstruction of spectra
            Factor: multiply the grid by this factor
            y_scale_log: log scale for y axis

        """
    if emulation_data.masked_k_grid is not None:
        k_grid=np.power(10,emulation_data.masked_k_grid)#*1000)
    else:
        k_grid=emulation_data.lin_k_grid#*1000
    

    ratio_dict={}
    for ts,md in(zip(emulation_data.fullspace,emulation_data.matrix_ratios_dict["theo"])):
        ts=np.array(ts).flatten()
        ratio_dict[tuple(ts)]=md


    # paramvalue = np.atleast_1d(paramvalue)
    # if emulation_data.multiple_z == True:
    #     paramindex = (np.abs(emulation_data.test_samples[:,1:] - paramvalue)).argmin()
    # else:
    #     paramindex = (np.abs(emulation_data.test_samples[:] - paramvalue)).argmin()

    paramvalue =emulation_data.test_samples[paramindex]
    
    

    fig,ax =plt.subplots(2, figsize=(10,8), dpi=200,facecolor='w')

    #ax[0].semilogx(k_grid,ratio_test_dict[tuple(paramvalue)])
    prediction=predictions[tuple(paramvalue)].flatten()
    
    if ratio_mode== True :
        truth = emulation_data.matrix_datalearn_dict["theo"]["test"][paramindex].flatten()
    else:
        index = emulation_data.get_index_param(list(paramvalue ),multiple_redshift=emulation_data.multiple_z)
        truth =  emulation_data.df_ext.loc[index].values.flatten()[emulation_data.mask_true]
    
    
    ax[0].semilogx(k_grid,truth,color ='green', label = 'test data')
    ax[0].semilogx(k_grid,prediction,color ='red',label = 'prediction')
    
    if plot_training_vectors  == True and ratio_mode == True:
        for i,trv in enumerate (emulation_data.train_samples):
            training_vector = emulation_data.matrix_datalearn_dict["theo"]["train"][i].flatten()
            ax[0].semilogx(k_grid,training_vector ,color =cm.Blues(i*50), label = func_label(emulation_data,trv))

    ax[1].set_ylabel("Residuals")
    ax[0].set_xlabel(xlabel)

    ax[0].legend(loc='upper left')
    #plt.plot(kgrid_cod,emu_beta_EXP_codec/powerLCDM_cod)
    #plt.xscale('log')
    residuals = np.abs(1- (prediction/truth))
    label_str = func_label(emulation_data,paramvalue)
    if ratio_mode ==  True:
        title =  "Ratio for " +label_str +f_redshift(emulation_data)
        ax[0].set_ylabel("Ratio")
    else: 
        title =  "Spectra for " +label_str +f_redshift(emulation_data)
        ax[0].set_ylabel("Spectra")  
    ax[1].semilogx(k_grid, residuals, '--v',
               color='purple', lw=1, ms=1, markevery=1,
         alpha=0.8, label=title)
    ax[1].set_xlabel(xlabel)

        
    if y_scale_log == True:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')


    fig.suptitle(title, fontsize=14)
    if save_path!=None:
        plt.savefig(save_path+name_of_plot+'.png')
    return truth,prediction

def f_redshift(emulation_data):
    if emulation_data.train_redshift  == emulation_data.test_redshift :
        if len(emulation_data.train_redshift)>1:
            return ' at redshifts' + str(emulation_data.train_redshift)
        else:
            return ' at redshift ' + str(emulation_data.train_redshift[0])
    else :
        if len(emulation_data.train_redshift)>1:
            string = ' redshifts ' + str(emulation_data.train_redshift) +'for train vectors'
        else:
             string  =  ' redshift ' + str(emulation_data.train_redshift[0]) +'for train vectors'
        if len(emulation_data.test_redshift)>1:
            string += '\n redshifts ' + str(emulation_data.test_redshift) +'for test vectors'
        else:
             string  += '\n redshift ' + str(emulation_data.test_redshift[0]) +'for test vectors'
        return string         
                      
                 


def func_label(emulation_data,paramvalues):
    parnamlist=list(emulation_data.paramnames_dict.values())
    if emulation_data.multiple_z == False:
        string=parnamlist[0]+' = '+'{:.3f}'.format(paramvalues[0])+' '
        ZIP = zip(parnamlist[1:],paramvalues[1:])
    else :
        string=parnamlist[0]+' = '+'{:.3f}'.format(paramvalues[1])+' '
        ZIP = zip(parnamlist[1:],paramvalues[2:])

    for param,paramvalue in (ZIP):
        string += '--'+param+ ' = '+'{:3f}'.format(paramvalue)+' '
    return string

def get_param_array(data_df, parval, quanty):
    resuarr = [[ii, data_df.loc[ii][data_df.loc[ii].parameter_value==parval][quanty].values[0]] \
               for ii in np.unique(data_df.index.values)]
    return np.array(resuarr)

def plot_RMSE(zchoice,datatest_df_dict,noi='theo',turnoff_LIN=True,turnoff_PCA=True,turnoff_GP=True,turnoff_DL=True,
              y_scale_log = False,ratio_mode = True,plot_every = 2):
    """Plot RMSE at given redshift
        Args:
            zchoice: redshift's indice to plot
            datatest_df_dict: frame containing the stats
            noi='theo':
            turnoff_LIN:
            turnoff_PCA:
            turnoff_GP:
            turnoff_DL
            y_scale_log:
            ratio_mode: if True will plot rmse of ratios, otherwise the RMSE of the spectra
            plot_every:
        """
    
    
    
    fig=plt.figure(1, figsize=(20,12), dpi=80,facecolor='w')
    h1=h2=h3=h4=None
    labPCA=labLIN=labDL=labGP=None
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

    dat_df =[dd[noi] for dd in datatest_df_dict]

    ax[0].text(0.5,0.95, '$z=$'+zst, fontsize=12, ha='center', transform=ax[1].transAxes)
    ax[1].text(0.5,0.95, '$z=$'+zst, fontsize=12, ha='center', transform=ax[1].transAxes)

    if ratio_mode == False:
        
        ylabdict = dict(zip(['global_spectra_mean_rmse','global_spectra_mean_maxerr'],
                                     ['RMSE','Max. relative error']))
        ZIP = zip(['single_rmse','single_maxerr'],
                                     ['global_spectra_mean_rmse','global_spectra_mean_maxerr'])
    else :
        ylabdict = dict(zip(['global_mean_rmse','global_mean_maxerr'],
                                     ['RMSE','Max. relative error']))
        ZIP = zip(['single_rmse','single_maxerr'],
                                     ['global_mean_rmse','global_mean_maxerr'])
    for xx, (sn,gl) in enumerate(ZIP ):
        ### *** PCA

        color_iter=iter(color_list)
        c=next(color_iter)

        pca_col = c
        pca_namedcol = c

        if turnoff_PCA==False:
            labPCA = "mean PCA"
            h1 = Line2D([0,0.4], [0,0],  color=pca_namedcol, linestyle='--', linewidth=3,  alpha=0.6 )
            xvals = np.unique(dat_df[0].loc['PCA'].index.values)
            Yvals = np.array([dd.loc['PCA'][gl].groupby('n_train').mean() for dd in dat_df ]).reshape((len(dat_df),len(xvals)))
            yvals = np.mean(Yvals,axis =0)
            ystd = np.std(Yvals,axis =0)/len(xvals)
            #[dat_df.loc['PCA',ii][gl].values[0] for ii in xvals]

            ax[xx].plot(xvals,yvals, '--', color=pca_namedcol, lw=3, ms=5,  alpha=0.6, label=lab_pca  )
           #
           # ax[xx].plot(xvals,yvals+2*ystd , '--', color='red', lw=3, ms=5,  alpha=0.6, label=lab_pca  )
           # ax[xx].plot(xvals,yvals-2*ystd , '--', color='red', lw=3, ms=5,  alpha=0.6, label=lab_pca  )


            #parvals = np.unique(dat_df[0].loc['PCA']['parameter_value'].values)
            #ppvals=parvals


           # for jj,pp in enumerate(parvals):

            #    xy = get_param_array(dat_df.loc['PCA'], pp, sn);

            #    xscat = xy[:,0]###dat_df.loc['PCA'].index.values
            #    yscat = xy[:,1]  ##dat_df.loc['PCA']['single_rmse'].values
            ##    alpha_pp = jj/len(parvals)
            #    print(yscat)
            #    ax[xx].plot(xscat,yscat, 's', color=pca_col, lw=1.9, ms=5,  alpha=0.5,   label=lab_pca)

              #  yinterv = (yvals[0]-yvals.min())/len(parvals)
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
            labLIN = "mean LIN"
            h2 = Line2D([0,0.4], [0,0],  color=lin_namedcol, linestyle='-.', linewidth=3, alpha=0.6  )
            
            xvals = np.unique(dat_df[0].loc['LIN'].index.values)
            Yvals = np.array([dd.loc['LIN'][gl].groupby('n_train').mean() for dd in dat_df ]).reshape((len(dat_df),len(xvals)))
            yvals = np.mean(Yvals,axis =0)
            ystd = np.std(Yvals,axis =0)/len(xvals)
            ax[xx].plot(xvals,yvals, '--', color=lin_namedcol, lw=3, ms=5,  alpha=0.6, label=lab_lin    )
          

        c=next(color_iter)

        dl_col = c
        dl_namedcol= c

        ### *** DL
        if turnoff_DL ==False:
            labDL = "mean DL"
            h3 = Line2D([0,0.4], [0,0],  color=dl_namedcol, linestyle=':', linewidth=3, alpha=0.6  )
            xvals = np.unique(dat_df[0].loc['DL'].index.values)
            Yvals = np.array([dd.loc['DL'][gl].groupby('n_train').mean() for dd in dat_df ]).reshape((len(dat_df),len(xvals)))
            yvals = np.mean(Yvals,axis =0)
            ystd = np.std(Yvals,axis =0)/len(xvals)
            #[dat_df.loc['PCA',ii][gl].values[0] for ii in xvals]

            ax[xx].plot(xvals,yvals, '--', color=dl_namedcol, lw=3, ms=5,  alpha=0.6)

        ### *** GP
        c=next(color_iter)

        gp_col = c
        gp_namedcol= c
        if turnoff_GP==False:
            labGP = "mean GP"
            h4 = Line2D([0,0.4], [0,0],  color=gp_namedcol, linestyle=':', linewidth=3, alpha=0.6  )
            xvals = np.unique(dat_df[0].loc['GP'].index.values)
            Yvals = np.array([dd.loc['GP'][gl].groupby('n_train').mean() for dd in dat_df ]).reshape((len(dat_df),len(xvals)))
            yvals = np.mean(Yvals,axis =0)
            ystd = np.std(Yvals,axis =0)/len(xvals)
            #[dat_df.loc['PCA',ii][gl].values[0] for ii in xvals]

            ax[xx].plot(xvals,yvals, '--', color=gp_namedcol, lw=3, ms=5,  alpha=0.6)
           #
          #  ax[xx].plot(xvals,yvals+2*ystd , '--', color='red', lw=3, ms=5,  alpha=0.6  )
         #   ax[xx].plot(xvals,yvals-2*ystd , '--', color='red', lw=3, ms=5,  alpha=0.6  )



        ax[xx].set_ylabel(ylabdict[gl], fontsize=18)
        #ax[xx].set_xlabel('# training vectors', fontsize=18)
        #ax[xx].legend(loc='lower right', fontsize=15)
        #ax[xx].set_xticks(xvals)
        ax[xx].tick_params(axis='both', which='both', labelsize=18, length=5)
        #ax[xx].set_ylim((1e-2,2e-2))
        ax[xx].set_xlim((xvals.min(),xvals.max()))

    ########################################################
    #ax[0].set_xlabel('# training vectors', fontsize=18)
    ax[1].set_xlabel('# training vectors', fontsize=18)
    #legend_elements = [mpatches.Patch(facecolor=lin_col, edgecolor='w',
    #                             label='LIN'),
    #                           mpatches.Patch(facecolor=pca_col, edgecolor='w',
    #                             label='PCA') ]



    #lines = ax[0].get_lines()
    #print(len(lines))
    #legend1 = ax[0].legend([lines[::5]], ["mean PCA", "mean LIN"], loc='upper left', fontsize=16,
    #                    markerscale=1.2, frameon = False)
    

   
    
    
    
    mlabs = [labPCA,labLIN,labDL,labGP]

    legend01 = ax[1].legend([h1,h2,h3,h4], mlabs,
                           loc='upper right', fontsize=18, handlelength=3.0, frameon = False)
    ax[1].add_artist(legend01)

    legend02 = ax[0].legend([h1,h2,h3,h4],mlabs,
                           loc='upper right', fontsize=18, handlelength=3.0, frameon = False)

    ax[0].add_artist(legend02)

    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    #ax[0].add_artist(legend1)
    
    if y_scale_log == True:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')

    return plt.show()
