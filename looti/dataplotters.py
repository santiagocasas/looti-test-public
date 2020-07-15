import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
from future.utils import viewitems

def plotCompareDataNext(xarra, array1, array2, logplot=True):
    colorr=iter(cm.rainbow(np.linspace(0,1,len(array1))))
    myplo = plt.loglog
    if logplot==False:
        myplo=plt.plot
    for recopk, truepk in zip(array2, array1):
        c=next(colorr)
        myplo(xarra, recopk, c=c, ls=':')
        myplo(xarra, truepk, c=c, ls='-.')
        plt.ylabel("P(k)")
        plt.xlabel("log10(k)")

def plotRatioDataNext(xarra, array1, array2, logplot=False):
    colorr=iter(cm.rainbow(np.linspace(0,1,len(array1))))
    myplo = plt.plot
    if logplot==True:
        myplo=plt.loglog
    for recopk, truepk in zip(array2, array1):
        c=next(colorr)
        myplo(xarra, recopk/truepk, c=c, ls=':')

def plotCompareDataMissing(xarra, fulldata, traindatadict, fullkeylist):
    colorr=iter(cm.rainbow(np.linspace(0,1,len(fulldata))))
    for zi,pkz in enumerate(fulldata):
        c=next(colorr)
        plt.plot(xarra, pkz, c=c, ls='-.', alpha=0.6)
        for (zk, pkd) in viewitems(traindatadict):
            if zk==fullkeylist[zi]:
                plt.plot(xarra, pkd, c=c, ls=':', label=str(zk))
    plt.xlim(-3, 1)
    plt.legend()
    plt.ylabel("P(k)")
    plt.xlabel("log10(k)")


def plotComponents(matr, xarray='', transpose=False, iterlength='', logplot=True):
    if iterlength=='':
        itl=len(matr)
    else:
        itl=iterlength
    colorr=iter(cm.rainbow(np.linspace(0,1,itl)))
    fig, ax = plt.subplots()
    if xarray=='':
        xarray = np.arange(matr.shape[1])
    datamatr=matr
    if transpose==True:
        datamatr=(np.transpose(matr)[0:,:])
    for rep in (datamatr):
        c=next(colorr)
        if logplot==True:
            ax.semilogy(xarray, np.abs(rep), ls='-', marker='o',c=c)
        else:
            ax.plot(xarray, np.abs(rep), ls='-', marker='o',c=c)
    if transpose==False:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def plotInterpols(xvals_train, xvals_test, yMat, repIntFuncs, sliceind=[0,-1], coefflen=None):
    if coefflen==None:
        lencoeffs=yMat.shape[1]
    else:
        lencoeffs=coefflen
    colorr=iter(cm.rainbow(np.linspace(0,1,lencoeffs)))
    ini=sliceind[0]
    fini=sliceind[1]
    for ii, pkrep in enumerate(yMat[ini:fini,:]):
        c=next(colorr)
        #plt.semilogy(xvals_train, np.abs(pkrep), "o",c=c, alpha=0.4, label="train points")
        #plt.semilogy(xvals_test,np.abs(repIntFuncs[ii](xvals_test)), c=c, ls='-', label="interpolation")
        maxmin = np.max(pkrep)-np.min(pkrep)  ## maxmin used to normalize the curves
        if maxmin == 0: maxmin=1
        plt.plot(xvals_train, (pkrep)/maxmin, "o",c=c, alpha=0.4, label="train points")
        plt.plot(xvals_test,(repIntFuncs[ii+ini](xvals_test))/maxmin, c=c, ls='-', label="interpolation")
        plt.legend(bbox_to_anchor=(1.05, 1.05), ncol=int(lencoeffs/10)+1)

def plotRatioOfReconstructedData(xgrid,valispace,reconDatdict,fullDatdict,plottitle="PCA interpol at",
                                 secondDatdict=None, xlim=(None,None), ylim=(None,None), label1='method1', label2='method2',
                                 array_mask=np.array([]), array_mask2 = np.array([])):
    #f, axarr = plt.subplots(len(xvalsmiss), sharex=True)
    fig = plt.figure(figsize=(18,10))
    sublen=len(valispace)
    if sublen==1:
        sublen=2
    axes= fig.subplots(sublen, sharex=True)
    #axes=[ax1,ax2,ax3]
    colorr=iter(cm.rainbow(np.linspace(0,1,len(valispace)+1 )))
    xxgrid = np.power(10, xgrid)
    #print('0', xxgrid.shape)
    for pi,vi in enumerate(valispace):
        c=next(colorr)
        fullData = fullDatdict[vi]
        #print('1',fullData.shape)
        if array_mask.any() != False:
            fullData = fullData[array_mask]
            #print('2', fullData.shape)
        plarra1 = 100*(reconDatdict[vi]-fullData)/fullData
        axes[pi].semilogx(xxgrid, plarra1, c=c, ls='-.', label=label1)
        if secondDatdict is not None:
            secondData = secondDatdict[vi]
            if array_mask.any() != False:
                secondData = secondData[array_mask2]
            plarra2 = 100*(secondData-fullData)/fullData
            axes[pi].semilogx(xxgrid, plarra2, c='k', ls=':', alpha=0.6, label=label2)
        #axes[pi].set_ylabel("% difference")
        axes[pi].set_title(plottitle+" par="+str('{:.2f}'.format(vi)))
        axes[pi].legend()
        axes[pi].set_xlim(xlim)
        axes[pi].set_ylim(ylim)
        axes[pi].relim()
        axes[pi].autoscale_view(True, True, True)
        #pi=pi+1
    fig.text(-0.1, 0.5, '% diff of reconstructed to true data', ha='center', va='center', rotation='vertical')
    axes[-1].set_xlabel("k in h/Mpc")

def plotPkCompaBig(traindat,learndat,xgrid,validata=None,ylabel='OT',xlabel='k scale',alleps=np.arange(1,5),
                   trainingeps=[],validateps=[],valilablpre='z=',trainlablpre='z=',
                  xmini=0,xmaxi=1,ymini=0,ymaxi=1):

    G = gridspec.GridSpec(4,4)
    fig=plt.figure(1, figsize=(20,12), dpi=80,facecolor='w')
    collist=['b','g','r','c','Indigo','Olive','OrangeRed','SkyBlue','Pink', 'violet']
    col2=50*['skyblue', 'violet']
    markers=50*['-p', '-v', '-o','-s']

    axes2 = fig.add_subplot(G[:2,:])

    for (ii,orig) in enumerate(traindat):
        axes2.plot(xgrid,orig, color=col2[ii], lw=4, ms=5, markevery=1, alpha=1.0,
                   label=trainlablpre+str(trainingeps[ii]))
    for (jj,lu) in enumerate(learndat):
        axes2.plot(xgrid,lu, color='Orange', lw=1+0.6*jj, ms=5, markevery=1, alpha=0.2)
#plot curve not used for OT
    if validata is not None:
        for (kk,vd) in enumerate(validata):
            axes2.plot(xgrid,vd, markers[kk], label=valilablpre+str(validateps[kk]),
                       color='r', lw=2, ms=6, markevery=3, alpha=0.8)


    xminidf,xmaxidf,yminidf,ymaxidf = axes2.axis('tight')
    axes2.axis(ymax=ymaxidf*1.1, xmin=xminidf*0.9, xmax=xmaxidf*1.1, ymin=yminidf)
    axes2.legend(loc='best',markerscale=1.5,prop={'size':12},numpoints=2,handlelength=2, ncol=3)
    axes2.grid(True,which="major",ls=":")
    axes2.tick_params(which='both',length=6, width=1, labeltop=False)
    axes2.set_ylabel(ylabel,size='x-large')
    axes2.set_xlabel(xlabel,size='x-large')
    #axes1.xaxis.set_minor_locator(minorLocator)
    #minorLocator   = ticker.LogLocator(0.1,[0.01,0.02])
    #minorLocator   = ticker.LogLocator(0.1,[0.01,0.02])
    #axes2.xaxis.set_minor_locator(minorLocator)

    axes4 = fig.add_subplot(G[2:,:])
    for (ii,orig) in enumerate(traindat):
        axes4.plot(xgrid,orig, color=col2[ii], lw=4, ms=5, markevery=1, alpha=1.0,
                   label=trainlablpre+str(trainingeps[ii]))
    for (jj,lu) in enumerate(learndat):
        axes4.plot(xgrid,lu, color='Orange', lw=2, ls='-.' ,ms=5, markevery=1, alpha=0.6)
#plot curve not used for OT
    if validata is not None:
        for (kk,vd) in enumerate(validata):
            axes4.plot(xgrid,vd, markers[kk], label=valilablpre+str(validateps[kk]),
                       color='r', lw=2, ms=6, markevery=3, alpha=0.8)


    axes4.axis(ymax=ymaxi, xmin=xmini, xmax=xmaxi, ymin=ymini)
    axes4.legend(loc='best',markerscale=1.5,prop={'size':12},numpoints=2,handlelength=2, ncol=3)
    axes4.grid(True,which="major",ls=":")
    axes4.tick_params(which='both',length=6, width=1, labeltop=False)
    axes4.set_ylabel("Zoom-in "+ylabel,size='x-large')
    #axes1.xaxis.set_minor_locator(minorLocator)
    axes4.set_xlabel(xlabel,size='x-large')
    #minorLocator   = ticker.LogLocator(0.1,[0.01,0.02])
    #minorLocator   = ticker.LogLocator(0.1,[0.01,0.02])
    #axes4.xaxis.set_minor_locator(minorLocator)

    plt.show()
