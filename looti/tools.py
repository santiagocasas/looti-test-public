from __future__ import print_function
import numpy as np
from scipy import interpolate
import os
import errno
import pickle
# tools module

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)





def takeRatio(d0,d1,logit=False):

    print ('d1'," ",len(d1),'\n')
    print ('d0',' ',len(d0),'\n')
    dif = len(d0)-len(d1)
    d0x=d0[:,0]
    d0y=d0[:,1]
    d1x=d1[:,0]
    d1y=d1[:,1]
    if logit==True:
        d0x=np.log10(d0x)
        d0y=np.log10(d0y)
        d1x=np.log10(d1x)
        d1y=np.log10(d1x)
    if dif > 0:
        ratioDat=np.zeros((len(d1),2))
        fInterp = interpolate.pchip(d0x,d0y)   #!!!Important pchip = monotonic interpolation
        condition=d1x<=max(d0x)
        ratio=d1y[condition]/fInterp(d1x[condition])
        ratioDat=ratioDat[:len(ratio),:]
        ratioDat=np.transpose([d1x[condition],ratio])
        print (str(dif)+' elements deleted')
    elif dif < 0:
        ratioDat=np.zeros((len(d0),2))
        fInterp = interpolate.pchip(d1x,d1y)
        condition=d0x<=max(d1x)
        print (d0x[condition],'\n',d1x)
        ratio=fInterp(d0x[condition])/d0y[condition]
        ratioDat=ratioDat[:len(ratio),:]
        ratioDat=np.transpose([d0x[condition],ratio])
        print (str(dif)+' elements deleted')
    else:
        ratioDat=np.zeros((len(d0),2))
        fInterp = interpolate.pchip(d1x,d1y)
        condition=d0x<=max(d1x)

        ratio=fInterp(d0x[condition])/d0y[condition] #in all cases data d1 is divided through data d0
        ratioDat=ratioDat[:len(ratio),:]
        ratioDat=np.transpose([d0x[condition],ratio])
        print ('equal size')
    #print ratioDat, '\n'
    #print darray1, '\n'
    return ratioDat

def derivatives(x,y,order=1):
    yderi=np.diff(y)/np.diff(x)
    xderi=(x[1:]+x[:-1])/2
    if order==2:
        xderi2, yderi2 = derivatives(xderi,yderi,1)
        return xderi,yderi,xderi2,yderi2
    elif order==1:
        return xderi,yderi
    else:
        print ("error, function does not return "+str(order)+"order derivatives")
        return None

def calcMaxLim(x,y,xmin,maxArra,minArra):
    arra = np.transpose([x,y])
    arra = arra[arra[:,0]>=xmin]
    maxY = max(arra[:,1])
    minY = min(arra[:,1])
    maxArra.append(maxY)
    minArra.append(minY)
    return maxArra, minArra

def mkdirp(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def fileexists(filename):
    #flags =  os.O_WRONLY

    #try:
    file_bool = os.path.isfile(filename)
    #except:

    return file_bool
    #except OSError as e:
    #    if e.errno == errno.EEXIST:  # Failed as the file already exists.
    #        return True
    #    else:  # Something unexpected went wrong so reraise the exception.
    #        raise
    #else:
    #    return False


def condprint(*str, **kwargs):
    if kwargs['verbosity'] >= kwargs['level']:
        print(str)


def factors(n):
    return set(x for tup in ([i, n//i]
                for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)   ##from stackexchange

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def set_z_space(zspace):
    oldzkeys=['z'+'{:d}'.format(ii) for ii in range(len(zspace))]
    newzkeys=['z'+'{:02d}'.format(ii) for ii in range(len(zspace))]
    replace_zdict=dict(zip(oldzkeys,newzkeys))
    #replace_zdict
    znumstr=['{:.2f}'.format(ii) for ii in zspace]

    zlabel_to_key = dict(zip(znumstr,newzkeys))
    key_to_zlabel = {v: k for k, v in zlabel_to_key.items()}
    znum_to_key = dict(zip(zspace,newzkeys))
    return [oldzkeys, newzkeys, zlabel_to_key, key_to_zlabel, znum_to_key, replace_zdict]
    #key_to_znum


def zvalue_to_key(zz, zspace, znum_dict):
    zkey=znum_dict[find_nearest(zspace, zz)]
    return zkey


def root_mean_sq_err(obs_vec, true_vec, formatted=False, digits=3):
    if len(obs_vec)!=len(true_vec):
        print("Observed vector and true vector do not have same length")
        return None
    nn = len(obs_vec)
    mse  = (1/nn) * np.sum((obs_vec-true_vec)**2/(true_vec)**2)
    rmse = np.sqrt(mse)
    if formatted:
        return f'{rmse:.{digits}e}'
    else:
        return rmse


def minmax_abs_err(obs_vec, true_vec, formatted=False, digits=2, percentage=True):
    if len(obs_vec)!=len(true_vec):
        print("Observed vector and true vector do not have same length")
        return None
    rat  = (obs_vec-true_vec)/true_vec
    factor = 1
    if(percentage):
        factor=100
    mini = np.min(np.abs(rat))*factor
    maxi = np.max(np.abs(rat))*factor
    if formatted:
        return f'{mini:.{digits}e}'+', '+f'{maxi:.{digits}f}'
    else:
        return (mini,maxi)
