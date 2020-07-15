from __future__ import print_function
import numpy as np
import sys
import copy
import scipy.interpolate as interpol

interpmethods = {
            "int1d" : lambda x,z : interpol.interp1d(x,z, kind="slinear"),
            "bar1d" : lambda x,z : interpol.BarycentricInterpolator(x,z),
            "lag1d" : lambda x,z: interpol.lagrange(x,z),
            "spl1d" : lambda x,z: interpol.UnivariateSpline(x,z, k=2, s=0.01),
         }

class Interpolators:

    def __init__(self, method, dim=1, interp_opts=dict()):
        self.method = method
        self.dim = dim
        self.interp_opts=interp_opts

    def __call__(self, xvals, zMat, yvals=None):
        if self.dim==1:
            intpFuncs = self.interpolCoeffs_1D(xvals, zMat, self.method, **self.interp_opts)
        elif self.dim==2:
            intpFuncs = self.interpolCoeffs_2D(xvals, zMat, **self.interp_opts)
        return intpFuncs
    @staticmethod
    def interpolCoeffs_1D(xvals, zMat, method, **kwargs):
        repIntFuncs=[]
        if method=='int1d':
            intpfunc = lambda x,z : interpol.interp1d(x,z, **kwargs)
        elif method=='intspl1d':
            intpfunc = lambda x,z : interpol.InterpolatedUnivariateSpline(x,z, **kwargs)
        elif method=='spl1d':
            intpfunc = lambda x,z : interpol.UnivariateSpline(x,z, **kwargs)

        #intpfunc = interpmethods.get(method, interpmethods['int1d']) ##use int1d if key not found in methods
        for zrep in zMat:
            ddi=intpfunc(xvals,zrep)
            repIntFuncs.append(ddi)
        return(repIntFuncs)

    @staticmethod
    def interpolCoeffs_2D(vals, zMat, **kwargs):
     
        xvals,yvals =vals[:,0],vals[:,1]
        intpfunc = lambda x,y,z : interpol.interp2d(x, y, z, kind='cubic')
        repIntFuncs=[]
        for zrep in zMat:
            ddi=intpfunc(xvals, yvals, zMat)#interpol.SmoothBivariateSpline(x,y,zrep, **kwargs)
            repIntFuncs.append(ddi)
        return(repIntFuncs)
