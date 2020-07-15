import numpy as np
import theano
import theano.tensor as T
import time

#############
# UTILITIES #
#############


def simpleCost(xgrid):
    N = len(xgrid)
    C = np.zeros((N,N))
    for i in range(N):
        #for j in range(N):
        for j in range(i):
            #val = (np.log10(xgrid[i]) - np.log10(xgrid[j]))**2
            val = (xgrid[i] - xgrid[j])**2
            #val = (i*1.2 - j*1.1)**2
            C[i,j] = val
            C[j,i] = val
    return C





def EuclidCost(Nr, Nc, divmed=False, timeit=False, trunc=False, maxtol=745.13, truncval=745.13):
    if timeit:
        start = time.time()
    N = Nr * Nc
    C = np.zeros((N,N))
    for k1 in range(N):
        for k2 in range(k1):
            r1, r2 = int(float(k1) / Nc)+1, int(float(k2) / Nc)+1
            c1, c2 = k1%Nc + 1, k2%Nc + 1
            C[k1, k2] = (r1-r2)**2 + (c1-c2)**2
            C[k2, k1] = C[k1, k2]
    if timeit:
        print('cost matrix computed in '+str(time.time()-start)+'s.')
    if divmed:
        C /= np.median(C)
    if trunc:
        C[C>maxtol] = truncval
    return C

def LogCost(kv, divmed=False, timeit=False, trunc=False):
    if timeit:
        start = time.time()
    N = len(kv)
    C = np.zeros((N,N))
    for k1 in range(N):
        for k2 in range(k1):
            val = (np.log10(kv[k1])-np.log10(kv[k2]))**2
            C[k1, k2] = val
            C[k2, k1] = C[k1, k2]
    if timeit:
        print('cost matrix computed in '+str(time.time()-start)+'s.')
    return C

def alphatolbda(alpha):
    return (np.exp(alpha).T / np.sum(np.exp(alpha), axis=1)).T

##############
### THEANO ###
##############
# define Theano variables
Datapoint = T.vector('Datapoint')
Cost = T.matrix('Cost')
Gamma = T.scalar('Gamma')
Ker = T.exp(-Cost/Gamma)
n_iter = T.iscalar('n_iter')
Tau = T.scalar('Tau')
Rho = T.scalar('Rho')

# variable change (for simplex constraint)
def varchange(newvar):
    return T.exp(newvar)/T.sum(T.exp(newvar))

# define weights and dictionary
Newvar_lbda = T.vector('Newvar_lbda')
Newvar_D = T.matrix('Newvar_D')
lbda = varchange(Newvar_lbda)
D, D_varchange_updates = theano.scan(varchange, sequences=[Newvar_D.T])
D = D.T
theano_varchange = theano.function([Newvar_D], D)

## Logdomain version ##
logD = T.log(D)
Epsilon = T.scalar('Epsilon')

# Stabilized kernel computation
def StabKer(Cost, alpha, beta, Gamma):
    M = -Cost.dimshuffle(0,1,'x') + alpha.dimshuffle(0,'x',1) + beta.dimshuffle('x',0,1)
    M = T.exp(M / Gamma)
    return M

# Log Sinkhorn iteration
def log_sinkhorn_step(alpha, beta, logp, logD, lbda, Gamma, Cost, Tau, Epsilon):
    M = StabKer(Cost,alpha,beta,Gamma)
    newalpha = Gamma * (logD - T.log(T.sum(M,axis=1) + Epsilon)) + alpha
    alpha = Tau*alpha + (1.-Tau)*newalpha
    M = StabKer(Cost,alpha,beta,Gamma)
    lKta = T.log(T.sum(M, axis=0) + Epsilon) - beta/Gamma
    logp = T.sum(lbda*lKta, axis=1)
    newbeta = Gamma * (logp.dimshuffle(0,'x') - lKta)
    beta = Tau*beta + (1.-Tau)*newbeta
    return alpha, beta, logp

# Log Sinkhorn algorithm
log_result, log_updates = theano.scan(log_sinkhorn_step, outputs_info=[T.zeros_like(logD),
                                      T.zeros_like(logD), T.ones_like(logD[:,0])],
                                      non_sequences=[logD,lbda,Gamma,Cost,Tau,Epsilon],
                                      n_steps=n_iter)

# keep only final barycenter
log_bary = T.exp(log_result[2][-1])

# Log Theano barycenter function
log_Theano_wass_bary = theano.function([D,lbda,Gamma,Cost,n_iter,theano.In(Tau,value=0),
                                        theano.In(Epsilon,value=1e-200)],
                                        log_bary)
