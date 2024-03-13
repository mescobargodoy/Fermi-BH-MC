import numpy as np
from simple_mc_funcs import SB_factor

"""
Set of functions for the likelihood of a photon's traversed
path in a detector drawn from an exponentially decaying 
distribution.

"""

def L_exp(x,cs=4.0e-23,n=1.045e22):
    """
    Likelihood function of particles' traversed path
    drawn from an exponential distribution.
    
    It is the user's responsibility to make sure
    units work out.

    User can give an array for cs as input of 
    different size than x. This way now L_exp
    is a function of cs.

    Parameters
    ----------
    x : array
        Traversed length in detector of some number
        density n.
        Any value of x that is zero won't be considered.
    cs : float
        Cross section of process that determines 
        mean free path.
        Defaults to 4.0e-23 [cm^2] CsI pair prod. cross section.
    n : float
        Number density of material
        Defaults to 1.045e22 [atom/cm^3].
    
    Returns
    ----------
    likelihood :  array
        Likelihood function exp(-x/(n*cs))
        of size cs.
    """

    # Zeros are not taken as input
    x=x[x!=0]

    # If varying cs, using array broadcasting
    if type(cs)!=float:
        if len(cs)>1:
            cs=cs[:,np.newaxis]
            x=np.broadcast_to(x,(len(cs),len(x)))
            Lambda=1/(n*cs) # Mean free path
            likelihood=np.prod((np.exp(-x/Lambda)/Lambda),axis=1)
            return likelihood
        
    Lambda=1/(n*cs) # Mean free path
    likelihood=np.prod((np.exp(-x/Lambda)/Lambda))

    return likelihood

def L_LIV(x,E,E_LIV=1.22e16,cs=4.0e-23,n=1.045e22,n_LIV=-1):
    """
    
    Likelihood function of particles' traversed path
    drawn from an exponential distribution assuming 
    Lorentz invariance violation as proposed in
    Vankov and Stanev 2002.

    Parameters
    ----------
    x : array
        Traversed length in detector of some number
        density n.
        Any value of x that is zero won't be considered.
    E : float
        energy of gamma ray
    E_LIV : float, optional 
        LIV energy scale, defaults to Planck scale [TeV].
    cs : float, optional
        Cross section of process that determines 
        mean free path.
        Defaults to 4.0e-23 [cm^2] CsI pair prod. cross section. 
    n : float, optional
        Number density of material.
        Defaults to 1.045e22 [atom/cm^3]
    n_LIV : integer, optional
        order of the LIV effect, can be +-integers.
        Defaults to -1 sub-luminal case.

    Returns
    ----------
    likelihood : array
        Likelihood function exp(-x/(n*cs*LIV_factor(E,E_LIV,n_LIV)))
        of size x assuming there are no zeros on x.
    """

    x=x[x!=0]
    E=E[x!=0]
    LIV_factor=SB_factor(E=E,n=n_LIV,E_LIV=E_LIV)

    # If varying cs, using array broadcasting
    if type(cs)!=float:
        if len(cs)>1:
            cs=cs[:,np.newaxis]
            x=np.broadcast_to(x,(len(cs),len(x)))
            Lambda=1/(n*cs*LIV_factor) # Mean free path LIV included
            likelihood=np.prod(np.exp(-x/Lambda)/Lambda,axis=1)
            return likelihood

    elif type(E_LIV)!=float:
        if len(E_LIV)>1:
            E_LIV=E_LIV[:,np.newaxis]
            x=np.broadcast_to(x,(len(E_LIV),len(x)))
            Lambda=1/(n*cs*LIV_factor) # Mean free path LIV included
            likelihood=np.prod(np.exp(-x/Lambda)/Lambda,axis=1)
            return likelihood

    Lambda=1/(n*cs*LIV_factor) # Mean free path LIV included
    likelihood=np.prod(np.exp(-x/Lambda)/Lambda)

    return likelihood