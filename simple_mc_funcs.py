import numpy as np
import astropy.units as u
"""
Set of functions required to generate the energy and
mean free path distributions of events that would be 
captured by Fermi.     
"""

def powlaw_inversecdf(u,E_0=5.,gamma=2.5,E_max=0.):

    """
    Power law inverse cumulative distribution function.
    Generates pseudo-random data following power law.

    :param u: unform distribution [0,1]
    :type u: float or numpy array of floats
    
    :param E_0: minimium energy value allowed
    :type E_0: float [TeV]

    :param gamma: power law index, defaults to 2.5
    :type gamma: float, optional
    """
    assert gamma>1, "gamma can only take values > 1"
    assert E_0>0, "E_0 can only takes values > 0"
    photon = (E_0*(1-u)**(1/(1-gamma)))

    # Re-sampling event if it is past upper bound
    if E_max!=0.:
        # Generating new large uniform distribution to sample from
        newudist = np.random.uniform(0,1,10000)
        # Indices corresponding to events greater than E_max
        indices = np.where(photon>E_max)[0]
        # Looping over indices
        for i in indices:
            # Sampling event until it is less than E_max
            while photon[i] > E_max:
                newu = np.random.choice(newudist)
                photon[i] = (E_0*(1-newu)**(1/(1-gamma)))
        return photon
    
    else:
        return photon


def exp_inversecdf(u, meanfreepath=2.39):
    """
    Exponential inverse cumulative distribution function.
    Generates pseudo-random data following exponential
    distribution.

    CsI lambda = 2.39 cm
    
    :param u: unform distribution [0,1]
    :type E: float

    :param meanfreepath: mean free path as given by cross section
    :type meanfreepath: float

    """

    return (-meanfreepath*np.log(1-u))

def bh_cs_ultra(Z=108.0):
    """
    Returns the pair production or Bethe-Heitler cross section in the 
    ultra-relativistic limit as presented in F. A. Ahronia - Astrophysics 
    at High Energies 2013. 
    Units are [atom/cm^2].
    N_2 Z = 14
    Tungsten Z = 74
    CsI Z = 108

    :param Z: Atomic number, defaults to 7
    :type Z: float, optional
    """
    alpha = 1/137      # unitless 
    r_e = 5.291772e-9 # atomic radius centimeters
    return ((28/9)*alpha*(r_e**2)*Z*(Z+1)*np.log(183*Z**(-1/3))/(1+0.12*(Z/82)**2))

def bhcs_mean_path(n=1.05e22, cs = bh_cs_ultra, Z=108.0):
    """
    Returns the mean free path of a gamma ray traversing some material
    of number density n in units of centimeters.

    air n = 0.02504e21 cm^-3
    tungsten n = 6.31e22 cm^-3
    CsI n  = 1.05e22 cm^-3

    :param n: number density [atoms/cm^3], default is air 
    :type n: float

    :param cs: returns pair production cross section
    :type cs    : function
    """
    sigma = cs(Z)
    mean_free_path = 1/(n*sigma)
    
    return mean_free_path

def SB_factor(E, n=-1, E_LIV=1.22e16):
    """
    Returns suppresion/boost factor of pair production 
    cross section due to LIV as presented by 
    Vankov and Stanev 2002.

    n=1 sub-luminal case
    n=-1 super-luminal case
    
    Parameters
    ----------

    E : float
    energy of primary, should be units of TeV
    
    E_qg : float 
        LIV energy scale, defaults to Planck scale [TeV]
    
    n : integer 
        order of the LIV effect, can be +-integers
    """
    m_e = 511e-9 # TeV
    factor = 1+n*(E**3)/(4*E_LIV*m_e**2)
    return 1/factor

def histogramlogspacing(powerlawdist, measuredpathdist, bin_number=-1, min=10, max=-1):
    """
    Returns histogram (bins and counts) with logarithmic bin spacing as well as the binned average traversed
    path and statistical uncertainty to each bin.

    Parameters
    ----------
    powerlawdist : numpy array
        Distribution of events following power law
    bin_number : int,
        The number of bins to have.
        Will default to square root of length of powerlawdist
    min : int, 5
        Minimum energy for bin edge, by default 5
    max : int, optional
        Maximum energy for bin edge.
        Will default to max+1 of powerlawdist.
    """
    assert bin_number ==-1 or bin_number>0, "Use valid bin number"

    if bin_number==-1:
        bin_number = int(np.ceil(np.sqrt(len(powerlawdist)))) 

    if max==-1:
        max = np.max(powerlawdist)+1
    # Logarithmic binning, histogram and bin indices
    bins = np.logspace(np.log10(min), np.log10(max), num=bin_number)
    histo = np.histogram(powerlawdist,bins=bins)
    indices = np.digitize(powerlawdist,histo[1])

    # Initializing arrays to store traversed path and error on the mean
    traversedpathaverageperbin = np.zeros(shape=len(histo[0]))
    traversedpatherrorperbin = np.zeros(shape=len(histo[0]))
    traversedsinglepathmeasurements = []

    # Reducing array of bin indices to array without repeats
    reduced_indeces = np.unique(indices)

    for arrindex,element in enumerate(reduced_indeces):
        mask = np.where(element==indices)
        # Grouping measured average path in bin to do whatever we want with them
        binnedlambdas = measuredpathdist[mask]
        # Average lambda per bin
        binnedaveragelambda = np.mean(binnedlambdas)
        # RMS error per bin
        rmse=np.sqrt(np.sum(((binnedlambdas-binnedaveragelambda)**2))/(len(binnedlambdas)-1))
        normedrmse=rmse/np.sqrt(len(binnedlambdas))
            
        traversedpathaverageperbin[arrindex]=binnedaveragelambda
        traversedpatherrorperbin[arrindex]=normedrmse
        traversedsinglepathmeasurements.append(binnedlambdas)

    return histo, traversedpathaverageperbin, traversedpatherrorperbin, traversedsinglepathmeasurements

def filltraversedpathbins(E1,E2,binsizes,meanpathdist,binspacing=1):
    """
    Returns arrays of evenly spaced energy, mean traversed path and 
    traversed path RMS error.
    
    Energy array runs from E1 to E2.
    Traversed path array entries will be determined by sampling
    numbers from meanpathdist. The amount of numbers sampled
    per bin is set by binsize.  

    Parameters
    ----------
    E1 :float
        Start of energy bins.
    E2 : float
        End of energy bins. Might not be included depending
        on spacing of bins
    binsize : int or list
        Number of events per energy bin.
    meanpathdist : array
        Decaying exponential distrbution
        with mean free path lambda
    spacing : int, optional
        Spacing between energy bins not necessarily
        integer spacing,  by default 1
    """
    energybins = np.arange(E1,E2,binspacing)
    binsizes = np.array(binsizes)
    # Each element of list below will be an array
    # of size energybins containing average traversed
    # path per bin.
    traversedpathdists = []
    traversedpatherrors = []

    for bins in binsizes:
        # Initialize array to store average paths and rmse
        x=np.zeros(shape=len(energybins))
        xerr=np.zeros(shape=len(energybins))
        
        # Iterating through every element in x.
        for k,bin in enumerate(x):
            i=0
            w=0
            xerrbin = np.zeros(shape=int(bins))
            # Sampling events from distribution
            while i<bins:
                t=np.random.choice(meanpathdist) # random sampled path
                w+=t                             # adding paths
                xerrbin[i]=t                     # storing sampled path
                i+=1                             # increasing iterator
            # Averaging traversed path per bin
            #bin=w/bins
            x[k]=w/bins
            # RMS error
            rms=np.sqrt(np.sum(((xerrbin-np.mean(xerrbin))**2))/len(xerrbin))
            normedrms = rms/np.sqrt(len(xerrbin))
            xerr[k]=normedrms

        traversedpathdists.append(x)
        traversedpatherrors.append(xerr)

    traversedpathdists = np.array(traversedpathdists)
    traversedpatherrors = np.array(traversedpatherrors)

    return energybins,traversedpathdists,traversedpatherrors
