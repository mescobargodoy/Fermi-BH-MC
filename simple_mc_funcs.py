"""
Set of functions required to generate the energy and
mean free path distributions of events that would be 
captured by Fermi.     
"""

def powerlaw_prob(E,E_0=0.001,gamma=2.5):
    """
    Power law probability

    :param E: Energy in units of your choosing
    :type E: float
    
    :param E_0: minimium energy value allowed for our case, 
                defaults to 0.001
    :type E_0: float

    :param gamma: power law index, defaults to 2.5
    :type gamma: float, optional
    """
    assert gamma>0, "gamma can only take values > 0"
    assert E_0>0, "E_0 can only take values >0"
    gamma=-gamma
    return (E/E_0)**(gamma)

def powerlaw_cum(E, E_0=0.001, gamma=2.5):

    """
    Power law cumulative distribution function

    :param E: Energy in units of your choosing
    :type E: float
    
    :param E_0: minimium energy value allowed for our case, 
                defaults to 0.001
    :type E_0: float

    :param gamma: power law index, defaults to 2.5
    :type gamma: float, optional
    """
    assert gamma>0, "gamma can only take values > 0"
    gamma = -gamma
    return (E**(gamma+1))*(E_0**gamma)/(gamma+1)


def powlaw_inversecdf(E, E_0=0.001, gamma=2.5):

    """
    Power law inverse cumulative distribution function

    :param E: Energy in units of your choosing
    :type E: float
    
    :param E_0: minimium energy value allowed for our case, 
                defaults to 0.001
    :type E_0: float

    :param gamma: power law index, defaults to 2.5
    :type gamma: float, optional
    """
    assert gamma>0, "gamma can only take values > 0"
    gamma=-gamma
    return (((E*(gamma+1))/E_0**(gamma))**(1/(gamma+1)))
