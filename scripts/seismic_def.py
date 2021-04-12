# Contains definitions to read seismic files
import numpy as np


def read_tensor(filename, dtype, dshape):
    '''
    reads the given tensor data from the filename 
    '''
    data = np.fromfile(filename, dtype=dtype)
    data = np.reshape(data, dshape)
    return data

def e_lami(E, nu):
    '''
    Changes modulus elastic to lami constants
    '''
    mu = 0.5*E/(1.0+nu)
    lam = E*nu/((1.0+nu)*(1.0-2.0*nu))
    
    return lam, mu

def v_lami(Cp, Cs, scalar_rho):
    '''
    Change velicity modulus to lami constants
    '''
    scalar_mu = Cs*Cs*scalar_rho
    scalar_lam = Cp*Cp*scalar_rho - 2.0*scalar_mu
    return scalar_lam, scalar_mu

def w_vel(lam, mu, rho):
    '''
    Get wave velocities from lami parameters
    '''
    Cp = np.sqrt((lam+2*mu)/rho)
    Cs = np.sqrt(mu/rho)
    return Cp, Cs
    