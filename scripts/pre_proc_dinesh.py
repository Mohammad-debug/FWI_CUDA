# preprocessing of the files
import numpy as np
from seismic_def import read_tensor
import matplotlib.pyplot as plt

ftype = np.float64

len = 10.0+12.0+3.0+12.0+3 # Meters
dep = 0.5+4.0+0.5 # Depth


# Getting the input directly in this preprocessor file
# Geometric data
dt = 0.1e-3; dz = 0.1; dx = 0.1 # grid intervals
nt = 3000; nz = np.int(dep/dz)+21; nx = np.int(len/dx)+21 # grid numbers (adding for PMLs as well)

# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 10; npml_bottom = 10; npml_left = 10; npml_right = 10

# Surface grid index in each direction (0 = no surface)
surf = False # surface exists
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0

snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = np.int32(10 + 2.5/dz); snap_z2 = np.int32(10 + 5.5/dz); snap_x1 = np.int32(10 + 8.0/dx); snap_x2 = np.int32(10 + 26.0/dx) # snap boundaries
snap_dt = 1; snap_dz = 1; snap_dx = 1; # the snap intervals

#nshot = 1 #; nsrc = 3; nrec = 10; 
stf_type = 1; rtf_type = 0
fdorder = 2; fpad = 1


#Boolen values
fwinv = False
if (fwinv):
    accu_save = False; seismo_save=True
    mat_save_interval = 1; rtf_meas_true = True # RTF field measurement exists
else:
    accu_save = True; seismo_save=True
    mat_save_interval = -1; rtf_meas_true = False # RTF field measurement exists
    
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
    
# Adding material properties for different materials
rho_air = 1.25
lam_air, mu_air = v_lami(300, 0.5, rho_air)

rho_water = 1000.0
lam_water, mu_water = v_lami(1850, 475, rho_water)

rho_sub = 1800.0
lam_sub, mu_sub = e_lami(181.0e+6, 0.4)

rho_sand = 1700.0
lam_sand, mu_sand = e_lami(315.0e+6, 0.33)

rho_sand_wet = 1830.0
lam_sand_wet = lam_sand*1.1
mu_sand_wet = mu_sand*1.1

# --------------------------------------------
# scalar material variables
Cp = 2000.0
Cs = 700.0
scalar_rho = 0.5
scalar_mu = 0.5
scalar_lam = 0.5
mat_grid = 1 # 0 for scalar and 1 for grid
# --------------------------------------------

# --------------------------------------------------
# preparing  the starting material arrays (Fill with Air)
lam = np.full((nz, nx), lam_air)
mu = np.full((nz, nx), mu_air)
rho = np.full((nz, nx), rho_air)

# add for dam sand layer

for iz in range(0, nz):
    for ix in range(0, nx):
        # for sand dam
        if (iz>np.int(10+1.0/dz)): # top boundary 
            if (iz - np.int(10+1.0/dz) > 0.33*(ix - np.int(10+25.0/dx))):
                if (iz - np.int(10+1.0/dz) > -0.33*(ix - np.int(10+22.0/dx))):
                    lam[iz][ix] = lam_sand
                    mu[iz][ix] = mu_sand
                    rho[iz][ix] = rho_sand
                    
                    # Additional modification for original layering
                    if (fwinv==False):
                        if (iz - np.int(10+4.0/dz) > 0.2*(ix - np.int(10+14.0/dx))):
                            lam[iz][ix] = lam_sand_wet
                            mu[iz][ix] = mu_sand_wet
                            rho[iz][ix] = rho_sand_wet
                    
        if (iz>np.int(10+2.0/dz)): # water level boundary
            if (iz - np.int(10+1.0/dz) < -0.33*(ix - np.int(10+22.0/dx))):
                lam[iz][ix] = lam_water
                mu[iz][ix] = mu_water
                rho[iz][ix] = rho_water
                
        if (iz>np.int(10+5.0/dz)): # subsurface boundary
            lam[iz][ix] = lam_sub
            mu[iz][ix] = mu_sub
            rho[iz][ix] = rho_sub


# modifying density parameter
'''
if (fwinv==False):
    for iz in range(0, nz):
        for ix in range(0, nx):
            if (((nx/2-ix)**2+(nz/2-iz)**2)<(nx*nx/49)):
                #rho[iz][ix] = 1.5 * rho[iz][ix]
                mu[iz][ix] = mu1
                lam[iz][ix] = lam1
'''



# PML factors
pml_npower_pml = 2.0
damp_v_pml = Cp
rcoef = 0.001
k_max_pml = 1.0
freq_pml = 50.0 # PML frequency in Hz


# Creating source locations

xsrc = np.array([10 + np.int32(1.0/dx), 10 + np.int32(4.0/dx), 10 + np.int32(7.0/dx)], dtype=np.int32)
zsrc = np.full((xsrc.size,), 10+ np.int32(2.2/dz), dtype=np.int32)

nsrc = zsrc.size
# Creating source to fire arrays
src_shot_to_fire = np.arange(0,nsrc,1, dtype=np.int32)
#src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

nshot = nsrc # fire each shot separately

# Creating reciever locations
xrec = np.zeros((15,), dtype=np.int32)
zrec = np.zeros((15,), dtype=np.int32)
for ir in range(0,zrec.size):
    ix = (12.0)/15.0
    iz = (4.0)/15.0
    xrec[ir] = np.int32(10 + (25.0+ir*ix-0.1)/dx)
    zrec[ir] = np.int32(10 + (1.0+ir*iz+0.1)/dz)
    
    
    
print("xrec:", xrec)
print("zrec:", zrec)
nrec = zrec.size



# Plotting modified material
print('Plotting initial materials')
plt.figure(1)
plt.subplot(221)
plt.imshow(lam)
plt.plot(xsrc,zsrc, ls = '', marker= 'x', markersize=2)
plt.plot(xrec,zrec, ls = '', marker= '+', markersize=2)
plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
plt.subplot(222)
plt.imshow(mu)
plt.subplot(223)
plt.imshow(rho)
plt.show()
# modifying starting material
# --------------------------------------------------


# ---------------------------------------------------------------------------------
# Creating boolen arrays
metabool = np.array([surf, pml_z, pml_x, accu_save, seismo_save, fwinv, rtf_meas_true], dtype=np.bool_)
# Creating integer arrays and subsequent concatenation of the related fields
metaint = np.array([nt, nz, nx, snap_t1, snap_t2, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, nsrc, nrec, nshot, stf_type, rtf_type, fdorder, fpad, mat_save_interval, mat_grid], dtype=np.int32)
metaint = np.concatenate((metabool, metaint), axis=None)

intarray = np.array([npml_top, npml_bottom, npml_left, npml_right, isurf_top, isurf_bottom, isurf_left, isurf_right], dtype=np.int32)
intarray  = np.concatenate((intarray, zsrc), axis=None)
intarray  = np.concatenate((intarray, xsrc), axis=None)
intarray  = np.concatenate((intarray, src_shot_to_fire), axis=None)
intarray  = np.concatenate((intarray, zrec), axis=None)
intarray  = np.concatenate((intarray, xrec), axis=None)

print("Metaint: ", metaint)
# Creating float arrays and subsequent concatenation
metafloat = np.array([dt, dz, dx, pml_npower_pml, damp_v_pml, rcoef, k_max_pml, freq_pml, scalar_lam, scalar_mu, scalar_rho], dtype=np.float64)
print("Metafloat: ", metafloat)

material_inp = np.concatenate((lam, mu), axis = None)
material_inp = np.concatenate((material_inp, rho), axis = None)



# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# WRITING ARRAYS TO BINARY FILE, READABLE IN C++ KERNELS
# -------------------------------------------------------------------------
metaint.tofile('./bin/metaint.bin')
intarray.tofile('./bin/intarray.bin')
metafloat.tofile('./bin/metafloat.bin')
material_inp.tofile('./bin/mat.bin')

# TO WRITE IN INPUT FILE
# Integers:
# nt, nz, nx, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, 
# nsrc, nrec, nshot, stf_type, rtf_type, rtf_true, fdorder, fpad, 
# npml[4], isurf[4], 

# Doubles:
# dt, dz, dx, scalar_lam, scalar_mu, scalar_rho, 
# 

# Booleans
# surf, pml_z, pml_x, accu_save, fwinv

# integer and reciever arrays
# *zsrc, *xsrc, *src_shot_to_fire, 
# *rec_z, *rec_x

# Double arrays
# *hc: holberg coefficient (may be parameters for holberg coefficients is better for it)
# **lam, **mu, **rho // materials
# **stf_z, **stf_x
# **rft_z_true, **rtf_x_true
