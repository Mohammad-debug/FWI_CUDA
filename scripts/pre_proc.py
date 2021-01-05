# preprocessing of the files
import numpy as np
from seismic_def import read_tensor
import matplotlib.pyplot as plt

ftype = np.float64

# Getting the input directly in this preprocessor file
# Geometric data
nt = 1200; nz = 401; nx = 151 # grid numbers

dt = 0.25e-3; dz = 2.0; dx = 2.0 # grid intervals

snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = 0; snap_z2 = nz-1; snap_x1 = 0; snap_x2 = nx-1 # snap boundaries
snap_dt = 3; snap_dz = 2; snap_dx = 2; # the snap intervals

nshot = 1 #; nsrc = 3; nrec = 10; 
stf_type = 1; rtf_type = 0
fdorder = 2; fpad = 1

# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 10; npml_bottom = 10; npml_left = 10; npml_right = 10

# Surface grid index in each direction (0 = no surface)
surf = False # surface exist
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0

# scalar material variables
Cp = 4000.0
Cs = 2310.
scalar_rho = 1800.0
scalar_mu = Cs*Cs*scalar_rho
scalar_lam = Cp*Cp*scalar_rho - 2.9*scalar_mu
mat_grid = 1 # 0 for scalar and 1 for grid


# --------------------------------------------------
# preparing  the starting material arrays
lam = np.full((nz, nx), scalar_lam)
mu = np.full((nz, nx), scalar_mu)
rho = np.full((nz, nx), scalar_rho)

# modifying density parameter
'''
for iz in range(0, nz):
    for ix in range(0, nx):
        if (((nx/2-ix)**2+(nz/2-iz)**2)<(nx*nx/16)):
            rho[iz][ix] = 1.5*rho[iz][ix]
''' 
# Plotting modified material
print('Plotting initial materials')
plt.figure(1)
plt.subplot(221)
plt.imshow(lam)
plt.subplot(222)
plt.imshow(mu)
plt.subplot(223)
plt.imshow(rho)
plt.show()
# modifying starting material
# --------------------------------------------------

# PML factors
pml_npower_pml = 2.0
damp_v_pml = Cp
rcoef = 0.001
k_max_pml = 1.0
freq_pml = 50.0 # PML frequency in Hz

#Boolen values
 
accu_save = False; seismo_save=False; fwinv = True; mat_save_interval = 2
rtf_meas_true = True # RTF field measurement exists

# Creating source locations
zsrc = np.array([nz/4, nz/2, 3*nz/4], dtype=np.int32)
xsrc = np.array([npml_left*2, npml_left*2, npml_left*2], dtype=np.int32)
nsrc = zsrc.size
# Creating source to fire arrays
src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

# Creating reciever locations
zrec = np.array([nz/10, 2*nz/10, 3*nz/10, 4*nz/10, 5*nz/10, 6*nz/10, 7*nz/10, 8*nz/10, 9*nz/10], dtype=np.int32)
xrec = np.array([nx-npml_right*2, nx-npml_right*2, nx-npml_right*2, nx-npml_right*2, nx-npml_right*2, nx-npml_right*2, nx-npml_right*2,nx-npml_right*2, nx-npml_right*2], dtype=np.int32)
nrec = zrec.size


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
