# preprocessing of the files
import numpy as np
from seismic_def import read_tensor
import matplotlib.pyplot as plt

ftype = np.float64

# Getting the input directly in this preprocessor file
# Geometric data
dt = 0.1e-3; dz = 0.4; dx = 0.4 # grid intervals
nt = 3000; nz = 401; nx = 201 # grid numbers

# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 10; npml_bottom = 10; npml_left = 10; npml_right = 10

# Surface grid index in each direction (0 = no surface)
surf = False # surface exists
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0

snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = 50; snap_z2 = nz-51; snap_x1 = 25; snap_x2 = 175 # snap boundaries
snap_dt = 1; snap_dz = 1; snap_dx = 1; # the snap intervals

#nshot = 1 #; nsrc = 3; nrec = 10; 
stf_type = 1; rtf_type = 0
fdorder = 2; fpad = 1

#Boolen values
fwinv = True
if (fwinv):
    accu_save = False; seismo_save=True
    mat_save_interval = 1; rtf_meas_true = True # RTF field measurement exists
else:
    accu_save = True; seismo_save=True
    mat_save_interval = -1; rtf_meas_true = False # RTF field measurement exists
    
# scalar material variables
Cp = 2000.0
Cs = 700.0
scalar_rho = 1500.0
scalar_mu = Cs*Cs*scalar_rho
scalar_lam = Cp*Cp*scalar_rho - 2.0*scalar_mu
mat_grid = 1 # 0 for scalar and 1 for grid


# --------------------------------------------------
# preparing  the starting material arrays
lam = np.full((nz, nx), scalar_lam)
mu = np.full((nz, nx), scalar_mu)
rho = np.full((nz, nx), scalar_rho)


# scalar material variables
Cp1 = 1800.0
Cs1 = 500.0
scalar_rho = 1500.0
mu1 = Cs1*Cs1*scalar_rho
lam1 = Cp1*Cp1*scalar_rho - 2.0*scalar_mu
mat_grid = 1 # 0 for scalar and 1 for grid

# modifying density parameter
if (fwinv==False):
    for iz in range(0, nz):
        for ix in range(0, nx):
            if (((nx/2-ix)**2+(nz/2-iz)**2)<(nx*nx/49)):
                #rho[iz][ix] = 1.5 * rho[iz][ix]
                mu[iz][ix] = mu1
                lam[iz][ix] = lam1

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


# Creating source locations
zsrc = np.array([nz/4, nz/2, 3*nz/4], dtype=np.int32)
xsrc = np.full((zsrc.size,), 20, dtype=np.int32)

nsrc = zsrc.size
# Creating source to fire arrays
src_shot_to_fire = np.arange(0,nsrc,1, dtype=np.int32)
#src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

nshot = nsrc # fire each shot separately

# Creating reciever locations
zrec = np.arange(20, 381, 2, dtype=np.int32)
xrec = np.full((zrec.size,), 180, dtype=np.int32)
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
