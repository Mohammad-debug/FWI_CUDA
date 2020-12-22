# preprocessing of the files
import numpy as np
# create a sample numpy array
data = np.array([1.0e-3, 2.0e-7], dtype=np.float64)
print(data)
data.tofile("./bin/sample.npy")

# Getting the input directly in this preprocessor file
# Geometric data
nt = 1/0.2e-3; nz = 801; nx = 801 # grid numbers

dt = 0.2e-3; dz = 2.0; dx = 2.0 # grid intervals

snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = 0; snap_z2 = nz-1; snap_x1 = 0; snap_x2 = nx-1 # snap boundaries
snap_dt = 100; snap_dz = 2; snap_dx = 2; # the snap intervals

nsrc = 1; nrec = 3; nshot = 1
stf_type = 1; rtf_type = 0
fdorder = 2; fpad = 1

# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 20; npml_bottom = 20; npml_left = 20; npml_right = 20

# Surface grid index in each direction (0 = no surface)
surf = False # surface exist
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0

# scalar material variables
Cp = 4000.0
Cs = 2310.
scalar_rho = 1800.0
scalar_mu = Cs*Cs*scalar_rho
scalar_lam = Cp*Cp*scalar_rho - 2.9*scalar_mu

# --------------------------------------------------
# preparing  the starting material arrays
lam = np.full((nz, nx), scalar_lam)
mu = np.full((nz, nx), scalar_mu)
rho = np.full((nz, nx), scalar_rho)
# --------------------------------------------------

# PML factors
pml_npower_pml = 2.0
damp_v_pml = Cp
rcoef = 0.001
k_max_pml = 1.0
freq_pml = 15.0 # PML frequency in Hz

#Boolen values
 
accu_save = True; fwinv = False; 
rtf_meas_true = False # RTF field measurement exists

# Creating source locations
zsrc = np.array([nz/2], dtype=np.int32)
xsrc = np.array([nx/2], dtype=np.int32)

# Creating source to fire arrays
src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

# Creating reciever locations
zrec = np.array([nz-3-npml_bottom, nz-3-npml_bottom, nz-3-npml_bottom], dtype=np.int32)
xrec = np.array([nx/3, nx/2, 3*nx/4], dtype=np.int32)


# ---------------------------------------------------------------------------------
# Creating boolen arrays
metabool = np.array([surf, pml_z, pml_x, accu_save, fwinv, rtf_meas_true], dtype=np.bool_)
# Creating integer arrays and subsequent concatenation of the related fields
metaint = np.array([nt, nz, nx, snap_t1, snap_t2, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, nsrc, nrec, nshot, stf_type, rtf_type, fdorder, fpad], dtype=np.int32)
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
metafloat = np.concatenate((metafloat, lam), axis = None)
metafloat = np.concatenate((metafloat, mu), axis = None)
metafloat = np.concatenate((metafloat, rho), axis = None)

print("Metafloat: ", metafloat)

# ----------------------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------
# WRITING ARRAYS TO BINARY FILE, READABLE IN C++ KERNELS
# -------------------------------------------------------------------------
metaint.tofile('./bin/metaint.bin')
intarray.tofile('./bin/intarray.bin')
metafloat.tofile('./bin/metafloat.bin')

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
