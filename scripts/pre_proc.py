#%%
# # preprocessing of the files
import numpy as np
from seismic_def import read_tensor
import matplotlib.pyplot as plt

ftype = np.float64

##---------------------------------------------------------------------
# COMPUTATION IN GPU OR CPU
#---------------------------------------------------------------------

cuda_computation = True# True: computation in GPU, False: in CPU

#forward only or fWI?
fwinv = False # True: FWI, False: Forward only

#---------------------------------------------------------------------




#---------------------------------------------------------------------
# GRID PARAMETERS
#--------------------------------------------------------------------


# Geometric data
dt = 0.1e-3; dz = 0.4; dx = 0.4 # grid intervals
nt = 1000; nz = 3501; nx = 3001 # grid numbers (adding for PMLs as well)


# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 10; npml_bottom = 10; npml_left = 10; npml_right = 10


# Surface grid index in each direction (0 = no surface)
surf = False # surface exists
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0


snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = 20; snap_z2 = 380  # snap boundaries z
snap_x1 = 20; snap_x2 = 180 # snap boundaries x
snap_dt = 3; snap_dz = 1; snap_dx = 1; # the snap intervals


# Taper position
nz_snap = snap_z2 - snap_z1
nx_snap = snap_x2 - snap_x1

# taper relative to the total grid
# t: top, b: bottom, l: left, r: right
taper_t1 = snap_z1 + np.int32(nz_snap*0.05); taper_t2 = taper_t1 + np.int32(nz_snap*0.1)
taper_b1 = snap_z2 - np.int32(nz_snap*0.05); taper_b2 = taper_b1 - np.int32(nz_snap*0.1)

taper_l1 = snap_x1 + np.int32(nx_snap*0.05); taper_l2 = taper_l1 + np.int32(nx_snap*0.1)
taper_r1 = snap_x2 - np.int32(nx_snap*0.05); taper_r2 = taper_r1 - np.int32(nx_snap*0.1)

#------------------------------------------------------------------------------





# -------------------------------------------------------------------------
# FINITE DIFFERENCE PARAMETERS
# --------------------------------------------------------------------------

fdorder = 2 # finite difference order 
fpad = 1 # number of additional grids for finite difference computation



# Internal parameters for different cases 
if (fwinv):
    accu_save = False; seismo_save=True
    mat_save_interval = 1; rtf_meas_true = True # RTF field measurement exists
else:
    accu_save = True; seismo_save=True
    mat_save_interval = -1; rtf_meas_true = False # RTF field measurement exists

# ---------------------------------------------------------------------------------
    
#------------------------------------------------------------------
# MEDIUM (MATERIAL) PARAMETERS
#-----------------------------------------------------------------

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


# scalar material variables (For original layers)
Cp1 = 1800.0
Cs1 = 500.0
rho1 = 1800.0
mu1 = Cs1*Cs1*rho1
lam1 = Cp1*Cp1*rho1 - 2.0*mu1
mat_grid = 1 # 0 for scalar and 1 for grid

# modifying density parameter (in original layers)
if (fwinv==False):
    for iz in range(0, nz):
        for ix in range(0, nx):
            if (((nx/2-ix)**2+(nz/2-iz)**2)<(nx*nx/49)):
                #rho[iz][ix] = 1.5 * rho[iz][ix]
                mu[iz][ix] = mu1
                lam[iz][ix] = lam1
                rho[iz][ix] = rho1

#------------------------------------------------------------



# -----------------------------------------------------
# PML VALUES TO BE USED FOR COMPUTATION
# -----------------------------------------------------

# PML factors
pml_npower_pml = 2.0
damp_v_pml = Cp
rcoef = 0.001
k_max_pml = 1.0
freq_pml = 50.0 # PML frequency in Hz

# -----------------------------------------------------




#-----------------------------------------------------
# SOURCES AND RECIEVERS
#--------------------------------------------------------

# source and reciever time functions type
stf_type = 1; rtf_type = 0 # 1:velocity, 2:displacement

# Creating source locations
zsrc = np.array([nz/4, nz/2, 3*nz/4], dtype=np.int32)
xsrc = np.full((zsrc.size,), 20, dtype=np.int32)
nsrc = zsrc.size # counting number of sources from the source location data


# Creating source to fire arrays
src_shot_to_fire = np.arange(0,nsrc,1, dtype=np.int32)
#src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

nshot = nsrc # fire each shot separately

# Creating reciever locations
zrec = np.arange(20, 381, 2, dtype=np.int32)
xrec = np.full((zrec.size,), 180, dtype=np.int32)
nrec = zrec.size



# -----------------------------------------------------
# PLOTTING INPUTS
#---------------------------------------------------

# print('Plotting initial materials')
# plt.figure(1)
# plt.subplot(221)
# plt.imshow(lam) # lamda parameter
# plt.plot(xsrc,zsrc, ls = '', marker= 'x', markersize=4) # source positions
# plt.plot(xrec,zrec, ls = '', marker= '+', markersize=3) # reciever positions
# plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
# plt.plot([taper_l1, taper_r1, taper_r1, taper_l1, taper_l1], [taper_t1, taper_t1, taper_b1, taper_b1, taper_t1], ls = '--')
# plt.plot([taper_l2, taper_r2, taper_r2, taper_l2, taper_l2], [taper_t2, taper_t2, taper_b2, taper_b2, taper_t2], ls = '--')
# plt.subplot(222)
# plt.imshow(mu)
# plt.subplot(223)
# plt.imshow(rho)
# plt.show()

#--------------------------------------------------------




# -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# PROCESSING TO PREPARE THE ARRAYS (DO NOT MODIFY)
# -------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# -----------------------------------------------------
# CREATING BINARY INPUT METADATA
# ---------------------------------------------------

# Creating boolen arrays
metabool = np.array([cuda_computation, surf, pml_z, pml_x, accu_save, seismo_save, fwinv, rtf_meas_true], dtype=np.bool_)

# Creating integer arrays and subsequent concatenation of the related fields
metaint = np.array([nt, nz, nx, snap_t1, snap_t2, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, \
                    taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2,\
                    nsrc, nrec, nshot, stf_type, rtf_type, fdorder, fpad, mat_save_interval, mat_grid], dtype=np.int32)
metaint = np.concatenate((metabool, metaint), axis=None) # concatination of boolen and integer as integers

# additional concatenation of int arrays
intarray = np.array([npml_top, npml_bottom, npml_left, npml_right, isurf_top, isurf_bottom, isurf_left, isurf_right], dtype=np.int32)
intarray  = np.concatenate((intarray, zsrc), axis=None)
intarray  = np.concatenate((intarray, xsrc), axis=None)
intarray  = np.concatenate((intarray, src_shot_to_fire), axis=None)
intarray  = np.concatenate((intarray, zrec), axis=None)
intarray  = np.concatenate((intarray, xrec), axis=None)

print("Metaint: ", metaint)

# Creating float arrays and subsequent concatenation
metafloat = np.array([dt, dz, dx, pml_npower_pml, damp_v_pml, rcoef, k_max_pml, freq_pml, \
                     scalar_lam, scalar_mu, scalar_rho], dtype=np.float64)
print("Metafloat: ", metafloat)

# Creating material arrays
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
#--------------------------------------------------------

#--------------------------------------------------------
#-------------------------------------------------------



<<<<<<< HEAD
<<<<<<< HEAD
# %%
=======
# %%
>>>>>>> cuda_fwi_integration
=======
# %%
>>>>>>> 10a335c17ee112c245526a7bcd6a0d2ab22577e8
