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

x = 20
z = 35

# Geometric data
nt = 1000; nz = 701; nx = 401 # grid numbers (adding for PMLs as well)
dt = 0.1e-3; dz = z/(nz-1); dx = x/(nx-1) # grid intervals

print('Spacing: ', dt, dz, dx)
# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 10; npml_bottom = 10; npml_left = 10; npml_right = 10


# Surface grid index in each direction (0 = no surface)
surf = False # surface exists
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0


snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = npml_top; snap_z2 = nz-npml_bottom  # snap boundaries z
snap_x1 = npml_left; snap_x2 = nx-npml_right # snap boundaries x
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

print('snaps: ', snap_z1, snap_z2, snap_x1, snap_x2)



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
Cp = 800.0
Cs = 400.0
scalar_rho = 1700.0
scalar_mu = Cs*Cs*scalar_rho
scalar_lam = Cp*Cp*scalar_rho - 2.0*scalar_mu
mat_grid = 1 # 0 for scalar and 1 for grid


# --------------------------------------------------
# preparing  the starting material arrays
lam = np.full((nz, nx), scalar_lam)
mu = np.full((nz, nx), scalar_mu)
rho = np.full((nz, nx), scalar_rho)


# scalar material variables
Cp1 = 1500
Cs1 = 900
rho1 = 2100
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
freq_pml = 200.0 # PML frequency in Hz

# -----------------------------------------------------




#-----------------------------------------------------
# SOURCES AND RECIEVERS
#--------------------------------------------------------

# source and reciever time functions type
stf_type = 1; rtf_type = 0 # 1:velocity, 2:displacement

# Creating source locations
zsrc = np.array([nz/6, 2*nz/6, 3*nz/6, 4*nz/6, 5*nz/6], dtype=np.int32)
xsrc = np.full((zsrc.size,), npml_left+10, dtype=np.int32)
nsrc = zsrc.size # counting number of sources from the source location data


# Creating source to fire arrays
src_shot_to_fire = np.arange(0,nsrc,1, dtype=np.int32)
#src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

nshot = nsrc # fire each shot separately

# Creating reciever locations
zrec = np.arange((npml_top+10), (npml_bottom-10), 2, dtype=np.int32)
xrec = np.full((zrec.size,), nx-npml_right-10, dtype=np.int32)
nrec = zrec.size



# -----------------------------------------------------
# PLOTTING INPUTS
#---------------------------------------------------

print('Plotting initial materials')
plt.figure(1)
plt.subplot(221)
plt.imshow(lam) # lamda parameter
plt.plot(xsrc,zsrc, ls = '', marker= 'x', markersize=4) # source positions
plt.plot(xrec,zrec, ls = '', marker= '+', markersize=3) # reciever positions
plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
plt.plot([taper_l1, taper_r1, taper_r1, taper_l1, taper_l1], [taper_t1, taper_t1, taper_b1, taper_b1, taper_t1], ls = '--')
plt.plot([taper_l2, taper_r2, taper_r2, taper_l2, taper_l2], [taper_t2, taper_t2, taper_b2, taper_b2, taper_t2], ls = '--')
plt.subplot(222)
plt.imshow(mu)
plt.subplot(223)
plt.imshow(rho)
plt.tight_layout()
#plt.show()
plt.savefig('./fwi_pre_proc.png', format='png', bbox_inches='tight')

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



# %%
