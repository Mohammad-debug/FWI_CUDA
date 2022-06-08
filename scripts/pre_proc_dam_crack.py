#%%
# # preprocessing of the files
import numpy as np
from seismic_def import read_tensor, e_lami, v_lami, w_vel
import matplotlib.pyplot as plt

ftype = np.float64

# -------------------------------------------------------------------------
# FINITE DIFFERENCE PARAMETERS
# --------------------------------------------------------------------------

fdorder = 2 # finite difference order 
fpad = 1 # number of additional grids for finite difference computation

#forward only or fWI?
fwinv = False # True: FWI, False: Forward only

# Internal parameters for different cases 
if (fwinv):
    accu_save = False; seismo_save=True
    mat_save_interval = 1; rtf_meas_true = True # RTF field measurement exists
else:
    accu_save = True; seismo_save=True
    mat_save_interval = -1; rtf_meas_true = False # RTF field measurement exists

# ---------------------------------------------------------------------------------


##---------------------------------------------------------------------
# COMPUTATION IN GPU OR CPU
#---------------------------------------------------------------------

cuda_computation = False # True: computation in GPU, False: in CPU

#---------------------------------------------------------------------



#---------------------------------------------------------------------
# GRID PARAMETERS
#--------------------------------------------------------------------

# Field parameters
l_uadd = 1.0
l_usl = 12.0
l_top = 3.0
l_dsl = 12.0
l_dadd = 1.0
len = l_uadd + l_usl + l_top + l_dsl + l_dadd  # Meters

d_top = 0.5
d_wt = 1.0
d_sub = 4.0
d_tot = 7.0
dep = d_tot # Depth


# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 20; npml_bottom = 20; npml_left = 20; npml_right = 20

# Geometric data
dt = 1.85e-5; dz = 0.05; dx = 0.05; # grid intervals
nt = 10000.0; 
nz = fpad + npml_top + np.int32(dep/dz) + npml_bottom + fpad+ 1
nx = fpad + npml_left + np.int32(len/dx) + npml_right + fpad + 1 # grid numbers (adding for PMLs as well)


# Surface grid index in each direction (0 = no surface)
surf = True # surface exists
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0



snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = fpad #+npml_top+np.int32(d_wt/dz) 
snap_z2 = nz - fpad #- npml_bottom #fpad+npml_top+np.int32(d_tot/dz) # snap boundaries z
snap_x1 = fpad #np.int32(nx/2 - 1.0*l_top/dz) 
snap_x2 = nx - fpad #np.int32(nx/2  + 1.0*l_top/dz) # snap boundaries x
snap_dt = 10; snap_dz = 1; snap_dx = 1; # the snap interval

# Taper position
nz_snap = snap_z2 - snap_z1
nx_snap = snap_x2 - snap_x1

# taper relative to the total grid
# t: top, b: bottom, l: left, r: right
taper_t1 = snap_z1 + np.int32(nz_snap*0.00); taper_t2 = taper_t1 + np.int32(nz_snap*0.00)
taper_b1 = snap_z2 - np.int32(nz_snap*0.00); taper_b2 = taper_b1 - np.int32(nz_snap*0.00)

taper_l1 = snap_x1 + np.int32(nx_snap*0.0); taper_l2 = taper_l1 + np.int32(nx_snap*0.00)
taper_r1 = snap_x2 - np.int32(nx_snap*0.0); taper_r2 = taper_r1 - np.int32(nx_snap*0.00)
#snap_z1 = 1; snap_z2 = nz-1  # snap boundaries z
#snap_x1 = 1; snap_x2 = nx-1 # snap boundaries x

#------------------------------------------------------------------------------







#------------------------------------------------------------------
# MEDIUM (MATERIAL) PARAMETERS
#-----------------------------------------------------------------

# material input values scalar or arrays
mat_grid = 1 # 0 for scalar and 1 for grid

#---------------------------------------------------
# Adding material properties for different materials
rho_air = 1.25
lam_air, mu_air = v_lami(0.0, 0.0, rho_air)

rho_water = 1000
lam_water, mu_water = v_lami(1482, 0.0, rho_water)

rho_sub = 2100.0
lam_sub, mu_sub = v_lami(1900, 700, rho_sub)

rho_sand = 1700.0
lam_sand, mu_sand = v_lami(800, 400, rho_sand)

rho_sand_sat = 1950.0
lam_sand_sat, mu_sand_sat = v_lami(1450, 400, rho_sand_sat)

rho_sand_grout = 1000.0 #2000.0
lam_sand_grout, mu_sand_grout = v_lami(1600, 100.0, rho_sand_grout)

# --------------------------------------------

# Getting wave velocities for hardest layers
Cp, Cs = w_vel(lam_sub, mu_sub, rho_sub)

# Scalar material values to pass to the kernels
scalar_rho = rho_sub
scalar_mu = mu_sub
scalar_lam = lam_sub

# --------------------------------------------------
# preparing  the starting material arrays (Fill with Air)
lam = np.full((nz, nx), lam_air)
mu = np.full((nz, nx), mu_air)
rho = np.full((nz, nx), rho_air)

#-----------------------------------------------------------------------------
# add for dam sand layer
for iz in range(0, nz):
    for ix in range(0, nx):
        # for sand dam
        if (iz>np.int(fpad + npml_top+d_top/dz)): # top boundary 
            if (iz - np.int(fpad + npml_top+d_top/dz) >= 0.33*(ix - (fpad + npml_left + np.int((l_uadd + l_usl + l_top)/dx)))):
                if (iz - (fpad + npml_top + np.int(d_top/dz)) >= -0.33*(ix - (fpad + npml_left + np.int((l_uadd + l_usl)/dx)))):
                    lam[iz][ix] = lam_sand
                    mu[iz][ix] = mu_sand
                    rho[iz][ix] = rho_sand
                    
                    if (iz-42) > 0.00065*(ix-248)*(ix-248):
                        lam[iz][ix] = lam_sand_sat
                        mu[iz][ix] = mu_sand_sat
                        rho[iz][ix] = rho_sand_sat

            
                    
                    
        if (iz>np.int(fpad + npml_top+d_wt/dz)): # water level boundary
            if (iz - (fpad + npml_top + np.int(d_top/dz)) < -0.33*(ix - (fpad + npml_left + np.int((l_uadd + l_usl)/dx)))):
                lam[iz][ix] = lam_water
                mu[iz][ix] = mu_water
                rho[iz][ix] = rho_water
                
        if (iz>np.int(fpad + npml_top +d_sub/dz)): # subsurface boundary
            lam[iz][ix] = lam_sub
            mu[iz][ix] = mu_sub
            rho[iz][ix] = rho_sub

        # Additional modificatoin for th
        # +e modified material
        if (fwinv==False):
            #if ((iz > fpad + npml_top + (d_top+1.50)/dz) and (iz <= fpad + npml_top + (d_sub-0.5)/dz)): # top and bottom boundary
            #    if ((ix > fpad + npml_left + (l_uadd+ l_usl +l_top/2 - 1.0)/dx - 0.25*np.cos(((iz*dz)-fpad-npml_top-d_top-1.5)*np.pi)/dz) and \
            #        (ix < fpad + npml_left + (l_uadd+ l_usl +l_top/2 + 1.0)/dx + 0.25*np.cos(((iz*dz)-fpad-npml_top-d_top-1.5)*np.pi)/dz)): # left and right boundarz


            if (iz*dz > nz*dz/2.5):
                if (ix*dx > (nx*dx/2+(iz*dz -nz*dz/2.5))):
                    if (ix*dx-0.4 < (nx*dx/2+(iz*dz -nz*dz/2.5))):
                        #if ((ix*dx - (nx*dx/2))**2 +(iz*dx - (nz*dz/2+0.4))**2 < 0.5):
                        lam[iz][ix] = lam_sand_grout
                        mu[iz][ix] = mu_sand_grout
                        rho[iz][ix] = rho_sand_grout

#------------------------------------------------------------




# -----------------------------------------------------
# PML VALUES TO BE USED FOR COMPUTATION
# -----------------------------------------------------

# PML factors
pml_npower_pml = 2.0
damp_v_pml = Cp
rcoef = 0.001
k_max_pml = 1.0
freq_pml = 800.0 # PML frequency in Hz

# -----------------------------------------------------




#-----------------------------------------------------
# SOURCES AND RECIEVERS
#--------------------------------------------------------

# source and reciever time functions type
stf_type = 1; rtf_type = 0 # 1:velocity, 2:displacement

# Creating source locations
#xsrc = np.array([10 + np.int32(1.0/dx), 10 + np.int32(4.0/dx), 10 + np.int32(7.0/dx)], dtype=np.int32)
#xsrc = np.array([npml_left + np.int32((4.0)/dx)], dtype=np.int32)
#zsrc = np.full((xsrc.size,), npml_top+ np.int32((d_wt+0.5)/dz), dtype=np.int32)

# Creating reciever locations
xsrc = np.zeros((3,), dtype=np.int32)
zsrc = np.zeros((3,), dtype=np.int32)
nsrc = zsrc.size # counting number of sources from the source location data

for isr in range(0,zsrc.size):
    ix = (12.0)/nsrc
    iz = (4.0)/nsrc
    xsrc[isr] = np.int32(fpad + npml_left + (l_uadd + l_usl - isr*ix + 0.1)/dx)
    zsrc[isr] = np.int32(fpad + npml_top + (d_top +isr*iz+0.1)/dz)



# Creating source to fire arrays
src_shot_to_fire = np.arange(0,nsrc,1, dtype=np.int32)
#src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

nshot = nsrc # fire each shot separately

# Creating reciever locations
xrec = np.zeros((15,), dtype=np.int32)
zrec = np.zeros((15,), dtype=np.int32)
nrec = zrec.size # counting the recievers from the locations

for ir in range(0,zrec.size):
    ix = (12.0)/nrec
    iz = (4.0)/nrec
    xrec[ir] = np.int32(fpad + npml_left + (l_uadd + l_usl + l_top + ir*ix - 0.1)/dx)
    zrec[ir] = np.int32(fpad + npml_top + (d_top +ir*iz+0.1)/dz)
    
# overwrite the recorder to the last source location
xrec[0] = xsrc[2]
zrec[0] = zsrc[2]


# -----------------------------------------------------
# PLOTTING INPUTS
#---------------------------------------------------

Cs = np.sqrt(mu/rho)
Cp = np.sqrt((lam + 2 * mu)/rho)
    

print('Plotting initial materials')
plt.figure(1)
plt.subplot(221)
plt.imshow(Cp) # lamda parameter
plt.plot(xsrc,zsrc, ls = '', marker= 'x', markersize=2) # source positions
plt.plot(xrec,zrec, ls = '', marker= '+', markersize=2) # reciever positions
plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
plt.plot([taper_l1, taper_r1, taper_r1, taper_l1, taper_l1], [taper_t1, taper_t1, taper_b1, taper_b1, taper_t1], ls = '--')
plt.plot([taper_l2, taper_r2, taper_r2, taper_l2, taper_l2], [taper_t2, taper_t2, taper_b2, taper_b2, taper_t2], ls = '--')
plt.subplot(222)
plt.imshow(Cs)
plt.plot(xsrc,zsrc, ls = '', marker= 'x', markersize=2) # source positions
plt.plot(xrec,zrec, ls = '', marker= '+', markersize=2) # reciever positions
plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
plt.plot([taper_l1, taper_r1, taper_r1, taper_l1, taper_l1], [taper_t1, taper_t1, taper_b1, taper_b1, taper_t1], ls = '--')
plt.plot([taper_l2, taper_r2, taper_r2, taper_l2, taper_l2], [taper_t2, taper_t2, taper_b2, taper_b2, taper_t2], ls = '--')
plt.subplot(223)
plt.imshow(rho)
plt.plot(xsrc,zsrc, ls = '', marker= 'x', markersize=2) # source positions
plt.plot(xrec,zrec, ls = '', marker= '+', markersize=2) # reciever positions
plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
plt.plot([taper_l1, taper_r1, taper_r1, taper_l1, taper_l1], [taper_t1, taper_t1, taper_b1, taper_b1, taper_t1], ls = '--')
plt.plot([taper_l2, taper_r2, taper_r2, taper_l2, taper_l2], [taper_t2, taper_t2, taper_b2, taper_b2, taper_t2], ls = '--')
plt.show()

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


