#%%
# # preprocessing of the files
import numpy as np
import random
from seismic_def import read_tensor, e_lami, v_lami, w_vel
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
import pickle

ftype = np.float64

# -------------------------------------------------------------------------
# FINITE DIFFERENCE PARAMETERS
# --------------------------------------------------------------------------

fdorder = 2 # finite difference order 
fpad = 1 # number of additional grids for finite difference computation

#forward only or fWI?
fwinv = True # True: FWI, False: Forward only

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

# read the profile 
xy_profile = np.loadtxt('./io/profile_P2.csv', delimiter=', ').T

# getting x and y along the desired x

# import sampling data from seismics
infile = open('./io/xseis.pkl','rb')
data_seis = pickle.load(infile)
infile.close()
nsrc = data_seis[1]
nrec = data_seis[2]
nSample = data_seis[3]

dt_field = data_seis[4]
xsrc_cartesian = data_seis[5]
xrec_cartesian = data_seis[6]
shot_fire = data_seis[7]

#---------------------------------------------------------------------
# GRID PARAMETERS
#--------------------------------------------------------------------

# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml_top = 20; npml_bottom = 20; npml_left = 20; npml_right = 20

# Geometric data
x0 = -0.5; x1 = 18.0 # the start and end of model in meters
dt = dt_field; dz = 0.25/2; dx = 0.25/2; # grid intervals
nt = nSample #int(nSample*5/3) # taking portion of total samples

range_y= abs(max(xy_profile[1]) - min(xy_profile[1]))

nx = fpad + npml_left + int(abs(x1-x0)/dx) + npml_right + fpad+ 1
z_add1 = 10 # adding some layer above the ground profile
z_add2 = 20 # adding some layers below the ground profile
nz = fpad + npml_top + z_add1 + int(range_y/dz) + z_add2 + npml_bottom + fpad + 1 # adding some additional grids z_add1 and z_add2

# Surface grid index in each direction (0 = no surface)
surf = True # surface exists
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0

snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = fpad +npml_top
snap_z2 = nz - fpad - npml_bottom #fpad+npml_top+np.int32(d_tot/dz) # snap boundaries z
snap_x1 = fpad +npml_left + 15
snap_x2 = nx - fpad - npml_bottom - 15 # snap boundaries x
snap_dt = 1; snap_dz = 1; snap_dx = 1; # the snap interval

# Taper position
nz_snap = snap_z2 - snap_z1
nx_snap = snap_x2 - snap_x1

# taper relative to the total grid
# t: top, b: bottom, l: left, r: right
taper_t1 = snap_z1 + np.int32(nz_snap*0.05); taper_t2 = taper_t1 + np.int32(nz_snap*0.05)
taper_b1 = snap_z2 - np.int32(nz_snap*0.05); taper_b2 = taper_b1 - np.int32(nz_snap*0.05)

taper_l1 = snap_x1 + np.int32(nx_snap*0.05); taper_l2 = taper_l1 + np.int32(nx_snap*0.05)
taper_r1 = snap_x2 - np.int32(nx_snap*0.05); taper_r2 = taper_r1 - np.int32(nx_snap*0.05)
#snap_z1 = 1; snap_z2 = nz-1  # snap boundaries z
#snap_x1 = 1; snap_x2 = nx-1 # snap boundaries x

#------------------------------------------------------------------------------
# updating the profile in the grid
xy_profile[1] = max(xy_profile[1])-xy_profile[1] # setting the top most point at zero

# getting maximum and minimum values for the grid including npml and fpads
xmin = x0 - (npml_left+fpad)*dx
xmax = x1 + (npml_right+fpad)*dx
zmin = min(xy_profile[1]) - (z_add1 + npml_top+fpad)*dz
zmax = max(xy_profile[1]) + (z_add2 + npml_bottom+fpad)*dz

xy_profile[1] += min(xy_profile[1])-zmin # adding additional part for additional top, npml top and fpad
surf_fx = interpolate.interp1d(xy_profile[0], xy_profile[1]) # spline interpolation 

x_coord = np.linspace(xmin, xmax, nx) # the x coordinate along the grid
z_coord = np.linspace(zmin, zmax, nz) # the x coordinate along the grid

# getting the surface profile of z along the surface for each x value
z_coord_surf = surf_fx(x_coord)

surf_idz = np.zeros(x_coord.size, dtype=int)
# now getting the x z pair of grid index along the surface (z value for each x)
for ii in range(0, nx):
    # find the corresponding grid index for z coord along the surface
    surf_idz[ii] = int(z_coord_surf[ii]/dz)
    

#------------------------------------------------------------------
# MEDIUM (MATERIAL) PARAMETERS
#-----------------------------------------------------------------

# material input values scalar or arrays
mat_grid = 1 # 0 for scalar and 1 for grid

#---------------------------------------------------
# Adding material properties for different materials
rho_air = 1.25
lam_air, mu_air = v_lami(0.0, 0.0, rho_air)



rho_sand = 1700.0
lam_sand, mu_sand = v_lami(700, 200, rho_sand)


'''
rho_water = 1000.0
lam_water, mu_water = v_lami(1500, 0.0, rho_water)
#rho_sub = 1800.0
#lam_sub, mu_sub = v_lami(1400, 700, rho_sub)

rho_sub = 2000.0
lam_sub, mu_sub = v_lami(1400, 700, rho_sub)
rho_sand_grout = 1000.0 #2000.0
lam_sand_grout, mu_sand_grout = v_lami(300, 100.0, rho_sand_grout)
'''
# --------------------------------------------
# The scalar material values
scalar_lam = lam_sand
scalar_mu = mu_sand
scalar_rho = mu_sand

# Getting wave velocities for hardest layers
Cp, Cs = w_vel(lam_sand, mu_sand, rho_sand)

# --------------------------------------------------
# preparing  the starting material arrays (Fill with Air)
lam = np.full((nz, nx), lam_air)
mu = np.full((nz, nx), mu_air)
rho = np.full((nz, nx), rho_air)

#-----------------------------------------------------------------------------
# add for dam sand layer
for iz in range(0, nz):
    for ix in range(0, nx):
        if iz>=surf_idz[ix]:
            lam[iz][ix] = lam_sand *(1+ 0.05*random.uniform(0,1))
            mu[iz][ix] = mu_sand *(1+ 0.05*random.uniform(0,1))
            rho[iz][ix] = rho_sand *(1+ 0.05*random.uniform(0,1))

#------------------------------------------------------------




# -----------------------------------------------------
# PML VALUES TO BE USED FOR COMPUTATION
# -----------------------------------------------------

# PML factors
pml_npower_pml = 2.0
damp_v_pml = Cp
rcoef = 0.001
k_max_pml = 1.0
freq_pml = (50.0 + 120)/2.0 # PML frequency in Hz

# -----------------------------------------------------




#-----------------------------------------------------
# SOURCES AND RECIEVERS
#--------------------------------------------------------

# source and reciever time functions type
stf_type = 1; rtf_type = 1 # 1:velocity, 0:displacement

# finding the source and reciever locations
xsrc = np.zeros((nsrc,), dtype=np.int32)
zsrc = np.zeros((nsrc,), dtype=np.int32)
xrec = np.zeros((nrec,), dtype=np.int32)
zrec = np.zeros((nrec,), dtype=np.int32)
print('xsrc cart:', xsrc_cartesian)
print(xmin, xmax, zmin, zmax)
print(nx, nz, nt)
for ii in range(0, nsrc):
    print('ii: ', ii)
    xsrc[ii] = int((xsrc_cartesian[ii]-xmin)/dx)
    print('src:', xsrc[ii])
    zsrc[ii] = surf_idz[xsrc[ii]]
    print('src_z:', zsrc[ii])
    
for ii in range(0, nrec):
    xrec[ii] = int((xrec_cartesian[ii] - xmin)/dx)
    zrec[ii] = surf_idz[xrec[ii]]

#plt.plot(x_coord, z_coord_surf, marker='x')
plt.plot(np.arange(0, nx), surf_idz)
plt.plot(xrec, zrec, marker='+', ls=' ')
plt.plot(xsrc, zsrc, marker='o', ls=' ')
plt.grid()
plt.show()

# Creating source to fire arrays
#src_shot_to_fire = np.arange(0,nsrc, dtype=np.int32)
src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)
for ii in range(0, src_shot_to_fire.size):
    src_shot_to_fire[ii] = shot_fire[ii]
nshot = max(src_shot_to_fire)+1 # fire each shot separately


# -----------------------------------------------------
# PLOTTING INPUTS
#---------------------------------------------------

Cs = np.sqrt(mu/rho)
Cp = np.sqrt((lam + 2 * mu)/rho)
    

print('Plotting initial materials')
plt.figure(1)
plt.subplot(111)
plt.imshow(Cp, cmap=cm.Paired) # lamda parameter
plt.plot(xsrc,zsrc, ls = '', marker= 'o', markersize=4, color='k') # source positions
plt.plot(xrec,zrec, ls = '', marker= '+', markersize=4, color='k') # reciever positions
plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
plt.plot([taper_l1, taper_r1, taper_r1, taper_l1, taper_l1], [taper_t1, taper_t1, taper_b1, taper_b1, taper_t1], ls = '--')
plt.plot([taper_l2, taper_r2, taper_r2, taper_l2, taper_l2], [taper_t2, taper_t2, taper_b2, taper_b2, taper_t2], ls = '--')
'''
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
'''
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



# %%
