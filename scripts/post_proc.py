# Post processing in python
#%%

# reading the output arrays
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

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
npml_fpad = 21

def read_metaint(filename, dtype):
    
    '''
    reads the int metadata from input file
    '''
    global ndim, snap, fwinv, nsrc, nrec
    intdata = np.fromfile(filename, dtype=dtype)
    # nt, nz, nx
    print(intdata)
    fwinv = intdata[6]
    ndim = np.array(intdata[8:11], dtype = dtype) # [nt, nz, nx]
    nsrc = intdata[28]
    nrec = intdata[29]
    print("ndim: ", ndim)
    # snap_t1, snap_t2, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx
    snap = np.array(intdata[11:20], dtype=dtype) # the snap data
    print("snap: ", snap)
    




def read_tensor(filename, dtype, dshape):
    '''
    reads the given tensor data from the filename 
    '''
    data = np.fromfile(filename, dtype=dtype)
    data = np.reshape(data, dshape)
    return data



# reading the input data for the array size
read_metaint("./bin/metaint.bin", np.int32)
snap_nt = np.int32(1 + (ndim[0]-1)//snap[6])
snap_nz = 1 + (snap[3] - snap[2])//snap[7]
snap_nx = 1 + (snap[4] - snap[5])//snap[8]

# plotting the reciever data

if (fwinv):
    print("Plotting material for iteration in fwi")
    maxiter = 14
    for ii in range(0,maxiter,1):
        # reading data from csv file
        mat_dat = read_tensor("./bin/mat.bin", np.float64, (3, ndim[1], ndim[2]))
        #mat_dat = read_tensor("../io/mat_save/iter"+np.str(ii)+"_mat copy.bin", np.float64, (3, ndim[1], ndim[2]))
        
        lam = mat_dat[0][:][:]
        mu = mat_dat[1][:][:]
        rho = mat_dat[2][:][:]
        Cs = np.sqrt(mu/rho)
        Cp = np.sqrt((lam + 2 * mu)/rho)
        plt.figure(1)
        
        plt.subplot(221)
        plt.imshow(Cp, animated=True)#, cmap=cm.seismic, interpolation='nearest')#, vmin=1700, vmax=1900)
        plt.colorbar()
        #plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]')
        plt.title(r'$C_p \ (m/s)$')
        plt.subplot(222)
        plt.imshow(Cs, animated=True)#, cmap=cm.seismic,  interpolation='nearest')#, vmin=1700, vmax=1900)
        plt.colorbar()
        #plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]')
        plt.title(r'$C_s \ (m/s)$')
        plt.subplot(223)
        plt.imshow(rho, animated=True)#, cmap=cm.seismic,  interpolation='nearest')#, vmin=1700, vmax=1900)
        plt.colorbar()
        plt.title('Density '+r'$(kg/m^3)$')
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]')
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        #plt.grid()
        #pyplot.savefig('../io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
        #plt.show()
        #plt.draw()
        if (ii==(maxiter-1)):
            plt.show()
        else:
            plt.pause(0.005)
            plt.clf()
        
        #print('Figure '+np.str(ii)+' plotted.')

    
else:
    # Forward wavefield
    print("Plotting forward wavefield")
    
    
    # Plot the rtf first
    print("NREC: ", nrec)
    rtf_uz = read_tensor("./bin/shot0_rtf_uz.bin", np.float64, (nrec, ndim[0]))
    rtf_ux = read_tensor("./bin/shot0_rtf_ux.bin", np.float64, (nrec, ndim[0]))
    
    
    # Plotting the RTF functions
    plt.figure(1)
    plt.subplot(211)
    for ii in range(0, nrec,10):   
        plt.plot(rtf_uz[ii][:])
    plt.grid()
    plt.subplot(212)
    for ii in range(0, nrec,10):
        plt.plot(rtf_ux[ii][:])
    plt.grid()
    plt.show()
    plt.savefig('./rtf_signals.png', format='png', bbox_inches='tight')
    
    vz_dat = read_tensor("./bin/shot0_vz.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    vx_dat = read_tensor("./bin/shot0_vx.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    clip_nt = snap_nt
    clip_pz = np.amax(vz_dat[:clip_nt])
    clip_mz = np.amin(vz_dat[:clip_nt])
    clipz = 0.3*max([clip_pz, np.abs(clip_mz)])
    
    clip_px = np.amax(vx_dat[:clip_nt])
    clip_mx = np.amin(vx_dat[:clip_nt])
    clipx = 0.3*max([clip_px, np.abs(clip_mx)])
    
    for ii in range(0,clip_nt, 1):
        # reading data from csv file
        vz = vz_dat[ii,:,:]
        vx = vx_dat[ii,:,:]  
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipz, vmax=clipz)
        plt.colorbar()
        plt.title('Vz [Time snap '+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]'+np.str(ii))
        plt.ylabel('Z [no. of grids]')
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        plt.grid()
        plt.subplot(212)
        plt.imshow(vx, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx)
        plt.colorbar()
        plt.title('Vx [Time snap '+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]'+np.str(ii))
        plt.ylabel('Z [no. of grids]')
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        plt.grid()
        #pyplot.savefig('../io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
        #plt.show()
        #plt.draw()
        plt.pause(0.01)
        plt.clf()
        
        #if (ii<100):
        #    pyplot.clf()
        #else:
        #    pyplot.show()
        
        #print('Figure '+np.str(ii)+' plotted.')
        del vz
# %%
