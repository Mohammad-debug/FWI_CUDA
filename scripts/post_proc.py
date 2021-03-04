# Post processing in python


# reading the output arrays
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

def read_metaint(filename, dtype):
    
    '''
    reads the int metadata from input file
    '''
    global ndim, snap, fwinv, nsrc, nrec
    intdata = np.fromfile(filename, dtype=dtype)
    # nt, nz, nx
    print(intdata)
    fwinv = intdata[5]
    ndim = np.array(intdata[7:10], dtype = dtype) # [nt, nz, nx]
    nsrc = intdata[19]
    nrec = intdata[20]
    print("ndim: ", ndim)
    # snap_t1, snap_t2, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx
    snap = np.array(intdata[10:19], dtype=dtype) # the snap data
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
    maxiter = 99
    for ii in range(0,maxiter,1):
        # reading data from csv file
        mat_dat = read_tensor("./bin/iter"+np.str(ii)+"_mat.bin", np.float64, (3, ndim[1], ndim[2]))
        #mat_dat = read_tensor("./io/mat_save/iter"+np.str(ii)+"_mat copy.bin", np.float64, (3, ndim[1], ndim[2]))
        
        lam = mat_dat[0][:][:]
        mu = mat_dat[1][:][:]
        rho = mat_dat[2][:][:]
        Cs = np.sqrt(mu/rho)
        Cp = np.sqrt((lam + 2 * mu)/rho)
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(Cp, animated=True)#, cmap=cm.seismic, interpolation='nearest')#, vmin=1700, vmax=1900)
        plt.colorbar()
        #plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]'+np.str(ii))
        plt.subplot(132)
        plt.imshow(Cs, animated=True)#, cmap=cm.seismic,  interpolation='nearest')#, vmin=1700, vmax=1900)
        plt.colorbar()
        #plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]'+np.str(ii))
        plt.subplot(133)
        plt.imshow(rho, animated=True)#, cmap=cm.seismic,  interpolation='nearest')#, vmin=1700, vmax=1900)
        plt.colorbar()
        plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]'+np.str(ii))
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        plt.grid()
        #pyplot.savefig('./io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
        #plt.show()
        #plt.draw()
        if (ii==(maxiter-1)):
            plt.show()
        else:
            plt.pause(0.05)
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
    for ii in range(0, nrec):   
        plt.plot(rtf_uz[ii][:])
        plt.grid()
    plt.subplot(212)
    for ii in range(0, nrec):
        plt.plot(rtf_ux[ii][:])
        plt.grid()
    plt.show()
    
    vz_dat = read_tensor("./bin/shot2_vz.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    vx_dat = read_tensor("./bin/shot2_vx.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    
    clip_pz = np.amax(vz_dat)
    clip_mz = np.amin(vz_dat)
    clipz = 0.3*max([clip_pz, np.abs(clip_mz)])
    
    clip_px = np.amax(vx_dat)
    clip_mx = np.amin(vx_dat)
    clipx = 0.3*max([clip_px, np.abs(clip_mx)])
    
    
    for ii in range(1,snap_nt):
        # reading data from csv file
        vz = vz_dat[ii,:,:]
        vx = vx_dat[ii,:,:]  
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipz, vmax=clipz)
        plt.colorbar()
        plt.title('Vz [Time snap '+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]'+np.str(ii))
        plt.ylabel('Z [no. of grids]')
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        plt.grid()
        plt.subplot(122)
        plt.imshow(vx, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx)
        plt.colorbar()
        plt.title('Vx [Time snap '+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]'+np.str(ii))
        plt.ylabel('Z [no. of grids]')
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        plt.grid()
        #pyplot.savefig('./io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
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
