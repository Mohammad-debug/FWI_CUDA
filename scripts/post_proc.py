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
    
    for ii in range(0,20,2):
        # reading data from csv file
        mat_dat = read_tensor("./bin/iter"+np.str(ii)+"_mat.bin", np.float64, (3, ndim[1], ndim[2]))
        print(mat_dat)
        plt.figure(1)
        plt.subplot(221)
        plt.imshow(mat_dat[0][:][:], animated=True, interpolation='nearest')#, vmin=-clip, vmax=clip)
        plt.colorbar()
        plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]'+np.str(ii))
        plt.subplot(222)
        plt.imshow(mat_dat[1][:][:], animated=True,  interpolation='nearest')#, vmin=-clip, vmax=clip)
        plt.colorbar()
        plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]'+np.str(ii))
        plt.subplot(223)
        plt.imshow(mat_dat[2][:][:], animated=True,  interpolation='nearest')#, vmin=-clip, vmax=clip)
        plt.colorbar()
        plt.title('Material [Iteration'+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]')
        plt.ylabel('Z [no. of grids]'+np.str(ii))
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        plt.grid()
        #pyplot.savefig('./io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
        #pyplot.show()
        plt.draw()
        plt.pause(2.0)
        plt.show()
        #plt.clf()
        
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
    plt.subplot(212)
    for ii in range(0, nrec):
        plt.plot(rtf_ux[ii][:])
    plt.show()
    
    vz_dat = read_tensor("./bin/shot0_vx.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    
    clip_plus = np.amax(vz_dat)
    clip_minus = np.amin(vz_dat)
    clip = max([clip_plus, np.abs(clip_minus)])
    print("Vz: max = ", snap_nt, ", min = ", clip_minus, ", clip = ", clip, ".")
    for ii in range(1,snap_nt):
        # reading data from csv file
        data = vz_dat[ii,:,:] # np.fromfile("../bin/shot0_vz", dtype=np.float64)
        # removing nan from the end due to extra ',' in the end
        vz = np.zeros((data.shape[0]-10, data.shape[1]-11))
        
        for j in range(0,vz.shape[0]):
            for i in range(0,vz.shape[1]):
                vz[j][i] = data[j+5][i+5]
                
        del data
        
        plt.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clip, vmax=clip)
        plt.colorbar()
        plt.title('Vz [Time snap '+np.str(ii)+']', y=-0.2)
        plt.xlabel('X [no. of grids]'+np.str(ii))
        plt.ylabel('Z [no. of grids]')
        #pyplot.gca().invert_yaxis()
        #pyplot.axis('equal')
        plt.grid()
        #pyplot.savefig('./io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
        #pyplot.show()
        plt.draw()
        plt.pause(0.01)
        
        plt.clf()
        
        #if (ii<100):
        #    pyplot.clf()
        #else:
        #    pyplot.show()
        
        #print('Figure '+np.str(ii)+' plotted.')
        del vz
