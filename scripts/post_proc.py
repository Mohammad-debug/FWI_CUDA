# Post processing in python


# reading the output arrays
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

import sys

def read_metaint(filename, dtype):
    
    '''
    reads the int metadata from input file
    '''
    global ndim, snap, fwinv
    intdata = np.fromfile(filename, dtype=dtype)
    # nt, nz, nx
    ndim = np.array(intdata[6:9], dtype = dtype) # [nt, nz, nx]
    print("ndim: ", ndim)
    # snap_t1, snap_t2, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx
    snap = np.array(intdata[9:18], dtype=dtype) # the snap data
    print("snap: ", snap)
    fwinv = intdata[4]




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

vz_dat = read_tensor("./bin/shot0_vz.bin", np.float64, (snap_nt, snap_nz, snap_nx))


#for i in range(0, snap_nt):
#    print(i, vz_dat.dtype, vz_dat[i,:,:])
#    print("...........................")

clip = 0.01
for ii in range(1,snap_nt):
    # reading data from csv file
	data = vz_dat[ii,:,:] # np.fromfile("../bin/shot0_vz", dtype=np.float64)
	# removing nan from the end due to extra ',' in the end
	vz = np.zeros((data.shape[0]-10, data.shape[1]-11))
	for j in range(0,vz.shape[0]):
		for i in range(0,vz.shape[1]):
			vz[j][i] = data[j+5][i+5]
	del data

	#pyplot.matshow(vz, cmap=cm.seismic, interpolation='nearest', vmin=-clip, vmax=clip)
	pyplot.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clip, vmax=clip)
	pyplot.colorbar()
	pyplot.title('Vz [Time snap '+np.str(ii)+']', y=-0.2)
	pyplot.xlabel('X [no. of grids]')
	pyplot.ylabel('Z [no. of grids]')
	#pyplot.gca().invert_yaxis()
	#pyplot.axis('equal')
	pyplot.grid()
	#pyplot.savefig('./io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
	#pyplot.show()
	pyplot.draw()
	pyplot.pause(1.0)

	if (ii<100):
		pyplot.clf()
	else:
		pyplot.show()
		
	print('Figure '+np.str(ii)+' plotted.')
	del vz
