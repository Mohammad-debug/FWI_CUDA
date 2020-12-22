# Post processing in python


# reading the output arrays
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

import sys

def read_tensor(filename, dtype, dshape):
    '''
    reads the given tensor data from the filename 
    '''
    data = np.fromfile(filename, dtype=dtype)
    data = np.reshape(data, dshape)
    return data

vz_dat = read_tensor("../bin/shot0_vz.bin", np.float64, (50, 401, 401))

for i in range(0, 5):
    print(i, vz_dat.dtype, vz_dat[i,:,:])
    print("...........................")
clip = 0.01
for ii in range(1,50,1):
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
