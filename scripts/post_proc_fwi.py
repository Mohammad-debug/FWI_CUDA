# Post processing in python
#%%

# reading the output arrays
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from thesis_plot import tex_fonts, set_size
# Set the rc parameter updates
plt.rcParams.update(tex_fonts)


import sys

# Field parameters
l_uadd = 1.0
l_usl = 12.0
l_top = 3.0
l_dsl = 12.0
l_dadd = 1.0
len = l_uadd + l_usl + l_top + l_dsl + l_dadd  # Meters

d_top = -0.5
d_wt = -1.0
d_sub = -4
d_tot = -7.0
dep = d_tot # Depth
npml_dz = 50
npml_fpad = 20

dt = 0.3e-4; dz = 0.1; dx = 0.1; # grid intervals

def ricker_wavelet(nt, dt, amp, fc, ts):
    #// Create signal
    #// **signal: The array in which signal is to be written
    #// nt: number of time steps, dt: time step size, ts: time shift
    #// fc: peak frequency, amp: amplitude of the signal

    fci = 1.0/fc

    for it in range(0, nt):
        t = it * dt
        tau = math.pi * (t - 1.5 * fci - ts) / (1.5 * fci)
        signal[it] = amp*(1.0 - 2.0 * tau * tau) * exp(-2.0 * tau * tau)

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




out_folder = "../bin_fwi_005_250hz_301_601"

# reading the input data for the array size
read_metaint(out_folder+"/metaint.bin", np.int32)
snap_nt = np.int32(1 + (ndim[0]-1)//snap[6])
snap_nz = 1 + (snap[3] - snap[2])//snap[7]
snap_nx = 1 + (snap[4] - snap[5])//snap[8]

# plotting the reciever data



if (fwinv):
    # Plotting the norm value

    fw, fh = set_size('thesis', subplots=(1,1)) # Getting the appropriate figure size
    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)
    fig, axs = plt.subplots(1,1, figsize=(fw*0.6, fh*0.6))
    L2_value =  np.array([2.10697e-06, 1.2298e-06, 8.82353e-07, 7.6485e-07, 6.99908e-07, 5.48706e-07, 5.4087e-07, 5.12752e-07, 3.37711e-07,\
     2.85707e-07, 2.5777e-07, 2.41264e-07, 1.81128e-07, 1.77516e-07, 1.71681e-07, 1.3195e-07, 1.27057e-07, 1.24284e-07, 1.22512e-07, \
     1.11081e-07, 1.06817e-07, 1.04266e-07, 9.23261e-08, 9.16882e-08, 9.09441e-08, 8.47406e-08, 8.21689e-08, 8.05121e-08, 7.42662e-08,\
      7.35254e-08, 7.21805e-08, 6.35775e-08, 6.31845e-08, 6.27997e-08, 6.02882e-08, 5.9557e-08, 5.90321e-08, 5.61836e-08, 5.48985e-08,\
       5.34787e-08, 4.38285e-08, 4.28885e-08, 4.22354e-08, 4.17351e-08, 3.86051e-08, 3.78265e-08, 3.73319e-08, 3.69398e-08, 3.43905e-08,\
        3.38015e-08, 3.35092e-08, 3.32817e-08, 3.19743e-08, 3.16029e-08, 3.12236e-08, 2.95511e-08, 2.95414e-08, 2.91593e-08, 2.89251e-08, \
        2.87567e-08, 2.81102e-08, 2.8094e-08, 2.80432e-08, 2.74592e-08, 2.70922e-08, 2.57038e-08, 2.56882e-08, 2.54371e-08, 2.53202e-08, \
        2.51661e-08, 2.44283e-08, 2.43777e-08, 2.4281e-08, 2.34147e-08, 2.31301e-08, 2.29845e-08, 2.28817e-08, 2.24503e-08, 2.24312e-08, \
        2.23887e-08, 2.19416e-08, 2.16951e-08, 2.15347e-08, 2.06103e-08, 2.0448e-08, 2.03771e-08, 2.03172e-08, 1.99822e-08, 1.99328e-08, \
        1.9884e-08, 1.95956e-08, 1.95598e-08, 1.95242e-08, 1.93176e-08, 1.92802e-08, 1.92366e-08, 1.88538e-08, 1.86657e-08, 1.85349e-08, \
        1.79217e-08, 1.78897e-08, 1.78137e-08, 1.78126e-08, 1.77168e-08, 1.7634e-08, 1.70766e-08], dtype=float)
    
    L2_norm = L2_value[0:100]/L2_value[0]
    plt.plot(L2_norm, label=r'$L_2-$'+'norm')
    plt.ylabel(r'$\frac{|L_2|}{|L_2(0)|}$')
    plt.xlabel('iterations')
    plt.legend()
    #plt.xscale('log')
    #fig.savefig('./fwi_L2_norm.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    #  print("Plotting material for iteration in fwi")
    # iterations to plot
    iter = [0, 4, 14, 29, 49, 74, 99]

    fw, fh = set_size('thesis', subplots=(2,4)) # Getting the appropriate figure size
    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)
    # initialize the figure plots
    # ----------------------------------------------------------------
    # -----------------------------------------------------------------
    # PLOTS FOR CP
    # ----------------------------------------------
    fig, axs = plt.subplots(2,4, figsize=(fw, fh*3.2))
    fig.tight_layout() 
    plt.subplots_adjust(left=0.08, right=0.976, top=0.986, bottom=0.14, hspace=0.13, wspace=0.54)
    
    clip_min = 400
    clip_max = 900
    
    # 1
    mat_dat = read_tensor(out_folder+"/start_mat.bin", np.float64, (3, ndim[1], ndim[2]))
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 0].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,0].set_title('Start model', loc=('center'))
    
    
    # 2
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[1])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 1].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,1].set_title('Iteration:'+np.str(iter[1]+1), loc=('center'))
    
    # 3
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[2])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 2].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,2].set_title('Iteration:'+np.str(iter[2]+1), loc=('center'))
    
    # 4
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[3])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 3].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,3].set_title('Iteration:'+np.str(iter[3]+1), loc=('center'))
    
    # ---------------------------------------------------------------------------
    
    # 5
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[4])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[1, 0].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,0].set_title('Iteration:'+np.str(iter[4]+1), loc=('center'))
    
    # 6
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[5])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[1, 1].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,1].set_title('Iteration:'+np.str(iter[5]+1), loc=('center'))
    
    # 7
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[6])+"_mat.bin", np.float64, (3, ndim[1], ndim[2]))
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    get_im = axs[1, 2].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,2].set_title('Iteration:'+np.str(iter[6]+1), loc=('center'))
    
    # 8
    mat_dat = read_tensor(out_folder+"/true_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[1, 3].imshow(Cp, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,3].set_title('True model', loc=('center'))
    
    cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    cb =fig.colorbar(get_im, cax=cbar_ax, orientation="horizontal")
    cb.set_label(r'$c_1(m/s)$')

    # set labels 
    for ii in range(0,2):
        for jj in range(0,4):
            axs[ii, jj].set_xlabel(r'$x \ (m)$')
            axs[ii, jj].set_ylabel(r'$z \ (m)$')
            axs[ii, jj].set_xticks(np.linspace(0,15,4))
            
    
    fig.savefig('./fwi_cp.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    # ----------------------------------------------------------------
    # -----------------------------------------------------------------
    # PLOTS FOR CS
    # ----------------------------------------------
    fig, axs = plt.subplots(2,4, figsize=(fw, fh*3.2))
    fig.tight_layout() 
    plt.subplots_adjust(left=0.08, right=0.976, top=0.986, bottom=0.14, hspace=0.13, wspace=0.54)
    
    clip_min = 250
    clip_max = 450
    
    # 1
    mat_dat = read_tensor(out_folder+"/start_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 0].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,0].set_title('Start model', loc=('center'))
   
    
    # 2
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[1])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 1].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,1].set_title('Iteration:'+np.str(iter[1]+1), loc=('center'))
    
    # 3
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[2])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 2].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,2].set_title('Iteration:'+np.str(iter[2]+1), loc=('center'))
    
    # 4
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[3])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[0, 3].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[0,3].set_title('Iteration:'+np.str(iter[3]+1), loc=('center'))
    
    # ---------------------------------------------------------------------------
    
    # 5
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[4])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[1, 0].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,0].set_title('Iteration:'+np.str(iter[4]+1), loc=('center'))
    
    # 6
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[5])+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[1, 1].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,1].set_title('Iteration:'+np.str(iter[5]+1), loc=('center'))
    
    # 7
    mat_dat = read_tensor(out_folder+"/iter"+np.str(iter[6])+"_mat.bin", np.float64, (3, ndim[1], ndim[2]))
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    get_im = axs[1, 2].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,2].set_title('Iteration:'+np.str(iter[6]+1), loc=('center'))
    
    # 8
    mat_dat = read_tensor(out_folder+"/true_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
    lam = mat_dat[0][:][:]
    mu = mat_dat[1][:][:]
    rho = mat_dat[2][:][:]
    Cs = np.sqrt(mu/rho)
    Cp = np.sqrt((lam + 2 * mu)/rho)
    axs[1, 3].imshow(Cs, animated=True, interpolation='nearest', vmin=clip_min, vmax=clip_max, cmap=cm.Blues, extent=[0, 15, 30, 0]) 
    axs[1,3].set_title('True model', loc=('center'))
    
    cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    cb =fig.colorbar(get_im, cax=cbar_ax, orientation="horizontal")
    cb.set_label(r'$c_2(m/s)$')

    # set labels 
    for ii in range(0,2):
        for jj in range(0,4):
            axs[ii, jj].set_xlabel(r'$x \ (m)$')
            axs[ii, jj].set_ylabel(r'$z \ (m)$')
            axs[ii, jj].set_xticks(np.linspace(0,15,4))
    
    fig.savefig('./fwi_cs.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    
    exit()
    
    
    maxiter = 700
    for ii in range(600,maxiter,1):
        # reading data from csv file
        mat_dat = read_tensor(out_folder+"/iter"+np.str(ii)+"_mat.bin", np.float64, (3, ndim[1], ndim[2])) 
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
    rtf_uz = read_tensor(out_folder+"/shot2_rtf_uz.bin", np.float64, (nrec, ndim[0]))
    rtf_ux = read_tensor(out_folder+"/shot2_rtf_ux.bin", np.float64, (nrec, ndim[0]))
    
    #Plotting only the wavelets (source and receivers)
    # create an array
    dt = 0.3e-4
    time = np.arange(0, rtf_uz.shape[1])*dt
    # Plot figure for the paper
    # create the standard size for the figure
    fw, fh = set_size('thesis', subplots=(1,2))
    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)
    
    
    # initialize the figure plots
    fig, axs = plt.subplots(1,2, figsize=(fw, fh))
    fig.tight_layout() 
    plt.subplots_adjust(top=1, bottom=0.12, hspace=0)
    # Hide the top and right spines of the axis
    # The first figure plot
    axs[0].plot(time, rtf_uz[1], label='rec 1')
    axs[0].plot(time, rtf_uz[4], label='rec 2')
    axs[0].plot(time, rtf_uz[7], label='rec 3')
    axs[0].plot(time, rtf_uz[10], label='rec 4')


    # setting axis limits
    #axs[0].set_xlim(-1.5, 1.5)
    #axs[0].set_ylim(0, 5)
    axs[0].set_xlabel( 'Time'+ r'$(sec)$')
    axs[0].set_ylabel(r'$|U_z (m)|$')
    axs[0].legend(ncol=2, loc=(0., 0.75))
    axs[0].set_title('(a)', loc=('left'))

    axs[1].plot(time, rtf_ux[1], label='rec 1')
    axs[1].plot(time, rtf_ux[4], label='rec 1')
    axs[1].plot(time, rtf_ux[7], label='rec 1')
    axs[1].plot(time, rtf_ux[10], label='rec 1')

    # setting axis limits
    #axs[1].set_xlim(-1.5, 1.5)
    #axs[1].set_ylim(0, 5)
    axs[1].set_xlabel(r'$x_1/a$')
    axs[1].set_ylabel(r'$|U_i|$')
    axs[1].legend(ncol=2, loc=(0., 0.75))
    axs[1].set_title('(b)', loc=('left'))

    # Save and remove excess whitespace

    fig.savefig('./seismograms_dam.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    exit()
    
    vz_dat = read_tensor(out_folder+"/shot2_vz.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    vx_dat = read_tensor(out_folder+"/shot2_vx.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    
    clip_pz = np.amax(vz_dat)
    clip_mz = np.amin(vz_dat)
    clipz = 0.3*max([clip_pz, np.abs(clip_mz)])
    
    clip_px = np.amax(vx_dat)
    clip_mx = np.amin(vx_dat)
    clipx = 0.3*max([clip_px, np.abs(clip_mx)])
    
    
    
    
    fw, fh = set_size('thesis', subplots=(4,1)) # from local module thesis_plot.py
    fig = plt.figure(figsize=(fw, fh*0.5))
    t_step = 350
    dt = 0.3e-4
    vz = np.flipud(vz_dat[t_step,:,:])
    vx = np.flipud(vx_dat[t_step,:,:]) 
    
    vz = vz[41:][:]
    vx = vx[40:][:]
    plt.subplots_adjust(top=1, bottom=0.12, hspace=0.1)
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # subplot 1
    #plt.subplot(411)
    ax1 = fig.add_subplot(411)
    t_step = 100
    dt = 0.3e-4
    vz = np.flipud(vz_dat[t_step,:,:])
    vx = np.flipud(vx_dat[t_step,:,:]) 
    
    vz = vz[41:][:]
    vx = vx[40:][:]
    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
    plt.plot(((l_uadd+l_usl)/dx+npml_fpad, (l_uadd+l_usl+l_top)/dx+npml_fpad), \
        (d_top/dz+npml_dz+1, d_top/dz+npml_dz+1), color='k', lw =1.0) # top of the dam
    plt.plot(((l_uadd)/dx+2*npml_fpad-8, (l_uadd+l_usl)/dx+npml_fpad), \
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# upslope of the dam 
    plt.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (l_uadd+l_usl+l_top)/dx+npml_fpad),\
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# downslope of the dam   
    plt.plot((0 , (l_uadd)/dx+2*npml_fpad-8), ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface up
    plt.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (len+l_dadd)/dx+2*npml_fpad+3), \
        ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface down
    plt.plot((0, l_usl/dx+npml_fpad-8), (d_wt/dz+npml_dz, d_wt/dz+npml_dz), color='b', lw =1.0) # water level
    #-------------------------------------------------------------------------------------------------------
    
    ax1.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx)
    
    plt.xlim((20,300))
    plt.ylim((0,60))
    plt.xlabel('X'+r'$(dm)$')
    plt.ylabel('Z '+r'$(dm)$')
    #plt.colorbar()
    plt.text(50, 50, 'Time = '+np.format_float_scientific(t_step*5*dt, precision=2)+'s', fontsize=10)
    #plt.gca().invert_yaxis()
    #plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    
    # subplot 2
    plt.subplot(412)
    t_step = 200
    dt = 0.3e-4
    vz = np.flipud(vz_dat[t_step,:,:])
    vx = np.flipud(vx_dat[t_step,:,:]) 
    vz = vz[41:][:]
    vx = vx[40:][:]
    
    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
    plt.plot(((l_uadd+l_usl)/dx+npml_fpad, (l_uadd+l_usl+l_top)/dx+npml_fpad), \
        (d_top/dz+npml_dz+1, d_top/dz+npml_dz+1), color='k', lw =1.0) # top of the dam
    plt.plot(((l_uadd)/dx+2*npml_fpad-8, (l_uadd+l_usl)/dx+npml_fpad), \
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# upslope of the dam 
    plt.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (l_uadd+l_usl+l_top)/dx+npml_fpad),\
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# downslope of the dam   
    plt.plot((0 , (l_uadd)/dx+2*npml_fpad-8), ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface up
    plt.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (len+l_dadd)/dx+2*npml_fpad+3), \
        ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface down
    plt.plot((0, l_usl/dx+npml_fpad-8), (d_wt/dz+npml_dz, d_wt/dz+npml_dz), color='b', lw =1.0) # water level
    #-------------------------------------------------------------------------------------------------------
    
    plt.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx)
    
    plt.xlim((20,300))
    plt.ylim((0,60))
    plt.xlabel('X'+r'$(dm)$')
    plt.ylabel('Z '+r'$(dm)$')
    #plt.colorbar()
    plt.text(50, 50, 'Time = '+np.format_float_scientific(t_step*5*dt, precision=2)+'s', fontsize=10)
    #plt.gca().invert_yaxis()
    #plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    
    
    
    # subplot 3
    ax3 = fig.add_subplot(413)
    t_step = 300
    dt = 0.3e-4
    vz = np.flipud(vz_dat[t_step,:,:])
    vx = np.flipud(vx_dat[t_step,:,:]) 
    vz = vz[41:][:]
    vx = vx[40:][:]
    
    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
    ax3.plot(((l_uadd+l_usl)/dx+npml_fpad, (l_uadd+l_usl+l_top)/dx+npml_fpad), \
        (d_top/dz+npml_dz+1, d_top/dz+npml_dz+1), color='k', lw =1.0) # top of the dam
    ax3.plot(((l_uadd)/dx+2*npml_fpad-8, (l_uadd+l_usl)/dx+npml_fpad), \
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# upslope of the dam 
    ax3.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (l_uadd+l_usl+l_top)/dx+npml_fpad),\
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# downslope of the dam   
    ax3.plot((0 , (l_uadd)/dx+2*npml_fpad-8), ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface up
    ax3.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (len+l_dadd)/dx+2*npml_fpad+3), \
        ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface down
    ax3.plot((0, l_usl/dx+npml_fpad-8), (d_wt/dz+npml_dz, d_wt/dz+npml_dz), color='b', lw =1.0) # water level
    #-------------------------------------------------------------------------------------------------------
    
    ax3.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx)
    
    plt.xlim((20,300))
    plt.ylim((0,60))
    plt.xlabel('X'+r'$(dm)$')
    plt.ylabel('Z '+r'$(dm)$')
    #plt.colorbar()
    plt.text(50, 50, 'Time = '+np.format_float_scientific(t_step*5*dt, precision=2)+'s', fontsize=10)
    #plt.gca().invert_yaxis()
    #plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    
    # subplot 4
    #plt.subplot(414)
    ax4 = fig.add_subplot(414)
    t_step = 400
    dt = 0.3e-4
    vz = np.flipud(vz_dat[t_step,:,:])
    vx = np.flipud(vx_dat[t_step,:,:]) 
    vz = vz[41:][:]
    vx = vx[40:][:]
    
    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
    ax4.plot(((l_uadd+l_usl)/dx+npml_fpad, (l_uadd+l_usl+l_top)/dx+npml_fpad), \
        (d_top/dz+npml_dz+1, d_top/dz+npml_dz+1), color='k', lw =1.0) # top of the dam
    ax4.plot(((l_uadd)/dx+2*npml_fpad-8, (l_uadd+l_usl)/dx+npml_fpad), \
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# upslope of the dam 
    ax4.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (l_uadd+l_usl+l_top)/dx+npml_fpad),\
        ((d_sub)/dz+npml_dz, d_top/dz+npml_dz+1), color='k', lw =1.0)# downslope of the dam   
    ax4.plot((0 , (l_uadd)/dx+2*npml_fpad-8), ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface up
    ax4.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (len+l_dadd)/dx+2*npml_fpad+3), \
        ((d_sub)/dz+npml_dz, (d_sub)/dz+npml_dz), color='k', lw =1.0) # subsurface down
    ax4.plot((0, l_usl/dx+npml_fpad-8), (d_wt/dz+npml_dz, d_wt/dz+npml_dz), color='b', lw =1.0) # water level
    #-------------------------------------------------------------------------------------------------------
    
    im_ax4= ax4.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx)
    plt.text(50, 50, 'Time = '+np.format_float_scientific(t_step*5*dt, precision=2)+'s', fontsize=10)
    plt.xlim((20,300))
    plt.ylim((0,60))
    plt.xlabel('X'+r'$(dm)$')
    plt.ylabel('Z '+r'$(dm)$')

    cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    cb =fig.colorbar(im_ax4, cax=cbar_ax, orientation="horizontal")
    cb.set_label('Velocity '+r'$(m/s)$')
    #plt.gca().invert_yaxis()
    #plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    
   
    fig.savefig('./forward_dam_vz.pdf', format='pdf', bbox_inches='tight')
    plt.show()
  