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
snap_dt = 10; dt = 1.85e-5; dz = 0.02; dx = 0.02; # grid intervals

# Field parameters
#l_shift = 0.2 # shifting of the coord in discrete blocks in plots
l_uadd = 2.75
l_usl = 10.6
l_top = 3.0 +6*dx # correction for extended edges in figure
l_dsl = 10.6
l_dadd = 2.75
len = l_uadd + l_usl + l_top + l_dsl + l_dadd  # Meters

d_edge = -1.5
d_cor = -0.5
d_top = 0.5 + d_cor
d_wt = 1.0 + d_cor
d_sub = 4 + d_cor
d_tot = 7.0
dep = d_tot # Depth
npml_dz = 120
npml_fpad = 20



def ricker_wavelet(nt, dt, amp, fc, ts):
    #// Create signal
    #// **signal: The array in which signal is to be written
    #// nt: number of time steps, dt: time step size, ts: time shift
    #// fc: peak frequency, amp: amplitude of the signal

    fci = 1.0/fc;
    signal = np.zeros(nt)
    for it in range(0, nt):
        t = it * dt
        tau = math.pi * (t - 1.5 * fci - ts) / (1.5 * fci)
        signal[it] = amp*(1.0 - 2.0 * tau * tau) * math.exp(-2.0 * tau * tau)
        
    return signal

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
read_metaint("../FWD_PLOT_OUT/metaint.bin", np.int32)
snap_nt = np.int32(1 + (ndim[0]-1)//snap[6])
snap_nz = 1 + (snap[3] - snap[2])//snap[7]
snap_nx = 1 + (snap[4] - snap[5])//snap[8]

plot_ricker= True
plot_seismograms=False
plot_vel = False

if plot_ricker==True:
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Plotting the seismograms
    # ------------------------------------------------------------------------------
    rtf_uz = read_tensor("../FWD_PLOT_OUT/shot1_rtf_uz.bin", np.float64, (nrec, ndim[0]))

    nt = rtf_uz.shape[1]
    time = np.arange(0, nt*300)*dt
    tsignal = ricker_wavelet(nt*300, dt, 1.0, 80, 0)
    
    # Fourier transform
    fsignal = np.fft.fft(tsignal)
    freq = np.fft.fftfreq(tsignal.shape[-1], dt)
    #freq = freq[freq.size//2:-freq.size//2]
    #fsignal = fsignal[freq.size//2:-freq.size//2]
    arr1inds = freq.argsort()
    freq = freq[arr1inds[::-1]]
    fsignal = fsignal[arr1inds[::-1]]
    fs_max = np.max(abs(fsignal))
    fsignal = fsignal/fs_max
    
    fw, fh = set_size('thesis', subplots=(1,2))
    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)

    # initialize the figure plots
    fig, axs = plt.subplots(1,2, figsize=(fw, fh))
    #fig.tight_layout() 
    plt.subplots_adjust(top=0.85, bottom=0.12, hspace=0, wspace=0.4)
    # Hide the top and right spines of the axis
    # The first figure plot
    #axs.plot((-1, 10), (0, 0), color='k') # base line
    axs[0].plot(time[0:nt], tsignal[0:nt], ls='-', label='Ricker wavelet')

    # setting axis limits
    #axs[0].set_xlim(0, dt*nt)
    #axs.set_ylim(-0.15, 0.25)
    axs[0].set_xlabel( 'Time'+ r'$(s)$')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].set_title('(a)', loc='left')
    
    axs[1].plot(freq, np.abs(fsignal), ls='-', label='Real')
    #axs[1].plot(freq, fsignal.imag, ls='-', label='Imag')

    # setting axis limits
    axs[1].set_xlim(-255, 350)
    #axs.set_ylim(-0.15, 0.25)
    axs[1].set_xlabel( 'frequency'+ r'$(Hz)$')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc='upper right')
    axs[1].set_title('(b)', loc='left')
 
    
    fig.savefig('./ricker_wavelet.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if plot_seismograms==True:
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Plotting the seismograms
    # ------------------------------------------------------------------------------

    # Plot the rtf first
    print("NREC: ", nrec)
    rtf_uz = read_tensor("../FWD_PLOT_OUT/shot1_rtf_uz.bin", np.float64, (nrec, ndim[0]))
    rtf_ux = read_tensor("../FWD_PLOT_OUT/shot1_rtf_ux.bin", np.float64, (nrec, ndim[0]))

    #Plotting only the wavelets (source and receivers)
    # create an array
    nt = rtf_uz.shape[1]
    time = np.arange(0, nt)*dt
    # Plot figure for the paper
    # create the standard size for the figure
    fw, fh = set_size('thesis', subplots=(1,2))
    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)

    # initialize the figure plots
    fig, axs = plt.subplots(1,2, figsize=(fw, fh))
    fig.tight_layout() 
    plt.subplots_adjust(top=1, bottom=0.12, hspace=0, wspace=0.18)
    # Hide the top and right spines of the axis
    # The first figure plot
    axs[0].plot((-1, 10), (0, 0), color='k') # base line
    
    axs[0].plot(time, rtf_uz[1], ls='-', label='rec 1')
    axs[0].plot(time, rtf_uz[4], ls='-', label='rec 2')
    axs[0].plot(time, rtf_uz[7], ls='-', label='rec 3')
    axs[0].plot(time, rtf_uz[10], ls='-', label='rec 4')
    #axs[0].plot(time, rtf_uz[0], ls='-', color = 'k', label='Source')

    # setting axis limits
    axs[0].set_xlim(0, dt*nt)
    axs[0].set_ylim(-0.26, 0.46)
    axs[0].set_xlabel( 'Time'+ r'$(s)$')
    #axs[0].set_ylabel(r'$U_z$')
    axs[0].legend(ncol=2, loc=(0.06, 0.75))
    axs[0].set_title('(a) '+r'$V_z$', loc=('left'))
    #axs[0].grid()
    
    axs[1].plot((-1, 10), (0, 0), color='k') # base line
    
    axs[1].plot(time, rtf_ux[1], ls='-', label='rec 1')
    axs[1].plot(time, rtf_ux[4], ls='-', label='rec 2')
    axs[1].plot(time, rtf_ux[7], ls='-', label='rec 3')
    axs[1].plot(time, rtf_ux[10], ls='-', label='rec 4')
    #axs[1].plot(time, rtf_ux[0], ls='-', color = 'k', label='Source')

    # setting axis limits
    axs[1].set_xlim(0, dt*nt)
    axs[1].set_ylim(-0.15, 0.25)
    axs[1].set_xlabel( 'Time'+ r'$(s)$')
    #axs[1].set_ylabel(r'$U_x$')
    axs[1].legend(ncol=2, loc=(0.06, 0.75))
    axs[1].set_title('(b) '+r'$V_x$', loc=('left'))
    #axs[1].grid()

    # Save and remove excess whitespace

    fig.savefig('./seismograms_dam.pdf', format='pdf', bbox_inches='tight')
    plt.show()
   




if plot_vel==True:
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Plotting the velocity models 
    # ------------------------------------------------------------------------------
    
    vz_dat = read_tensor("../FWD_PLOT_OUT/shot1_vz.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    vx_dat = read_tensor("../FWD_PLOT_OUT/shot1_vx.bin", np.float64, (snap_nt, snap_nz, snap_nx))

    clip_pz = np.amax(vz_dat)
    clip_mz = np.amin(vz_dat)
    clipz = 0.5*max([clip_pz, np.abs(clip_mz)])*dt

    clip_px = np.amax(vx_dat)
    clip_mx = np.amin(vx_dat)
    clipx = 0.5*max([clip_px, np.abs(clip_mx)])*dt




    fw, fh = set_size('thesis', subplots=(4,1)) # from local module thesis_plot.py
    fig = plt.figure(figsize=(fw, fh*0.5))
    t_step = 350
    dt = 0.3e-4
    #vz = np.flipud(vz_dat[t_step,:,:])
    #vx = np.flipud(vx_dat[t_step,:,:]) 
    vz = vz_dat[t_step,:,:]
    vx = vx_dat[t_step,:,:]
    
    vz = vz[:vz.shape[0]-30,10:vz.shape[1]-10]*dt
    vx = vx[:vx.shape[0]-30,10:vx.shape[1]-10]*dt

    
    ez0 = d_edge
    ez1 = ez0 + vz.shape[0]*dz
    ex0 = -0.5*vz.shape[1]*dx + 0.5*dx
    ex1 = ex0 + vz.shape[1]*dx
    

    #vz = vz[41:][:]
    #vx = vx[40:][:]
    plt.subplots_adjust(top=0.99, bottom=0.15, hspace=0.5)
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # subplot 1
    #plt.subplot(411)
    ax1 = fig.add_subplot(411)
    t_step = 100
  
    vz = vz_dat[t_step,:,:]
    vx = vx_dat[t_step,:,:]
    vz = vz[:vz.shape[0]-30,10:vz.shape[1]-10]*dt
    vx = vx[:vx.shape[0]-30,10:vx.shape[1]-10]*dt

    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
     
    plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
    plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
    plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
    plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
    plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
    plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
    
    #-------------------------------------------------------------------------------------------------------

    plt.imshow(vx, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx, extent=[ex0, ex1, ez1, ez0])

    plt.xlabel('X'+r'$(m)$')
    plt.ylabel('Z '+r'$(m)$')
    #plt.colorbar()
    plt.text(5, 0, 'Time = '+np.format_float_scientific(t_step*dt, precision=2)+'s', fontsize=10)
    #plt.gca().invert_yaxis()
    plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # subplot 2
    #plt.subplot(412)
    ax2 = fig.add_subplot(412)
    t_step = 160
    
    #vz = np.flipud(vz_dat[t_step,:,:])
    #vx = np.flipud(vx_dat[t_step,:,:]) 
    vz = vz_dat[t_step,:,:]
    vx = vx_dat[t_step,:,:]
    vz = vz[:vz.shape[0]-30,10:vz.shape[1]-10]*dt
    vx = vx[:vx.shape[0]-30,10:vx.shape[1]-10]*dt

    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
     
    plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
    plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
    plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
    plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
    plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
    plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
    
    #-------------------------------------------------------------------------------------------------------

    plt.imshow(vx, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx, extent=[ex0, ex1, ez1, ez0])

    plt.xlabel('X'+r'$(m)$')
    plt.ylabel('Z '+r'$(m)$')
    #plt.colorbar()
    plt.text(5, 0, 'Time = '+np.format_float_scientific(t_step*dt, precision=2)+'s', fontsize=10)
    #plt.gca().invert_yaxis()
    plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------



    # subplot 3
    ax3 = fig.add_subplot(413)
    t_step = 240
   
    vz = vz_dat[t_step,:,:]
    vx = vx_dat[t_step,:,:]
    vz = vz[:vz.shape[0]-30,10:vz.shape[1]-10]*dt
    vx = vx[:vx.shape[0]-30,10:vx.shape[1]-10]*dt

    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
     
    plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
    plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
    plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
    plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
    plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
    plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
    
    #-------------------------------------------------------------------------------------------------------

    plt.imshow(vx, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx, extent=[ex0, ex1, ez1, ez0])

    plt.xlabel('X'+r'$(m)$')
    plt.ylabel('Z '+r'$(m)$')
    #plt.colorbar()
    plt.text(5, 0, 'Time = '+np.format_float_scientific(t_step*dt, precision=2)+'s', fontsize=10)
    #plt.gca().invert_yaxis()
    plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # subplot 4
    #plt.subplot(414)
    ax4 = fig.add_subplot(414)
    t_step = 300
 
    vz = vz_dat[t_step,:,:]
    vx = vx_dat[t_step,:,:]
    vz = vz[:vz.shape[0]-30,10:vz.shape[1]-10]*dt
    vx = vx[:vx.shape[0]-30,10:vx.shape[1]-10]*dt

    # plotting dam parameters
    # plotting dam parameters
    #------------------------------------------------------------------------------------------
     
    plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
    plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
    plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
    plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
    plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
    plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
    
    #-------------------------------------------------------------------------------------------------------

    plt.imshow(vx, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx, extent=[ex0, ex1, ez1, ez0])

    plt.xlabel('X'+r'$(m)$')
    plt.ylabel('Z '+r'$(m)$')
    #plt.colorbar()
    plt.text(5, 0, 'Time = '+np.format_float_scientific(t_step*dt, precision=2)+'s', fontsize=10)
    #plt.gca().invert_yaxis()
    plt.axis('equal')

    im_ax4= ax4.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx, extent=[ex0, ex1, ez1, ez0])

    cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    cb =fig.colorbar(im_ax4, cax=cbar_ax, orientation="horizontal")
    #cb.set_label('Velocity '+r'$(m/s)$')
    cb.set_label('Normalized Velocity')
    #plt.gca().invert_yaxis()
    #plt.axis('equal')
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------


    fig.savefig('./forward_dam_vx.pdf', format='pdf', bbox_inches='tight')
    plt.show()
