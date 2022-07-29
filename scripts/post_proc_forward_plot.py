# Post processing in python
#%%

# reading the output arrays
from warnings import WarningMessage
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.mplot3d import Axes3D
from fractions import Fraction as frac
from matplotlib.ticker import FuncFormatter, MultipleLocator
#from .pi_formatter import 

from thesis_plot import tex_fonts, set_size
# Set the rc parameter updates
plt.rcParams.update(tex_fonts)

import sys

def pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """
    format label properly
    for example: 0.6666 pi --> 2π/3
               : 0      pi --> 0
               : 0.50   pi --> π/2  
    """
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val/np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator
    
    fmt2 = "%s" % d 
    if n == 0:
        fmt1 = "0"
    elif n == 1:
        fmt1 = pi
    else:
        fmt1 = r"%s%s" % (n,pi)
        
    fmtstring = "$" + minus + (fmt1 if d == 1 else r"{%s}/{%s}" % (fmt1, fmt2)) + "$"
    
    return fmtstring

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
    
def read_metafloat(filename, dtype):
    
    '''
    reads the float metadata from input file
    '''
    global dt, dz, dx, freq
    floatdata = np.fromfile(filename, dtype=dtype)
    # nt, nz, nx
    print(floatdata)
    dt = floatdata[0]
    dz = floatdata[1]
    dx = floatdata[2]
    freq = floatdata[7]
    
def read_tensor(filename, dtype, dshape):
    '''
    reads the given tensor data from the filename 
    '''
    data = np.fromfile(filename, dtype=dtype)
    data = np.reshape(data, dshape)
    return data



# reading the input data for the array size
read_metaint("./FWD_PLOT_OUT/metaint.bin", np.int32)
snap_dt = snap[6]
snap_nt = np.int32(1 + (ndim[0]-1)//snap[6])
snap_nz = 1 + (snap[3] - snap[2])//snap[7]
snap_nx = 1 + (snap[4] - snap[5])//snap[8]

read_metafloat("./FWD_PLOT_OUT/metafloat.bin", np.float64) # reads dt, dz, dx, freq

# Field parameters
#l_shift = 0.2 # shifting of the coord in discrete blocks in plots
l_uadd = 2.75
l_usl = 12
l_top = 3.0 #+6*dx # correction for extended edges in figure
l_dsl = 12
l_dadd = 2.75
len = l_uadd + l_usl + l_top + l_dsl + l_dadd  # Meters

d_edge = -1.5
d_cor = -1.5
d_top = 0.5 + d_cor
d_wt = 1.0 + d_cor
d_sub = 4.5 + d_cor
d_tot = 7.0
dep = d_tot # Depth
npml_dz = 120
npml_fpad = 20


plot_ricker= False
plot_seismograms = True
plot_vel = False
plot_wiggle = False

# time step plots
ts_plot = [50, 100, 200, 400]
plt_fmt = 'pdf' # 'pdf' or 'png' 



if plot_ricker==True:
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Plotting the seismograms
    # ------------------------------------------------------------------------------
    rtf_uz = read_tensor("./FWD_PLOT_OUT/shot0_rtf_uz.bin", np.float64, (nrec, ndim[0]))

    nt = int(0.01/dt) #rtf_uz.shape[1]
    time = np.arange(0, nt)*dt*1.0e3 # milli seconds
    tsignal = ricker_wavelet(nt, dt, 1.0, freq, 0)
    
    # Fourier transform
    fsignal = np.fft.fft(tsignal)
    freq = np.fft.fftfreq(tsignal.shape[-1], dt)
    #freq = freq[freq.size//2:-freq.size//2]
    #fsignal = fsignal[freq.size//2:-freq.size//2]
    #arr1inds = freq.argsort()
    #freq = freq[arr1inds[::-1]]
    #fsignal = fsignal[arr1inds[::-1]]
    fs_max = np.max(abs(fsignal))
    fsignal_amp = np.abs(fsignal)/fs_max
    freq = freq*1.0e-3
    
    fw, fh = set_size('thesis', subplots=(1,2))
    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)
    #plt.rcParams["figure.figsize"] = [fw, fh]
    #plt.rcParams["figure.autolayout"] = True
    # initialize the figure plots
    fig, axs = plt.subplots(1,2, figsize=(fw*1.1, fh*1.2))
    #fig.tight_layout() 
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9, hspace=0.0, wspace=0.3)
    # Hide the top and right spines of the axis
    # The first figure plot
    #axs.plot((-1, 10), (0, 0), color='k') # base line
    axs[0].plot(time[0:nt], tsignal[0:nt], ls='-', label='Ricker wavelet')

    # setting axis limits
    #axs[0].set_xlim(0, dt*nt)
    #axs.set_ylim(-0.15, 0.25)
    axs[0].set_xlabel( 'Time '+ r'$(m s)$')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].set_title('(a)', loc='left')
    
    axs[1].plot(freq[:freq.size//2], np.abs(fsignal_amp[:freq.size//2]), ls='-', label='Amplitude')
    #axs[1].plot(freq, fsignal.imag, ls='-', label='Imag')

    # setting axis limits
    axs[1].set_xlim(0, 2.5)
    #axs.set_ylim(-0.15, 0.25)
    axs[1].set_xlabel( 'Frequency '+ r'$(kHz)$')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc=(0.5, 0.87))
    axs[1].set_title('(b)', loc='left')
    
    ax2 = axs[1].twinx()
    color = 'tab:red'
    ticklen = np.pi/2
    theta = np.angle(fsignal[:freq.size//2]) #/np.pi
    ax2.set_ylabel('Phase') #, color=color)  # we already handled the x-label with ax1
    ax2.plot(freq[:freq.size//2], theta, color=color, label='Phase')
    ax2.tick_params(axis='y') #, labelcolor=color)
    ax2.set_ylim(-3.2, 4.6)
    ax2.legend(loc=(0.5, 0.74))
    # setting ticks labels
    ax2.yaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
    # setting ticks at proper numbers
    ax2.yaxis.set_major_locator(MultipleLocator(base=ticklen))
    
 
    
    fig.savefig('./ricker_wavelet.pdf', format='pdf', bbox_inches='tight')
    plt.show()

if plot_seismograms==True:
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Plotting the seismograms
    # ------------------------------------------------------------------------------

    # Plot the rtf first
    print("NREC: ", nrec)
    rtf_uz = read_tensor("./FWD_PLOT_OUT/shot0_rtf_uz.bin", np.float64, (nrec, ndim[0]))
    rtf_ux = read_tensor("./FWD_PLOT_OUT/shot0_rtf_ux.bin", np.float64, (nrec, ndim[0]))

    # Plotting only the wavelets (source and receivers)
    # Create an array
    nrec = rtf_uz.shape[0]
    nt = rtf_uz.shape[1]
    time = np.arange(0, nt)*dt
    # Plot figure for the paper
    # create the standard size for the figure
    fw, fh = set_size('thesis', subplots=(1,1))
    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)

    # initialize the figure plots
    fig, axs = plt.subplots(1,2, figsize=(fw, fh))
    fig.tight_layout() 
    plt.subplots_adjust(top=0.935,bottom=0.125,left=0.09,right=0.95,hspace=0.455,wspace=0.31)
    
    x_spacing = 0.5 #(l_usl+l_top+l_dsl)/(nrec-1)
    xsrc = 12 # Position of the source from the left end
    
    #nt = int(0.06/dt)
    rtf = rtf_uz
    time = np.arange(0, nt)*dt
    axs[0].set_title('(a) ', loc=('left'))
    axs[1].set_title('(b) ', loc=('left'))
    outfile = './seismograms_dam_vz.pdf'
    #axs[0].set_xlim(0, dt*nt*0.6)
    
    #axs[0].plot((-1, 10), (0, 0), color='k') # base line
    
    counter = 0
    for ii in range(24, nrec, 6):
        dist = ii*x_spacing-13.5
        dist_str = "{:.2f}".format(dist)
        if dist<0:
            ls = '-'
        else:
            ls = '-'
            
        axs[0].plot(time, counter+3*rtf[ii][0:nt], ls=ls, label='x = '+dist_str+r' $m$')
        axs[0].set_xlim(-0.001, 0.06)
        
        # Frequency domain plot
        fsignal = np.fft.fft(rtf[ii][0:nt])
        freq = np.fft.fftfreq(rtf[ii][0:nt].shape[-1], dt)
        fs_max = np.max(abs(fsignal))
        fsignal_amp = np.abs(fsignal)/fs_max
        freq = freq*1.0e-3
        
        axs[1].plot(freq[:freq.size//2], counter+np.abs(fsignal_amp[:freq.size//2]), ls=ls, label='x = '+dist_str+r' $m$')
        
        # writing the text file of the frequency data
        dat_csv = np.zeros((2, nt), dtype = float)
        dat_csv[0] = time
        dat_csv[1] = rtf[ii][0:nt]
        dt_str =  "{:.5e}".format(dt)
        dat_head = 'loc: X = '+ dist_str+'m; \n time step: dt = '+dt_str+ 's \n col1: time(s); \n col2: amplitude (normalized to the source wavelet)'
        np.savetxt('./dyke_simulation_rtf_'+dist_str+'m.csv', dat_csv.T, delimiter=', ', header=dat_head, fmt='%.5e')
        
        counter +=1
    
    # setting axis limits
    
    #axs[0].set_ylim(-0.26, 0.46)
    axs[0].set_xlabel( 'Time '+ r'$(s)$')
    axs[0].set_ylabel('Transducer position')
    #axs[0].legend(ncol=3)#, loc=(0.2, -0.6))
    
    axs[1].set_xlabel( 'Frequency '+ r'$(kHz)$')
    axs[1].set_ylabel('Transducer position')
    #axs[1].legend(ncol=1)#, loc=(0.2, -0.6))
    axs[1].set_xlim(0, 2.5)
    #axs.set_title('(a) '+r'$v_z$', loc=('left'))
    '''
    # initialize the figure plots
    fig, axs = plt.subplots(1,2, figsize=(fw, fh))
    fig.tight_layout() 
    plt.subplots_adjust(top=0.895, bottom=0.35, left=0.117, right=0.97,  hspace=0.0, wspace=0.4)
    # Hide the top and right spines of the axis
    # The first figure plot
    x_spacing = 0.5 #(l_usl+l_top+l_dsl)/(nrec-1)
    xsrc = 12 # Position of the source from the left end
    
    axs[0].plot((-1, 10), (0, 0), color='k') # base line
    axs[1].plot((-1, 10), (0, 0), color='k') # base line
    for ii in range(24, nrec, 6):
        dist = ii*x_spacing-12.0
        dist_str = "{:.2f}".format(dist)
        if dist<0:
            ls = '--'
        else:
            ls = '-'
        axs[0].plot(time, rtf_uz[ii], ls=ls, label=dist_str+r' $m$')
        axs[1].plot(time, rtf_ux[ii], ls=ls, label=dist_str+r' $m$')
    #axs[1].plot(time, rtf_ux[0], ls='-', color = 'k', label='Source')

    # setting axis limits
    axs[0].set_xlim(0, dt*nt*2/3)
    #axs[0].set_ylim(-0.26, 0.46)
    axs[0].set_xlabel( 'Time'+ r'$(s)$')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(ncol=4, loc=(0.2, -0.6))
    axs[0].set_title('(a) '+r'$v_z$', loc=('left'))
        
    # setting axis limits
    axs[1].set_xlim(0, dt*nt*2/3)
    #axs[1].set_ylim(-0.15, 0.25)
    axs[1].set_xlabel( 'Time'+ r'$(s)$')
    axs[1].set_ylabel('Amplitude')
    #axs[1].legend(ncol=2, loc=(0.06, 0.75))
    axs[1].set_title('(b) '+r'$v_x$', loc=('left'))
    #axs[1].grid()
    '''

    # Save and remove excess whitespace

    fig.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.show()
    
    
if plot_wiggle == True:
    '''
    wiggle plots of the receiver time functions
    '''
    # Plot the rtf first
    print("NREC: ", nrec)
    rtf_uz = read_tensor("./FWD_PLOT_OUT/shot0_rtf_uz.bin", np.float64, (nrec, ndim[0]))
    rtf_ux = read_tensor("./FWD_PLOT_OUT/shot0_rtf_ux.bin", np.float64, (nrec, ndim[0]))

    #Plotting only the wavelets (source and receivers)
    # create an array
    nt = 2*rtf_uz.shape[1]//3
    nrec = rtf_uz.shape[0]
    xrec = np.linspace(-13.5, 13.5, nrec)
    time = np.arange(0, nt)*dt
    
    fw, fh = set_size('thesis', subplots=(1,2)) # from local module thesis_plot.py
    fig, axs = plt.subplots(1, 2, figsize=(fw, fh*2))
    fig.tight_layout() 
    plt.subplots_adjust(top=0.898, bottom=0.149, left=0.115, right=0.975, hspace=0.5, wspace=0.308)
    
    x_spacing = (l_usl+l_top+l_dsl)/(nrec-1)
    for kk in range(nrec):
        # Normalizing the amplitudes
        amp = abs(np.max(rtf_uz[kk][:]) - np.min(rtf_uz[kk][:]))
        trz = xrec[kk] + 1*x_spacing*rtf_uz[kk][:]/amp
        axs[0].plot(time[:nt], trz[:nt],  lw=0.5, color='k')
        axs[0].plot(time[:nt], trz[:nt],  lw=0.5, color='k')
          
        amp = abs(np.max(rtf_ux[kk][:]) - np.min(rtf_ux[kk][:]))*0.5
        trx = xrec[kk] + 1*x_spacing*rtf_ux[kk][:]/amp
        axs[1].plot(time[:nt], trz[:nt],  lw=0.5, color='k')
        axs[1].plot(time[:nt], trz[:nt],  lw=0.5, color='k')
    axs[0].plot((0.0), (-1.5), ls ='', marker='*', markersize= 7,
                markerfacecolor="tab:green", markeredgecolor="tab:green", label = 'Source position') # source position
    axs[0].legend(loc=(0.9, -0.15)) 
    axs[1].plot((0.0), (-1.5), ls ='', marker='*', markersize= 7,
                markerfacecolor="tab:green", markeredgecolor="tab:green", label = 'Source position') # source position
    axs[0].set_ylabel('horizontal distance (m)')
    axs[0].set_xlabel('time (s)')
    axs[1].set_ylabel('horizontal distance (m)')
    axs[1].set_xlabel('time (s)')
    

    #axs.set_xlim(-0.5, 18)
    #axs.set_ylim(0.0, 0.3)
    axs[0].set_title('(a) ' + r'$v_{z}$', loc=('left'))
    axs[1].set_title('(b) ' + r'$v_{x}$', loc=('left'))
    
    fig.savefig('./traces_wiggle_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if plot_vel == True:
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Plotting the velocity models 
    # ------------------------------------------------------------------------------    

    for ii in range(2):
        if ii == 0: # Plot in z direction
            plt_dir = 'z'
            v_dat = read_tensor("./FWD_PLOT_OUT/shot0_vz.bin", np.float64, (snap_nt, snap_nz, snap_nx))
            
        elif ii == 1: #plot in x direction
            plt_dir = 'x'
            v_dat = read_tensor("./FWD_PLOT_OUT/shot0_vx.bin", np.float64, (snap_nt, snap_nz, snap_nx))
            
        clip_p = np.amax(v_dat)
        clip_m = np.amin(v_dat)
        clip = 0.1*max([clip_p, np.abs(clip_m)])#*dt

        fw, fh = set_size('thesis', subplots=(4,1)) # from local module thesis_plot.py
        fig = plt.figure(figsize=(fw, fh*0.5))
    
        #vz = np.flipud(vz_dat[t_step,:,:])
        #vx = np.flipud(vx_dat[t_step,:,:]) 
        
        v = v_dat[0,:,:]
        
        # The extents of the plot in Cartesian coordinates
        ez0 = d_edge
        ez1 = ez0 + v.shape[0]*dz
        ex0 = -0.5*v.shape[1]*dx + 0.5*dx
        ex1 = ex0 + v.shape[1]*dx

        #vz = vz[41:][:]
        #vx = vx[40:][:]
        plt.subplots_adjust(top=0.99, bottom=0.15, hspace=0.5)
        # ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # subplot 1
        #plt.subplot(411)
        ax1 = fig.add_subplot(411)
        t_step = ts_plot[0]
         
        v = v_dat[t_step,:,:]
    
        # plotting dam parameters
        #------------------------------------------------------------------------------------------
        
        plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
        plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
        plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
        plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
        plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
        plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
        
        #-------------------------------------------------------------------------------------------------------

        plt.imshow(v, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clip, vmax=clip, extent=[ex0, ex1, ez1, ez0])
        plt.plot((-l_top/2), (d_top), ls ='', marker='*', markersize= 7,
                markerfacecolor="tab:green", markeredgecolor="tab:green", label = 'Source position') # source position
        
        plt.xlabel('X'+r'$(m)$')
        plt.ylabel('Z '+r'$(m)$')
        #plt.colorbar()
        plt.text(7, -1, 'Time = '+np.format_float_scientific(t_step*snap_dt*dt, precision=2)+'s', fontsize=10)
        #plt.gca().invert_yaxis()
        plt.axis('equal')
        
        # ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # subplot 2
        #plt.subplot(412)
        ax2 = fig.add_subplot(412)
        t_step = ts_plot[1]
        
        v = v_dat[t_step,:,:]
    
        # plotting dam parameters
        #------------------------------------------------------------------------------------------
        
        plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
        plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
        plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
        plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
        plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
        plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
        
        #-------------------------------------------------------------------------------------------------------

        plt.imshow(v, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clip, vmax=clip, extent=[ex0, ex1, ez1, ez0])
        plt.plot((-l_top/2), (d_top), ls ='', marker='*', markersize= 7,
                markerfacecolor="tab:green", markeredgecolor="tab:green", label = 'Source position') # source position
        
        plt.xlabel('X'+r'$(m)$')
        plt.ylabel('Z '+r'$(m)$')
        #plt.colorbar()
        plt.text(7, -1, 'Time = '+np.format_float_scientific(t_step*snap_dt*dt, precision=2)+'s', fontsize=10)
        #plt.gca().invert_yaxis()
        plt.axis('equal')
        # ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # subplot 3
        ax3 = fig.add_subplot(413)
        t_step = ts_plot[2]
    
        v = v_dat[t_step,:,:]

        # plotting dam parameters
        #------------------------------------------------------------------------------------------
        
        plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
        plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
        plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
        plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
        plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
        plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
        
        #-------------------------------------------------------------------------------------------------------

        plt.imshow(v, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clip, vmax=clip, extent=[ex0, ex1, ez1, ez0])
        plt.plot((-l_top/2), (d_top), ls ='', marker='*', markersize= 7,
                markerfacecolor="tab:green", markeredgecolor="tab:green", label = 'Source position') # source position

        plt.xlabel('X'+r'$(m)$')
        plt.ylabel('Z '+r'$(m)$')
        #plt.colorbar()
        plt.text(7, -1, 'Time = '+np.format_float_scientific(t_step*snap_dt*dt, precision=2)+'s', fontsize=10)
        #plt.gca().invert_yaxis()
        plt.axis('equal')
        # ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # subplot 4
        #plt.subplot(414)
        ax4 = fig.add_subplot(414)
        t_step = ts_plot[3]
    
        v = v_dat[t_step,:,:]

        # plotting dam parameters
        #------------------------------------------------------------------------------------------
        
        plt.plot(((-l_top/2), (l_top/2)), (d_top, d_top), color='k', lw =1.0) # top of the dam
        plt.plot(((-l_top/2), (-l_top/2-l_usl)), (d_top, d_sub), color='k', lw =1.0) # upslope of the dam
        plt.plot(((l_top/2), (l_top/2+l_dsl)), (d_top, d_sub), color='k', lw =1.0) # downslope of the dam
        plt.plot(((-l_top/2-l_usl-l_uadd), (-l_top/2-l_usl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface up
        plt.plot(((l_top/2+l_dsl+l_dadd), (l_top/2+l_dsl)), (d_sub, d_sub), color='k', lw =1.0) # subsurface down
        plt.plot(((-l_top/2-l_usl-l_uadd), -l_top/2-(l_usl*(d_wt-d_top)/(d_sub-d_top))), (d_wt, d_wt), color='b', lw =1.0) # water level
        
        #-------------------------------------------------------------------------------------------------------

        im_ax4= plt.imshow(v, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clip, vmax=clip, extent=[ex0, ex1, ez1, ez0])
        plt.plot((-l_top/2), (d_top), ls ='', marker='*', markersize= 7,
                markerfacecolor="tab:green", markeredgecolor="tab:green", label = 'Source position') # source position
        plt.legend(loc=(0.7, -0.4))
        
        plt.xlabel('X'+r'$(m)$')
        plt.ylabel('Z '+r'$(m)$')
        plt.xlim(-15,15)
        #plt.colorbar()
        plt.text(7, -1, 'Time = '+np.format_float_scientific(t_step*snap_dt*dt, precision=2)+'s', fontsize=10)
        #plt.gca().invert_yaxis()
        plt.axis('equal')
        
        cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
        cb =fig.colorbar(im_ax4, cax=cbar_ax, orientation="horizontal")
        #cb.set_label('Velocity '+r'$(m/s)$')
        cb.set_label('Normalized Velocity')
        #plt.gca().invert_yaxis()
        #plt.axis('equal')
        # ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        fig.savefig('./forward_dam_v'+plt_dir+'.'+ plt_fmt, format=plt_fmt, bbox_inches='tight')
        plt.show()