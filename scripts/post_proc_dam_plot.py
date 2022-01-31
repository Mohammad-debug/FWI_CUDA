# Post processing in python
#%%

# reading the output arrays
import numpy as np
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
read_metaint("../bin/metaint.bin", np.int32)
snap_nt = np.int32(1 + (ndim[0]-1)//snap[6])
snap_nz = 1 + (snap[3] - snap[2])//snap[7]
snap_nx = 1 + (snap[4] - snap[5])//snap[8]

# plotting the reciever data

if (fwinv):
    print("Plotting material for iteration in fwi")
    maxiter = 700
    for ii in range(600,maxiter,1):
        # reading data from csv file
        mat_dat = read_tensor("../bin/iter"+np.str(ii)+"_mat.bin", np.float64, (3, ndim[1], ndim[2]))
        #mat_dat = read_tensor("../io/mat_save/iter"+np.str(ii)+"_mat copy.bin", np.float64, (3, ndim[1], ndim[2]))
        
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
    rtf_uz = read_tensor("../bin/shot2_rtf_uz.bin", np.float64, (nrec, ndim[0]))
    rtf_ux = read_tensor("../bin/shot2_rtf_ux.bin", np.float64, (nrec, ndim[0]))
    
    #Plotting only the wavelets (source and receivers)
    # create an array
    dt = 0.3e-4
    time = np.arange(0, rtf_uz.shape[1])*dt
    # Plot figure for the paper
    # create the standard size for the figure
    fw, fh = set_size('thesis', subplots=(1,2))
    # Using seaborn's style
    #plt.style.use('seaborn')
    #width = 345

    # Set the rc parameter updates
    plt.rcParams.update(tex_fonts)
    
    
    # initialize the figure plots
    fig, axs = plt.subplots(1,2, figsize=(fw, fh))
    fig.tight_layout() 
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
    
    vz_dat = read_tensor("../bin/shot2_vz.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    vx_dat = read_tensor("../bin/shot2_vx.bin", np.float64, (snap_nt, snap_nz, snap_nx))
    
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
    exit()
    
    for ii in range(1,snap_nt, 5):
        # reading data from csv file
        vz = vz_dat[ii,:,:]
        vx = vx_dat[ii,:,:]  
        plt.figure(1)
        plt.subplot(211)
        
        # plotting dam parameters
        #------------------------------------------------------------------------------------------
        plt.plot(((l_uadd+l_usl)/dx+npml_fpad, (l_uadd+l_usl+l_top)/dx+npml_fpad), \
            (d_top/dz+npml_fpad-1, d_top/dz+npml_fpad-1), color='k', lw =1.0) # top of the dam
        
        plt.plot(((l_uadd)/dx+2*npml_fpad-8, (l_uadd+l_usl)/dx+npml_fpad), \
            ((d_sub)/dz+npml_fpad, d_top/dz+npml_fpad-1), color='k', lw =1.0)# upslope of the dam
        
        plt.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (l_uadd+l_usl+l_top)/dx+npml_fpad),\
            ((d_sub)/dz+npml_fpad, d_top/dz+npml_fpad-1), color='k', lw =1.0)# downslope of the dam
        
        
        plt.plot((0 , (l_uadd)/dx+2*npml_fpad-8), ((d_sub)/dz+npml_fpad, (d_sub)/dz+npml_fpad), color='k', lw =1.0) # subsurface up
        
        plt.plot(((l_uadd+l_usl+l_top+l_dsl)/dx+npml_fpad-10, (len+l_dadd)/dx+2*npml_fpad+3), \
            ((d_sub)/dz+npml_fpad, (d_sub)/dz+npml_fpad), color='k', lw =1.0) # subsurface down
        plt.plot((0, l_usl/dx+npml_fpad-8), (d_wt/dz+npml_fpad, d_wt/dz+npml_fpad), color='b', lw =1.0) # water level
        #-------------------------------------------------------------------------------------------------------
        
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
