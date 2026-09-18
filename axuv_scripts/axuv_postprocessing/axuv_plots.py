import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.gridspec as gridspec

# Need to save plots to /mnt/n/whamdata/shot_plots/yr/mo/dy/AXUV/ path # DE added 25/04/02
filepath = '/mnt/n/whamdata/shot_plots/'
# Script needs to identify if '.../yr/mo/dy/AXUV/' exists, and if not to recursively create directories. 
# Use the same procedure from /home/whamdata/post_shot_plots/save_plot_ech.py which uses 
# shotstr = str(shotnum)
# directory = filepath + shotstr[:2]+'/'+shotstr[2:4]+'/'+shotstr[4:6]+'/AXUV/'
# Path(directory).mkdir(parents=True,exist_ok=True)
from pathlib import Path

class AXUV_plots:
    def __init__(self, analysis, plt_freq, walltime_window, cm_R_plot_limit, diode_key=None, write_mds=True):  # CHANGE 3: Added diode_key and write_mds params
        
        self.analysis = analysis
        self.da_name      = analysis.da_name
        self.diode_key = diode_key  # CHANGE 4a: Store diode_key
        self.write_mds = write_mds  # CHANGE 4b: Store write_mds flag
        original_fs  = 1/np.mean(np.diff(analysis.time_coor))*1e-3 #kHz
        sampling_fs  = analysis.ft_dig_khz #kHz
        frame        = int(original_fs/sampling_fs)

        plt_frame        = int(original_fs/plt_freq)
        time_steps       = list(range(0, len(analysis.time_coor), frame))
        time_steps_plt   = list(range(0, len(analysis.time_coor), plt_frame))
        self.shotnum     = analysis.shotnum
        
        self.timewall_min = walltime_window[0]
        self.timewall_max = walltime_window[1]
        self.cm_R_plot_limit = cm_R_plot_limit
        
        
        self.plot_R_edge = self.analysis.x_padded[0,0]*100

        self.kW_nA       = analysis.kW_nA

        # Plot Initialization
        plt.close('all')
        fig = plt.figure(figsize=(23, 25), dpi=20)
        cmap = colormaps['magma']
        norm = plt.Normalize(min(time_steps), 1.5*max(time_steps))
        
        gs = gridspec.GridSpec(4, 4, figure=fig) 
        ax   = fig.add_subplot(gs[0:2, 2:4], projection='3d')  # One subplot on the left, spanning all rows
        ax2  = fig.add_subplot(gs[2:4, 2:4], projection='3d')  # One subplot on the left, spanning all rows
        ax3   = fig.add_subplot(gs[0:2, 0:2])  # One subplot on the left, spanning all rows
        ax4 = fig.add_subplot(gs[2:4, 0:2])
        inner_gs = gridspec.GridSpecFromSubplotSpec(4, 1, 
                                                    subplot_spec=gs[2:4, 0:2],
                                                    hspace=0.12)
        ax4_1 = fig.add_subplot(inner_gs[0])
        ax4_2 = fig.add_subplot(inner_gs[1])
        ax4_3 = fig.add_subplot(inner_gs[2])
        ax4_4 = fig.add_subplot(inner_gs[3])
        
        
        #Plot AX1: RaW Signal and Instability Chi
        self.plot_rawsignal_chi(ax , time_steps_plt, cmap, norm)
        #Plot AX2: M0 Profile and M1 Amp.
        self.plot_abelsignal_m1(ax2, time_steps_plt, cmap, norm)
        #Plot AX3: Shot Over view
        self.plot_whamplasma_ov(ax3, num_levels=8, color="plasma", resolution_factor=3)
        #Plot AX4: Parameter time plots

        self.plot_CM(ax4_1, time_steps_plt, cmap, norm)
        self.plot_R(ax4_2, time_steps_plt, cmap, norm)
        self.plot_spectrogram(ax4_3, r"$R_{m_{0}}$", analysis.mode0_t, analysis.mode0_f, analysis.mode0_P+1e-12, showxtick = 0)
        self.plot_spectrogram(ax4_4, r"$m_{1}$",     analysis.mode1_t, analysis.mode0_f, analysis.mode1_P+1e-12, showxtick = 1)

        ax4.set_title("Sampling at " + str(int(analysis.ft_dig_khz)) +
                      " kHz | Plot at "+str(int(plt_freq))+" kHz", fontsize=18)
        for spine in ax4.spines.values():
            spine.set_visible(False)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ## overlay Chi
        # ax4_3.plot(analysis.detv_list["1_t"]*1000,
        #            1+3*(np.log10(analysis.detv_list["1_dev"])-min(np.log10(analysis.detv_list["1_dev"]))),
        #            lw = 3)
        #Plot touchup
        fig.text(0.8, 0.85, "Raw Signal", 
                 fontsize=28, color='navy', va='center', ha='center')
        fig.text(0.8, 0.45, r"$m_{0}$ Radial Profile", 
                 fontsize=28, color='navy', va='center', ha='center')
        fig.text(0.54, 0.4, r"$m_{1}$ [%]",
                 fontsize=18, rotation=23, va='center', ha='center')
        fig.text(0.54, 0.81, 
                 "instability\n"+r"$\log (e\chi)$",
                 fontsize=18, rotation=23, va='center', ha='center', color='darkred')#+r"$\chi$ [$cm^{2}$]"
        fig.text(0.16, 0.53, "E",
                 fontsize=42, va='center', ha='center',
                 color = '#FF9933')  # or use hex: '#FFA500'
        fig.text(0.46, 0.53, "W",
                 fontsize=42, va='center', ha='center',
                 color = '#FF9933')  # kind of soft white        #fig.text(0.5, 0.93, analysis.diode_name+":"+self.da_name+":"+str(self.shotnum),
        #         fontsize=24, va='center', ha='center',
        #         color = 'k')  # kind of soft white
        # Combine string pieces
        left_text   = f"{analysis.diode_name}:"
        middle_text = self.da_name
        right_text  = f":{self.shotnum}"
        # Common position and style
        x0, y0 = 0.5, 0.93  # center position
        fontsize = 24
        # Draw left part (black)
        fig.text(x0-0.05, y0, left_text, fontsize=fontsize, va='center', ha='right', color='k')
        # Draw middle part (orange)
        fig.text(x0, y0, middle_text, fontsize=fontsize, va='center', ha='center', color='orange',fontweight='bold')
        # Draw right part (black)
        fig.text(x0+0.05, y0, right_text, fontsize=fontsize, va='center', ha='left', color='k')
        
        #plt.show()

        shotstr = str(self.shotnum) # DE added 25/04/02
        directory = filepath + shotstr[:2]+'/'+shotstr[2:4]+'/'+shotstr[4:6]+'/AXUV/'
        Path(directory).mkdir(parents=True,exist_ok=True)
        fig.savefig(directory+"AXUV_"+str(self.shotnum)+"_"+analysis.diode_name+"_overview.png", dpi=500)#, bbox_inches='tight')


    

        # CHANGE 5: Call save_to_mdsplus() at end of __init__
        try:
            self.save_to_mdsplus()
        except Exception as e:
            print(f"[axuv_plots] MDSPlus save failed (unhandled): {e}")
    def plot_spectrogram(self, ax, name, t, f, power, showxtick = 0):
        from scipy.signal import butter, filtfilt
        from scipy.signal import stft
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    
        power_db = 10*np.log10(power)
    
        pcm = ax.pcolormesh(
            t * 1000+self.timewall_min, f / 1000, power_db,
            shading='nearest', cmap='magma',
            vmax=np.mean(power_db)+np.std(power_db),
            vmin=np.mean(power_db))
    
        # Overlay line on a second y-axis
        # ax2 = ax.twinx()
        # # Plot the mean power vs time
        # ax2.plot(t * 1000, mean_power_time, color='white', lw=2)
        
        # # Label the secondary y-axis
        # ax2.set_ylabel("Mean Power [dB]", fontsize=14, color='white')
        
        # # Optionally adjust y-limits or tick params
        # ax2.set_ylim(np.min(mean_power_time), np.max(mean_power_time))
        # ax2.tick_params(axis='y', colors='white')
    
        # Colorbar inside the plot
        cax = inset_axes(ax,
                         width="2%",  # width of colorbar (relative to plot)
                         height="80%",  # height of colorbar
                         loc='upper right',  # position inside the plot
                         borderpad=0.2)
        fig = plt.gcf()  # get current figure associated with your ax
        cbar = fig.colorbar(pcm, cax=cax)  
        cbar.set_label("Power [dB]", fontsize=16)
        if showxtick == 0:
            ax.set_xticklabels([])
        elif showxtick == 1:
            ax.set_xlabel("Time [ms]", fontsize = 18)

        ax.set_ylabel("f{"+name+"} [kHz]", fontsize = 18)
        #ax.set_ylim(0, max(f / 1000))
        #ax.set_ylim(0, 10)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='y', labelsize=16)
        #ax.set_xlim(t[0] * 1000, t[-1] * 1000)
        ax.set_xlim(self.timewall_min, self.timewall_min+t[-1] * 1000)
    def plot_CM(self, ax, time_steps, cmap, norm):
        max_time_steps =  max(time_steps)
        len_time_steps =  len(time_steps)
        x = self.analysis.time_coor[time_steps]*1000
        y = self.analysis.centroid_hr[time_steps]
        ax.plot(x, y,
               lw=3, c = cmap(norm(min(time_steps))))
        if self.cm_R_plot_limit is not None:
            ax.set_ylim(-self.cm_R_plot_limit[0], self.cm_R_plot_limit[0])
        else:
            ax.set_ylim(-5, 5)
        
        #ax.set_xlim(self.analysis.time_coor[0], self.analysis.time_coor[max_time_steps]*1000)
        ax.set_xlim(self.timewall_min, self.analysis.time_coor[max_time_steps]*1000)
        ax.set_ylabel("Centroid [cm]" , fontsize = 18)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='y', labelsize=14)
        
    def plot_R(self, ax, time_steps, cmap, norm):
        max_time_steps =  max(time_steps)
        len_time_steps =  len(time_steps)
        x = self.analysis.time_coor[time_steps]*1000
        y = self.analysis.radius_hr[time_steps]
        ax.plot(x, y,
               lw=3, c = cmap(norm(max(time_steps))))
        if self.cm_R_plot_limit is not None:
            ax.set_ylim(0, self.cm_R_plot_limit[1])
        else:
            ax.set_ylim(0, self.plot_R_edge*1.5)
        ax.set_ylabel("Radius [cm]" , fontsize = 18)
        #ax.set_xlim(self.analysis.time_coor[0], self.analysis.time_coor[max_time_steps]*1000)
        ax.set_xlim(self.timewall_min, self.analysis.time_coor[max_time_steps]*1000)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='y', labelsize=14)
        


    def plot_whamplasma_ov(self, ax, num_levels, color, resolution_factor):
        from scipy.interpolate import RectBivariateSpline
        import matplotlib.patches as patches
        #set contours colors
        cmin = self.analysis.CM_2D.min()
        cmax = self.analysis.CM_2D.max()
        cmin_r = self.analysis.R_2D.min()
        cmax_r = self.analysis.R_2D.max()    
        num_levels_r = num_levels #num_levels = 20
        levels   = np.linspace(cmin, cmax, num_levels)
        levels_r = np.linspace(cmin_r, cmax_r, 3)
        
        #plot impact b line
        
        theta = (self.analysis.diag_loc[1] + 90) * np.pi / 180  # angle in radians
        #print(self.analysis.diag_loc[1])
        xplt = np.linspace(-self.analysis.wham_r*np.abs(np.cos(theta)), self.analysis.wham_r*np.abs(np.cos(theta)))*100
        #yplt = xplt * np.sin(theta)
        yplt = xplt * np.tan(theta)
        ax.plot(xplt, yplt, 'k--', lw=3, alpha=0.3, zorder = 0)
        # impact b line ticks 
        tick_spacing = 5   # distance between ticks [cm]
        tick_length = 1    # tick mark length (half-length on each side)
        num_ticks = 7      # number of ticks in each direction
        origin = np.array([0, 0])    
            # Direction vector of the line
        dx = 1
        #dy = np.sin(theta)
        dy = np.tan(theta)
        line_dir = np.array([dx, dy]) / np.linalg.norm([dx, dy])
            # Perpendicular direction (for tick orientation)
        perp_dir = np.array([-line_dir[1], line_dir[0]])
            # Tick positions (positive and negative sides)
        tick_offsets = np.arange(-num_ticks, num_ticks + 1) * tick_spacing  
        tick_coords = [origin + offset * line_dir for offset in tick_offsets]
        # if theta >0 and theta<=180 :
        #     print(tick_offsets)
        #     tick_offsets = tick_offsets[::-1]
        #     print(tick_offsets)
        #print(tick_offsets)
            # Plot tick marks and labels
        rot_txt_ang = np.degrees(theta) 
        theta_wrapped = (rot_txt_ang + 180) % 360 - 180
        if theta_wrapped > 90:   
            #print("in")
            rot_txt_ang = np.degrees(theta)-180
            tick_offsets = tick_offsets[::-1]
        elif theta_wrapped < -90: 
            #print("in")
            rot_txt_ang = np.degrees(theta)+180
            tick_offsets = tick_offsets[::-1]
        for offset, pos in zip(tick_offsets, tick_coords):
            # Tick endpoints
            p1 = pos - tick_length * perp_dir
            p2 = pos + tick_length * perp_dir    
            # Draw tick line
               
            # Label (with signed distance)
            if offset == 0:
                p1 = pos - 2*tick_length * perp_dir
                p2 = pos + 2*tick_length * perp_dir 
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='r', lw=2, alpha=0.4) 
                outprint = f"{offset:d}"
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', lw=1, alpha=0.3) 
                outprint = f"{offset:+d}"
    
    
            ax.text(pos[0] + 2*np.cos(theta-np.pi/2), pos[1] + 2*np.sin(theta-np.pi/2), outprint,
                    fontsize=12,
                    rotation=0,
                    va='center', ha='left', alpha=0.3, zorder = 10)
        ax.text(tick_coords[-3][0]+8*np.cos(theta+np.pi/2), tick_coords[-3][1] + 8*np.cos(theta+np.pi/2), r"$b$ (cm)",
                fontsize=16,
                rotation=0,
                va='center', ha='left', alpha=0.6, zorder = 10)
        #print(theta_wrapped)
        # (Eye candy) make higer resolution
        if resolution_factor > 1:
            x = self.analysis.X[0, :] * 100
            y = self.analysis.Y[:, 0] * 100
            z_r  = self.analysis.R_2D
            z_cm = self.analysis.CM_2D 
            spline_r  = RectBivariateSpline(y, x, z_r)
            spline_cm = RectBivariateSpline(y, x, z_cm)
            # Finer grid
            xi = np.linspace(x.min(), x.max(), int(len(x)*resolution_factor))
            yi = np.linspace(y.min(), y.max(), int(len(x)*resolution_factor))
            Xi, Yi = np.meshgrid(xi, yi)
            Zi_r   = spline_r(yi, xi)
            Zi_cm  = spline_cm(yi, xi)
            cmin   = Zi_cm.min()
            cmax   = Zi_cm.max()
            cmin_r = Zi_r.min()
            cmax_r = Zi_r.max()
            levels   = np.linspace(cmin, cmax, num_levels)
            levels_r = np.linspace(cmin_r, cmax_r, num_levels)
        else:
            Xi     = self.analysis.X* 100
            Yi     = self.analysis.Y* 100
            Zi_r   = self.analysis.R_2D
            Zi_cm  = self.analysis.CM_2D
    
        #plot R contours
        ax.contour(Xi, Yi, Zi_r, 
                   levels=levels_r[1:],
                   cmap=color,
                   extend='max', zorder = 0)
        #plot CM contours

#         ax.contour(Xi, Yi, Zi_cm, 
#                    levels=levels[1:],
#                    cmap=color,
#                    extend='max', zorder = 0)
        if not np.all(np.isnan(Zi_cm)) and np.nanmin(Zi_cm) != np.nanmax(Zi_cm):           
            ax.contour(Xi, Yi, Zi_cm, 
                    levels=levels[1:],
                    cmap=color,
                    extend='max', zorder = 0)
        else:
            print("Skipping contour: Zi_cm invalid or flat")
        
        # Create WHAM
        circle = patches.Circle((0, 0), radius=self.analysis.wham_r*100, 
                                edgecolor='k', facecolor='none', lw=6, zorder = 1)
        ax.add_patch(circle)
    
        # Create Diode
        ax.scatter(self.analysis.diag_loc_cart[0]*100,
                   self.analysis.diag_loc_cart[1]*100, edgecolor='#98FB98', facecolor='none', lw=6)
        # Create viewchord
        def line_circle_intersection(x0, y0, m, R):
            # Coefficients of the quadratic equation: Ax^2 + Bx + C = 0
            A = 1 + m**2
            B = 2 * m * (y0 -  m * x0) 
            C = (y0 - m * x0)**2 - R**2
        
            # Solve the quadratic
            discriminant = B**2 - 4 * A * C
            if discriminant < 0:
                return []  # No intersection
        
            sqrt_disc = np.sqrt(discriminant)
            x1 = (-B + sqrt_disc) / (2 * A)
            x2 = (-B - sqrt_disc) / (2 * A)
        
            # Compute corresponding y values
            y1 = m * (x1 - x0) + y0
            y2 = m * (x2 - x0) + y0
        
            return [x1, x2], [y1, y2]
        for i in range(len(self.analysis.diag_view_chord)):
            chrod_x, chrod_y = line_circle_intersection(x0=self.analysis.diag_loc_cart[0]*100, y0=self.analysis.diag_loc_cart[1]*100,
                                                     m=np.tan(np.radians(self.analysis.diag_view_chord[i])), R=self.analysis.wham_r*100)
            #plot chrods
##================hot fix Kai DA3 1, 3 death ch ======================================================
            if self.analysis.diode_name == 'DIODEARRAY3':
                ax.text(
                1.0, 0.75,
                "Dead CH4",
                transform=ax.transAxes,
                fontsize=16,
                color='red',
                ha='center',
                va='center'
                )
                ax.text(
                1.02, 0.65,
                "Dead CH2",
                transform=ax.transAxes,
                fontsize=16,
                color='red',
                ha='center',
                va='center'
                )


                if (i == 1) or (i == 3):
                    ax.plot(chrod_x, chrod_y,
                    c='r', linestyle='--', dashes=(5, 6), lw =0.3, alpha=0.6, zorder = -2)
                else:
            #plot chrods
                    ax.plot(chrod_x, chrod_y,
                    c='k', linestyle='--', dashes=(5, 6), lw =0.3, alpha=0.6, zorder = -2) 
            else:
                ax.plot(chrod_x, chrod_y,
                c='k', linestyle='--', dashes=(5, 6), lw =0.3, alpha=0.6, zorder = -2)
#===========================================================================================================


           #ax.plot(chrod_x, chrod_y,
                    #c='k', linestyle='--', dashes=(5, 6), lw =0.3, alpha=0.6, zorder = -2) 
    
        #ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-38, 38)
        ax.set_ylim(-38, 38)
        # Hide ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])        
        # Hide the plot frame (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)


            

    def plot_abelsignal_m1(self, ax, time_steps, cmap, norm):
        W_to_mW        = 1000 # change  W to mW
        x_yz_edge      = -self.plot_R_edge #cm
        min_time_wall  =  self.timewall_min  #ms
        max_time_wall  =  self.timewall_max#ms
 
        time_axis = np.array([self.analysis.time_coor[t] for t in time_steps])*1000 #ms
        max_time_steps =  max(time_steps)
        len_time_steps =  len(time_steps)
        
        maxabel = np.max(self.analysis.radial_profile[time_steps])
        #plotmax        =  1  #a.u.import sys
        plotmax        =  maxabel*W_to_mW  #a.u.
        
        if x_yz_edge <= -self.analysis.def_plasma_edge*100:
            edge = self.analysis.def_plasma_edge
        else:
            edge = -x_yz_edge/100
        
        dnumhr_sub_edge  = self.analysis.dnumhr - edge
        zero_crossings   = np.where((dnumhr_sub_edge[:-1] < 0) & (dnumhr_sub_edge[1:] > 0))[0][0]
        n_edge_idx       = len(self.analysis.dnumhr)-zero_crossings-1
        p_edge_idx       = zero_crossings
        #print(n_edge_idx,p_edge_idx)
        #np.savez(str(self.shotnum)+"rprofile_data.npz", r=self.analysis.dnumhr[self.analysis.radial_profile.shape[1]-1:]*100, t=time_axis, s=self.analysis.radial_profile)
        for idx, it in enumerate(time_steps):
            t_prenc = it/max_time_steps
        
            x           = self.analysis.dnumhr[n_edge_idx:p_edge_idx]*100
            y           = np.full_like(x, time_axis[idx])        # t [ms]
            profile     = self.analysis.radial_profile[it]*W_to_mW
            sym_profile = np.concatenate([profile[::-1], profile[1:]])
            #z           = sym_profile[n_edge_idx:p_edge_idx]/maxabel
            z           = sym_profile[n_edge_idx:p_edge_idx]
            
            proj_x      = x_yz_edge
            proj_y      = time_axis[idx]
            proj_z      = self.analysis.c1n1_cm1[it]
#            plotproj_z = sc_factor*proj_z
##
#            if plotproj_z>=1:
#                plotproj_z= 1
#        #m=1 amp. value plot
#            ax.scatter(proj_x, proj_y, plotproj_z,
#                        color=cmap(norm(it)), s=1, alpha=0.2)
        #time m=0 signals plot    
            ax.plot(x, y, z, 
                    color=cmap(norm(it)),
                    lw=1+0.3*t_prenc,
                    alpha=0.6+0.4*t_prenc,
                    zorder=len_time_steps-idx)    
        #time m=0  signals plot on wall    import sys
            ax.plot(x, np.full_like(x, max_time_wall), z,
            color=cmap(norm(it)),
            lw=1.2,
            alpha=0.2+0.4*t_prenc,
            zorder=-len_time_steps+idx)

        #m=1 amp. value trace plot
#        print(self.analysis.c1n1_cm1)
        
        sc_factor = 4*W_to_mW*maxabel
        plotm1 = sc_factor*self.analysis.c1n1_cm1[time_steps]
        plotm1[plotm1>=W_to_mW*maxabel] =W_to_mW*maxabel
        ax.plot(np.full_like(time_axis, x_yz_edge), time_axis, plotm1,
        color='k', lw=0.8)

        
        
        # m=1 amp. ticks and label plot
        z_ticks =  np.linspace(0, plotmax, 6)
#        ax.plot([x_yz_edge, x_yz_edge], [min_time_wall, min_time_wall], [0, 1],
#                color='gray', lw=1.5)   
        for idx, i in enumerate(z_ticks):
            ax.plot([x_yz_edge-1, x_yz_edge], [min_time_wall, min_time_wall], [i, i]
                    , color='gray', lw=1)
            # sc_factor = 4 >>>i*25
            ax.text(x_yz_edge-1, min_time_wall, i,
                    f"{i*100/sc_factor:.0f}",
                    fontsize=16, ha='right', va='center')
        
        # plot ticks and labels
        ax.set_zticks(z_ticks)
        ax.tick_params(axis='x', labelsize = 16)
        ax.tick_params(axis='y', labelsize = 16)
        ax.tick_params(axis='z', labelsize = 16)
        ax.set_xlabel(r"Impact parameter $b$ (cm)",
                      fontsize = 18, labelpad=20)
        ax.set_ylabel(r"Time $t$ (ms)",
                      fontsize = 18, labelpad=20)
        ax.set_zlabel(r"$m_{0}$ Emissivity [$mW/(cm^3)$]",
                      fontsize=18, labelpad=20)
        ax.view_init(elev=35, azim=-50)
        #ax2.set_box_aspect([0.3, 0.6, 0.5])  # [X:Y:Z] ratio
        ax.set_xlim3d(x_yz_edge, -x_yz_edge)
        ax.set_ylim3d(min_time_wall, max_time_wall)
        ax.set_zlim3d(0, plotmax)



    def plot_rawsignal_chi(self, ax, time_steps, cmap, norm):
        x_yz_edge      = -self.plot_R_edge #cm
        min_time_wall  =  self.timewall_min  #ms
        max_time_wall  =  self.timewall_max #ms
        time_axis = np.array([self.analysis.time_coor[t] for t in time_steps])*1000 #ms
        max_time_steps =  max(time_steps)
        len_time_steps =  len(time_steps)
        
        max_peak_signal = np.max(self.analysis.y_padded[:, time_steps])
        #print("max print", max_peak_signal)
        #max_peak_signal = np.max(self.analysis.y_padded[:, time_steps])*100
        z_proj_sum      = np.sum(self.analysis.y_padded, 0)
        maxsumstep      = max(np.sum(self.analysis.y_padded[:, time_steps], 0))
        z_proj_sum_norm = z_proj_sum/maxsumstep*max_peak_signal
        log_max_psiganl = int(np.log10(max_peak_signal))
        pow_10_siganl   = 10**log_max_psiganl
        max_1o5_siganl  = np.ceil((max_peak_signal - pow_10_siganl)/(pow_10_siganl/5))
        plotmax         = pow_10_siganl+(pow_10_siganl/5)*max_1o5_siganl
        

        max_pow_signal   = maxsumstep*self.kW_nA
        log_max_psiganl2 = int(np.log10(max_pow_signal))
        pow_10_siganl2   = 10**log_max_psiganl2
        max_1o5_siganl2  = np.ceil((max_pow_signal - pow_10_siganl2)/(pow_10_siganl2/5))
        plotmax2         = pow_10_siganl2+(pow_10_siganl2/5)*max_1o5_siganl2

        for idx, it in enumerate(time_steps):
            t_prenc = it/max_time_steps
            
            x = self.analysis.x_padded[:, 0]* 100          # b [cm]
            y = np.full_like(x, time_axis[idx])        # t [ms]
            #z = self.analysis.y_padded[:, it]* 100         # signal [mV]
            z = self.analysis.y_padded[:, it]       # signal [mV]      
            proj_x = x_yz_edge
            proj_y = time_axis[idx]
            proj_z = z_proj_sum_norm[it]
        #total signals plot
           # ax.scatter(proj_x, proj_y, proj_z,
                       #color='blue', s=1)#color=cmap(norm(it)), s=1)    
        #time signals plot    
            ax.plot(x, y, z, 
                    color=cmap(norm(it)),
                    lw=1.0+0.3*t_prenc,
                    alpha=0.6+0.4*t_prenc,
                    zorder=  len_time_steps-idx)
        #time signals plot        
            ax.plot(x, np.full_like(x, max_time_wall),  z,
                    color=cmap(norm(it)),
                    lw=1.2,
                    alpha=0.2+0.4*t_prenc,
                    zorder= -len_time_steps+idx)
        #print("=== DEBUG SHAPES BEFORE PLOT ===")
        #print("type(x_yz_edge):", type(x_yz_edge), "shape:", np.shape(x_yz_edge))
        #print("type(time_axis):", type(time_axis), "shape:", np.shape(time_axis))
        #print("type(z_proj_sum_norm):", type(z_proj_sum_norm), "shape:", np.shape(z_proj_sum_norm))
        #print("type(time_steps):", type(time_steps), "value:", time_steps)
        #print("z_proj_sum_norm[time_steps] shape:", np.shape(z_proj_sum_norm[time_steps]))
        #print("First 5 of x_yz_edge:", x_yz_edge[:5])
        #print("First 5 of time_axis:", time_axis[:5])
        #print("First 5 of z_proj_sum_norm[time_steps]:", z_proj_sum_norm[time_steps][:5])

       #total signals trace plot
        ax.plot(np.full_like(time_axis, x_yz_edge), time_axis,
        z_proj_sum[time_steps]/maxsumstep*plotmax*max_pow_signal/plotmax2,
                color='blue', lw=2)
        
        #Chi Value and re-scaling
        chi_f1_t = self.analysis.detv_list[str(1)+'_t']*1000
        chi_f1_s = self.analysis.detv_list[str(1)+'_dev']*10*10*np.exp(1)
        mask_p0  = chi_f1_s > 0
        chi_f1_t_clean     = chi_f1_t[mask_p0]
        log_chi_f1_s_clean = np.log(chi_f1_s[mask_p0])
        chi_cen_idx        = slice(len(chi_f1_t)//4,3*len(chi_f1_t)//4)
        chi_mean           = np.mean(log_chi_f1_s_clean[chi_cen_idx])
        chi_std            = np.std(log_chi_f1_s_clean)
#        plt_y_nlim         = -4#np.floor(chi_mean-2*chi_std).astype(int)
#        plt_y_plim         =  3#np.ceil(chi_mean+2*chi_std).astype(int)
        plt_y_nlim         = -2#np.floor(chi_mean-2*chi_std).astype(int)
        plt_y_plim         = 8#np.ceil(chi_mean+2*chi_std).astype(int)
        plt_y_range        = plt_y_plim-plt_y_nlim
        rescale_log_chi    = (log_chi_f1_s_clean-plt_y_nlim)/(plt_y_range)*plotmax####plotmax
        plt_chi_ticks      = np.linspace(plt_y_nlim, plt_y_plim, plt_y_range+1)
        plt_pow_ticks      = np.linspace(0, plotmax2, plt_y_range+1)
        
        rescale_log_chi[rescale_log_chi<0] = 0
        rescale_log_chi[rescale_log_chi>=plotmax] = plotmax
        #Chi plot
        #dt = int(np.diff(time_steps)[0])
        #time_steps_chi = list(range(0, len(), dt))
        ax.plot(np.full_like(chi_f1_t_clean, x_yz_edge), chi_f1_t_clean, rescale_log_chi,
        color='darkred', lw=1)

        
        # Chi ticks and label plot
        z_ticks =  np.linspace(0, plotmax, plt_y_range+1)
        
#        ax.plot([x_yz_edge, x_yz_edge], [min_time_wall, min_time_wall], [0, 1],
#                color='gray', lw=1.5)
        ax.set_zticks(z_ticks)

        if (plotmax >= 0.1 and plotmax < 1):
            ax.set_zticklabels([f"{z:.2f}" for z in z_ticks])
        elif (plotmax >= 1 and plotmax < 10) :
            ax.set_zticklabels([f"{z:.1f}" for z in z_ticks])
        elif plotmax >= 10 :
            ax.set_zticklabels([f"{z:.0f}" for z in z_ticks])
        
        for idx, i in enumerate(z_ticks):
            ax.plot([x_yz_edge-1, x_yz_edge], [min_time_wall, min_time_wall], [i, i]
                    , color='darkred', lw=1)
            ax.text(x_yz_edge-1, min_time_wall, i, 
                    r"${:.0f}$".format(plt_chi_ticks[idx]),
                    fontsize=16, ha='right', va='center', color='darkred') #r"$10^{{{:.0f}}}$"
        for idx, i in enumerate(z_ticks[::2]):
            ax.plot([x_yz_edge-1-5, x_yz_edge], [min_time_wall, min_time_wall], [i, i]
                    , color='blue', lw=1)

            if   plotmax2 < 0.1 :
                ax.text(x_yz_edge-1-5, min_time_wall, i,
                    r"${:.2f}$".format(plt_pow_ticks[::2][idx]),
                    fontsize=16, ha='right', va='center', color='blue')
            elif (plotmax2 >= 0.1 and plotmax2 < 1):
                ax.text(x_yz_edge-1-5, min_time_wall, i, 
                    r"${:.2f}$".format(plt_pow_ticks[::2][idx]),
                    fontsize=16, ha='right', va='center', color='blue') #r"$10^{{{:.0f}}}$"
            elif (plotmax2 >= 1 and plotmax2 < 10) :
                ax.text(x_yz_edge-1-5, min_time_wall, i, 
                    r"${:.1f}$".format(plt_pow_ticks[::2][idx]),
                    fontsize=16, ha='right', va='center', color='blue') #r"$10^{{{:.0f}}}$"
            elif plotmax2 >= 10 :
                ax.text(x_yz_edge-1-5, min_time_wall, i, 
                    r"${:.0f}$".format(plt_pow_ticks[::2][idx]),
                    fontsize=16, ha='right', va='center', color='blue') #r"$10^{{{:.0f}}}$"         
        
        ax.text2D(
        -0.08, 0.5,
        r"$P_{tot}$ [kW]",
        transform=ax.transAxes,
        fontsize=22,      
        rotation=90,
        color='blue',
        ha='center',
        va='center'
        )



        # plot ticks and labels
        ax.tick_params(axis='x', labelsize = 16)
        ax.tick_params(axis='y', labelsize = 16)
        ax.tick_params(axis='z', labelsize = 16)
        ax.set_xlabel(r"Impact parameter $b$ (cm)",
                      fontsize = 18, labelpad=20)
        ax.set_ylabel(r"Time $t$ (ms)",
                      fontsize = 18, labelpad=20)
        ax.set_zlabel(r"Raw Signal $s$ [nA]",
                      fontsize = 18, labelpad=20)
        ax.view_init(elev=35, azim=-50)
        #ax.set_box_aspect([0.3, 0.6, 0.5])  # [X:Y:Z] ratio
        ax.set_xlim3d(x_yz_edge, -x_yz_edge)
        ax.set_ylim3d(min_time_wall, max_time_wall)
        ax.set_zlim3d(0, plotmax)

    
    def plot_mode_decomp(self, x_coor, singal,
                         m_name ='', amp_factor=None, yaxis_factor=None,
                         time_window=1, fft_dig_khz=50):
    
        #import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        #import numpy as np
        from scipy.signal import stft
        
        original_fs = 1 / np.mean(np.diff(self.analysis.time_coor)) / 1000
        frame = int(original_fs / fft_dig_khz)
        time_steps = list(range(0, len(self.analysis.time_coor), frame))
        time = self.analysis.time_coor[time_steps]
        #X = self.analysis.radial_profile.T[:, time_steps]
        X = singal[:, time_steps]
        # Get real detector x-positions (flattened and scaled if needed)
        x_axis = x_coor
        if x_axis[-1] < x_axis[0]:
            x_axis = x_axis[::-1]
            X      = X [::-1]
        
        # Preprocessing & SVD
        #X_centered = X - np.mean(X, axis=1, keepdims=True)
        X_centered = X
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        num_modes = 4
        cmap = colormaps['plasma']
        norm = plt.Normalize(0, num_modes)
        
        
        # U shape: (sensors, modes) → Transpose to (modes, sensors)
        U_display = U[:, :num_modes].T
        
        fig = plt.figure(figsize=(14, 12), dpi=60)
        gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1], hspace=0.8)
        
        # --- 1. Spatial Mode Heatmap (top row, full width) ---
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot([0, 0],[0-0.5,num_modes-0.5], 'k', lw =2)
        
        im = ax0.imshow(U_display, aspect='auto', cmap='Greys', interpolation='nearest',
                        extent=[x_axis.min(), x_axis.max(), num_modes - 0.5, -0.5],
                        vmin=-np.max(np.abs(U_display)), vmax=np.max(np.abs(U_display)),
                        alpha = 0.7)
        ax0.set_xlabel("Impact parameter (cm)", fontsize=14)
        ax0.set_ylabel("Mode", fontsize=14)
        ax0.set_title(m_name+" Mode Structures", fontsize=15)
        ax0.set_yticks(np.arange(num_modes))
        ax0.set_yticklabels([fr"$m_{{{m_name}{i}}}$" for i in range(num_modes)])
        fig.colorbar(im, ax=ax0, orientation='vertical',
                     label="Sensor Strength")
        
        # --- 2. Temporal Amplitudes (second row, full width) ---
        ax1 = fig.add_subplot(gs[0, 1])
        for i in range(num_modes):
            
            ax0.plot(x_axis, x_axis*0+0.5+i, 'k'  ,lw = 1)
            color = cmap(norm(i))
            ax0.plot(x_axis, i-U_display[i], color=color  ,lw = 1)
            ax0.plot(x_axis, i-0*U[:, 1]/2, 'w--' ,lw = 0.7)
            ax0.set_ylim(0-0.5, num_modes-0.5)
            
            amp = S[i] * VT[i, :]
            t_len = VT.shape[1]
            val = np.abs(np.max(np.concatenate([
                ((S[0, None] * VT[0, int(t_len/6):int(5*t_len/6)]) / 10)[None, :],  # now shape (1, m)
                S[1:4, None] * VT[1:4,int(t_len/6):int(5*t_len/6)]
            ], axis=0)))

            vallog =np.floor(np.log10(val))  # single scalar
            
            if amp_factor is None:
                amp_factor = -(int(vallog)-1)
                yaxis_factor = int(2.5*int(val/(10**(vallog-1)))/10)*10
            else:
                amp_factor = int(amp_factor)
#            print(amp_factor, yaxis_factor )
#            print(amp.shape, time.shape)
            
            
            if i == 0 :
                # ax1.plot(time * 1000, amp*10**amp_factor/10,
                #          color = color, linestyle="--", label=fr"$m_{{{m_name}{i}}}$/10")
            # else:    
            #     ax1.plot(time * 1000, amp*10**amp_factor, 
            #              color = color, label=fr"$m_{{{m_name}{i}}}$", alpha=0.7)                
                ax1.plot(time * 1000, -amp*10**amp_factor/10,
                         color = color, linestyle="--", label=fr"$m_{{{m_name}{i}}}$/10")
            else:    
                ax1.plot(time * 1000, -amp*10**amp_factor, 
                         color = color, label=fr"$m_{{{m_name}{i}}}$", alpha=0.7)
        
        
        ax1.tick_params(axis='both', direction='in')        
        ax1.set_xlabel("Time [ms]", fontsize=14)
        if amp_factor > 0:
            ax1.set_ylabel(fr"Amp. [$10^{{-{amp_factor}}}$]", fontsize=14)
        elif amp_factor < 0:
            ax1.set_ylabel(fr"Amp. [$10^{{{-amp_factor}}}$]", fontsize=14)
        elif amp_factor == 0:
            ax1.set_ylabel(fr"Amp. ", fontsize=14)
        ax1.legend(ncol=4, fontsize=10)
        ax1.set_title("Amp. Evolution", fontsize=15)
        ax1.set_ylim(-yaxis_factor, yaxis_factor)
        #ax1.grid(True)
        
        # --- 3. STFT Spectrograms (3rd and 4th rows) ---
        fs = 1 / np.mean(np.diff(time))
        nperseg = int(fs * time_window * 1e-3)
        noverlap = nperseg // 2
        
        for i in range(num_modes):
        
            ax = fig.add_subplot(gs[1 + i // 2, i % 2])
            amp = S[i] * VT[i, :]
            f, t_stft, Zxx = stft(amp, fs=fs, nperseg=nperseg, noverlap=noverlap)
            power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        
            pcm = ax.pcolormesh(t_stft * 1000, f / 1000, power_db,
                                shading='nearest', cmap='magma',
                                vmax=np.mean(power_db) + np.std(power_db),
                                vmin=np.mean(power_db))
            ax.set_title(fr"$m_{{{m_name}{i}}}$", fontsize=16)
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Freq [kHz]")
            ax.set_ylim(0, fs / 2 / 1000)
            fig.colorbar(pcm, ax=ax, label="Power [dB]")

                             
        if m_name =="A":
            p_name = "Abel Profile"
        else:
            p_name = "Line Integral"

        fig.suptitle(str(self.shotnum)+":"+self.da_name+":"+p_name, fontsize=18, fontweight='bold', y=0.98) 
        #plt.show()
        #fig.savefig("AXUV_"+str(self.analysis.shotnum)+"_"+
                     #self.analysis.diode_name+"_"+m_name+"_mode.png", dpi=300, bbox_inches='tight')

        shotstr = str(self.shotnum) # DE added 25/04/02
        directory = filepath + shotstr[:2]+'/'+shotstr[2:4]+'/'+shotstr[4:6]+'/AXUV/Mode_Decomposition/Signal_'+m_name+'/'
        Path(directory).mkdir(parents=True,exist_ok=True)
        fig.savefig(directory+"AXUV_"+str(self.shotnum)+"_"+self.analysis.diode_name+"_"+m_name+"_mode.png", dpi=300, bbox_inches='tight')


    # CHANGE 6: New method to save derived quantities to MDSPlus tree
    def save_to_mdsplus(self):
        if not self.write_mds:
            return

        import MDSplus as mds
        if self.diode_key not in ("DIODEARRAY1", "DIODEARRAY2", "DIODEARRAY3"):
            print(f"[axuv_plots] skipping MDSPlus save: unknown diode_key={self.diode_key}")
            return

        base = f"DIAG.AXUV.{self.diode_key}.ANALYSIS"
        node_names = ["CENTROID_HR", "RADIUS_HR", "C1N1_CM1", "CHI_F1", "CHI_F1_LOG"]

        # Phase 1: ensure the ANALYSIS structure + signal nodes exist. Adding
        # nodes mutates the tree structure, which requires the tree be opened in
        # EDIT mode and committed with write(); data is written separately below
        # in normal mode. This makes the method self-bootstrapping on historical
        # shots whose stored model predates these nodes.
        try:
            etree = mds.Tree("wham", self.shotnum, "EDIT")
            added = False
            try:
                analysis = etree.getNode(base)
            except mds.mdsExceptions.TreeNNF:
                analysis = etree.getNode(f"DIAG.AXUV.{self.diode_key}").addNode("ANALYSIS", "STRUCTURE")
                added = True

            for node_name in node_names:
                try:
                    etree.getNode(f"{base}.{node_name}")
                except mds.mdsExceptions.TreeNNF:
                    analysis.addNode(node_name, "SIGNAL")
                    added = True

            if added:
                etree.write()
            etree.quit()
        except Exception as e:
            print(f"[axuv_plots] failed to create nodes: {e}")
            return

        # Phase 2: reopen normally to write data into the (now-existing) nodes.
        try:
            tree = mds.Tree("wham", self.shotnum)
        except Exception as e:
            print(f"[axuv_plots] failed to open tree: {e}")
            return

        t_s = self.analysis.time_coor
        n   = len(t_s)
        t0  = float(t_s[0])
        dt  = float((t_s[-1] - t_s[0]) / (n - 1)) if n > 1 else 1.0

        win   = "BUILD_WINDOW(0,$1-1,BUILD_WITH_UNITS($2,'s'))"
        axis  = ("BUILD_WITH_UNITS(BUILD_RANGE($2,$2+(1/d_float($3))*($1-1),"
                 "1/d_float($3)),'s')")
        tbase = "BUILD_DIM(" + win + "," + axis + ")"
        fs    = 1.0 / dt

        def put_uniform(node_rel, values, units):
            sig = mds.Data.compile(
                f"BUILD_SIGNAL(BUILD_WITH_UNITS($VALUE,'{units}'),$4,{tbase})",
                n, t0, fs, values.astype("float32"),
            )
            tree.getNode(f"{base}.{node_rel}").putData(sig)

        try:
            put_uniform("CENTROID_HR", self.analysis.centroid_hr, "m")
            put_uniform("RADIUS_HR",   self.analysis.radius_hr,   "m")
            put_uniform("C1N1_CM1",    self.analysis.c1n1_cm1,    "1")
        except Exception as e:
            print(f"[axuv_plots] failed to save uniform-grid signals: {e}")
            return

        try:
            chi_t = self.analysis.detv_list["1_t"]
            chi_s = self.analysis.detv_list["1_dev"] * 10 * 10 * np.exp(1)
            mask  = chi_s > 0

            chi_t_m   = chi_t[mask].astype("float32")
            chi_s_m   = chi_s[mask].astype("float32")
            chi_log_m = np.log(chi_s_m)

            tree.getNode(f"{base}.CHI_F1").putData(
                mds.Signal(mds.WithUnits(chi_s_m, "1"), None, mds.WithUnits(chi_t_m, "s"))
            )
            tree.getNode(f"{base}.CHI_F1_LOG").putData(
                mds.Signal(mds.WithUnits(chi_log_m, "1"), None, mds.WithUnits(chi_t_m, "s"))
            )
        except Exception as e:
            print(f"[axuv_plots] failed to save chi signals: {e}")
            return

        print(f"[axuv_plots] saved DIAG.AXUV.{self.diode_key}.ANALYSIS to shot {self.shotnum}")

