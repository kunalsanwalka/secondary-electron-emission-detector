import numpy as np
class Tool:
    def corners(self, diode_list_one):
        from scipy.signal import find_peaks

        axuv_all_f = diode_list_one
        # peaks, _ = find_peaks(np.diff(axuv_all_f), height=0.1, distance = len(axuv_all_f))
        # lpeaks, _ = find_peaks(-np.diff(axuv_all_f)[::-1], height=0.1, distance = len(axuv_all_f))
        # lpeaks = len(axuv_all_f)- lpeaks
        mi_peak  = np.argmax(axuv_all_f)
        #mi_peak  = len(axuv_all_f)//2
        peaks   = [mi_peak]
        lpeaks  = [mi_peak]
        con_rais = np.argmax(axuv_all_f[peaks[0]]/peaks[0]* np.arange(0, peaks[0])- axuv_all_f[:peaks[0]])

        fall_m = np.diff((axuv_all_f[[lpeaks[0], -1]])/(len(axuv_all_f)-lpeaks[0]))[0]
        fall_c = axuv_all_f[-1]
        con_fall = np.argmax( -(axuv_all_f[lpeaks[0]:] -fall_m*(-np.arange(0,len(axuv_all_f)-lpeaks[0])[::-1])-fall_c)  )+lpeaks[0]
        
        
        corners = np.array([con_rais,con_fall])

        return corners
class AXUV_analysis:
    
    def __init__(self, diode, time_range_ms, xmesh=101, recon_center=3,
                 test=0, chi_signal_dig_khz=[20, 1], digi_Mhz=1, fft_dig_khz = 50, FFT_time_window = 1,
                 t0_ms =-5, hard_noise=0):

        # recon_center  = 0 # 0 => 0; 1 => peak; 2 => c.m.; 3=> B center
        # test          = 0
        # xmesh         = 301 #have to be odd
        self.digi_Mhz        = digi_Mhz
        start                =   int(time_range_ms[0]-t0_ms)*int(digi_Mhz*1e3)
        stop                 =   int(time_range_ms[1]-t0_ms)*int(digi_Mhz*1e3)
        defined_plamsa_edge  =   diode.def_plasma_edge #cm
        self.da_name         =   diode.da_name             
        self.def_plasma_edge =   defined_plamsa_edge
        nosie                =   diode.signal_noise
        nosie_p              =   diode.signal_noise_pc
        nosie_h              =   hard_noise
        self.diag_loc        =   diode.diag_loc
        self.diag_loc_cart   =   diode.diag_loc_cart
        self.diag_view_chord =   diode.diag_view_chord
        self.dnumhr          =   np.linspace(-diode.wham_r, diode.wham_r, xmesh)
        self.signal          =   diode.axuvData_cal[:, start:stop]
        self.time_coor       =   diode.time[start:stop]
        lv_signal            =   self.signal - nosie - nosie_h
        y_original           =   lv_signal   - nosie_p*np.max(lv_signal, 0)
        self.signal_ready    =   y_original
        y_original[y_original < 0] = 0
        x_original           =   diode.R_impact
        self.R_impact        =   diode.R_impact
        self.chi_signal_dig_khz= chi_signal_dig_khz
        self.wham_r          =   diode.wham_r
        self.diode_name      =   diode.name
        self.shotnum         =   diode.shotnum
        self.ft_dig_khz      =   fft_dig_khz
        self.FFT_time_window =   FFT_time_window  #ms
        # filling edge with linear fal-off 
        self.x_padded, self.y_padded   = self.edge_filling(y_original, x_original, defined_plamsa_edge)
        self.AB_light_Cal    =  diode.AB_light_Cal #0.0019819008635497135/0.9 # z of fiber and AXUV have 12cm long in z. 0.9 is the transimition of Ha filter
        self.attenuation     =  diode.attenuation
        self.kW_nA           =  diode.kW_nA
        self.AXUV_plasma_len =  diode.AXUV_plasma_len

        # Calculating plasma parameters input cm output cm
        
        (
        self.centroid, 
        self.radius, 
        self.c1n1_cm1, 
        self.detv_list,
        self.centroid_hr,
        self.radius_hr,
        self.signal_hr, 
        self.dnumhr_shift, 
        self.sym_axuvData_cal_hr_smooth,
        self.asy_axuvData_cal_hr_smooth
        )                              = self.plasma_analysis_more(test, self.x_padded*100, self.y_padded, 
                                                                   self.dnumhr*100, chi_signal_dig_khz, 
                                                                   recon_center, self.time_coor)

        # Abel invetion and Calculating 2D centriod
        self.X, self.Y                 = np.meshgrid(self.dnumhr, self.dnumhr)
        (
        self.gaussian_2d_in_time, 
        self.m1Gp, 
        self.num_gaussians_in_time,
        self.m0_3d_array,
        self.integrated_signal_in_time,
        self.radial_profile,
        self.cm_loc_ab,
        self.rms_loc_ab
        )                              = self.m1_abel_2D(test, self.X, self.Y,
                                                         self.dnumhr_shift, self.dnumhr,
                                                         self.asy_axuvData_cal_hr_smooth, 
                                                         self.sym_axuvData_cal_hr_smooth)

        # 2D centriod Gen
        # self.X, self.Y                 = np.meshgrid(self.dnumhr, self.dnumhr)
        # centroid radius are non even mesh, result are biased, centroid_hr radius_hr are even, use them
        (
        self.CM_2D, 
        self.R_2D, 
        self.CM_2D_Gvector,
        self.R_2D_Gvector,
        self.CM_2D_Gtest,
        self.R_2D_Gtest
        )                              = self.CM_2D(self.centroid_hr/100, self.radius_hr/100, 20,
                                                    self.time_coor, y_original)


        (self.mode0_f,
         self.mode1_f, 
         self.mode0_t, 
         self.mode1_t,
         self.mode0_P,
         self.mode1_P,
         self.mode0_P_db, 
         self.mode1_P_db,
         self.mode0_P_mean, 
         self.mode1_P_mean,
         self.signal_ind
        )                              = self.mode_FFT(self.ft_dig_khz, self.FFT_time_window)




    def mode_FFT(self, fft_dig_khz, FFT_time_window = 1, stop_signal_perc=0.02):

        from scipy.signal import butter, filtfilt
        from scipy.signal import stft


        time_window    = FFT_time_window
        original_fs    = 1 / np.mean(np.diff(self.time_coor))/1000
        frame          = int(original_fs/fft_dig_khz)
        time_steps     = list(range(0, len(self.time_coor), frame))

        max_time_steps =  max(time_steps)
        len_time_steps =  len(time_steps)
        totalsegsignal = np.sum(self.y_padded[:, time_steps], 0)
        #signal_ind = (totalsegsignal < max(totalsegsignal) * stop_signal_perc)

        

        y1 = self.c1n1_cm1[time_steps]
        y0 = self.rms_loc_ab[time_steps]
        x  = self.time_coor[time_steps]

        from scipy.signal import savgol_filter
        from scipy.ndimage import label
        
        # Find connected regions within signal_ind
        signal_ind = (totalsegsignal < np.max(totalsegsignal) * stop_signal_perc)
  
        


        def moving_average(x, w):
            return np.convolve(x, np.ones(w)/w, mode='same')

        def lowsignal_smoothing(y, signal_ind, method):
                    # Copy original signal
            y1_smoothed = y.copy()
            labeled_array, num_features = label(signal_ind)
            # Apply smoothing patch by patch
            for region in range(1, num_features + 1):
                region_mask = labeled_array == region
                indices = np.where(region_mask)[0]
            
                # Skip very short regions that can't support the smoothing window
                if len(indices) < 5:
                    continue
                if method == 1:
                    
                    window = min(20, len(indices))  # window size can be adjusted
                    if window % 2 == 0:
                        window -= 1  # make sure window size is odd
                
                    y1_smoothed[indices] = moving_average(y1[indices], w=window)            
                elif  method == 2:
                    # Use Savitzky-Golay filter on this segment
                    segment_len = len(indices)
                    if segment_len >= 5:  # Minimum needed for polyorder=2 and window ≥ 5
                        window = min(segment_len // 2 * 2 + 1, 15)  # Make sure it's odd
                        window = max(window, 5)
                        window = min(window, segment_len)  # Ensure window does not exceed segment
                        smoothed_patch = savgol_filter(y1[indices], window_length=window, polyorder=1)
                        y1_smoothed[indices] = smoothed_patch
    
            
                    y1_smoothed[indices] = smoothed_patch
                    
            return y1_smoothed
        



        y1_s = lowsignal_smoothing(y1, signal_ind, 2)
        y0_s = lowsignal_smoothing(y0, signal_ind, 2)
        
        #plt.plot(x*1000, y1)

        
        # y1[signal_ind] = 0
        # y0[signal_ind] = 0


        
        #plt.plot(x*1000, y1_s)
        #plt.show()
        # y1[(y > 1) | (y < 0)] = 0
        # y0[(y > 1) | (y < 0)] = 0
        
        # Sampling rate
        fs           = 1 / np.mean(np.diff(x))  # samples per second
        nperseg      = int(fs*time_window*1e-3)
        noverlap     = nperseg // 2
        f1, t1, Zxx1 = stft(y1_s, fs=fs, nperseg=nperseg, noverlap=noverlap)
        f0, t0, Zxx0 = stft(y0_s, fs=fs, nperseg=nperseg, noverlap=noverlap)
        #print(nperseg)
        amp0   = np.abs(Zxx0)
        amp1   = np.abs(Zxx1)
        power0 = np.abs(Zxx0) ** 2
        power1 = np.abs(Zxx1) ** 2
        
        # Power spectrum (dB scale)
        power_db0 = 20 * np.log10(np.abs(Zxx0) + 1e-12)
        power_db1 = 20 * np.log10(np.abs(Zxx1) + 1e-12)
        # power_db has shape (n_frequencies, n_times)
        mean_power_db0 = np.mean(power_db0, axis=0)  # shape = (n_times,)
        mean_power_db1 = np.mean(power_db1, axis=0)  # shape = (n_times,)

        #in_range_P_std  = np.std(power_db[])
        #in_range_P_mean = np.mean(power_db[])

        return f0, f1, t0, t1, power0, power1, power_db0, power_db1, mean_power_db0, mean_power_db1, signal_ind
        
    
    def CM_2D(self, cm, rms, bint, time, signal2D, centroid_r=0, xy_relsolution=0, stop_signal_perc=0.02):
        from scipy.stats import kstest
        from scipy.integrate import dblquad    
        from scipy.optimize import curve_fit        
        
        #mean_centriod = np.mean(fitfit, 0)[1]
        # x = np.linspace(-centroid_r, centroid_r, xy_relsolution)
        # X, Y = np.meshgrid(x, x)  # Meshgrid for both X and Y from the same array
        X, Y = self.X, self.Y
        Z    = np.zeros_like(self.X)   # Create an array of zeros matching X's shape
        Z_r  = np.zeros_like(self.X)
        
        # Define Gaussian function
        def Zgaussian(x, sigma):
            return 1/(np.sqrt(2*np.pi)*sigma)**2 * np.exp(-x / (2 * sigma**2))

        signal   = np.sum(signal2D, 0 )
        noms     = cm.shape[0]//bint
        fitfit   = np.zeros([noms, 3])
        fitfit_r = np.zeros([noms, 3])
        G_test   = np.zeros([noms, 2])
        G_test_r = np.zeros([noms, 2])
        # Loop through the data in segments
        da           = self.diag_loc[1]*np.pi/180
        data_segment   = np.array([])
        data_segment_t = np.array([])
        self.klist = []
        max_signal = max(signal)
        for k in range(noms):
            st_ind = k * bint
            if k == noms -1 : 
                ed_ind = -1
            else:
                ed_ind = (k + 1) * bint
                
            data_segment   = np.hstack((data_segment, cm[st_ind:ed_ind]))
            data_segment_r = rms[st_ind:ed_ind]
            data_segment_t = np.hstack((data_segment_t, time[st_ind:ed_ind]))
            totalsegsignal = np.sum(signal[st_ind:ed_ind])
            if totalsegsignal/max_signal >= stop_signal_perc:
                initial_guess   = [1  , np.mean(data_segment)  , np.std(data_segment)]
                initial_guess_r = [1, np.mean(data_segment_r), np.std(data_segment_r)]            
                A, mu, std       = initial_guess  # Handle fit failures gracefully
                A_r, mu_r, std_r = initial_guess_r
                G_test[k,:]     = kstest(data_segment  , 'norm', args=(mu  , std  ))
                G_test_r[k,:]   = kstest(data_segment_r, 'norm', args=(mu_r, std_r))
                #print(len(data_segment), G_test[k,1])
                meant         = np.mean(data_segment_t)   
                
                if G_test[k,1] >= 0.05 and k >= 1:
                    
                    data_segment   = np.array([])
                    data_segment_t = np.array([])
                    
                    RRsq   = (X-(-mu*np.sin(da)))**2 + (Y-( mu*np.cos(da)))**2
                    RRsq_r = (np.sqrt(RRsq)-mu_r)**2 
                    
                    Z     = Z   + Zgaussian(RRsq  , std  )/noms
                    Z_r   = Z_r + Zgaussian(RRsq_r, std_r)/noms
                    
                    self.klist.append( k)
                elif G_test[k,1] <= G_test[k-1,1] and G_test[k,1] != 0 and k >= 1:
                   
                    data_segment   = cm[st_ind:ed_ind]
                    data_segment_t = time[st_ind:ed_ind]
                    
                    RRsq   = (X-(-fitfit[k-1,1]*np.sin(da)))**2 + (Y-( fitfit[k-1,1]*np.cos(da)))**2
                    RRsq_r = (np.sqrt(RRsq)-fitfit_r[k-1,1])**2               
                    Z      = Z   + Zgaussian(RRsq  , fitfit[k-1,2]  )/noms
                    Z_r    = Z_r + Zgaussian(RRsq_r, fitfit_r[k-1,2])/noms
                    self.klist.append(k-1)
    
    
                fitfit[k,:]   = [meant, mu, std]
                fitfit_r[k,:] = [meant, mu_r, std_r]

        d_area    = (X[0,1]-X[0,0])*(Y[1,0]-Y[0,0])
        return Z, Z_r/np.sum(Z_r*d_area+1e-20), fitfit[self.klist], fitfit_r[self.klist], G_test[self.klist], G_test_r[self.klist]
  
                    

    

    def m1_abel_2D(self, test, X, Y, xin_time_s, x_sym_in_time_s, yin_time_s, y_sym_in_time_s):
        import abel
        from scipy.signal import find_peaks
        if test == 0:
            run_f = yin_time_s.shape[1]
            offset = 0
        else:
            offset = test
            run_f = 1
            test  = 1
               
        xmesh = len(x_sym_in_time_s)
        m1Gp  = {}
        num_gaussians_in_time = np.zeros(run_f)
        # Initialize the 2D Gaussian sum
        gaussian_2d_sum = np.zeros((len(xin_time_s), len(xin_time_s)))  # Example 2D array
        #gaussian_3d_array = np.stack([gaussian_2d_sum] * run_f, axis=0)
        gaussian_3d_array = np.zeros((run_f, len(xin_time_s), len(xin_time_s)))
        m0_3d_array       = np.zeros((run_f, len(xin_time_s), len(xin_time_s)))
        integrated_signal_in_time = np.zeros((xmesh, run_f))
        
        projection_smoothed = y_sym_in_time_s[xmesh//2:, :].T
        radial_profile      = abel.basex.basex_transform(projection_smoothed,
                                                         direction="inverse", verbose=False)*self.AB_light_Cal/self.attenuation
        
        x           = self.dnumhr*100
        profile     = radial_profile
        sym_profile = np.hstack([profile[:,::-1], profile[:,1:]])
        
        cm_loc_ab = np.sum(sym_profile * x, axis=1) / (np.sum(sym_profile, axis=1) + 1e-20)
        rms_loc_ab = 2 * np.sqrt(
            np.sum(sym_profile * (x - cm_loc_ab[:, np.newaxis]) ** 2, axis=1) /
            (np.sum(np.abs(sym_profile), axis=1) + 1e-20))

        

  
        return gaussian_3d_array, m1Gp, num_gaussians_in_time, m0_3d_array, integrated_signal_in_time, radial_profile, cm_loc_ab, rms_loc_ab


        
    def plasma_analysis_more(self, test, x_padded, y_padded, dnumhr, 
                             chi_signal_dig_khz, recon_center, time_coor):
        
        from scipy.ndimage import gaussian_filter
        from scipy.interpolate import RectBivariateSpline
        
        import pandas as pd

        if x_padded[:,0][1] < x_padded[:,0][0]:
            x_padded = x_padded[::-1] 
            y_padded = y_padded[::-1]
        #y_padded[y_padded < 0] = 0
        signal      = np.maximum((y_padded), 0)
        totalsignal = np.sum(signal, axis=0)+1e-20
        # Compute RMS Radius evolution in time
        meanloc     = np.sum(signal * x_padded, axis=0) / totalsignal
        cm_loc      = meanloc
        rms_loc     = 2*np.sqrt(np.sum(signal*(x_padded-meanloc)**2, axis=0) / totalsignal)        
              
       # print(x_padded[:,0])   
        interp_func         = RectBivariateSpline(x_padded[:,0], np.arange(signal.shape[1]), y_padded, kx=3, ky=3)
        signal_hr           = interp_func(dnumhr, np.arange(signal.shape[1]))
        signal_hr           = np.maximum((signal_hr), 0)

        #gsbsdfds

        
        cm_loc_hr = (np.sum(signal_hr*dnumhr.reshape(dnumhr.shape[0],1),0)/(np.sum(signal_hr,0)+1e-20))
        rms_loc_hr= 2*np.sqrt(np.sum(signal_hr*(dnumhr.reshape(dnumhr.shape[0],1)-cm_loc_hr)**2, axis=0) / (np.sum(signal_hr,0)+1e-20))  
        #print(cm_loc_hr) 
        # for at in [2500,3500,4000]:
        #     cc   = np.sum(x_padded[:,at]*signal[:,at])/np.sum(signal[:,at])
        #     cchr = np.sum(dnumhr*signal_hr[:,at])/np.sum(signal_hr[:,at])

        #     #if np.abs(cc-cchr)>3.6:
        #     plt.plot(dnumhr, signal_hr[:,at],'k')
        #     plt.plot(x_padded[:,at], signal[:,at],'b')
        #     print(cc,cchr)
        #     print(at)
        #     plt.show()
        # print(dnumhr)
        # print(x_padded[:,2500])
        
        if test == 0:
            run_f = signal_hr.shape[1]
        else:
            offset = test
            run_f = 1
            test  = 1
            
    
        asy_axuvData_cal_hr_smooth = np.zeros([len(dnumhr), run_f])
        sym_axuvData_cal_hr_smooth = np.zeros([len(dnumhr), run_f]) 
        dnumhr_shift               = np.zeros([len(dnumhr), run_f])
    
        peak_element = (np.argmax(signal_hr, 0)).astype(int)
        cm_element   = (cm_loc_hr//(dnumhr[1]-dnumhr[0])).astype(int)+dnumhr.shape[0]//2   
        
        if recon_center == 0:
            shift_element = np.zeros(cm_element.shape[0])+dnumhr.shape[0]//2   
        elif  recon_center == 1:    
            shift_element = peak_element
        elif  recon_center == 2:
            shift_element = cm_element
        elif  recon_center == 3:
            shift_element = np.zeros(cm_element.shape[0])+dnumhr.shape[0]//2
            
        backshift_for_x = (shift_element).astype(int)

        if recon_center == 3:
            for look in range(run_f):
        
                mid_ind = np.mod(np.linspace(shift_element[look]- (dnumhr.shape[0])//2,
                                             len(dnumhr)+shift_element[look]-1- (dnumhr.shape[0])//2,
                                             len(dnumhr)).astype(int), dnumhr.shape[0])
        
                ### symetery =====================================================
                asy_axuvData_cal_hr = (signal_hr[mid_ind, look] - signal_hr[mid_ind, look][::-1]) / 2
                sym_axuvData_cal_hr = (signal_hr[mid_ind, look] + signal_hr[mid_ind, look][::-1]) / 2

                
                asy_axuvData_cal_hr_smooth[:, look] = gaussian_filter(asy_axuvData_cal_hr, sigma=2)
                sym_axuvData_cal_hr_smooth[:, look] = gaussian_filter(sym_axuvData_cal_hr, sigma=2)
                dnumhr_shift[:, look]               = dnumhr+dnumhr[backshift_for_x[look]]
            max_ind         =   np.argmax(asy_axuvData_cal_hr_smooth, 0)
            asy_cm_loc_hr   =  (np.sum(asy_axuvData_cal_hr_smooth*dnumhr.reshape(dnumhr.shape[0],1),0)/
                                (np.sum(np.abs(asy_axuvData_cal_hr_smooth),0)+1e-20))
        
            max_ind_Sym     = np.argmax(sym_axuvData_cal_hr_smooth, 0)
            rhot_c1         =  dnumhr_shift[max_ind, np.arange(len(max_ind))]
            
        else :
            for look in range(run_f):
                
                mid_ind = np.mod(np.linspace(shift_element[look]- (dnumhr.shape[0])//2,
                                             len(dnumhr)+shift_element[look]-1- (dnumhr.shape[0])//2,
                                             len(dnumhr)).astype(int), dnumhr.shape[0])
        
                
                asy_axuvData_cal_hr = np.maximum((-signal_hr[mid_ind, look][::-1]
                                                  +signal_hr[mid_ind, look]),0)
                sym_axuvData_cal_hr = signal_hr[mid_ind, look] - np.maximum((asy_axuvData_cal_hr), 0)
                ### symetery =====================================================
                
                asy_axuvData_cal_hr_smooth[:, look] = gaussian_filter(asy_axuvData_cal_hr, sigma=2)
                sym_axuvData_cal_hr_smooth[:, look] = gaussian_filter(sym_axuvData_cal_hr, sigma=2)
                dnumhr_shift[:, look]               = dnumhr+dnumhr[backshift_for_x[look]]
    
            max_ind         =   np.argmax(asy_axuvData_cal_hr_smooth, 0)
            asy_cm_loc_hr   =  (np.sum(asy_axuvData_cal_hr_smooth*dnumhr.reshape(dnumhr.shape[0],1),0)/
                                (np.sum(asy_axuvData_cal_hr_smooth,0)+1e-20))
        
            max_ind_Sym     = np.argmax(sym_axuvData_cal_hr_smooth, 0)
            rhot_c1         =  dnumhr_shift[max_ind, np.arange(len(max_ind))]
    
        
        
 
        whamc = dnumhr[backshift_for_x]

        if recon_center == 2:
            c1n1         = np.sum(asy_axuvData_cal_hr_smooth,0)/np.sum(signal_hr,0)
            c1n1_cm1     = c1n1
        #elif (recon_center == 0 or recon_center == 1 ):
        else:
            c1n1         = abs((cm_loc_hr-whamc)/(rhot_c1-whamc+1e-20))
            c1n1_cm1     = abs((cm_loc_hr-whamc)/(asy_cm_loc_hr-whamc+1e-20)) 
#        elif recon_center == 3:
#            c1n1         = (cm_loc_hr-whamc)/(asy_cm_loc_hr-whamc+1e-20)
#            c1n1_cm1     = c1n1
#
#        print("m1 = ", c1n1_cm1.shape)
#        print("c1n1_cm1 = ", c1n1_cm1[1600])
#        print("c1n1_cm1 = ", c1n1_cm1)
        determinants = []

        cm_loc = np.nan_to_num(cm_loc, nan=0)
        rms_loc = np.nan_to_num(rms_loc, nan=0)
        c1n1_cm1 = np.nan_to_num(c1n1_cm1, nan=0)
        
        data = {
            'CM': cm_loc_hr, #cm_loc, #cm_coor cm
            'R' : rms_loc_hr/2 #rms_loc #rms_coor cm
        }
        df = pd.DataFrame(data)
        
        maxdev = 0
        realmaxdev = 0
        detv_list = {}
        for jjh in range(len(chi_signal_dig_khz)):
    
            window_size = 1/chi_signal_dig_khz[jjh]*1000
            wt = int((window_size))
            rolling_cov_matrix = df.rolling(window=wt).cov()
            determinants = []
            
            for i in range(wt - 1, len(df)):
                # Extract the rolling covariance matrix at each index
                cov_matrix = rolling_cov_matrix.loc[i].values.reshape(len(df.columns), -1)    
                # Calculate the determinant if the matrix is square
                det = np.abs(np.linalg.det(cov_matrix))
                #det = np.nan_to_num(det, nan=0.0)

                determinants.append(np.sqrt(det))
                #determinants.append(np.sqrt(cov_matrix[0,0]))
                #print(f"Determinant of rolling covariance matrix at index {i}: {det*1e6:.4f}")
        
            detv = np.array(determinants)
            detv_list[f"{jjh}_dev"] = detv
            detv_list[f"{jjh}_t"]   = time_coor[np.arange(wt - 1, len(df))-wt//2]#[np.arange(wt - 1, len(df))-wt//2]: [np.arange(wt - 1, len(df))]
            
            maxdev = max(detv)
            realmaxdev = max([realmaxdev, maxdev])
    
        if test == 1:
            plt.plot(x_original, y_original[:, look])
            #plt.plot(dnumhr, signal_hr[mid_ind, look])
            
            plt.plot(dnumhr_shift[:, look], asy_axuvData_cal_hr_smooth[:, look])
            plt.plot(dnumhr_shift[:, look], sym_axuvData_cal_hr_smooth[:, look])
            plt.plot(dnumhr_shift[:, look], sym_axuvData_cal_hr_smooth[:, look]+
                                   asy_axuvData_cal_hr_smooth[:, look], 
                    linestyle='--')
            plt.xlim([-24, 24])
            plt.show()
        
        return  cm_loc, rms_loc, c1n1_cm1, detv_list, cm_loc_hr, rms_loc_hr,\
        signal_hr, dnumhr_shift, sym_axuvData_cal_hr_smooth, asy_axuvData_cal_hr_smooth
        
        
    def edge_filling(self, y_original, x_original, plasma_edge, mode="l"):
        flip = 0
 ########======supper hot fix ============Kai x_original[1 and 3] DA3 broke=============================
        if self.diode_name == 'DIODEARRAY3':
            for i in (1, 3):
                x0, y0 = x_original[i-1], y_original[i-1]
                x1, y1 = x_original[i+1], y_original[i+1]
                xi = x_original[i]

                denom = (x1 - x0)
                w = np.divide((xi - x0), denom, out=np.zeros_like(xi, dtype=float), where=denom!=0)

                y_original[i] = y0 + w * (y1 - y0)
  ########================supper hot fix ================================================


        if x_original[1] > x_original[0]:
            x_original = x_original[::-1] 
            y_original = y_original[::-1]
            flip       = 1

        plasma_edge_p =  plasma_edge
        plasma_edge_n = -plasma_edge
        if x_original[0]  >= plasma_edge:   # postive end
            plasma_edge_p = x_original[0]+0.01 #1cm
        if x_original[-1] <= -plasma_edge:  # negative end
            plasma_edge_n = x_original[-1]-0.01 #1cm     
        left_x_mid  = np.mean([plasma_edge_p , x_original[0 ]])
        right_x_mid = np.mean([plasma_edge_n , x_original[-1]])


        left_x_mid12 = np.linspace(plasma_edge_p , x_original[0], 4)[1:3]
        right_x_mid12 = np.linspace(plasma_edge_n ,x_original[-1], 4)[1:3]
        #print(left_x_mid12)
        
        left_xinter = plasma_edge_p
        righ_xinter = plasma_edge_n
        

        # Calculate slopes and intercepts using the mean of adjacent edge elements
        leftslope = ((y_original[0] + y_original[1]) / 2 - (y_original[1] + y_original[2]) / 2) / \
            ((x_original[0] + x_original[1]) / 2 - (x_original[1] + x_original[2]) / 2)
        
        rightslope = ((y_original[-1] + y_original[-2]) / 2 - (y_original[-2] + y_original[-3]) / 2) / \
            ((x_original[-1] + x_original[-2]) / 2 - (x_original[-2] + x_original[-3]) / 2)






        def Gfalling(x0, y0, m, plasma_edge, sigma_nm):
            sigma = np.abs(x0 - plasma_edge)/sigma_nm
            G_mu  = x0+(m/(y0+1e-20))*sigma**2
            G_A   = y0*np.exp(m**2*sigma**2/(2*y0**2+1e-20))
            return  G_mu, G_A, sigma       
        totalsegsignal = np.sum(y_original,0)
        max_signal     = np.max(y_original,0)
        #print(max_signal.shape)
        stop_signal_perc  = 0.050
        inG_r = (y_original[-1]/(max_signal+1e-20) >=stop_signal_perc) & (totalsegsignal > 0.02)
        inG_l = (y_original[0] /(max_signal+1e-20) >=stop_signal_perc) & (totalsegsignal > 0.02)
        
        left_mu, left_A, left_sigma = Gfalling(x_original[0], y_original[0, inG_l],
                                               leftslope[inG_l] , plasma_edge=plasma_edge_p, sigma_nm=2)
        righ_mu, righ_A, righ_sigma = Gfalling(x_original[-1], y_original[-1,inG_r],
                                               rightslope[inG_r], plasma_edge=plasma_edge_n, sigma_nm=2)
        
        # print(x_original[0], x_original[-1])
        # print(left_xinter, left_x_mid, right_x_mid, righ_xinter)

                #print(left_mu, left_A, left_sigma)





        
        
        leftslope[leftslope>=0]   = -leftslope[leftslope>=0]
        rightslope[rightslope<=0] = -rightslope[rightslope<=0]
        left_intercept  = (y_original[0] + y_original[1])  / 2 - leftslope * (x_original[0] + x_original[1]) / 2
        right_intercept = (y_original[-1] + y_original[-2]) / 2 - rightslope * (x_original[-1] + x_original[-2]) / 2
        
        left_c  = y_original[0] -leftslope *x_original[0]
        right_c = y_original[-1]-rightslope*x_original[-1]







        

        # plt.plot(left_mu)
        # plt.plot(x_original[0]+left_mu*0, 'k')
        # plt.plot(left_x_mid+left_mu*0, 'r')
        # plt.plot(plasma_edge_p+left_mu*0, 'b')
        # plt.show()
        # plt.plot(righ_mu)
        # plt.plot(x_original[-1]+righ_mu*0, 'k')
        # plt.plot(right_x_mid+righ_mu*0, 'r')
        # plt.plot(plasma_edge_n+righ_mu*0, 'b')
        # plt.show()
        # plt.plot(left_A)
        # plt.show()   





        
        left_y_mid  = left_x_mid*leftslope+left_c
        right_y_mid = right_x_mid*rightslope+right_c
        left_yinter = left_xinter*leftslope+left_c
        righ_yinter = righ_xinter*rightslope+right_c



        left_y_mid12  =  left_x_mid12.reshape(left_x_mid12.shape[0],1) *leftslope +left_c
        right_y_mid12 = right_x_mid12.reshape(right_x_mid12.shape[0],1)*rightslope+right_c
        left_y_mid12[left_y_mid12 < 0] = 0
        right_y_mid12[right_y_mid12 < 0] = 0

        # print(left_y_mid12)
        # print(right_y_mid12)
        
        right_y_mid[right_y_mid < 0] = 0
        left_y_mid[left_y_mid < 0] = 0
        
        right_y_mid[inG_r]= righ_A*np.exp(-(right_x_mid-righ_mu)**2/(2*righ_sigma**2))
        left_y_mid[inG_l] = left_A*np.exp(-(left_x_mid -left_mu)**2/(2*left_sigma**2 ))

        left_y_mid12[0,inG_l]  = left_A*np.exp(-(left_x_mid12[0] -left_mu)**2/(2*left_sigma**2 ))
        right_y_mid12[0,inG_r] = righ_A*np.exp(-(right_x_mid12[0]-righ_mu)**2/(2*righ_sigma**2))
        left_y_mid12[1,inG_l]  = left_A*np.exp(-(left_x_mid12[1] -left_mu)**2/(2*left_sigma**2 ))
        right_y_mid12[1,inG_r] = righ_A*np.exp(-(right_x_mid12[1]-righ_mu)**2/(2*righ_sigma**2))
        
        # totalsegsignal = np.sum(y_original,0)
        # max_signal     = np.max(totalsegsignal,0)
        # # print(totalsegsignal.shape)
        # stop_signal_perc  = 0.05

        # right_sigma =   y_original[-1]/(rightslope+1e-20)
        # left_sigma  =  -y_original[0]/(leftslope+1e-20)

        # right_A =   rightslope *right_sigma *np.sqrt(np.e)
        # left_A  =   leftslope * left_sigma*np.sqrt(np.e)

        # right_mu =   x_original[-1]+right_sigma
        # left_mu  =   x_original[0]-left_sigma
        # inG = y_original[-1]/max_signal >=stop_signal_perc
        # #plt.plot(right_A[y_original[-1]/max_signal >=stop_signal_perc])
        # plt.plot(right_sigma[inG])       
        # plt.show()
        # # right_y_mid_G = right_y_mid.copy()
        # # right_y_mid_G[inG] = right_A[inG]*np.exp(-(right_x_mid-right_mu[inG])**2/(2*right_sigma[inG]**2))
        # #right_y_mid[y_original[-1]/max_signal >=stop_signal_perc] = right_A
        # plt.plot(right_A[inG]*np.exp(-(right_x_mid-right_mu[inG])**2/(2*right_sigma[inG]**2)))
        # plt.plot(right_y_mid[inG])        
        # plt.plot(y_original[-1][inG])
        # plt.show()

        y_padded = np.concatenate([
            np.zeros((1, y_original.shape[1])),  # Top padding
            left_y_mid12,  # Left padding
            y_original,  # Original data
            right_y_mid12[::-1],  # Right padding
            np.zeros((1, y_original.shape[1]))  # Bottom padding
        ], axis=0)
        y_padded = y_padded-np.max((y_padded[0,:], y_padded[-1,:]),0)
        x_original_repeated = np.tile(x_original, (y_original.shape[1], 1)).T  # Transpose to shape (20, 6000)


        
        # Concatenate with repeated x_original
        x_padded = np.concatenate([
            left_xinter+np.zeros((1, y_original.shape[1])),  # Top padding
            np.tile(left_x_mid12.reshape(2, 1), (1, y_original.shape[1])),  # Left padding
            x_original_repeated,  # Original data repeated for 6000 columns
            np.tile(right_x_mid12[::-1].reshape(2, 1), (1, y_original.shape[1])),
            righ_xinter+np.zeros((1, y_original.shape[1]))  # Bottom padding
        ], axis=0)        

        # print(x_padded[:,0])
        
        # y_padded = np.concatenate([
        #     np.zeros((1, y_original.shape[1])),  # Top padding
        #     left_y_mid.reshape(1, y_original.shape[1]),  # Left padding
        #     y_original,  # Original data
        #     right_y_mid.reshape(1, y_original.shape[1]),  # Right padding
        #     np.zeros((1, y_original.shape[1]))  # Bottom padding
        # ], axis=0)
        # y_padded = y_padded-np.max((y_padded[0,:], y_padded[-1,:]),0)
        # x_original_repeated = np.tile(x_original, (y_original.shape[1], 1)).T  # Transpose to shape (20, 6000)
        
        # # Concatenate with repeated x_original
        # x_padded = np.concatenate([
        #     left_xinter+np.zeros((1, y_original.shape[1])),  # Top padding
        #     left_x_mid+np.zeros((1, y_original.shape[1])),  # Left padding
        #     x_original_repeated,  # Original data repeated for 6000 columns
        #     right_x_mid+np.zeros((1, y_original.shape[1])),  # Right padding
        #     righ_xinter+np.zeros((1, y_original.shape[1]))  # Bottom padding
        # ], axis=0)

        if flip == 1:
            x_padded = x_padded[::-1] 
            y_padded = y_padded[::-1]  


        # cccc=np.linspace(1,len(totalsegsignal),len(totalsegsignal))
        # ccindx = cccc[inG].astype(int)
        # print(ccindx[0])

        # dfdf = left_A > 100

        # fft = np.linspace(1,len(totalsegsignal),len(totalsegsignal)).astype(int)
        # ffrv = fft[inG_l]
        # print(ffrv.shape)
        # ff = ffrv[dfdf]
        # print(left_A[dfdf])
        # plt.plot(x_padded[:,0], y_padded[:,ff], 'r--', alpha=0.4)
        # plt.plot(x_original, y_original[:,ff] )
        # plt.show()    
        return x_padded, y_padded

