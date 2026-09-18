import numpy as np

class ObjectInit:
    """Class to hold Diode and WHAM object structures."""
    
    class Diode:
        def __init__(self, name, da_name, shotnum, location, time, diag_loc, wham_r, nod, recal_dl, perspective,
                     ref_d_num, axuvData, Rarray, Rs, philist, view_ang, dnum, S_eff, def_p_edg,
                     def_noise, def_noise_p, attenuation, AB_light_Cal, kW_nA, AXUV_plasma_len,
                     def_noise_h=0, cal_mode=0):
            self.name             =  name
            self.da_name          =  da_name
            self.shotnum          =  shotnum
            self.location         =  location
            self.time             =  time  
            self.diag_loc_raw     =  diag_loc
            self.wham_r           =  wham_r
            self.nod              =  nod
            self.recal_dl         =  recal_dl
            self.perspective      =  perspective
            self.ref_d_num        =  ref_d_num
            self.axuvData         =  axuvData
            self.Rarray           =  Rarray
            self.Rs               =  Rs
            self.philist          = -philist  #go change in wham, then put this [philist*180/np.pi]
            self.view_ang         =  view_ang
            self.dnum             =  dnum.astype(int)
            self.S_eff            =  S_eff/max(S_eff)
            self.def_plasma_edge  =  def_p_edg    
            self.attenuation      = attenuation
            self.AB_light_Cal     = AB_light_Cal
            self.kW_nA            = kW_nA
            self.AXUV_plasma_len  = AXUV_plasma_len

            # The tree's CURRENT node is defined as VOLTAGE / RESISTOR [A]. When
            # RESISTOR is stored in Ohms (~1e4) that division is already done and
            # axuvData is a true current, so A -> nA is all that is left. Older
            # trees stored RESISTOR in kOhm (~10), leaving axuvData a voltage that
            # still needs dividing by the transimpedance.
            resistor_in_ohm           =  np.nanmedian(self.Rarray) > 1000
            if shotnum > 250700000 and not resistor_in_ohm:
                axuvData_cal          =  self.axuvData/(self.Rarray.reshape(len(self.Rarray),1)*1000)*1e9

            else:
                axuvData_cal          =  self.axuvData*1e9
            if cal_mode == 0:
                self.axuvData_cal     =  axuvData_cal/(self.S_eff.reshape(len(self.S_eff),1))
            elif cal_mode == 1:
                self.axuvData_cal     =  axuvData_cal
                print("In Cal Mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                raise ValueError(f"Invalid cal_mode: {cal_mode}. Expected 0 or 1.")
                
            self.axuvData_cal[self.axuvData_cal < 0] = 0
            self.diag_loc         =  self.diag_loc_raw + self.recal_dl
            self.diag_loc_cart    = [self.diag_loc[0]*np.cos(self.diag_loc[1]*np.pi/180),
                                     self.diag_loc[0]*np.sin(self.diag_loc[1]*np.pi/180),
                                     self.diag_loc[2]]
            self.diag_view_chord  =  self.philist + self.diag_loc[1]+self.perspective[0]
            #print( self.diag_loc[1])
            self.R_impact         =  self.impact_p()
            #print(self.R_impact)
            self.signal_noise_pc  =  def_noise_p
            self.signal_noise     =  def_noise
            
            self.centroid, self.R_rms     =  self.cm_rms(self.diag_view_chord , self.axuvData_cal) 
            self.centroid_r, self.R_rms_r =  self.cm_rms(self.R_impact  , self.axuvData_cal)
            self.centroid_n, self.R_rms_n =  self.cm_rms(self.dnum  , self.axuvData_cal)
            
        def impact_p(self):
            #b = -np.tan((self.philist+self.perspective[0])/180*np.pi)*self.diag_loc[0]+self.perspective[1]
            b = -np.sin((self.philist+self.perspective[0])/180*np.pi)*self.diag_loc[0]+self.perspective[1]

            return b
        def cm_rms(self, coordinate1d, signal):

            coordinate   =   (np.ones(signal.shape).T*coordinate1d).T
            nod_local    =   signal.shape[0]                              # [.....] -> array like input for each noT
            not_local    =   signal.shape[1]
            dnum         =   np.linspace(1, nod_local, nod_local).astype(int)
            totalsignal  =   np.sum(signal, axis=0)+1e-20
    
            # Compute C.M. and RMS Radius evolution in time
            meanphi      =   np.sum(signal * coordinate, axis=0) / totalsignal
            cm_coor      =   meanphi
            rms_coor     =   np.sqrt(np.sum(signal*(coordinate-meanphi)**2, axis=0) / totalsignal)
            return cm_coor, rms_coor

    class WHAM:
        def __init__(self, name, location, R):
            self.name     = name
            self.location = location
            self.R        = R
