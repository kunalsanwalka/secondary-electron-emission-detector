import MDSplus as mds
import numpy as np
from axuv_postprocessing import object_init


class AXUV_Read:
    """Main class to handle AXUV diagnostics and store multiple diodes."""

    def __init__(self, shotnum, tree='axuv', selected_indices_input=None,
                 cal_mode=0, plasma_edge = None,
                 recal = None):
        """Initialize MDSplus tree and create diode objects."""
        self.tree    = mds.Tree(tree, shotnum)  
        self.shotnum = shotnum
        self.diodes  = {}  
        self.wham    = {}
        self.cal_mode= cal_mode
        if recal is not None:
            self.recal_input = recal
        else:
            self.recal_input = [-1.5, 0]
        self.create_diodes(plasma_edge, selected_indices_input)
        self.create_wham()

        
    def create_wham(self):
        """Create wham wall objects."""
        self.wham["CentralCell"] = object_init.ObjectInit.WHAM(name="CentralCell", location=[0, 0, 0  ], R=0.362,)
        self.wham["EastBox"]     = object_init.ObjectInit.WHAM(name="EastBox"    , location=[0, 0, 0.4], R=0.362,) 

    def create_diodes(self, plasma_edge_inread, selected_indices=None, ):
        """Fetch diodes from MDSplus tree and create diode objects."""
        if self.tree.expt == "AXUV": # old style, from when it was a subtree
            nodes = self.tree.getNodeWild("*") 
            node_header = ''
        elif self.tree.expt == "WHAM":
            nodes = self.tree.getNodeWild("\\AXUV.*")
            node_header = '\\AXUV.'
        else:
            pass 
        self.num_diodes = len(nodes)
        
        
        # If no indices are provided, select all
        if selected_indices is None:
            selected_indices = list(range(1, len(nodes) + 1))  # 1-based index

        kk = 0
        for idx, node in enumerate(nodes, start=1):  # 1-based indexing
            #if idx in selected_indices:

            if str(node.getNodeName())[0:3] == "DIO":
            #if str(node.getNodeName()) == "DIODEARRAY1":
                #print(node)
                kk += 1
                node_name = str(node.getNodeName())
                node_name = f'{node_header}{node_name}'

                try:
                    self.DA_name = self.tree.getNode(f'{node_name}.NAME').getTag()[0]
                    #print(self.DA_name)
                except Exception as e: # This change is for when the node name has the full path
                    #print(f"Warning: failed to get tag for {node_name} ({e})")
                    if self.shotnum > 250700000:
                        #if node_name == "DIODEARRAY1":
                        if 'DIODEARRAY1' in node_name:
                            self.DA_name = "Broadband"
                        #elif node_name == "DIODEARRAY2":
                        elif 'DIODEARRAY2' in node_name:
                            self.DA_name = r"$H_{\alpha}$"+"\n[654-659 nm]"
                        #elif node_name == "DIODEARRAY3":
                        elif 'DIODEARRAY3' in node_name:
                            self.DA_name = "soft_X\n[10–1000 eV]"
                        else:
                            self.DA_name = "No name"
                    elif self.shotnum > 241000000:
                        if node_name == "DIODEARRAY1":
                            self.DA_name = "soft_X\n[10–1000 eV]"
                        else:
                            self.DA_name = "soft_X\n[10–1000 eV]"
                    else:
                        self.DA_name = "Broadband"
                        
                        
                location = "Unknown_Location"

                try:
                    time = self.tree.getNode(f'{node_name}.CH_01.CURRENT').getData().dim_of().data()
                except:
                    #print(f"Warning: Missing CURRENT time data for {node_name}")
                    time = self.tree.getNode(f'{node_name}.CH_01.PHOTOCURRENT').getData().dim_of().data()

                try:
                    diag_loc = np.array([
                        self.tree.getNode(f'{node_name}.diag_r').getData().data(),
                        self.tree.getNode(f'{node_name}.diag_0').getData().data(),
                        self.tree.getNode(f'{node_name}.diag_z').getData().data()])
                except:
                    #print(f"Warning: Missing diag_loc for {node_name}")
                    diag_loc = np.array([0.38, -120, 0])
#                    if kk == 1:
#                        diag_loc = np.array([0.38, -120, 0]) # [m deg m] -180 to -180
#                    if kk == 2:
#                        diag_loc = np.array([0.38, -30, 0]) # [m deg m] -180 to -180

                try:
                    recal_dl = np.array([
                        self.tree.getNode(f'{node_name}.re_diag_r').getData().data(),
                        self.tree.getNode(f'{node_name}.re_diag_0').getData().data(),
                        self.tree.getNode(f'{node_name}.re_diag_z').getData().data()
                    ])
                except:
                    #print(f"Warning: Missing recal_dl for {node_name}")
                    recal_dl = np.array([0, 0, 0]) # [m deg m]

                try:
                    perspective = np.array([
                        self.tree.getNode(f'{node_name}.per_spt_0').getData().data(),
                        self.tree.getNode(f'{node_name}.per_spt_a').getData().data()])
                except:
                    #print(f"Warning: Missing recal_dl for {node_name}")
                    perspective = np.array(self.recal_input) # [deg m][-1.66, 0] -3.55 seems new resluf for 2508 3 diode setup relative to vessel, but WHAM-R have not calibrate vessel with B. so i use -1, which more matach with experiment i soft-X.

                try:
                    wham_r = self.tree.getNode(f'{node_name}.wham_r').getData().data()
                except:
                    #print(f"Warning: Missing wham_r for {node_name}")
                    wham_r = 0.362

                try:
                    ref_d_num = self.tree.getNode(f'{node_name}.ref_d_num').getData().data()
                except:
                    #print(f"Warning: Missing ref_d_num for {node_name}")
                    ref_d_num = 11
                if plasma_edge_inread is None:
                    try:
                        def_p_edg = self.tree.getNode(f'{node_name}.def_plamsa_edge').getData().data()
                    except:
                        #print(f"Warning: Missing defing plasma edge data for {node_name}")
                        def_p_edg = 0.25
                else:
                        def_p_edg = plasma_edge_inread
                    
                try:
                    def_noise_p = self.tree.getNode(f'{node_name}.def_noise_percentage').getData().data()
                except:
                    #print(f"Warning: Missing manaully defind % of nosie-cutoff for {node_name}")
                    def_noise_p = 0.00
                    
                try:
                    def_noise = self.tree.getNode(f'{node_name}.def_noise').getData().data()
                except:
                    #print(f"Warning: Missing nosie data for {node_name}")
                    def_noise = 0

                try:
                    attenuation = self.tree.getNode(f'{node_name}.filter_attenuation').getData().data()
                except:
                    #print(f"Warning: Missing filter attenuation for {node_name}")
                    if self.shotnum > 250700000: # KDM 260420, this change is to allow cases where the node name has full path
                        if 'DIODEARRAY1' in node_name:
                        #if node_name == "DIODEARRAY1":
                            attenuation = 1                   
                        #elif node_name == "DIODEARRAY2":
                        elif 'DIODEARRAY2' in node_name:
                            attenuation = 0.9
                        #elif node_name == "DIODEARRAY3":
                        elif 'DIODEARRAY3' in node_name:
                            attenuation = 0.5
                        else:
                            attenuation = 1
                    elif self.shotnum > 241000000:
                        if node_name == "DIODEARRAY1":
                            attenuation = 0.5                   
                        else:
                            attenuation = 0.5
                    else:        
                        attenuation = 1

                try:
                    AXUV_plasma_len = self.tree.getNode(f'{node_name}.AXUV_plasma_len').getData().data()
                except:
                    #print(f"Warning: Missing AXUV in view plasma_len data for {node_name}")
                    AXUV_plasma_len = 12 #cm

                try:
                    AB_light_Cal    = self.tree.getNode(f'{node_name}.AB_light_Cal').getData().data()
                except:
                    #print(f"Warning: Missing emissity caliburation data for {node_name}")
                    AB_light_Cal    =  0.0197222583262867/0.9/AXUV_plasma_len #[W/cm^3 raw Abel[nA]] z of fiber and AXUV have 12cm, long in z. 0.9 is the transimtion for Ha filter old_valve0.023782810362596562

                kW_nA = 1/attenuation/AB_light_Cal/(4*np.pi)*1e-6/np.sqrt(2)        

                # Get child nodes
                #child_nodes = node.getChildren()
                child_nodes = node.descendants

                # Populate arrays with data from child nodes
                axuvData, Rarray, Rs, philist, view_ang, dnum, S_eff, nod = self.populate_arrays(node_name, child_nodes, time)
            
                # Store Diode object with all attributes
                self.diodes[node_name.replace(node_header,'')] = object_init.ObjectInit.Diode(
                    name=node_name, da_name=self.DA_name,  shotnum=self.shotnum, location=location, time=time, diag_loc=diag_loc, wham_r=wham_r,
                    nod=nod, recal_dl=recal_dl, perspective=perspective, ref_d_num=ref_d_num, axuvData=axuvData,
                    Rarray=Rarray, Rs=Rs, philist=philist, view_ang=view_ang, dnum=dnum, S_eff=S_eff, def_p_edg=def_p_edg,
                    def_noise=def_noise, attenuation=attenuation, AB_light_Cal=AB_light_Cal, kW_nA=kW_nA, AXUV_plasma_len=AXUV_plasma_len,
                    def_noise_p=def_noise_p, cal_mode=self.cal_mode
                )
    
    def populate_arrays(self, node_name, child_nodes, time):
        """Collects data from the tree and populates the arrays."""
        # Initialize NumPy arrays
        #ch_doide = [ch_data.getNodeName() for idx, ch_data in enumerate(child_nodes)]
        #nod      = sum([ch_name[0:2] == "CH" for ch_name in ch_doide])
        ch_diode = [ch.getNodeName() for ch in child_nodes]
        nod = sum(name.startswith("CH") for name in ch_diode)
        axuvData = np.zeros((nod, len(time))) if time is not None else None
        Rarray   = np.zeros(nod)
        Rs       = np.zeros_like(Rarray)
        philist  = np.zeros(nod) 
        view_ang = np.zeros_like(philist)
        dnum     = np.zeros_like(view_ang)
        S_eff    = np.zeros_like(dnum)
        #print(child_nodes)
        #print(len(S_eff))
        idxx = -1
        # 3 diode cal result, hard code here for temp. 2508
        # all reference to D1 D-numder = 11, which is indx = 10
        # also used -3.55 for perspective upstair, also seted shotnum > 250700000
        D1_philist = -np.array([-23.13, -20.95, -18.41, -15.98, -13.47,
                                -10.88,  -8.2 ,  -5.56,  -2.63,   0.  ,
                                  2.6 ,   5.47,   8.42,  11.  ,  13.68,
                                 16.18,  18.85,  21.19,  23.81,  26.11])

        D2_philist = -np.array([-22.05, -19.56, -17.01, -14.58, -12.01,
                                 -9.37,  -6.7 ,  -4.  ,  -1.29,   1.49,
                                  4.27,   7.01,   9.77,  12.46,  15.06,
                                 17.7 ,  20.12,  22.8 ,  24.99,  27.21])
        D3_philist = -np.array([-22.12, -19.99, -17.8 , -15.12, -12.75,
                                -10.05,  -7.61,  -4.97, -2.32,   0.4 ,
                                  3.08,   5.8 ,   8.47,  11.54,  14.04,
                                 16.61,  19.  ,  21.56,  24.82,  27.08])
                                
        D1_seff= np.array([1.01109865, 0.98362196, 1.00291853, 1.01502929, 0.99352676,
                           0.99570874, 1.03092762, 1.02113219, 1.00544264, 1.        ,
                           1.01230402, 1.00754269, 1.00592678, 1.0009473 , 1.00980598,
                           1.00641121, 1.00408031, 0.99623358, 0.97669736, 0.9818528 ])
        D2_seff= np.array([0.76814526, 0.78068402, 0.79688044, 0.86612483, 0.82181614,
                           0.82677999, 0.82278266, 0.83250041, 0.81204191, 0.85302393,
                           0.85014532, 0.82596381, 0.86456005, 0.82225204, 0.83754074,
                           0.81672052, 0.79657567, 0.74485527, 0.74341613, 0.75179168])
        D3_seff= np.array([0.76700681, 0.75093836, 0.76270771, 0.79223071, 0.81270142,
                           0.80752119, 0.7731693 , 0.80613255, 0.78507237, 0.84037358,
                           0.83780021, 0.82558822, 0.77435616, 0.80710916, 0.81599461,
                           0.78770072, 0.8150493 , 0.79894284, 0.81886034, 0.76211521])
        
        for idx, cnode in enumerate(child_nodes):
            cnode_name = str(cnode.getNodeName())
            if cnode_name[0:2] == "CH":
                idxx += 1
                #print(cnode_name, idxx)

                if self.shotnum > 250700000 :
                    #if node_name == "DIODEARRAY1":
                    if 'DIODEARRAY1' in node_name:
                        philist[idxx] = D1_philist[idxx]

                    #elif node_name == "DIODEARRAY2":
                    elif 'DIODEARRAY2' in node_name:
                        philist[idxx] = D2_philist[idxx]
                    elif 'DIODEARRAY3' in node_name:
                    #elif node_name == "DIODEARRAY3":
                        philist[idxx] = D3_philist[idxx]
                    else:
                        print("error, check philist")
                else:
                    try:
                        # i was changed to rad here and then change back to deg in ObjectInit, i stoped this craziness 250901  / 180 * np.pi
                        philist[idxx] = (
                            (self.tree.getNode(f'{node_name}.{cnode_name}.LOS_DEG').getData().data())
                        )
                        
                    except:
#                        ch1 = self.tree.getNode(f"{node_name}.CH_01")
#
#                        for n in ch1.getDescendants():
#                            print(n.getFullPath())
#                        phi = self.tree.getNode(f"\AXUV::TOP:DIODEARRAY1.{cnode_name}:B_IMPACT").getData().data()
#                        print(phi)
                        print(f"Warning: Missing LOS_DEG for {node_name}.{cnode_name}")
                        philist[idxx] = np.nan
            
                try:
                    view_ang[idxx] = self.tree.getNode(f'{node_name}.{cnode_name}.AOV_DEG').getData().data()
                except:
                    print(f"Warning: Missing AOV_DEG for {node_name}.{cnode_name}")
                    view_ang[idxx] = np.nan
                    
                if self.shotnum > 250700000 :
                    if 'DIODEARRAY1' in node_name:
                    #if node_name == "DIODEARRAY1":
                        S_eff[idxx] = D1_seff[idxx]
                    elif 'DIODEARRAY2' in node_name:
                    #elif node_name == "DIODEARRAY2":
                        S_eff[idxx] = D2_seff[idxx]
                    elif 'DIODEARRAY3' in node_name:
                    #elif node_name == "DIODEARRAY3":
                        S_eff[idxx] = D3_seff[idxx]
                    else:
                        print("error, check S_eff")
                else:
                    try:
                        S_eff[idxx] = self.tree.getNode(f'{node_name}.{cnode_name}.V_PEAK').getData().data()
                        
                    except:
                        print(f"Warning: Missing V_PEAK for {node_name}.{cnode_name}")
                        S_eff[idxx] = np.nan
                
                try:
                    dnum[idxx] = self.tree.getNode(f'{node_name}.{cnode_name}.DIODE_NUM').getData().data()
                except:
                    print(f"Warning: Missing DIODE_NUM for {node_name}.{cnode_name}")
                    dnum[idxx] = np.nan
        
                try:
                    axuvData[idxx, :] = self.tree.getNode(f'{node_name}.{cnode_name}.CURRENT').getData().data()
                except:
                    #print(f"Warning: Missing CURRENT for {node_name}.{cnode_name}")
                    axuvData[idxx, :] = self.tree.getNode(f'{node_name}.{cnode_name}.PHOTOCURRENT').getData().data()
        
                try:
                    Rarray[idxx] = self.tree.getNode(f'{node_name}.{cnode_name}.RESISTOR').getData().data()
                except:
                    print(f"Warning: Missing RESISTOR for {node_name}.{cnode_name}")
                    Rarray[idxx] = np.nan
        
                try:
                    Rs[idxx] = self.tree.getNode(f'{node_name}.{cnode_name}.R').getData().data()
                except:
                    #print(f"Warning: Missing R for {node_name}.{cnode_name}")
                    Rs[idxx] = np.nan
    
        # Return all populated arrays
        return axuvData, Rarray, Rs, philist, view_ang, dnum, S_eff, nod
                

    def display_diodes(self):
        """Print details of all diodes stored."""
        print(f"\nTotal Diodes: {self.num_diodes}\n")
        for name, diode in self.diodes.items():
            time_preview = diode.time[:5] if diode.time is not None else "No Data"
            print(f"Diode: {diode.name}, Location: {diode.location}, Time: {time_preview}")
