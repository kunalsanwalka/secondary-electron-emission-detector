#
# This processes the axuv data and pushes it to MDSplus

import numpy as np
#import scipy as sc
import MDSplus as mds
import os
import shutil
import logging
import argparse
import time
import abel
from scipy.signal import savgol_filter as savgol

# Path for the MDSplus model trees
#MDSplus_path = '/home/dtacqAdmin/mdsplus_trees/'

## Directory where all the data files get stored
#pulseDirectory = '/home/dtacqAdmin/data/'
#pulseDirectory = '/mnt/n/data2/'

treeName = 'axuv'

# axuv digitizer 639, 64 channel, 1 MS/s 
digIP = '192.168.113.164'  # CHECK
digName = 'acq1001_639'

logName = 'postproc_axuv'
#logDir = '/home/dtacqAdmin/procScripts/logs/'
logDir = '/home/whamdata/post_processing/logs/'
logPath = logDir+logName+'.log'

#statusDir = '/home/dtacqAdmin/armScripts/status/'
#statusPath = statusDir + 'acq1001_639.log'

# Current settings to store in MDSplus


######################################################

def runningMean(x,N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

######################################################

def initialize_logger(path):

# Create a logger for this file.
    logger = logging.getLogger(logName)
    logger.setLevel(logging.DEBUG)
# Capture other warnings
    logging.captureWarnings(True)
# Create a file handler to write to.
    #file_handler = logging.FileHandler("/home/dtacqAdmin/procScripts/logs/postproc_shinethru.log", "w")
    file_handler = logging.FileHandler(path, "w")

    file_handler.setLevel(logging.DEBUG)
# Create a formatter to make the logging look nice.
    formatter = logging.Formatter('%(asctime)s, %(name)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
# Add the file handler to the logger.
    logger.addHandler(file_handler)

    logger.info('Logger file created')

    return logger

######################################################

def parseArgs(logger):
#    """
#    This function gets the shot number if specified and then runs the post processing script.
#    """
    
    logger.info('parseArgs')

    # Create a parser object that allows us to pass multiple arguments of various types into run_single_shot with 1 object
    parser = argparse.ArgumentParser(description='post processing script arguments')
    # Shot number
    parser.add_argument('-s','--shotnum', metavar = 'shot number', type=int, default=0,
                        help = 'Shot number to post-process. If not specified, latest shot is post-processed by getting shot number from andrew')

    parser.add_argument('-p','--plot', metavar = 'plot boolean', type=int, default=0,
            help = 'If 1, will show plots. Default is 0')
    parser.add_argument('-w','--write', metavar = 'write boolean', type=int, default=1,
            help = 'If 1, will save data.. Default is 1')

    try:
        args = parser.parse_args()
        logger.info('parser.parse_args succeeded?')
    except Exception as e:
        logger.error(e)

    logger.info(vars(args))
        
    logger.info('Arguments parsed.')
    
    return args

######################################################

def check_status(status_path,logger):
    # This function checks the status of the mdsplus data files and waits until they've been written to proceed
    for line in open(status_path):
        pass
    status = line.strip()

    idx = 0
    while status != 'IDLE':
        time.sleep(1); idx += 1
        logger.info('dtacq status = '+status)

        if idx >= 30:
            logger.info('Waited {:1.0f} seconds. Terminating script.'.format(idx))
            exit()
            
######################################################

def run_post_process(logger,args):

    logger.info('run_post_process')

    if args.plot:
        import matplotlib.pyplot as plt


    shotnum = args.shotnum
    print('shotnum = ',+shotnum)
    logger.info('shotnum = {}'.format(shotnum))

    if shotnum == 0:
        logger.info('shotnum = 0. Connecting to andrew mdsplus to get current shot')
        # Open a connection to andrew
        MDS_SERVER = 'andrew.psl.wisc.edu'
        conn = mds.connection.Connection(MDS_SERVER)
        # Open the WHAM tree, get the current shot number and close the tree
        conn.openTree('wham', 0)
        shotnum = int(conn.get('$shot'))
        conn.closeTree('wham', 0)

    shotString = str(shotnum)
    year = shotString[:2] + '/'
    month = shotString[2:4] + '/'
    day = shotString[4:6] + '/'

    ### Open the raw data tree
    logger.info('Getting raw tree data')
    rawTree = mds.Tree('acq1001_639', shotnum)
    # Processed data tree 
    procTree = mds.Tree('axuv',shotnum,'edit')

# \WHAM::TOP.RAW.AC1001_631:CH_01
    #sig1a = Tree.getNode('RAW.ACQ1001_631:CH_01').getData().data()

# Resistor measurements on Dsub connections, Measured 24/08/07 DE
#    0, 219.1, 219.2, 219.4, 219.2, 219.6, 219.0, 219.2, 218.9, 220.9, 220.6, 0, 0 # Row 1 on Dsub, pins 1-13
#    0, 219.2, 218.8, 219.1, 218.9, 219.2, 218.8, 220.0, 220.4, 220.8, 0 # Row 2 on Dsub, pins 14-25
    
# resistor array sorts z0_p10 -> z0_p1
#                      z0_n01 -> z0_n10
    
# End of day on 08/07, the cables were swapped such that cable AAB went to channels 1-32, and cable BCC went to 33-64
# Before 08/08, the situation was reversed. 
    
# Expect that ch_31+32, 63+64 are empty
# Ch_33 goes to z0_n01
# Ch_42 goes to z0_n10
# Ch_21 goes to z0_p01
# Ch_30 goes to z0_p10

#   z0_n01  -> z0_n10
#   z0_p01 -> z0_p10
    dtacqToDiodeMap = {}
    if (shotnum >= 240808000 ): # Correct orientation
        dtacqToDiodeMap={'CH_42':'CH_20',   'CH_41':'CH_19',    'CH_40':'CH_18',    'CH_39':'CH_17',    'CH_38':'CH_16',
                         'CH_37':'CH_15',   'CH_36':'CH_14',    'CH_35':'CH_13',    'CH_34':'CH_12',    'CH_33':'CH_11',
                         'CH_21':'CH_10',   'CH_22':'CH_09',    'CH_23':'CH_08',    'CH_24':'CH_07',    'CH_25':'CH_06',
                         'CH_26':'CH_05',   'CH_27':'CH_04',    'CH_28':'CH_03',    'CH_29':'CH_02',    'CH_30':'CH_01' }
        R_array = np.array([219.1,219.2,219.2,218.8,219.4,219.1,219.2,218.9,219.6,219.2, # Sorted resistor array, 
                            219.2,219.0,218.8,219.2,220.0,218.9,220.4,220.9,220.8,220.6])*1e3 # pins 

    elif shotnum <240808000: # Cables swapped!
        dtacqToDiodeMap={'CH_10':'CH_01',   'CH_09':'CH_02',    'CH_08':'CH_03',    'CH_07':'CH_04',    'CH_06':'CH_05',
                         'CH_05':'CH_06',   'CH_04':'CH_07',    'CH_03':'CH_08',    'CH_02':'CH_09',    'CH_01':'CH_10',
                         'CH_53':'CH_11',   'CH_54':'CH_12',    'CH_55':'CH_13',    'CH_56':'CH_14',    'CH_57':'CH_15',
                         'CH_58':'CH_16',   'CH_59':'CH_17',    'CH_60':'CH_18',    'CH_61':'CH_19',    'CH_62':'CH_20' }
        R_array = np.array([10.,  10., 10., 10., 10., 10., 10., 10., 10., 10.,   
                            220.,220.,220.,220.,220.,220.,220.,220.,220.,220.]) * 1e3
        # I'm still not certain that the 10k gain isn't a better setting, but probably not. 

    
    dtacqToDiodeMap2={'CH_10':'CH_01',   'CH_09':'CH_02',    'CH_08':'CH_03',    'CH_07':'CH_04',    'CH_06':'CH_05',
                      'CH_05':'CH_06',   'CH_04':'CH_07',    'CH_03':'CH_08',    'CH_02':'CH_09',    'CH_01':'CH_10',
                      'CH_53':'CH_11',   'CH_54':'CH_12',    'CH_55':'CH_13',    'CH_56':'CH_14',    'CH_57':'CH_15',
                      'CH_58':'CH_16',   'CH_59':'CH_17',    'CH_60':'CH_18',    'CH_61':'CH_19',    'CH_62':'CH_20' }
    R_array2 = np.array([219.1,219.2,219.2,218.8,219.4,219.1,219.2,218.9,219.6,219.2, # Sorted resistor array, 
                         219.2,219.0,218.8,219.2,220.0,218.9,220.4,220.9,220.8,220.6])*1e3 # pins 

    dtacqToDiodeMap3={'CH_20':'CH_01',   'CH_19':'CH_02',    'CH_18':'CH_03',    'CH_17':'CH_04',    'CH_16':'CH_05',
                      'CH_15':'CH_06',   'CH_14':'CH_07',    'CH_13':'CH_08',    'CH_12':'CH_09',    'CH_11':'CH_10',
                      'CH_43':'CH_11',   'CH_44':'CH_12',    'CH_45':'CH_13',    'CH_46':'CH_14',    'CH_47':'CH_15',
                      'CH_48':'CH_16',   'CH_49':'CH_17',    'CH_50':'CH_18',    'CH_51':'CH_19',    'CH_52':'CH_20' }
    R_array3 = np.array([219.1,219.2,219.2,218.8,219.4,219.1,219.2,218.9,219.6,219.2, # Sorted resistor array, 
                         219.2,219.0,218.8,219.2,220.0,218.9,220.4,220.9,220.8,220.6])*1e3 # pins 

    diodeToDtacqMap  = {v : k for k, v in dtacqToDiodeMap.items()} # invert this mapping
    diodeToDtacqMap2 = {v : k for k, v in dtacqToDiodeMap2.items()} # invert this mapping
    diodeToDtacqMap3 = {v : k for k, v in dtacqToDiodeMap3.items()} # invert this mapping


    # Current configuration values 
    Rwall = 0.362 
    Rflange = 15.94*0.0254
#    Rflange - 0.047 # Radial distance in meters
    Phi = -120.*np.pi/180 # rad, phi location of slit 
    D = 0.037 # distance between slot and diode face
# DE 24/08/15 I think these values are more accurate. They agree better with visual inspection inside the chamber. Measure these upon removal!!
    Rslit = 0.380 # From solidwork, distance from machine axis to slit
    Zslit = 0.000 # axial location of slit
    D = 0.025 # From solidworks, Distance between slit and diode face. 
    d = 0.00095 # distance between diode centers
    ds = d/2 + d*np.arange(-10,10); #print('ds = ',ds) # Array of distances between each diode center and the slit-axis line. 
    hypotenii = np.sqrt(D**2 + ds**2) # 
    phis = np.arctan2(ds,D) # Difference between Phi and each diode line of sight
    # On 24/09/14, DE installed the flanges backwards. Now the diodes are pointing in the opposite way as before. To maintain consistency, flip. FIX this on the next vent. 
    if shotnum>240914000: phis = np.flip(phis)

    Rplasm = 0.16 # port this number from MDSplus for future analysis scripts

    # slit location
    XYslit = np.array([Rslit*np.cos(Phi),Rslit*np.sin(Phi)])
    # individual diode locations
    XYs = np.array([Rslit*np.cos(Phi)+hypotenii*np.cos(Phi+phis),Rslit*np.sin(Phi)+hypotenii*np.sin(Phi+phis)])
    # Phis, individual diode locations in RPhiZ cylindrical coordinates
    Phis = np.arctan2(XYs[1,:],XYs[0,:])
    # Radial location of individual diodes
    Rs = np.sqrt(XYs[0,:]**2 + XYs[1,:]**2) 
    # Axial location of individual diodes
    Zs = np.zeros_like(Phis)

    # New slopes of each chord
    ms = (XYslit[1] - XYs[1,:]) / (XYslit[0] - XYs[0,:])
    bs = np.zeros_like(ms) # impact parameters
    # Plot all chords
    for idx in range(len(ms)):
               # Find impact parameters b
        rs = np.sqrt((XYslit[0]+np.linspace(0,1,1001)*np.cos(np.arctan2(ms[idx],1)))**2 + (XYslit[1]+np.linspace(0,1,1001)*np.sin(np.arctan2(ms[idx],1)))**2)
        bs[idx] = np.min(rs)
    bs *= np.sign(ds) 

    if args.plot: # Plot of chord geometry
        fig,ax = plt.subplots(figsize=(6,8))
        ax.set_aspect('equal')
        ax.scatter([0,Rslit*np.cos(Phi)],[0,Rslit*np.sin(Phi)])
        ax.plot([0,Rslit*np.cos(Phi)],[0,Rslit*np.sin(Phi)],'k--')
        ax.plot(Rwall*np.cos(np.linspace(0,2*np.pi,1000)),Rwall*np.sin(np.linspace(0,2*np.pi,1000))) # plots vessel wall
        ax.scatter(XYs[0,:],XYs[1,:]) #
        for idx in range(len(ms)):     
            xs = np.linspace(XYslit[0],XYslit[0]+np.cos(np.arctan2(ms[idx],1)),100)
            ys = np.linspace(XYslit[1],XYslit[1]+np.sin(np.arctan2(ms[idx],1)),100)
            ax.plot(xs,ys,'k--')
            iidx = np.argwhere(np.sqrt(xs**2+ys**2)<Rplasma)
            ax.plot(xs[iidx],ys[iidx],lw=4,label='b={:1.3f}m'.format(bs[idx]))
        # Plot circular shells
#        for idx in range(15):
#            ax.plot(0.01*idx*np.cos(np.linspace(0,2*np.pi,1000)),.01*idx*np.sin(np.linspace(0,2*np.pi,1000)),'k--',lw=1)
        ax.set(xlabel='X [m]',ylabel='Y [m]',title='AXUV orientation')
        
        ax.legend()
        plt.tight_layout()
        plt.show()
   
    # Load parameters from MDSplus
    nsamp = rawTree.getNode('numsamples').getData().data()


    # Create time array from dim_of
    time = rawTree.getNode('CH_01').getData().dim_of().data()

#    if args.plot: # Raw data line plot
#        fig,ax = plt.subplots()
#        for i in range(len(axuv_raw[:,0])):
#            ax.plot(time,axuv_raw[i,:]+i*.01,label='CH_{:02d}'.format(i))
#        ax.set(xlabel='time [s]',ylabel='Voltage [V]',title='Shot {:1.0f}\nAXUV Uncalibrated Signal Amplitude'.format(shotnum))
#        plt.show()
#        plt.close()

#    if args.plot: # Processed data contour plot
#        fig,ax = plt.subplots()
##        im = ax.contourf(time,bs[:10],axuv_proc[:10,:],cmap='inferno')#,vmin=,vmax=)
##        im1 = ax.contourf(time,bs[10:],axuv_proc[10:,:],cmap='inferno')#,vmin=,vmax=)
#        im = ax.contourf(time,bs,axuv_proc,cmap='inferno')#,vmin=,vmax=)
      
#        plt.colorbar(im,label='photocurrent [A]')
#        ax.set(xlabel='time [s]',ylabel='Impact parameter b [m]',title='Shot {:1.0f}\nAXUV Calibrated Signal Amplitude'.format(shotnum))
#        plt.show()
#        plt.close()


 
    freq = rawTree.getNode('frequency').getData().data()
    trig_time = rawTree.getNode('trig_time').getData().data()
    delay = rawTree.getNode('delay').getData().data()
    dec = rawTree.getNode('decimation').getData().data()
    
    logger.info('freq = {:1.0f}. delay = {:1.1f}. trig_time = {:1.0f}. dec = {:1.0f}. nsamples = {:1.0f}'.format(freq,delay,trig_time+delay,dec,nsamp))

    # Time basis for signals
    # $1 --> NSAM | $2 --> TIME0 | $3 --> FREQ
    # This codeblock allows the signals to be displayed w.r.t. time vs. w.r.t. sample number
    win = "BUILD_WINDOW(0,$1-1,BUILD_WITH_UNITS($2,'S'))"
    axis = "BUILD_WITH_UNITS(BUILD_RANGE($2,$2+(1/d_float($3))*($1-1),1/d_float($3)),'S')"
    tbase = "BUILD_DIM("+win+","+axis+")"

    # Put base parameters
    nodename_list = ['DIODEARRAY1','DIODEARRAY2','DIODEARRAY3']
    for n in nodename_list:

        axuv_raw = np.zeros((20,nsamp))
    # import raw data from MDSplus 
        for i in range(len(axuv_raw[:,0])):
            axuv_raw[i,:] = -rawTree.getNode(diodeToDtacqMap['{}_CH_{:02d}'.format(n,i+1)]).getData().data()
    # Crude filter out highest frequency noise
        #wndw=3 ; ordr=1
        #axuv_proc = savgol(axuv_raw / R_array.reshape(20,1),wndw,ordr,axis=1)

        nodename = nodename_list[n]
        try: 
            procTree.getNode(nodename+'.PHI_SLIT').putData(Phi)
            procTree.getNode(nodename+'.D_SLIT_DIOD').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')", D))
            procTree.getNode(nodename+'.R_SLIT').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')", Rslit))
            procTree.getNode(nodename+'.Z_SLIT').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')", Zslit))
        except Exception as e:
            logger.warning(e)

        for idx in range(20):
            CH_Name = nodename+'.CH_{:02d}'.format(idx+1)
            logger.info(CH_Name)
            try:
                procTree.getNode(CH_Name+'.PHI').putData(Phis[idx])
                procTree.getNode(CH_Name+'.R').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')",Rs[idx]))
                procTree.getNode(CH_Name+'.Z').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')",Zs[idx]))
                procTree.getNode(CH_Name+'.RESISTOR').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'Ohm')",R_array[idx]))
                procTree.getNode(CH_Name+'.B_IMPACT').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')",bs[idx]))
                procTree.getNode(CH_Name+'.CURRENT').putData(mds.Data.compile('BUILD_SIGNAL($VALUE,$4,'+tbase+')',
                                 len(axuv_raw[idx,:]), trig_time+delay, freq/dec, axuv_raw[idx,:]))
            except Exception as e:
                logger.warning(e)
#    for idx in range(20):
#        nodeName = 'DIODEARRAY2.CH_{:02d}'.format(idx+1)
#        logger.info(nodeName)
#        try:
#            procTree.getNode(nodeName+'.PHI').putData(Phis[idx])
#            procTree.getNode(nodeName+'.R').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')",Rs[idx]))
#            procTree.getNode(nodeName+'.Z').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')",Zs[idx]))
#            procTree.getNode(nodeName+'.RESISTOR').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'Ohm')",R_array[idx]))
#            procTree.getNode(nodeName+'.B_IMPACT').putData(mds.Data.compile("BUILD_WITH_UNITS($1,'m')",bs[idx]))
#            procTree.getNode(nodeName+'.CURRENT').putData(mds.Data.compile('BUILD_SIGNAL($VALUE,$4,'+tbase+')',
#                            len(axuv_proc[idx,:]), trig_time+delay, freq/dec, axuv_proc[idx,:]))
#        except Exception as e:
#            logger.warning(e)



    logger.info('Finished putting data')

## Move MDSplus files to /mnt/n/data2/
#    # Filenames of the pulse files that were just made
#    charFile = treeName + '_' + shotString + '.characteristics'
#    dataFile = treeName + '_' + shotString + '.datafile'
#    treeFile = treeName + '_' + shotString + '.tree'
#
#    shotString = str(shotnum)
#    yr = shotString[:2] + '/'
#    mo = shotString[2:4] + '/'
#    dy = shotString[4:6] + '/'

#    try:
#        mdsDataPath = os.path.join(pulseDirectory,yr,mo,dy,treeName)
#        os.makedirs(mdsDataPath)
#    except FileExistsError as file_exists:
#        logger.debug('%s already exists.' %mdsDataPath)
#
#
#
#    try:
#    #    os.rename(mdsModelPath + treeName + '/' + charFile, mdsDataPath + '/' + charFile)
#    #    os.rename(mdsModelPath + treeName + '/' + dataFile, mdsDataPath + '/' + dataFile)
#    #    os.rename(mdsModelPath + treeName + '/' + treeFile, mdsDataPath + '/' + treeFile)
#        shutil.move(MDSplus_path + treeName + '/' + charFile, mdsDataPath + '/' + charFile) # works with /mnt/ directory
#        shutil.move(MDSplus_path + treeName + '/' + dataFile, mdsDataPath + '/' + dataFile)
#        shutil.move(MDSplus_path + treeName + '/' + treeFile, mdsDataPath + '/' + treeFile)
#        logger.debug('Moved the pulse files from model tree path to the data path')
#    except Exception as e:
#        logger.exception(e)



# FLUX LOOP EXAMPLE GRAMMAR
#    flSig1 = mds.Data.compile('BUILD_SIGNAL($VALUE,$4,'+tbase+')', len(fl1) , t0, freq / dec, fl1)
#    try:
#        flTree.getNode('FL1').putData(flSig1)
#    except:
#        logger.exception('{}: MDSplus write error on FL1')

#    try:
#        frequency_node = tree.getNode('frequency')
#        frequency_node.putData(mds.Data.compile("BUILD_WITH_UNITS($1,'Hz')", args.frq*1e6))# /args.dec)
#    except Exception as e:
#        logname.exception(e)











# execute main script

if __name__ == '__main__':

    logger = initialize_logger(logPath)

    args = parseArgs(logger)

    #    check_status(statusPath,logger)
    #Attempt to run the post process script. If fails, log error and try again (max 3 attempts)
    for i in range(3):
        try:
            run_post_process(logger,args)
            break
        except Exception as e:
            logger.info("Error running script:\n {}".format(e))
            print("Error running script: {}".format(e))

