# -*- coding: utf-8 -*-
"""
This code calculates the measured SEE plasma density for a given shot.

It adds the measured line-integrated density and its uncertainty to the SEE detector dictionary for each detector.

It adds the following 2 keys to the dictionary for each detector-
1. singledetDict['line_integrated_density'] = line-integrated density (m^-2) at the detector, calculated from the shine-thru signal
2. singledetDict['line_integrated_density_sigma'] = uncertainty in the line-integrated density (m^-2) at the detector
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import MDSplus as mds
import scipy as sc
from splines_forward_model_fitting import radial_profile_splines_forward_model

def parseArgs():

    parser = argparse.ArgumentParser(description='post processing script arguments')
    
    # Shot number
    parser.add_argument('-s','--shotnum', metavar = 'shot number', type=int, default=0,
                        help = 'Shot number to post-process')

    # Reference shot number
    parser.add_argument('-r','--ref_shotnum', metavar = 'reference shot number', type=int, default=0,
                        help = 'Reference shot number to calculate the radial density profile')

    # Debug plots
    parser.add_argument('-debug', metavar = 'Debug plots', type=bool, default=False,
                        help = 'Make debug plots and print debug statements')

    # Decimation
    parser.add_argument('-dec', metavar = 'Decimation', type=int, default=2000,
                        help = 'Decimation of the timeArr for data fitting. 2000 is a good value to use when debugging things. 2000 = 2ms')

    # Tries
    parser.add_argument('-tries', metavar='Number of tries', type=int, default=5,
                        help='Number of models to fit for every timestep.')

    # Model error
    parser.add_argument('-modelErrTime', metavar = 'Time of model error', type=float, default=0,
                        help = 'Calculate the model error at a given time.')
    
    args = parser.parse_args()
    
    # If the reference shot number is 0, check if it is in the MDSplus tree for that shot
    if args.ref_shotnum == 0:

        try:
            tree = mds.Tree('wham', args.shotnum)
            ref_shotnum = tree.getNode('diag.shinethru.ref_shotnum').getData().data()
            args.ref_shotnum = ref_shotnum
            tree.close()
        except Exception as e:
            print(e)
    
    return args

def get_shinethru_data(args, shotnum):
    
    # Get the data from the plasma shot
    tree = mds.Tree('wham', shotnum)

    # Beam dump detectors
    signals = []
    timeArr = []
    for i in range(1, 16):

        if i != 6:
            nodeName = 'diag.shinethru.detector_{:02d}'.format(i)

            if args.debug == True:
                print(nodeName)
                
            signals.append(tree.getNode(nodeName).getData().data())
            timeArr = tree.getNode(nodeName).dim_of().data() * 1e3
        else:
            continue

    # In vessel detectors
    for i in range(1, 7):

        nodeName = f'raw.acq1001_633.ch_{i:02d}'
        if args.debug == True:
            print(nodeName)
        
        node = tree.getNode(nodeName)

        tempData = node.getData().data() / 1e2
        tempTime = node.dim_of().data()

        # Filter the data
        tempData = sc.signal.savgol_filter(tempData, 200, 2)

        # Fix the time array
        startTime = tempTime[0]
        tempTime -= startTime
        tempTime *= 1e3
        tempTime += startTime

        # Re-basline the data onto the normal time array
        tempData = np.interp(timeArr, tempTime, tempData, left=0, right=0)

        signals.append(tempData)

    signals = np.array(signals)

    if args.debug == True:

        fig = plt.figure(figsize=(12, 8), tight_layout='True')
        ax = fig.add_subplot(111)

        for i in range(len(signals)):

            ax.plot(timeArr, signals[i])

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Signal [A]')
        ax.set_title('{}'.format(shotnum))

        plt.show()

    # Put the data into the dictionary
        
    return timeArr, signals

def calculate_line_densities(args, detDictList):

    tree = mds.Tree('wham', args.shotnum)
    
    # Get the data from the plasma and reference shot
    timeArr, currentSignals = get_shinethru_data(args, args.shotnum)
    timeArr, refSignals = get_shinethru_data(args, args.ref_shotnum)

    # Remove negative values
    currentSignals[currentSignals <= 0.0] = 1e-6
    refSignals[refSignals <= 0.0] = 1e-6

    # NBI parameters
    nbiVoltageArr = tree.getNode('nbi.v_beam').getData().data()
    timeArr = tree.getNode('nbi.v_beam').dim_of().data() * 1e3

    # Load the shine-thru signal lookup table
    try:
        gasType = tree.getNode('nbi.gas_type').getData().data()
        print(gasType)
    except:
        gasType = 'Hydrogen'

    if gasType == 'Hydrogen':
        lookupTable = np.load('/home/sanwalka/shinethru/lookup_tables/shine_thru_table.npz')
    elif gasType == 'Deuterium':
        lookupTable = np.load('/home/sanwalka/shinethru/lookup_tables/shine_thru_table_d.npz')
    lookupTable = np.load('/home/sanwalka/shinethru/lookup_tables/shine_thru_table_d.npz')

    shineThruTable = lookupTable['shineThruTable']
    lineDensScanArr = lookupTable['lineDensScanArr'] # [m^-2]
    nbiVoltageScanArr = lookupTable['nbiVoltageScanArr'] # [V]
    
    #### Calculate the fraction of the NBI that shines through the plasma (1 = no plasma, 0 = super dense plasma)
    shineThruFracArr = np.abs(currentSignals/refSignals)

    lineIntegratedDensArr = np.zeros_like(currentSignals)
    # Uncertainity in the measurement
    sigmaArr = np.zeros_like(currentSignals)

    # Go over each timestep
    for i in range(len(timeArr)):

        # If the shinethru signal is too small, then the NBI was probably not active and the density calculation can be skipped for this timestep
        if np.max(refSignals[:, i]) < 3e-4:
            continue
            
        # Current time value
        currTime = timeArr[i]
    
        # Get the NBI voltage at this timestep to the nearest kV [V]
        currNBIVoltage = np.round(nbiVoltageArr[i] / 1000) * 1000

        # Get the index of currNBIVoltage in nbiVoltageScanArr
        # The try-except block accounts for when the NBI voltage is below 1kV and sets the index such that the voltage is 1kV for the rest of the loop instance.
        try:
            nbiVoltageInd = np.where(nbiVoltageScanArr == currNBIVoltage)[0][0]
        except:
            nbiVoltageInd = 0
            lineIntegratedDensArr[:, i] = np.zeros(shape=len(lineIntegratedDensArr))
            continue
    
        # Get the shine-thru vs. density for this beam voltage
        currShineThruTable = shineThruTable[:, nbiVoltageInd]
    
        # Go over each shine-thru detector
        for j in range(len(shineThruFracArr)):
            
            # Index where the measured shine-thru matches closest to the lookup table
            lineDensInd = (np.abs(currShineThruTable - shineThruFracArr[j, i])).argmin()
        
            # Add that line integrated density to the array
            lineIntegratedDensArr[j, i] = lineDensScanArr[lineDensInd]

            # Calculate the uncertainty for this measurement
            # 2e-5 and 7e-5 are from looking at the raw data for a normal plasma and reference shot
            # 0.707 since we switch from a 45deg to a 90deg view
            term1 = 0.707 * lineDensScanArr[lineDensInd] / np.log(1/shineThruFracArr[j,i])
            term2 = 2e-5/refSignals[j,i]
            term3 = 7e-5/currentSignals[j,i]
            term4 = (term2**2 + term3**2)**0.5

            sigma = term1 * term4
            if term1 * term4 < 0:
                sigma = 1e20
            
            sigmaArr[j, i] = sigma

    # Divide the density by 2 to go from a 45deg view to a 90deg view
    # It should be sqrt(2) but there is an error in the lookup table somewhere that adds an extra sqrt(2) factor that is missing somewhere.
    lineIntegratedDensArr /= 2

    # Use a low pass butterworth filter to clean up the spikes
    sos = sc.signal.butter(1, 100, btype='lowpass', fs=1/(timeArr[1]-timeArr[0]), output='sos')
    lineIntegratedDensArr = sc.signal.sosfiltfilt(sos, lineIntegratedDensArr)

    if args.debug == True:

        # Plot the line integrated density at each detector
        fig = plt.figure(figsize=(12, 8), tight_layout='True')
        ax = fig.add_subplot(111)

        for i in range(len(lineIntegratedDensArr)):

            # Filter the data to make the plots a bit cleaner
            filteredData = sc.signal.savgol_filter(lineIntegratedDensArr[i], 200, 2)
            ax.plot(timeArr, filteredData, label='Detector {}'.format(i+1))

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
        ax.set_title('{}'.format(args.shotnum))

        plt.show()

        # Plot the raw SEE signals for the reference and plasma shot for each detector
        fig = plt.figure(figsize=(20, 8), tight_layout='True')
        # Reference shot
        ax1 = fig.add_subplot(121)
        # Plasma shot
        ax2 = fig.add_subplot(122)

        for i in range(len(currentSignals)):
            
            ax1.plot(timeArr, refSignals[i], label='Detector {}'.format(i+1))
            ax2.plot(timeArr, currentSignals[i], label='Detector {}'.format(i+1))

        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Signal [A]')
        ax1.set_title('Reference Shot: {}'.format(args.ref_shotnum))

        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Signal [A]')
        ax2.set_title('Plasma Shot: {}'.format(args.shotnum))
        ax2.legend()

        plt.show()
    
    # Add the line integrated densities and their uncertainties to the dictionary
    for i in range(len(lineIntegratedDensArr)):

        # Skip the 6th detector as it is not working
        if i < 5:
            detDictList[i]['line_integrated_density'] = lineIntegratedDensArr[i]
            detDictList[i]['line_integrated_density_sigma'] = sigmaArr[i]
            detDictList[i]['time'] = timeArr
        else:
            detDictList[i+1]['line_integrated_density'] = lineIntegratedDensArr[i]
            detDictList[i+1]['line_integrated_density_sigma'] = sigmaArr[i]
            detDictList[i+1]['time'] = timeArr

    # Put some None values for the 6th detector since it is not working
    detDictList[5]['line_integrated_density'] = None
    detDictList[5]['line_integrated_density_sigma'] = None
    detDictList[5]['time'] = timeArr

    tree.close()

    return timeArr, lineIntegratedDensArr, sigmaArr

def model_error_envelope(args, detDictList):

    # Line integrated data from the detectors
    timeArr, lineIntegratedDensArr, sigmaArr = calculate_line_densities(args, detDictList)

    # Detector vertical positions
    detectorVerticalPositions = []
    for i in range(len(detDictList)):
        if i != 5:
            detectorVerticalPositions.append(detDictList[i]['impact_param_vertical'])
    detectorVerticalPositions = np.array(detectorVerticalPositions) / 1e3 # convert to m
    
    # Radial positions where I want my reconstructed data
    radialPosArr = np.linspace(-0.25, 0.25, 100)
    
    # Index of when we want the centralValErr calculated
    timeIdx = (np.abs(timeArr - args.modelErrTime)).argmin()

    # The line-integrated density and error for the given timestep
    currLineIntegratedDensArr = lineIntegratedDensArr[:, timeIdx]
    currSigmaArr = sigmaArr[:, timeIdx]

    # Remove inf and -inf values from the arrays
    currLineIntegratedDensArr = np.nan_to_num(currLineIntegratedDensArr, nan=0.0, posinf=0.0, neginf=0.0)
    currSigmaArr = np.nan_to_num(currSigmaArr, nan=0.0, posinf=0.0, neginf=0.0)
    # Replace 0 values in currSigmaArr with a large value to avoid issues with the normal distribution
    currSigmaArr[currSigmaArr == 0] = np.max(currSigmaArr)
    
    # Number of iterations for the central value uncertainty analysis
    numIter = 100

    # Dummy time array for the fitting function
    timeArrDummy = np.linspace(0, 1, numIter)

    # Decimation should be 1 for this analysis
    decimation = args.dec
    args.dec = 1

    # Number of tries should be 5 for this analysis
    # Effectively, we are doing numIter*args.tries number of fits.
    numTriesFitting = args.tries
    args.tries = 5

    # Repeat currLineIntegredDensArr numIter times to create an array of shape (len(currLineIntegratedDensArr), numIter)
    normLineIntegratedDensArr = np.tile(currLineIntegratedDensArr, (numIter, 1)).T

    # Sigma array with the same shape as lineIntegratedDensArr
    normSigmaArr = np.tile(currSigmaArr, (numIter, 1)).T

    # Generate 100 radial density profiles
    print(f'Generating {numIter} radial density profiles for model error analysis...')
    
    _, radialFits, lineIntegratedFits = radial_profile_splines_forward_model(timeArrDummy, normLineIntegratedDensArr, normSigmaArr, detectorVerticalPositions, radialPosArr, args)

    # Save the fits in an .npz file
    np.savez(f'/home/sanwalka/shinethru/radial_profile_fits/{args.shotnum}_t={args.modelErrTime}.npz',
             radialFits = radialFits, 
             lineIntegratedFits = lineIntegratedFits, 
             radialPosArr = radialPosArr, 
             currLineIntegratedDensArr = currLineIntegratedDensArr, 
             currSigmaArr = currSigmaArr)

    # Plot the data
    if args.debug == True:

        fig = plt.figure(figsize=(10, 10), tight_layout='True')
        plt.rcParams.update({'font.size' : 22})
        
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        #### Plot the radial profiles
        
        import matplotlib.cm as cm
        cmap = cm.get_cmap('viridis', radialFits.shape[1]).colors

        for i in range(radialFits.shape[1]):
            ax1.plot(radialPosArr, radialFits[:, i], color=cmap[i], linewidth=3)

        #### Plot the line integrated profiles
        
        for i in range(radialFits.shape[1]):
            ax2.plot(radialPosArr, lineIntegratedFits[:, i], color=cmap[i], linewidth=3)

        # SEE data
        ax2.errorbar(detectorVerticalPositions, currLineIntegratedDensArr,
                    yerr=currSigmaArr,
                    fmt='o',
                    capsize=0,
                    color='k',
                    ecolor='k',
                    label='Data')

        ax1.set_xlim(-0.25, 0.25)
        ax2.set_xlim(-0.25, 0.25)
        ax1.set_xlabel('Radius [m]')
        ax2.set_xlabel('Radius [m]')

        ax1.set_ylim(0, None)
        ax2.set_ylim(0, 1.2*np.max(lineIntegratedFits))
        ax1.set_ylabel(r'$n_p$ [m$^{-3}$]')
        ax2.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')

        ax1.set_title('{}; t={}ms'.format(args.shotnum, args.modelErrTime), loc='right')
        ax2.legend(loc=(1.01, 0))

        # plt.savefig(f'/home/sanwalka/shinethru_paper_plots/plots/{args.shotnum}_t={args.modelErrTime}.png',
        #             dpi=600)

        plt.show()
    
    # Return decimation and tries to their original values
    args.dec = decimation
    args.tries = numTriesFitting
    
    return radialFits, lineIntegratedFits, radialPosArr

if __name__ == '__main__':

    # Load the SEE detector dictionary
    with open('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl', 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    # Calculate the line integrated densities at each detector
    args = parseArgs()
    timeArr, lineIntegratedDensArr, sigmaArr = calculate_line_densities(args, detDictList)