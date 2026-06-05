# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import abel
from astropy import modeling
from astropy.modeling import Fittable1DModel, Parameter, models, fitting
from astropy.modeling.models import custom_model
from forward_model_fitting import radial_profile_forward_model
from kai_model_fitting import radial_profile_kai_model
from splines_model_fitting import radial_profile_splines_model
from splines_forward_model_fitting import radial_profile_splines_forward_model
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size':22})

def parseArgs():

    parser = argparse.ArgumentParser(description='post processing script arguments')
    
    # Shot number
    parser.add_argument('-s','--shotnum', metavar = 'shot number', type=int, default=0,
                        help = 'Shot number to post-process')

    # Reference shot number
    parser.add_argument('-r','--ref_shotnum', metavar = 'reference shot number', type=int, default=0,
                        help = 'Reference shot number to calculate the radial density profile')

    # Debug plots
    parser.add_argument('-debug', metavar = 'Debug plots', type=str, default='False',
                        help = 'Make debug plots and print debug statements')

    # Decimation
    parser.add_argument('-dec', metavar = 'Decimation', type=int, default=2000,
                        help = 'Decimation of the timeArr for data fitting. 2000 is a good value to use when debugging things.')

    # Tries
    parser.add_argument('-tries', metavar='Number of tries', type=int, default=5,
                        help='Number of models to fit for every timestep.')

    # Model error
    parser.add_argument('-modelErrTime', metavar = 'Time of model error', type=float, default=0,
                        help = 'Calculate the model error at a given time.')

    # Edge detectors
    parser.add_argument('-edge', metavar = 'Use edge detectors', type=str, default='True',
                        help = 'Add the edge detectors for the density reconstruction. True- Use edge detectors, False- Do not use edge detectors.')
    
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

    print(args.shotnum)
    print(args.ref_shotnum)
    
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
    if args.edge == True:
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
        
    return timeArr, signals

def calculate_line_densities(args):

    tree = mds.Tree('wham', args.shotnum)
    
    # Get the data from the plasma and reference shot
    timeArr, currentSignals = get_shinethru_data(args, args.shotnum)
    timeArr, refSignals = get_shinethru_data(args, args.ref_shotnum)

    # Remove negative values
    currentSignals[currentSignals <= 0.0] = 1e-6
    refSignals[refSignals <= 0.0] = 1e-6

    # NBI parameters
    nbiVoltageArr = tree.getNode('nbi.v_beam').getData().data()
    nbiCurrentArr = tree.getNode('nbi.i_beam').getData().data()
    timeArr = tree.getNode('nbi.v_beam').dim_of().data() * 1e3

    # Load the shine-thru signal lookup table
    try:
        gasType = tree.getNode('nbi.gas_type').getData().data()
        print(gasType)
    except:
        gasType = 'Hydrogen'

    if gasType == 'Hydrogen':
        lookupTable = np.load('/home/sanwalka/shinethru_paper_plots/shine_thru_table.npz')
    elif gasType == 'Deuterium':
        lookupTable = np.load('/home/sanwalka/shinethru_paper_plots/shine_thru_table_d.npz')
    # lookupTable = np.load('/home/sanwalka/shinethru_paper_plots/shine_thru_table_d.npz')

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

    # Use a Savitsky-Golay filter to clean up the noise in the data
    # lineIntegratedDensArr = sc.signal.savgol_filter(lineIntegratedDensArr,
    #                                                 window_length=20,
    #                                                 polyorder=2,
    #                                                 mode='constant',
    #                                                 axis=1)

    # Use a low pass butterworth filter to clean up the spikes
    sos = sc.signal.butter(1, 100, btype='lowpass', fs=1/(timeArr[1]-timeArr[0]), output='sos')
    lineIntegratedDensArr = sc.signal.sosfiltfilt(sos, lineIntegratedDensArr)

    if args.debug == True:

        fig = plt.figure(figsize=(12, 8), tight_layout='True')
        ax = fig.add_subplot(111)

        for i in range(len(lineIntegratedDensArr)):

            ax.plot(timeArr, lineIntegratedDensArr[i])

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
        ax.set_title('{}'.format(args.shotnum))

        plt.show()
    
    return timeArr, lineIntegratedDensArr, sigmaArr

def impact_params(args):

    # Vertical positions of the 15 SEE detectors
    detectorVerticalPositions = np.array([2.75, 2.34, 1.7, 1.35, 0.79, 0, -0.71, -1.07, -1.66, -1.98, -2.55, -0.41, -0.13, 0.22, 0.42]) * 2.54 / 1e2 # Convert in to m
    # Remove the 6th detector that is now a calorimeter
    detectorVerticalPositions = np.delete(detectorVerticalPositions, (5))

    # In vessel detectors
    inVesselPositions = np.array([13.55, 12.18, 10.5, -10.5, -12.18, -13.55]) / 1e2 # Convert cm to m

    if args.edge == True:
        detectorVerticalPositions = np.concatenate((detectorVerticalPositions, inVesselPositions))
    
    return detectorVerticalPositions

def model_error(fittedTimeArr, timeArr, detectorVerticalPositions, radialPosArr, lineIntegratedDensArr, lineIntegratedRadialProfile, makeplot=False):
    """
    This function calculates the chi2 error between the measured line integrated density from the shinethru
    detectors vs the reconstructed profile.
    """

    # Array to store the error
    errorArr = np.zeros_like(fittedTimeArr)

    totError = 0
    numPoints = 0
    
    for i in range(len(fittedTimeArr)):

        currTime = fittedTimeArr[i]
        
        # Time index of the raw data
        timeIndexRaw = (np.abs(timeArr - currTime)).argmin()

        # Interpolation function of the fitted data
        interpFunc = sc.interpolate.CubicSpline(radialPosArr, lineIntegratedRadialProfile[:, i])

        # Fitted data at the same points as the shinethru data
        comparisonData = interpFunc(detectorVerticalPositions)

        # Shinethru data
        shinethruData = lineIntegratedDensArr[:, timeIndexRaw]
        
        # Calculate the chi2 error
        chi2 = np.sum(((comparisonData - shinethruData)/shinethruData)**2)

        # Check if there was a plasma at all
        if np.max(comparisonData) > 0:
            errorArr[i] = chi2
            numPoints += 1
        else:
            errorArr[i] = 0

    totError = np.sum(errorArr)
    avgError = totError/numPoints
        
    if makeplot == True:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(fittedTimeArr, errorArr)

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'$\chi^2$ error')

        plt.show()

    return errorArr, avgError

def plot_shinethru_vs_interferometer(args):

    #### Compare interferometer to shinethru

    timeArr, lineIntegratedDensArr, _ = calculate_line_densities(args)
    
    fig = plt.figure(figsize=(12, 8), tight_layout='True')

    # Plot the NBI and ECH power
    ax = fig.add_subplot(211)
    
    tree = mds.Tree('wham', args.shotnum)
    # ECH Power
    echPower = tree.getNode('ech.ech_proc.wg_monitor_f.filtered').getData().data()
    echTime = tree.getNode('ech.ech_proc.wg_monitor_f.filtered').dim_of().data() * 1e3 # s to ms
    nbiPower = tree.getNode('nbi.v_beam').getData().data() * tree.getNode('nbi.i_beam').getData().data() / 1e3 # W to kW
    nbiTime = tree.getNode('nbi.time_slow').getData().data() * 1e3 # s to ms

    ax.plot(nbiTime, nbiPower, color='red', linewidth=3, label='NBI')
    ax.plot(echTime, echPower, color='blue', linewidth=3, label='ECH')

    ax.legend(loc=1)
    ax.set_ylabel('Power [kW]')
    ax.set_title('{}'.format(args.shotnum), loc='right')
    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 1200)
    ax.set_xticks([])
    ax.set_ylim(0, None)
    
    ax = fig.add_subplot(212)

    # Plot the innermost chord of the shinethru
    ax.plot(timeArr, lineIntegratedDensArr[5], linewidth=3, label='Shinethrough', zorder=10)

    # Plot the interferometer data for this shot
    interferometerData = tree.getNode('diag.interferomtr.linedens').getData().data()
    interferometerTime = tree.getNode('diag.interferomtr.time').getData().data() * 1e3 # s to ms
    ax.plot(interferometerTime, interferometerData, linewidth=3, label='Interferometer')

    ax.legend(loc=1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 2.5e19)
    
    plt.savefig('/home/sanwalka/shinethru_paper_plots/plots/inf_vs_shinethru_{}.png'.format(args.shotnum), dpi=600)

    plt.show()
    
    return

def model_error_envelope(args):

    # Line integrated data from the detectors
    timeArr, lineIntegratedDensArr, sigmaArr = calculate_line_densities(args)

    # Detector vertical positions
    detectorVerticalPositions = impact_params(args)
    
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

    print(currLineIntegratedDensArr)
    print(currSigmaArr)
    
    # Number of iterations for the central value uncertainty analysis
    numIter = 100

    # Dummy time array for the fitting function
    timeArrDummy = np.linspace(0, 1, numIter)

    # Decimation should be 1 for this analysis
    decimation = args.dec
    args.dec = 1

    # Number of tries should be 5 for this analysis
    # Effectively, we are doing numIter*args.tries number of fits. i.e. 25 fits
    numTriesFitting = args.tries
    args.tries = 10

    # # Generate the line integrated density array with the values changing based on currSigmaArr
    # normLineIntegratedDensArr = np.random.normal(loc = currLineIntegratedDensArr[:, None],
    #                                              scale = currSigmaArr[:, None],
    #                                              size=(len(currLineIntegratedDensArr), numIter))

    # Repeat currLineIntegredDensArr numIter times to create an array of shape (len(currLineIntegratedDensArr), numIter)
    normLineIntegratedDensArr = np.tile(currLineIntegratedDensArr, (numIter, 1)).T

    # Sigma array with the same shape as lineIntegratedDensArr
    normSigmaArr = np.tile(currSigmaArr, (numIter, 1)).T

    # Generate 100 radial density profiles
    _, radialFits, lineIntegratedFits = radial_profile_splines_forward_model(timeArrDummy, normLineIntegratedDensArr, normSigmaArr, detectorVerticalPositions, radialPosArr, args)

    print(np.shape(radialFits))

    # Save the fits in an .npz file
    np.savez(f'/home/sanwalka/shinethru_paper_plots/data/{args.shotnum}_t={args.modelErrTime}_model_error_envelope.npz',
             radialFits = radialFits, 
             lineIntegratedFits = lineIntegratedFits, 
             radialPosArr = radialPosArr, 
             currLineIntegratedDensArr = currLineIntegratedDensArr, 
             currSigmaArr = currSigmaArr)

    # Calculate the standard deviations
    radialStdDev = np.std(radialFits, axis=1)
    lineIntStdDev = np.std(lineIntegratedFits, axis=1)

    # Calculate the minimum and maximum values
    radialMin = np.min(radialFits, axis=1)
    radialMax = np.max(radialFits, axis=1)
    lineIntMax = np.min(lineIntegratedFits, axis=1)
    lineIntMin = np.max(lineIntegratedFits, axis=1)

    # Calculate the average values
    radialAvg = np.mean(radialFits, axis=1)
    lineIntAvg = np.mean(lineIntegratedFits, axis=1)

    # Plot the data
    fig = plt.figure(figsize=(10, 10), tight_layout='True')
    plt.rcParams.update({'font.size' : 22})
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    #### Plot the radial profiles
    
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis', radialFits.shape[1]).colors

    for i in range(radialFits.shape[1]):
        ax1.plot(radialPosArr, radialFits[:, i], color=cmap[i], linewidth=3)

    # # Average value
    # ax1.plot(radialPosArr, radialAvg,
    #          linewidth=3, color='red', zorder=10, label='Average Fit')
    # # Minimum value
    # ax1.plot(radialPosArr, radialMin,
    #          linestyle='dashed', color='blue')
    # # Maximum value
    # ax1.plot(radialPosArr, radialMax,
    #          linestyle='dashed', color='blue')
    # # Fill between these values
    # ax1.fill_between(radialPosArr, radialMin, radialMax,
    #                  color='blue', alpha=0.2, label='Fit Envelope')
    # ax1.fill_between(radialPosArr, radialAvg+radialStdDev, radialAvg-radialStdDev,
    #                  color='blue', alpha=0.2, label='Fit Envelope')

    #### Plot the line integrated profiles
    
    for i in range(radialFits.shape[1]):
        ax2.plot(radialPosArr, lineIntegratedFits[:, i], color=cmap[i], linewidth=3)

    # # Average value
    # ax2.plot(radialPosArr, lineIntAvg,
    #          linewidth=3, color='red', label='Average Fit')
    # # Minimum value
    # ax2.plot(radialPosArr, lineIntMin,
    #          linestyle='dashed', color='blue')
    # # Maximum value
    # ax2.plot(radialPosArr, lineIntMax,
    #          linestyle='dashed', color='blue')
    # # Fill between these values
    # ax2.fill_between(radialPosArr, lineIntMin, lineIntMax,
    #                  color='blue', alpha=0.2, label='Fit Envelope')

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

    plt.savefig(f'/home/sanwalka/shinethru_paper_plots/plots/{args.shotnum}_t={args.modelErrTime}.png',
                dpi=600)

    plt.show()
    
    # Return decimation and tries to their original values
    args.dec = decimation
    args.tries = numTriesFitting
    
    return

def plot_model_reconstruction(args, model_type):

    # Line integrated data from the detectors
    timeArr, lineIntegratedDensArr, sigmaArr = calculate_line_densities(args)

    # # Density normalization
    # normArr = np.max(lineIntegratedDensArr, axis=0)
    # lineIntegratedDensArr /= normArr

    # Detector vertical positions
    detectorVerticalPositions = impact_params(args)

    print(np.shape(lineIntegratedDensArr))
    print(np.shape(detectorVerticalPositions))
    
    # Timepoints where I want to plot the radial profile
    timePointsPlot = np.array([3, 5, 7, 9, 11]) # ms
    timePointsPlot = np.array([7, 9, 11])

    # Radial positions where I want my reconstructed data
    radialPosArr = np.linspace(-0.25, 0.25, 100)
    
    # Data from the reconstruction
    fittedTimeArr, radialDensProfile, lineIntegratedRadialProfile = model_type(timeArr, lineIntegratedDensArr, sigmaArr, detectorVerticalPositions, radialPosArr, args)

    # Undo normalization
    # lineIntegratedRadialProfile *= normArr
    # lineIntegratedDensArr *= normArr
    # radialDensProfile *= normArr

    # Clean the data a bit
    lineIntegratedRadialProfile = np.nan_to_num(lineIntegratedRadialProfile)

    # print(np.shape(lineIntegratedRadialProfile))
    # print(np.shape(radialPosArr))
    
    # Least squares error of the model vs the data
    # errorArr has the same shape as fittedTimeArr
    errorArr, avgError = model_error(fittedTimeArr, timeArr, detectorVerticalPositions, radialPosArr, lineIntegratedDensArr, lineIntegratedRadialProfile, makeplot=True)
    
    modelTypeStr = model_type.__name__.split('_')[2]
    # print(modelTypeStr)
    # print(avgError)

    #### Compare the radial profile to data
    fig = plt.figure(figsize=(10, 10), tight_layout='True')
    plt.rcParams.update({'font.size' : 22})

    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis', len(timePointsPlot)).colors
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    for i in range(len(timePointsPlot)):

        timeIndex = (np.abs(timeArr - timePointsPlot[i])).argmin()
        timeIndexFitted = (np.abs(fittedTimeArr - timePointsPlot[i])).argmin()

        ax1.plot(radialPosArr, radialDensProfile[:, timeIndexFitted],
                 linewidth=3, color=cmap[i])
        
#        # Calculate the error in the central value
#        if args.centralVal == True:
#            centralVal, centralValErr, centralValLoc = centralVal_error(timeArr, timePointsPlot[i], lineIntegratedDensArr, sigmaArr, detectorVerticalPositions, radialPosArr, args)
#
#            # Index of the central value
#            centralValIdx = (np.abs(radialPosArr - centralValLoc)).argmin()
#
#            # Central value from the forward model
#            centralValModel = radialDensProfile[centralValIdx, timeIndexFitted]
#
#            ax1.errorbar([centralValLoc], [centralValModel],
#                         yerr=[centralValErr],
#                         fmt='o',
#                         capsize=0,
#                         color=cmap[i],
#                         ecolor=cmap[i])

        ax2.plot(radialPosArr, lineIntegratedRadialProfile[:, timeIndexFitted],
                 linewidth=3, color=cmap[i], label='{}ms'.format(timePointsPlot[i]))
        
        ax2.errorbar(detectorVerticalPositions, lineIntegratedDensArr[:, timeIndex],
                     yerr=sigmaArr[:, timeIndex],
                     fmt='o',
                     capsize=0,
                     color=cmap[i],
                     ecolor=cmap[i])

    ax1.set_xlim(-0.25, 0.25)
    ax2.set_xlim(-0.25, 0.25)
    ax1.set_xlabel('Radius [m]')
    ax2.set_xlabel('Radius [m]')

    ax1.set_ylim(0, None)
#    ax2.set_ylim(0, 1.2*np.max(lineIntegratedDensArr))
    ax2.set_ylim(0, 1.5e19)
    ax1.set_ylabel(r'$n_p$ [m$^{-3}$]')
    ax2.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')

    ax2.legend(loc=(1.01, 0))
    ax1.set_title('{}'.format(args.shotnum), loc='right')

    plt.savefig('/home/sanwalka/shinethru_paper_plots/plots/profile_reconstruction_'+modelTypeStr+'_{}.png'.format(args.shotnum), dpi=600)

    plt.show()

    #### 3D contour of the radial density profile
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    plt.rcParams.update({'font.size' : 16})
    ax = fig.add_subplot(111, projection='3d')

    # Time limits based on when the NBI was active
    maxDensArr = np.max(radialDensProfile, axis=0)
    idxArr = np.where(maxDensArr > 0)[0]
    startTime = fittedTimeArr[idxArr[0]] - 2
    endTime = fittedTimeArr[idxArr[-1]] + 2
    startInd = np.abs(fittedTimeArr - startTime).argmin()
    endInd = np.abs(fittedTimeArr - endTime).argmin()

    # Trim the data accordingly
    timeArrPlot = fittedTimeArr[startInd:endInd]
    radialProfPlot = radialDensProfile[:, startInd:endInd]
    
    X, Y = np.meshgrid(radialPosArr, timeArrPlot)
    ax.plot_surface(X, Y, radialProfPlot.T,
                    cmap=cm.inferno, alpha=1)

    ax.set_title(args.shotnum)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Time [ms]')
    ax.set_zlabel(r'$n_p$ [m$^{-3}$]')

    ax.set_ylim(startTime, endTime)
    ax.set_zlim(1e5, None)

    plt.show()

    return

def save_model_reconstruction(args, model_type):

    # Line integrated data from the detectors
    timeArr, lineIntegratedDensArr = calculate_line_densities(args)

    # Detector vertical positions
    detectorVerticalPositions = impact_params(args)

    # Radial positions where I want my reconstructed data
    radialPosArr = np.linspace(-0.25, 0.25, 100)
    
    # Data from the reconstruction
    fittedTimeArr, radialDensProfile, lineIntegratedRadialProfile = model_type(timeArr, lineIntegratedDensArr, sigmaArr, detectorVerticalPositions, radialPosArr, args)

    # Least squares error of the model vs the data
    # errorArr has the same shape as fittedTimeArr
    errorArr, avgError = model_error(fittedTimeArr, timeArr, detectorVerticalPositions, radialPosArr, lineIntegratedDensArr, lineIntegratedRadialProfile, makeplot=True)
    
    modelTypeStr = model_type.__name__.split('_')[2]

    # Save the data in an .npz file
    np.savez(f'/home/sanwalka/shinethru_paper_plots/data/{args.shotnum}_{modelTypeStr}.npz',
             timeArr = timeArr, timeArrUnits = 'ms',
             timeArrFitted = fittedTimeArr, timeArrFittedUnits = 'ms',
             shinethruData = lineIntegratedDensArr, shinethruDataUnits = 'm^-2',
             shinethruImpactParams = detectorVerticalPositions, shinethruImpactParamsUnits = 'm',
             radialDensityProfile = radialDensProfile, radialDensityProfileUnits = 'm^-3',
             reconstructedLineIntegral = lineIntegratedRadialProfile, reconstructedLineIntegralUnits = 'm^-2',
             radialPositions = radialPosArr, radialPositionsUnits = 'm',
             fitChi2Error = errorArr)

    return

def make_shot_plots(args):

    plot_shinethru_vs_interferometer(args)
    plot_model_reconstruction(args, radial_profile_kai_model)
    plot_model_reconstruction(args, radial_profile_forward_model)
    plot_model_reconstruction(args, radial_profile_splines_model)

    return

if __name__ == '__main__':

    # Shot - Reference shot
    # 241227100 - 241227056
    # 250324094 - 250324084
    # 250322085 - 250322062

    # Low density case- 260106057
    
    args = parseArgs()

    # Convert the debug and edge arguments from string to boolean
    if args.debug == 'True':
        args.debug = True
    else:
        args.debug = False

    if args.edge == 'True':
        args.edge = True
    else:
        args.edge = False

#    timeArr, signals = get_shinethru_data(args, args.shotnum)

#    timeArr, lineIntegratedDensities = calculate_line_densities(args)

    plot_shinethru_vs_interferometer(args)

    if args.modelErrTime != 0:
        model_error_envelope(args)
    else:
        plot_model_reconstruction(args, radial_profile_splines_forward_model)
#    save_model_reconstruction(args, radial_profile_splines_forward_model)
#    plot_model_reconstruction(args, radial_profile_kai_model)
#    plot_model_reconstruction(args, radial_profile_forward_model)
#    plot_model_reconstruction(args, radial_profile_splines_model)

#    make_shot_plots(args)
