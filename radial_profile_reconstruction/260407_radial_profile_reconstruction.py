# Stuff to make parallel processing work
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt
import argparse
import matplotlib
from functools import partial
from multiprocessing import Pool, cpu_count
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size':18})

from astropy import modeling
from astropy.modeling import Fittable1DModel, Parameter, models, fitting
from astropy.modeling.models import custom_model

def parseArgs():

    parser = argparse.ArgumentParser(description = 'Command line arguments for shinethru data reconstruction.')

    parser.add_argument('-s', '--shotnum', 
                        type=int, 
                        metavar='Shot number',
                        default=0,
                        help='The shot number to load the data from. Default is 0.')
    
    parser.add_argument('-p', '--debug',
                        type=bool,
                        metavar='Debug mode',
                        default=False,
                        help='Enable debug mode. Makes a lof of print statements and plots. Default is False.')

    parser.add_argument('-n', '--numFits',
                        type=int,
                        metavar='Number of tries in the fitting algorithm',
                        default=20, # determined via checks against synthetic data. After 20 fits, the error doesn't decrease much anymore and we tend to overfit.
                        help='The number of times to try the fit. Default is 20.')

    return parser.parse_args()

def get_shinethru_data():

    # Get the data from MDSplus
    tree = mds.Tree('wham', args.shotnum)

    impactParams = []
    shinethruData = []
    shinethruDataErr = []
    timeArr = []

    # There are 21 SEE detectors
    for i in range(21):

        # Skip the 14th detector. It is broken
        if i in [15]:
            continue

        # Detector node
        try:
            detNode = tree.getNode(f'diag.shinethru.det_{(i+1):02d}')
        except:
            print(f"Error occurred while accessing node diag.shinethru.det_{(i+1):02d}")
            pass

        # Vertical impact parameter
        try:
            impactParams.append(detNode.getNode('v_impact').data())
        except:
            impactParams.append(None)
            print(f"Error occurred while accessing node diag.shinethru.det_{(i+1):02d}.v_impact")
            pass

        # Shinethru data
        try:
            shinethruData.append(detNode.getNode('linedens').data())
        except:
            shinethruData.append(None)
            print(f"Error occurred while accessing node diag.shinethru.det_{(i+1):02d}.linedens")
            pass
        
        # Shinethru data error
        try:
            shinethruDataErr.append(detNode.getNode('linedens_err').data())
        except:
            shinethruDataErr.append(None)
            print(f"Error occurred while accessing node diag.shinethru.det_{(i+1):02d}.linedens_err")
            pass
        
        # Time array
        try:
            timeArr.append(detNode.getNode('linedens').dim_of().data())
        except:
            timeArr.append(None)
            print(f"Error occurred while accessing node diag.shinethru.det_{(i+1):02d}.linedens.dim_of()")
            pass

    # Put all the data onto the same timebase
    newTimeArr = timeArr[0]
    newShinethruData = []
    newShinethruDataErr = []
    newImpactParams = []

    for i in range(len(shinethruData)):

        # Interpolate the data onto the new timebase
        if shinethruData[i] is not None and shinethruDataErr[i] is not None and timeArr[i] is not None:
            newShinethruData.append(np.interp(newTimeArr, timeArr[i], shinethruData[i]))
            newShinethruDataErr.append(np.interp(newTimeArr, timeArr[i], shinethruDataErr[i]))
            newImpactParams.append(impactParams[i])

        elif shinethruData[i] is not None and timeArr[i] is not None:
            newShinethruData.append(np.interp(newTimeArr, timeArr[i], shinethruData[i]))
            newShinethruDataErr.append(np.zeros_like(newTimeArr))
            newImpactParams.append(impactParams[i])

    # Convert to numpy arrays
    impactParams = np.array(newImpactParams)
    shinethruData = np.array(newShinethruData)
    shinethruDataErr = np.array(newShinethruDataErr)

    # Clean up the data a bit
    shinethruDataErr = np.nan_to_num(shinethruDataErr, posinf=1e21, neginf=1e21)

    # Crop the data to where the density is non-zero. This will make the fitting faster and more accurate
    nonZeroIndices = np.where(np.any(shinethruData > 1e17, axis=0))[0]
    firstIndex = nonZeroIndices[0] - 100
    lastIndex = nonZeroIndices[-1] + 100
    shinethruData = shinethruData[:, firstIndex:lastIndex+1]
    shinethruDataErr = shinethruDataErr[:, firstIndex:lastIndex+1]
    newTimeArr = newTimeArr[firstIndex:lastIndex+1]

    if args.debug:

        print("Impact parameters:", impactParams)
        print("Shinethru data shape:", shinethruData.shape)
        print("Shinethru data error shape:", shinethruDataErr.shape)
        print("Time array shape:", newTimeArr.shape)

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        # Plot the data for a few time points
        timePointsPlot = np.array([7,8,9,10,11]) * 1e-3

        timeIndices = np.array([np.argmin(np.abs(newTimeArr - timePoint)) for timePoint in timePointsPlot])
        dataForPlotting = shinethruData[:, timeIndices]
        errorForPlotting = shinethruDataErr[:, timeIndices]

        for i in range(len(timePointsPlot)):

            timePoint = timePointsPlot[i]

            ax.errorbar(impactParams, dataForPlotting[:, i], 
                        yerr=errorForPlotting[:, i], 
                        fmt='o', 
                        label=f't={timePoint*1e3:.1f} ms',
                        ms=10)

        ax.set_ylim(0, np.max(dataForPlotting)*1.2)

        ax.set_xlabel('Vertical impact parameter [m]')
        ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
        ax.set_title('Data for shot ' + str(args.shotnum))
        ax.legend()
        plt.show()

        # Plot the time evolution of the data for all the detectors
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        for i in range(shinethruData.shape[0]):

            # Apply a strong savgol filter to the data to make it smoother for plotting
            plottingData = sc.signal.savgol_filter(shinethruData[i], window_length=101, polyorder=2)
            ax.plot(newTimeArr*1e3, plottingData, label=f'{i+1}')

        ax.set_xlim(5, 15)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
        ax.set_title('Time evolution of the data for shot ' + str(args.shotnum))
        ax.legend(ncols=2, title='Detector #')
        plt.show()

        # Make a contour plot of the data as a function of time and impact parameter
        fig = plt.figure(figsize=(15, 8), tight_layout=True)

        # Line integrated densities
        ax1 = fig.add_subplot(211)
        # Error bars
        ax2 = fig.add_subplot(212)

        timeGrid, impactParamGrid = np.meshgrid(newTimeArr*1e3, impactParams)

        pltObj1 = ax1.contourf(timeGrid, impactParamGrid, shinethruData, levels=100, cmap='inferno')
        pltObj2 = ax2.contourf(timeGrid, impactParamGrid, shinethruDataErr, levels=np.linspace(0, np.max(shinethruData), 100), cmap='inferno')
        
        cbar1 = fig.colorbar(pltObj1, ax=ax1, label=r'$\int n_p \cdot dl$ [m$^{-2}$]', ticks=np.linspace(0, np.max(shinethruData), 5))
        cbar2 = fig.colorbar(pltObj2, ax=ax2, label=r'$\delta \left( \int n_p \cdot dl \right)$ [m$^{-2}$]', ticks=np.linspace(0, np.max(shinethruData), 5))

        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Impact parameter [m]')
        ax1.set_title(str(args.shotnum))

        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Impact parameter [m]')

        plt.show()

    tree.close()

    return impactParams, shinethruData, shinethruDataErr, newTimeArr

def get_detector_positions():

    tree = mds.Tree('wham', args.shotnum)

    detectorPositions = []

    # There are 21 SEE detectors
    for i in range(21):

        # Detector node
        try:
            detNode = tree.getNode(f'diag.shinethru.det_{(i+1):02d}')
        except:
            print(f"Error occurred while accessing node diag.shinethru.det_{(i+1):02d}")
            pass

        # Detector position
        try:
            detectorPositions.append(detNode.getNode('v_impact').data())
        except:
            detectorPositions.append(None)
            print(f"Error occurred while accessing node diag.shinethru.det_{(i+1):02d}.v_impact")
            pass

    detectorPositions = np.array(detectorPositions)

    tree.close()

    return detectorPositions

def synthetic_circular_profile(nMax, rMax, rOffset, impactParams, noiseLevel=0, smoothingSigma=0.01):

    # Radial positions where we want the radial profile to be evaluated
    rArr = np.linspace(-0.2, 0.2, 100)

    # Radial density profile
    nArr = (rMax**2 - rArr**2)**0.5
    nArr[np.isnan(nArr)] = 0
    nArr /= np.max(nArr)
    nArr *= nMax
    # Smooth the profile a bit to make it more realistic
    nArr = sc.ndimage.gaussian_filter1d(nArr, sigma=smoothingSigma, mode='nearest')

    # Set all -ve values to 0
    nArr[nArr < 0] = 0
    # Set all nan values to 0
    nArr[np.isnan(nArr)] = 0

    # Interpolation function
    nInterpFunc = sc.interpolate.interp1d(rArr, nArr, bounds_error=False, fill_value=0)

    # Make a 2D grid of x,y positions
    xArr = np.linspace(-0.2, 0.2, 100)
    yArr = np.linspace(-0.2, 0.2, 100)
    xGrid, yGrid = np.meshgrid(xArr, yArr)

    # Get the density on the 2D grid
    rGrid = np.sqrt(xGrid**2 + yGrid**2)
    nGrid = nInterpFunc(rGrid)

    # Make a 2D interpolation function for the 2D grid
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGrid,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')
    

    # Offset the density profile by rOffset
    yGridOffset = yGrid - rOffset
    points = np.array([xGrid.flatten(), yGridOffset.flatten()]).T
    nGridNew = n2DInterpFunc(points).reshape(xGrid.shape)

    # Make a new interpolation function for the offset profile
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGridNew,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')
    
    # Radial density profile on the finer grid
    points = np.array([rArr, np.zeros_like(rArr)]).T
    nArrFine = n2DInterpFunc(points)
    
    # Go over each impact parameter and get the density along that detectors line of sight
    detectorDensities = np.zeros(shape=(len(impactParams), len(xArr)))
    for i in range(len(impactParams)):

        # y positions along the line of sight of the detector
        yPosArr = np.full_like(xArr, fill_value=impactParams[i])

        # Get the density along the line of sight of the detector
        points = np.array([yPosArr, xArr]).T

        detectorDensities[i] = n2DInterpFunc(points)

    # Calculate the line integrated density along the line of sight of each detector
    lineIntegratedDensities = np.trapezoid(detectorDensities, x=xArr, axis=1)

    # Line integrated density for each y position on the grid
    lineIntegratedDensityGrid = np.trapezoid(nGridNew, x=xArr, axis=1)

    # Add noise to the line integrated densities
    lineIntegratedDensities += lineIntegratedDensities * noiseLevel * np.random.normal(size=lineIntegratedDensities.shape)

    if args.debug:

        # Plot the radial profile

        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(221)

        ax.plot(rArr, nArrFine)

        ax.set_xlabel('r [m]')
        ax.set_ylabel(r'n [m$^{-3}$]')
        ax.set_title('n(r)')

        # Plot the 2D profile
        ax = fig.add_subplot(222)

        pltObj = ax.contourf(xGrid, yGrid, nGridNew, levels=100, cmap='inferno')

        # Add the detector lines of sight
        for i in range(len(impactParams)):
            yPos = impactParams[i]
            ax.plot(xArr, np.full_like(xArr, yPos), lw=2)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('2D density profile')
        ax.set_aspect('equal')
        cbar = fig.colorbar(pltObj, ax=ax, label=r'$n$ [m$^{-3}$]')

        # Plot the density along the line of sight of each detector
        ax = fig.add_subplot(223)

        for i in range(len(impactParams)):
            ax.plot(xArr, detectorDensities[i], label=f'Impact parameter = {impactParams[i]:.3f} m')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$n$ [m$^{-3}$]')
        ax.set_title('Density along the line of sight of each detector')

        # Plot the line integrated density along the line of sight of each detector
        ax = fig.add_subplot(224)

        for i in range(len(impactParams)):
            ax.scatter([impactParams[i]], [lineIntegratedDensities[i]])

        ax.plot(xArr, lineIntegratedDensityGrid, label='Line integrated density along y-axis')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$\int n \cdot dl$ [m$^{-2}$]')
        ax.set_title('Line integrated densities')

        plt.show()

    return lineIntegratedDensities, nArrFine, rArr

def synthetic_flat_profile(nMax, rMax, rOffset, impactParams, noiseLevel=0, smoothingSigma=0.01):

    # Radial positions where we want the radial profile to be evaluated
    rArr = np.linspace(-0.2, 0.2, 100)

    # Radial density profile
    nArr = np.where(np.abs(rArr) <= rMax, nMax, 0)

    # Set all -ve values to 0
    nArr[nArr < 0] = 0
    # Set all nan values to 0
    nArr[np.isnan(nArr)] = 0

    # Smooth the profile a bit to make it more realistic
    # nArr = np.convolve(nArr, np.ones(5)/5, mode='same')
    # nArr = sc.ndimage.median_filter(nArr, size=10, mode='nearest')
    nArr = sc.ndimage.gaussian_filter1d(nArr, sigma=smoothingSigma, mode='nearest')

    # Interpolation function
    nInterpFunc = sc.interpolate.interp1d(rArr, nArr, bounds_error=False, fill_value=0)

    # Make a 2D grid of x,y positions
    xArr = np.linspace(-0.2, 0.2, 100)
    yArr = np.linspace(-0.2, 0.2, 100)
    xGrid, yGrid = np.meshgrid(xArr, yArr)

    # Get the density on the 2D grid
    rGrid = np.sqrt(xGrid**2 + yGrid**2)
    nGrid = nInterpFunc(rGrid)

    # Make a 2D interpolation function for the 2D grid
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGrid,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')
    

    # Offset the density profile by rOffset
    yGridOffset = yGrid - rOffset
    points = np.array([xGrid.flatten(), yGridOffset.flatten()]).T
    nGridNew = n2DInterpFunc(points).reshape(xGrid.shape)

    # Make a new interpolation function for the offset profile
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGridNew,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')

    # Radial density profile on the finer grid
    points = np.array([rArr, np.zeros_like(rArr)]).T
    nArrFine = n2DInterpFunc(points)
    
    # Go over each impact parameter and get the density along that detectors line of sight
    detectorDensities = np.zeros(shape=(len(impactParams), len(xArr)))
    for i in range(len(impactParams)):

        # y positions along the line of sight of the detector
        yPosArr = np.full_like(xArr, fill_value=impactParams[i])

        # Get the density along the line of sight of the detector
        points = np.array([yPosArr, xArr]).T

        detectorDensities[i] = n2DInterpFunc(points)

    # Calculate the line integrated density along the line of sight of each detector
    lineIntegratedDensities = np.trapezoid(detectorDensities, x=xArr, axis=1)

    # Line integrated density for each y position on the grid
    lineIntegratedDensityGrid = np.trapezoid(nGridNew, x=xArr, axis=1)

    # Add noise to the line integrated densities
    lineIntegratedDensities += lineIntegratedDensities * noiseLevel * np.random.normal(size=lineIntegratedDensities.shape)

    if args.debug:

        # Plot the radial profile

        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(221)

        ax.plot(rArr, nArr)

        ax.set_xlabel('r [m]')
        ax.set_ylabel(r'n [m$^{-3}$]')
        ax.set_title('Unshifted n(r)')

        # Plot the 2D profile
        ax = fig.add_subplot(222)

        pltObj = ax.contourf(xGrid, yGrid, nGridNew, levels=100, cmap='inferno')

        # Add the detector lines of sight
        for i in range(len(impactParams)):
            yPos = impactParams[i]
            ax.plot(xArr, np.full_like(xArr, yPos), lw=2)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('2D density profile')
        ax.set_aspect('equal')
        cbar = fig.colorbar(pltObj, ax=ax, label=r'$n$ [m$^{-3}$]')

        # Plot the density along the line of sight of each detector
        ax = fig.add_subplot(223)

        for i in range(len(impactParams)):
            ax.plot(xArr, detectorDensities[i], label=f'Impact parameter = {impactParams[i]:.3f} m')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$n$ [m$^{-3}$]')
        ax.set_title('Density along the line of sight of each detector')

        # Plot the line integrated density along the line of sight of each detector
        ax = fig.add_subplot(224)

        for i in range(len(impactParams)):
            ax.scatter([impactParams[i]], [lineIntegratedDensities[i]])

        ax.plot(xArr, lineIntegratedDensityGrid, label='Line integrated density along y-axis')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$\int n \cdot dl$ [m$^{-2}$]')
        ax.set_title('Line integrated densities')

        plt.show()

    return lineIntegratedDensities, nArrFine, rArr

def synthetic_gaussian_profile(nMax, rMax, rOffset, impactParams, noiseLevel=0, smoothingSigma=0.01):

    # Radial positions where we want the radial profile to be evaluated
    rArr = np.linspace(-0.2, 0.2, 100)

    # Radial density profile
    nArr = nMax * np.exp(-rArr**2 / (2 * (rMax/2)**2))

    # Set all -ve values to 0
    nArr[nArr < 0] = 0
    # Set all nan values to 0
    nArr[np.isnan(nArr)] = 0

    # Interpolation function
    nInterpFunc = sc.interpolate.interp1d(rArr, nArr, bounds_error=False, fill_value=0)

    # Make a 2D grid of x,y positions
    xArr = np.linspace(-0.2, 0.2, 100)
    yArr = np.linspace(-0.2, 0.2, 100)
    xGrid, yGrid = np.meshgrid(xArr, yArr)

    # Get the density on the 2D grid
    rGrid = np.sqrt(xGrid**2 + yGrid**2)
    nGrid = nInterpFunc(rGrid)

    # Make a 2D interpolation function for the 2D grid
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGrid,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')
    

    # Offset the density profile by rOffset
    yGridOffset = yGrid - rOffset
    points = np.array([xGrid.flatten(), yGridOffset.flatten()]).T
    nGridNew = n2DInterpFunc(points).reshape(xGrid.shape)

    # Make a new interpolation function for the offset profile
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGridNew,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')
    
    # Radial density profile on the finer grid
    points = np.array([rArr, np.zeros_like(rArr)]).T
    nArrFine = n2DInterpFunc(points)
    
    # Go over each impact parameter and get the density along that detectors line of sight
    detectorDensities = np.zeros(shape=(len(impactParams), len(xArr)))
    for i in range(len(impactParams)):

        # y positions along the line of sight of the detector
        yPosArr = np.full_like(xArr, fill_value=impactParams[i])

        # Get the density along the line of sight of the detector
        points = np.array([yPosArr, xArr]).T

        detectorDensities[i] = n2DInterpFunc(points)

    # Calculate the line integrated density along the line of sight of each detector
    lineIntegratedDensities = np.trapezoid(detectorDensities, x=xArr, axis=1)

    # Line integrated density for each y position on the grid
    lineIntegratedDensityGrid = np.trapezoid(nGridNew, x=xArr, axis=1)

    # Add noise to the line integrated densities
    lineIntegratedDensities += lineIntegratedDensities * noiseLevel * np.random.normal(size=lineIntegratedDensities.shape)

    if args.debug:

        # Plot the radial profile

        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(221)

        ax.plot(rArr, nArr)

        ax.set_xlabel('r [m]')
        ax.set_ylabel(r'n [m$^{-3}$]')
        ax.set_title('Unshifted n(r)')

        # Plot the 2D profile
        ax = fig.add_subplot(222)

        pltObj = ax.contourf(xGrid, yGrid, nGridNew, levels=100, cmap='inferno')

        # Add the detector lines of sight
        for i in range(len(impactParams)):
            yPos = impactParams[i]
            ax.plot(xArr, np.full_like(xArr, yPos), lw=2)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('2D density profile')
        ax.set_aspect('equal')
        cbar = fig.colorbar(pltObj, ax=ax, label=r'$n$ [m$^{-3}$]')

        # Plot the density along the line of sight of each detector
        ax = fig.add_subplot(223)

        for i in range(len(impactParams)):
            ax.plot(xArr, detectorDensities[i], label=f'Impact parameter = {impactParams[i]:.3f} m')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$n$ [m$^{-3}$]')
        ax.set_title('Density along the line of sight of each detector')

        # Plot the line integrated density along the line of sight of each detector
        ax = fig.add_subplot(224)

        for i in range(len(impactParams)):
            ax.scatter([impactParams[i]], [lineIntegratedDensities[i]])

        ax.plot(xArr, lineIntegratedDensityGrid, label='Line integrated density along y-axis')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$\int n \cdot dl$ [m$^{-2}$]')
        ax.set_title('Line integrated densities')

        plt.show()

    return lineIntegratedDensities, nArrFine, rArr

def synthetic_hollow_profile(nMax, rMax, rOffset, impactParams, noiseLevel=0, smoothingSigma=0.01):

    # Radial positions where we want the radial profile to be evaluated
    rArr = np.linspace(-0.2, 0.2, 100)

    # Radial density profile
    nArr = np.where(np.abs(rArr) <= rMax, nMax, 0)
    nArr[np.abs(rArr) < rMax/2] = nMax/10

    # Set all -ve values to 0
    nArr[nArr < 0] = 0
    # Set all nan values to 0
    nArr[np.isnan(nArr)] = 0

    # Smooth the profile a bit to make it more realistic
    # nArr = sc.ndimage.median_filter(nArr, size=10)
    nArr = sc.ndimage.gaussian_filter1d(nArr, sigma=smoothingSigma, mode='nearest')

    # Interpolation function
    nInterpFunc = sc.interpolate.RegularGridInterpolator((rArr,), nArr, bounds_error=False, fill_value=0)

    # Make a 2D grid of x,y positions
    xArr = np.linspace(-0.2, 0.2, 100)
    yArr = np.linspace(-0.2, 0.2, 100)
    xGrid, yGrid = np.meshgrid(xArr, yArr)

    # Get the density on the 2D grid
    rGrid = np.sqrt(xGrid**2 + yGrid**2)
    nGrid = nInterpFunc(rGrid.flatten()).reshape(rGrid.shape)

    # Make a 2D interpolation function for the 2D grid
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGrid,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')
    

    # Offset the density profile by rOffset
    yGridOffset = yGrid - rOffset
    points = np.array([xGrid.flatten(), yGridOffset.flatten()]).T
    nGridNew = n2DInterpFunc(points).reshape(xGrid.shape)

    # Make a new interpolation function for the offset profile
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGridNew,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')

    # Radial density profile on the finer grid
    points = np.array([rArr, np.zeros_like(rArr)]).T
    nArrFine = n2DInterpFunc(points)
    
    # Go over each impact parameter and get the density along that detectors line of sight
    detectorDensities = np.zeros(shape=(len(impactParams), len(xArr)))
    for i in range(len(impactParams)):

        # y positions along the line of sight of the detector
        yPosArr = np.full_like(xArr, fill_value=impactParams[i])

        # Get the density along the line of sight of the detector
        points = np.array([yPosArr, xArr]).T

        detectorDensities[i] = n2DInterpFunc(points)

    # Calculate the line integrated density along the line of sight of each detector
    lineIntegratedDensities = np.trapezoid(detectorDensities, x=xArr, axis=1)

    # Line integrated density for each y position on the grid
    lineIntegratedDensityGrid = np.trapezoid(nGridNew, x=xArr, axis=1)

    # Add noise to the line integrated densities
    lineIntegratedDensities += lineIntegratedDensities * noiseLevel * np.random.normal(size=lineIntegratedDensities.shape)

    if args.debug:

        # Plot the radial profile

        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(221)

        ax.plot(rArr, nArr)

        ax.set_xlabel('r [m]')
        ax.set_ylabel(r'n [m$^{-3}$]')
        ax.set_title('Unshifted n(r)')

        # Plot the 2D profile
        ax = fig.add_subplot(222)

        pltObj = ax.contourf(xGrid, yGrid, nGridNew, levels=100, cmap='inferno')

        # Add the detector lines of sight
        for i in range(len(impactParams)):
            yPos = impactParams[i]
            ax.plot(xArr, np.full_like(xArr, yPos), lw=2)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('2D density profile')
        ax.set_aspect('equal')
        cbar = fig.colorbar(pltObj, ax=ax, label=r'$n$ [m$^{-3}$]')

        # Plot the density along the line of sight of each detector
        ax = fig.add_subplot(223)

        for i in range(len(impactParams)):
            ax.plot(xArr, detectorDensities[i], label=f'Impact parameter = {impactParams[i]:.3f} m')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$n$ [m$^{-3}$]')
        ax.set_title('Density along the line of sight of each detector')

        # Plot the line integrated density along the line of sight of each detector
        ax = fig.add_subplot(224)

        for i in range(len(impactParams)):
            ax.scatter([impactParams[i]], [lineIntegratedDensities[i]])

        ax.plot(xArr, lineIntegratedDensityGrid, label='Line integrated density along y-axis')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$\int n \cdot dl$ [m$^{-2}$]')
        ax.set_title('Line integrated densities')

        plt.show()

    return lineIntegratedDensities, nArrFine, rArr

def construct_spline_function(x, mean=0, limit=0.15, knot1Pos=0.1, knot2Pos=0.14, knot1Val=1, knot2Val=0, centralVal=0.5):

    # Make a spline function that goes through the points (0, centralVal), (knotPos, knotVal) and (limit, 0)
    knotPosArr = np.array([0, knot1Pos, knot2Pos, limit])
    knotPosArr = np.sort(knotPosArr)
    knotValArr = np.array([centralVal, knot1Val, knot2Val, 0])

    # Add stiffness
    bc_natural = ((1, 0.0), (1, 0.0)) # (order, value) for second derivative

    try:
        spline1Func = sc.interpolate.CubicSpline(knotPosArr, knotValArr, extrapolate=False, bc_type=bc_natural)
    except:
        print("Error occurred while creating the spline function. Check the knot positions and values.")
        print("Knot positions:", knotPosArr)
        print("Knot values:", knotValArr)
    # spline1Func = sc.interpolate.pchip(knotPosArr, knotValArr, extrapolate=False)
    spline1 = spline1Func(np.abs(x - mean))

    # Clean up the data by setting all -ve values to 0 and all nan values to 0
    # The density cannot be -ve and it definitiely cannot be nan
    spline1[spline1 < 0] = 0
    spline1[np.isnan(spline1)] = 0

    if args.debug:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(x, spline1, linewidth=3)

        ax.scatter(knotPosArr+mean, knotValArr, c='red', label='Knot points', s=200)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('Spline value')
        ax.set_title('Spline function')
        ax.legend()

        plt.show()

    return spline1

@custom_model
def integrated_spline_function(xInput, mean=[0], limit=[0.15], knot1Pos=[0.1], knot2Pos=[0.14], knot1Val=[1], knot2Val=[0], centralVal=[0.5]):

    # Sometimes xInput can be a list or an int. Convert it to a numpy array if it is not already
    xInput = np.array(xInput)

    # The fitter passes 1 element arrays
    try:
        mean = mean[0]
        limit = limit[0]
        knot1Pos = knot1Pos[0]
        knot2Pos = knot2Pos[0]
        knot1Val = knot1Val[0]
        knot2Val = knot2Val[0]
        centralVal = centralVal[0]
    except:
        pass

    # The fitter fits log values, so we need to exponentiate the parameters
    knot1Val = 10**knot1Val
    knot2Val = 10**knot2Val
    centralVal = 10**centralVal

    # Finer x array for integration
    xLim = limit * 1.5
    rArr = np.linspace(-xLim, xLim, 100)

    # Get the unshifted spline function
    nArr = construct_spline_function(rArr, 
                                     mean=0, 
                                     limit=limit, 
                                     knot1Pos=knot1Pos, 
                                     knot2Pos=knot2Pos, 
                                     knot1Val=knot1Val, 
                                     knot2Val=knot2Val, 
                                     centralVal=centralVal)

    # Interpolation function
    nInterpFunc = sc.interpolate.RegularGridInterpolator((rArr,), nArr, bounds_error=False, fill_value=0)

    # Make a 2D grid of x,y positions
    xArr = rArr
    yArr = rArr
    xGrid, yGrid = np.meshgrid(xArr, yArr)

    # Get the density on the 2D grid
    rGrid = np.sqrt(xGrid**2 + yGrid**2)
    nGrid = nInterpFunc(rGrid.flatten()).reshape(rGrid.shape)

    # Make a 2D interpolation function for the 2D grid
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGrid,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')
    

    # Offset the density profile by the mean
    yGridOffset = yGrid - mean
    points = np.array([xGrid.flatten(), yGridOffset.flatten()]).T
    nGridNew = n2DInterpFunc(points).reshape(xGrid.shape)

    # Make a new interpolation function for the offset profile
    n2DInterpFunc = sc.interpolate.RegularGridInterpolator((xArr, yArr), nGridNew,
                                                           bounds_error=False, 
                                                           fill_value=0,
                                                           method='linear')

    # Calculate the line integrated density along the line of sight of each detector
    X, Y = np.meshgrid(xInput, xArr, indexing='ij')

    # Build query points for interpolation
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # Evaluate all at once
    detectorDensities = n2DInterpFunc(points).reshape(len(xInput), len(xArr))

    # Integrate
    lineIntegratedDensities = np.trapezoid(detectorDensities, x=xArr, axis=1)

    if args.debug:

        # Line integrated density for each y position on the grid
        lineIntegratedDensityGrid = np.trapezoid(nGridNew, x=xArr, axis=1)

        # Plot the radial profile

        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(221)

        ax.plot(rArr, nArr, linewidth=3)

        # Add the knot points
        knotPosArr = np.array([0, knot1Pos, knot2Pos, limit])
        knotPosArr = np.sort(knotPosArr)
        knotValArr = np.array([centralVal, knot1Val, knot2Val, 0])
        ax.scatter(knotPosArr, knotValArr, c='red', label='Knot points', s=200)

        ax.set_xlabel('r [m]')
        ax.set_ylabel(r'n [m$^{-3}$]')
        ax.set_title('Unshifted n(r)')

        # Plot the 2D profile
        ax = fig.add_subplot(222)

        pltObj = ax.contourf(xGrid, yGrid, nGridNew, levels=100, cmap='inferno')

        # Add the detector lines of sight
        for i in range(len(xInput)):
            yPos = xInput[i]
            ax.plot(xArr, np.full_like(xArr, yPos), lw=2)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('2D density profile')
        ax.set_aspect('equal')
        cbar = fig.colorbar(pltObj, ax=ax, label=r'$n$ [m$^{-3}$]')

        # Plot the density along the line of sight of each detector
        ax = fig.add_subplot(223)

        for i in range(len(xInput)):
            ax.plot(xArr, detectorDensities[i], label=f'Impact parameter = {xInput[i]:.3f} m')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$n$ [m$^{-3}$]')
        ax.set_title('Density along the line of sight of each detector')

        # Plot the line integrated density along the line of sight of each detector
        ax = fig.add_subplot(224)

        for i in range(len(xInput)):
            ax.scatter([xInput[i]], [lineIntegratedDensities[i]])

        ax.plot(xArr, lineIntegratedDensityGrid, label='Line integrated density along y-axis')

        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'$\int n \cdot dl$ [m$^{-2}$]')
        ax.set_title('Line integrated densities')

        plt.show()

    return lineIntegratedDensities

def fit_data(lineIntegratedDensities, impactParams, errorBars=None):

    # Fit the data multiple times and take the best fit
    numFits = args.numFits

    meanArr = np.random.uniform(low=-0.05, high=0.05, size=numFits)
    limitArr = np.random.uniform(low=0.1, high=0.2, size=numFits)
    knotPosArr = np.random.uniform(low=0.05, high=0.15, size=numFits)
    knotValArr = np.random.uniform(low=1e18, high=1e20, size=numFits)
    centralValArr = np.random.uniform(low=1e18, high=5e20, size=numFits)

    models = []
    errors = []

    for i in range(numFits):

        model = integrated_spline_function()
        fitter = fitting.LevMarLSQFitter()

        # Some reasonable bounds on the model
        
        model.mean.max = 0.03
        model.mean.min = -0.03

        model.limit.max = 0.2
        model.limit.min = 0.1

        model.knotPos.max = 0.15
        model.knotPos.min = 0.01

        model.knotVal.max = 1e20
        model.knotVal.min = 1e18

        model.centralVal.max = 5e20
        model.centralVal.min = 1e18

        # First guess
        model.mean = meanArr[i]
        model.limit = limitArr[i]
        model.knotPos = knotPosArr[i]
        model.knotVal = knotValArr[i]
        model.centralVal = centralValArr[i]

        # Fit the model to the data
        # Sort the data before sending it to the fitter
        sortIdx = np.argsort(impactParams)
        sortedImpactParams = impactParams[sortIdx]
        sortedLineIntegratedDensities = lineIntegratedDensities[sortIdx]

        if errorBars is None:
            fittedModel = fitter(model, 
                                sortedImpactParams, 
                                sortedLineIntegratedDensities,
                                maxiter=200)
        else:
            sortedErrorBars = errorBars[sortIdx]
            fittedModel = fitter(model, 
                                sortedImpactParams, 
                                sortedLineIntegratedDensities,
                                weights=1/sortedErrorBars,
                                maxiter=200)

        # Get the fitted values
        fittedValues = fittedModel(impactParams)

        # Calculate the error between the fitted values and the data
        errorArr = ((fittedValues - lineIntegratedDensities)/lineIntegratedDensities)**2
        error = np.average(errorArr)

        models.append(fittedModel)
        errors.append(error)

    return models, errors

def best_fit(i, dataToFit, impactParams, errorBars=None):

    lineIntegratedDensities = dataToFit[:, i]

    # Sort the data before sending it to the fitter
    sortIdx = np.argsort(impactParams)
    xdata = impactParams[sortIdx]
    ydata = lineIntegratedDensities[sortIdx]
    if errorBars is not None:
        weights = 1/errorBars[sortIdx]
    else:
        weights = None

    # Limits on the model parameters for optimization
    meanLimits = (-0.03, 0.03)
    limitLimits = (0.1, 0.25)
    knot1PosLimits = (0.01, 0.2)
    knot1ValLimits = (18, 20)       # log scale
    knot2PosLimits = (0.02, 0.225)
    knot2ValLimits = (18, 20)       # log scale
    centralValLimits = (18, 20)     # log scale

    #### Perform a global optimization to get a good initial guess for the fitter ####
    bounds = [meanLimits,               # mean
              limitLimits,              # limit
              knot1PosLimits,           # knotPos
              knot2PosLimits,           # knotPos
              knot1ValLimits,           # knotVal (log scale)
              knot2ValLimits,           # knotVal (log scale)
              centralValLimits]         # centralVal (log scale)

    def objective(params):

        mean, limit, knot1Pos, knot2Pos, knot1Val, knot2Val, centralVal = params

        model = integrated_spline_function()

        # Set parameters
        model.mean = mean
        model.limit = limit
        model.knot1Pos = knot1Pos
        model.knot2Pos = knot2Pos
        model.knot1Val = knot1Val
        model.knot2Val = knot2Val
        model.centralVal = centralVal

        yfit = model(xdata)

        if weights is None:
            err = (yfit - ydata)
        else:
            err = (yfit - ydata) * (weights)

        return np.mean(err**2)

    result = sc.optimize.differential_evolution(objective,
                                                bounds,
                                                strategy='best1bin',
                                                maxiter=100,     # tune this
                                                popsize=15,      # tune this
                                                tol=1e-3,
                                                seed=0)          # makes it reproducible

    bestParams = result.x

    #### Refine the result with a local fitter ####
    fitter = fitting.LevMarLSQFitter()
    model = integrated_spline_function()

    # Apply bounds again
    model.mean.min, model.mean.max = meanLimits
    model.limit.min, model.limit.max = limitLimits
    model.knot1Pos.min, model.knot1Pos.max = knot1PosLimits
    model.knot2Pos.min, model.knot2Pos.max = knot2PosLimits
    model.knot1Val.min, model.knot1Val.max = knot1ValLimits
    model.knot2Val.min, model.knot2Val.max = knot2ValLimits
    model.centralVal.min, model.centralVal.max = centralValLimits

    # Set initial guess from global optimizer
    (model.mean,
    model.limit,
    model.knot1Pos,
    model.knot2Pos,
    model.knot1Val,
    model.knot2Val,
    model.centralVal) = bestParams

    # Calculate the error of the initial guess    
    initialFitValues = model(xdata)
    initialErrorArr = ((initialFitValues - lineIntegratedDensities)/lineIntegratedDensities)**2
    initialError = np.average(initialErrorArr)
    print(f'Initial error from global optimization: {initialError:.2e}')

    # Final fit
    if weights is None:
        fittedModel = fitter(model, xdata, ydata, maxiter=200)
    else:
        fittedModel = fitter(model, xdata, ydata, weights=weights, maxiter=200)

    # Error in the best model
    fittedValues = fittedModel(xdata)
    # print(f'{np.array([fittedValues, ydata]).T}')
    errorArr = ((fittedValues - ydata)/ydata)**2
    error = np.average(errorArr)

    print(f'Error after local optimization: {error:.2e}')

    return fittedModel, error

def fit_vs_input(model_type):

    nMax, rMax, rOffset, impactParams = 1e20, 0.15, 0.01, get_detector_positions()

    # Synthetic data
    lineIntegratedDensities, radialProfile, rArr = model_type(nMax, rMax, rOffset, impactParams, args)

    # Fit the data
    fittedModels, errors = fit_data(lineIntegratedDensities, impactParams)

    # Index of the best model
    bestModelIndex = np.argmin(errors)

    # Plot the fitted model against the input data
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    fig.suptitle('Fitted model vs input data')

    # Plot the radial density profiles
    ax = fig.add_subplot(121)

    # Plot the input data
    ax.plot(rArr, radialProfile, label='Input data', color='red')

    # Plot the fitted models
    for i in range(len(fittedModels)):

        fittedModel = fittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                mean=fittedModel.mean.value, 
                                                limit=fittedModel.limit.value, 
                                                knotPos=fittedModel.knotPos.value, 
                                                knotVal=fittedModel.knotVal.value, 
                                                centralVal=fittedModel.centralVal.value)
        if i == bestModelIndex:
            ax.plot(rArr, fittedDensity, label=f'Error: {errors[i]:.2e}', color='blue', linewidth=3, zorder=5)
        else:
            ax.plot(rArr, fittedDensity, color='green')

    ax.set_xlabel('r [m]')
    ax.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax.legend()

    # Plot the line integrated result
    ax = fig.add_subplot(122)

    # Plot the input data
    inputLineIntegratedDensities, _, _ = model_type(nMax, rMax, rOffset, rArr, args)
    ax.plot(rArr, inputLineIntegratedDensities, label='Input data', color='red')

    # Plot the data points
    ax.scatter(impactParams, lineIntegratedDensities, label='Data points', color='black', zorder=5)

    # Plot the fitted models
    for i in range(len(fittedModels)):
        fittedModel = fittedModels[i]
        if i == bestModelIndex:
            ax.plot(rArr, fittedModel(rArr), label=f'Error: {errors[i]:.2e}', color='blue', linewidth=3, zorder=5)
        else:
            ax.plot(rArr, fittedModel(rArr), color='green')

    ax.set_xlabel('r [m]')
    ax.set_ylabel(r'$\int n \cdot dl$ [m$^{-2}$]')
    ax.legend()
    plt.show()

    return

def effect_of_fit_count():

    nMax, rMax, rOffset, impactParams = 1e20, 0.15, 0.01, get_detector_positions()

    # Synthetic data
    circLineIntegratedDensities, circRadialProfileFine, rArr = synthetic_circular_profile(nMax, rMax, rOffset, impactParams)
    flatLineIntegratedDensities, flatRadialProfileFine, rArr = synthetic_flat_profile(nMax, rMax, rOffset, impactParams)
    gaussianLineIntegratedDensities, gaussianRadialProfileFine, rArr = synthetic_gaussian_profile(nMax, rMax, rOffset, impactParams)
    hollowLineIntegratedDensities, hollowRadialProfileFine, rArr = synthetic_hollow_profile(nMax, rMax, rOffset, impactParams)
    # # Add a small number to avoid /0 errors
    # circRadialProfileFine += 1e-6
    # flatRadialProfileFine += 1e-6
    # gaussianRadialProfileFine += 1e-6
    # hollowRadialProfileFine += 1e-6
    # Interpolate the radial profiles onto impactParams for error calculation later
    circRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, circRadialProfileFine, bounds_error=False, fill_value=0)
    flatRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, flatRadialProfileFine, bounds_error=False, fill_value=0)
    gaussianRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, gaussianRadialProfileFine, bounds_error=False, fill_value=0)
    hollowRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, hollowRadialProfileFine, bounds_error=False, fill_value=0)
    circRadialProfile = circRadialProfileInterpFunc(impactParams)
    flatRadialProfile = flatRadialProfileInterpFunc(impactParams)
    gaussianRadialProfile = gaussianRadialProfileInterpFunc(impactParams)
    hollowRadialProfile = hollowRadialProfileInterpFunc(impactParams)

    numFitsArr = np.array([1, 5, 10, 20, 50])
    errorsArr = np.zeros(shape=(4, len(numFitsArr)), dtype=float)

    # Fit the data for the maximum number of fits
    args.numFits = np.max(numFitsArr)
    circFittedModels, circErrors = fit_data(circLineIntegratedDensities, impactParams)
    flatFittedModels, flatErrors = fit_data(flatLineIntegratedDensities, impactParams)
    gaussianFittedModels, gaussianErrors = fit_data(gaussianLineIntegratedDensities, impactParams)
    hollowFittedModels, hollowErrors = fit_data(hollowLineIntegratedDensities, impactParams)

    for i in range(len(numFitsArr)):

        numFits = numFitsArr[i]

        # Get the best model for the current number of fits
        # Circular profile
        bestModelIndex = np.argmin(circErrors[:numFits])
        bestError = circErrors[bestModelIndex]
        errorsArr[0, i] = bestError
        # Flat profile
        bestModelIndex = np.argmin(flatErrors[:numFits])
        bestError = flatErrors[bestModelIndex]
        errorsArr[1, i] = bestError
        # Gaussian profile
        bestModelIndex = np.argmin(gaussianErrors[:numFits])
        bestError = gaussianErrors[bestModelIndex]
        errorsArr[2, i] = bestError
        # Hollow profile
        bestModelIndex = np.argmin(hollowErrors[:numFits])
        bestError = hollowErrors[bestModelIndex]
        errorsArr[3, i] = bestError

    
    trueErrorsArr = np.zeros_like(errorsArr, dtype=float)
    # Calculate the error against the radial profile itself
    for i in range(len(numFitsArr)):

        # Circular profile
        fittedModel = circFittedModels[i]
        fittedDensity = construct_spline_function(impactParams, 
                                                  mean=fittedModel.mean.value, 
                                                  limit=fittedModel.limit.value, 
                                                  knotPos=fittedModel.knotPos.value, 
                                                  knotVal=fittedModel.knotVal.value, 
                                                  centralVal=fittedModel.centralVal.value)
        errorArr = ((fittedDensity - circRadialProfile)/circRadialProfile)**2

        trueError = np.average(errorArr)
        trueErrorsArr[0, i] = trueError

        # Flat profile
        fittedModel = flatFittedModels[i]
        fittedDensity = construct_spline_function(impactParams, 
                                                  mean=fittedModel.mean.value, 
                                                  limit=fittedModel.limit.value, 
                                                  knotPos=fittedModel.knotPos.value, 
                                                  knotVal=fittedModel.knotVal.value, 
                                                  centralVal=fittedModel.centralVal.value)
        errorArr = ((fittedDensity - flatRadialProfile)/flatRadialProfile)**2
        trueError = np.average(errorArr)
        trueErrorsArr[1, i] = trueError

        # Gaussian profile
        fittedModel = gaussianFittedModels[i]
        fittedDensity = construct_spline_function(impactParams, 
                                                  mean=fittedModel.mean.value, 
                                                  limit=fittedModel.limit.value, 
                                                  knotPos=fittedModel.knotPos.value, 
                                                  knotVal=fittedModel.knotVal.value, 
                                                  centralVal=fittedModel.centralVal.value)
        errorArr = ((fittedDensity - gaussianRadialProfile)/gaussianRadialProfile)**2
        trueError = np.average(errorArr)
        trueErrorsArr[2, i] = trueError

        # Hollow profile
        fittedModel = hollowFittedModels[i]
        fittedDensity = construct_spline_function(impactParams, 
                                                  mean=fittedModel.mean.value, 
                                                  limit=fittedModel.limit.value, 
                                                  knotPos=fittedModel.knotPos.value, 
                                                  knotVal=fittedModel.knotVal.value, 
                                                  centralVal=fittedModel.centralVal.value)
        errorArr = ((fittedDensity - hollowRadialProfile)/hollowRadialProfile)**2
        trueError = np.average(errorArr)
        trueErrorsArr[3, i] = trueError

    # Plot the error vs number of fits
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    for i in range(errorsArr.shape[0]):
        if i == 0:
            label = 'Circular profile'
        elif i == 1:
            label = 'Flat profile'
        elif i == 2:
            label = 'Gaussian profile'
        elif i == 3:
            label = 'Hollow profile'
        
        ax.plot(numFitsArr, errorsArr[i], linewidth=5, label=label, color=f'C{i}')
        ax.scatter(numFitsArr, errorsArr[i], s=200, color=f'C{i}')

        ax.plot(numFitsArr, trueErrorsArr[i], linewidth=5, linestyle='--', color=f'C{i}')
        ax.scatter(numFitsArr, trueErrorsArr[i], s=200, marker='x', color=f'C{i}')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel('Number of fits')
    ax.set_ylabel(r'$\chi^2$ of best fit')
    ax.set_title('Effect of number of fits on error')
    ax.legend()

    plt.show(block=True)
    plt.close()

    # Plot the best fit for the maximum number of fits against the input data for each profile
    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    fig.suptitle('Best fit for maximum number of fits vs input data')

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Circular profile
    bestModelIndex = np.argmin(circErrors)
    bestModel = circFittedModels[bestModelIndex]
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knotPos=bestModel.knotPos.value, 
                                              knotVal=bestModel.knotVal.value, 
                                              centralVal=bestModel.centralVal.value)
    ax1.plot(rArr, circRadialProfileFine, label='Input data', color='red')
    ax1.plot(rArr, fittedDensity, label=f'Best fit (Error: {circErrors[bestModelIndex]:.2e})', color='blue', linewidth=3)

    ax1.set_xlabel('r [m]')
    ax1.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax1.set_title('Circular profile')
    ax1.legend()

    # Flat profile
    bestModelIndex = np.argmin(flatErrors)
    bestModel = flatFittedModels[bestModelIndex]
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knotPos=bestModel.knotPos.value, 
                                              knotVal=bestModel.knotVal.value, 
                                              centralVal=bestModel.centralVal.value)
    ax2.plot(rArr, flatRadialProfileFine, label='Input data', color='red')
    ax2.plot(rArr, fittedDensity, label=f'Best fit (Error: {flatErrors[bestModelIndex]:.2e})', color='blue', linewidth=3)

    ax2.set_xlabel('r [m]')
    ax2.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax2.set_title('Flat profile')
    ax2.legend()

    # Gaussian profile
    bestModelIndex = np.argmin(gaussianErrors)
    bestModel = gaussianFittedModels[bestModelIndex]
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knotPos=bestModel.knotPos.value, 
                                              knotVal=bestModel.knotVal.value, 
                                              centralVal=bestModel.centralVal.value)
    ax3.plot(rArr, gaussianRadialProfileFine, label='Input data', color='red')
    ax3.plot(rArr, fittedDensity, label=f'Best fit (Error: {gaussianErrors[bestModelIndex]:.2e})', color='blue', linewidth=3)

    ax3.set_xlabel('r [m]')
    ax3.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax3.set_title('Gaussian profile')
    ax3.legend()

    # Hollow profile
    bestModelIndex = np.argmin(hollowErrors)
    bestModel = hollowFittedModels[bestModelIndex]
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knotPos=bestModel.knotPos.value, 
                                              knotVal=bestModel.knotVal.value, 
                                              centralVal=bestModel.centralVal.value)
    ax4.plot(rArr, hollowRadialProfileFine, label='Input data', color='red')
    ax4.plot(rArr, fittedDensity, label=f'Best fit (Error: {hollowErrors[bestModelIndex]:.2e})', color='blue', linewidth=3)

    ax4.set_xlabel('r [m]')
    ax4.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax4.set_title('Hollow profile')
    ax4.legend()

    plt.show()

def global_optimizer_testing():

    nMax, rMax, rOffset, impactParams = 1e20, 0.15, 0.01, get_detector_positions()

    # Synthetic data
    circLineIntegratedDensities, circRadialProfileFine, rArr = synthetic_circular_profile(nMax, rMax, rOffset, impactParams)
    flatLineIntegratedDensities, flatRadialProfileFine, rArr = synthetic_flat_profile(nMax, rMax, rOffset, impactParams)
    gaussianLineIntegratedDensities, gaussianRadialProfileFine, rArr = synthetic_gaussian_profile(nMax, rMax, rOffset, impactParams)
    hollowLineIntegratedDensities, hollowRadialProfileFine, rArr = synthetic_hollow_profile(nMax, rMax, rOffset, impactParams)

    # Interpolate the radial profiles onto impactParams for error calculation later
    circRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, circRadialProfileFine, bounds_error=False, fill_value=0)
    flatRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, flatRadialProfileFine, bounds_error=False, fill_value=0)
    gaussianRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, gaussianRadialProfileFine, bounds_error=False, fill_value=0)
    hollowRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, hollowRadialProfileFine, bounds_error=False, fill_value=0)
    circRadialProfile = circRadialProfileInterpFunc(impactParams)
    flatRadialProfile = flatRadialProfileInterpFunc(impactParams)
    gaussianRadialProfile = gaussianRadialProfileInterpFunc(impactParams)
    hollowRadialProfile = hollowRadialProfileInterpFunc(impactParams)

    # Error bars for the line integrated densities (assuming 10% error)
    circErrorBars = 0.1 * circLineIntegratedDensities.reshape(-1, 1)
    flatErrorBars = 0.1 * flatLineIntegratedDensities.reshape(-1, 1)
    gaussianErrorBars = 0.1 * gaussianLineIntegratedDensities.reshape(-1, 1)
    hollowErrorBars = 0.1 * hollowLineIntegratedDensities.reshape(-1, 1)

    # Fit the data with the global optimizer
    circBestModel, circError = best_fit(0, circLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=circErrorBars)
    flatBestModel, flatError = best_fit(0, flatLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=flatErrorBars)
    gaussianBestModel, gaussianError = best_fit(0, gaussianLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=gaussianErrorBars)
    hollowBestModel, hollowError = best_fit(0, hollowLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=hollowErrorBars)

    # Calculate the error against the radial profile itself
    fittedDensity = construct_spline_function(impactParams, 
                                              mean=circBestModel.mean.value, 
                                              limit=circBestModel.limit.value, 
                                              knot1Pos=circBestModel.knot1Pos.value, 
                                              knot2Pos=circBestModel.knot2Pos.value, 
                                              knot1Val=10**circBestModel.knot1Val.value, 
                                              knot2Val=10**circBestModel.knot2Val.value, 
                                              centralVal=10**circBestModel.centralVal.value)
    errorArr = ((fittedDensity - circRadialProfile)/circRadialProfile)**2
    circError = np.average(errorArr)

    fittedDensity = construct_spline_function(impactParams, 
                                              mean=flatBestModel.mean.value, 
                                              limit=flatBestModel.limit.value, 
                                              knot1Pos=flatBestModel.knot1Pos.value, 
                                              knot2Pos=flatBestModel.knot2Pos.value, 
                                              knot1Val=10**flatBestModel.knot1Val.value, 
                                              knot2Val=10**flatBestModel.knot2Val.value, 
                                              centralVal=10**flatBestModel.centralVal.value)
    errorArr = ((fittedDensity - flatRadialProfile)/flatRadialProfile)**2
    flatError = np.average(errorArr)

    fittedDensity = construct_spline_function(impactParams, 
                                              mean=gaussianBestModel.mean.value, 
                                              limit=gaussianBestModel.limit.value, 
                                              knot1Pos=gaussianBestModel.knot1Pos.value, 
                                              knot2Pos=gaussianBestModel.knot2Pos.value, 
                                              knot1Val=10**gaussianBestModel.knot1Val.value, 
                                              knot2Val=10**gaussianBestModel.knot2Val.value, 
                                              centralVal=10**gaussianBestModel.centralVal.value)
    errorArr = ((fittedDensity - gaussianRadialProfile)/gaussianRadialProfile)**2
    gaussianError = np.average(errorArr)

    fittedDensity = construct_spline_function(impactParams, 
                                              mean=hollowBestModel.mean.value, 
                                              limit=hollowBestModel.limit.value, 
                                              knot1Pos=hollowBestModel.knot1Pos.value,
                                              knot2Pos=hollowBestModel.knot2Pos.value,
                                              knot1Val=10**hollowBestModel.knot1Val.value, 
                                              knot2Val=10**hollowBestModel.knot2Val.value, 
                                              centralVal=10**hollowBestModel.centralVal.value)
    errorArr = ((fittedDensity - hollowRadialProfile)/hollowRadialProfile)**2
    hollowError = np.average(errorArr)

    # Plot the best fit against the input data for each profile
    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    fig.suptitle('Best fit vs input data (Global -> Local optimization)')

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Circular profile
    bestModel = circBestModel
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knot1Pos=bestModel.knot1Pos.value, 
                                              knot2Pos=bestModel.knot2Pos.value, 
                                              knot1Val=10**bestModel.knot1Val.value, 
                                              knot2Val=10**bestModel.knot2Val.value, 
                                              centralVal=10**bestModel.centralVal.value)
    ax1.plot(rArr, circRadialProfileFine, label='Input data', color='red')
    ax1.plot(rArr, fittedDensity, label=f'Best fit (Error: {circError:.2e})', color='blue', linewidth=3)

    ax1.set_xlabel('r [m]')
    ax1.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax1.set_title('Circular profile')
    ax1.legend()

    # Flat profile
    bestModel = flatBestModel
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knot1Pos=bestModel.knot1Pos.value, 
                                              knot2Pos=bestModel.knot2Pos.value, 
                                              knot1Val=10**bestModel.knot1Val.value, 
                                              knot2Val=10**bestModel.knot2Val.value, 
                                              centralVal=10**bestModel.centralVal.value)
    ax2.plot(rArr, flatRadialProfileFine, label='Input data', color='red')
    ax2.plot(rArr, fittedDensity, label=f'Best fit (Error: {flatError:.2e})', color='blue', linewidth=3)

    ax2.set_xlabel('r [m]')
    ax2.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax2.set_title('Flat profile')
    ax2.legend()

    # Gaussian profile
    bestModel = gaussianBestModel
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knot1Pos=bestModel.knot1Pos.value, 
                                              knot2Pos=bestModel.knot2Pos.value, 
                                              knot1Val=10**bestModel.knot1Val.value, 
                                              knot2Val=10**bestModel.knot2Val.value, 
                                              centralVal=10**bestModel.centralVal.value)
    ax3.plot(rArr, gaussianRadialProfileFine, label='Input data', color='red')
    ax3.plot(rArr, fittedDensity, label=f'Best fit (Error: {gaussianError:.2e})', color='blue', linewidth=3)

    ax3.set_xlabel('r [m]')
    ax3.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax3.set_title('Gaussian profile')
    ax3.legend()

    # Hollow profile
    bestModel = hollowBestModel
    fittedDensity = construct_spline_function(rArr, 
                                              mean=bestModel.mean.value, 
                                              limit=bestModel.limit.value, 
                                              knot1Pos=bestModel.knot1Pos.value, 
                                              knot2Pos=bestModel.knot2Pos.value, 
                                              knot1Val=10**bestModel.knot1Val.value, 
                                              knot2Val=10**bestModel.knot2Val.value, 
                                              centralVal=10**bestModel.centralVal.value)
    ax4.plot(rArr, hollowRadialProfileFine, label='Input data', color='red')
    ax4.plot(rArr, fittedDensity, label=f'Best fit (Error: {hollowError:.2e})', color='blue', linewidth=3)

    ax4.set_xlabel('r [m]')
    ax4.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax4.set_title('Hollow profile')
    ax4.legend()

    plt.show()

def effect_of_noise():

    nMax, rMax, rOffset, impactParams = 1e20, 0.15, 0.01, get_detector_positions()

    noiseArr = np.array([0, 0.01, 0.05, 0.1, 0.25])   # noise level as a fraction of the maximum line integrated density

    # Fitted models and errors for each noise level
    circFittedModels = []
    circErrors = []
    flatFittedModels = []
    flatErrors = []
    gaussianFittedModels = []
    gaussianErrors = []
    hollowFittedModels = []
    hollowErrors = []    

    # Go over each noise level and fit the data
    for i in range(len(noiseArr)):

        noiseLevel = noiseArr[i]

        print(f'Fitting data with noise level: {noiseLevel:.2f}')

        # Synthetic data
        circLineIntegratedDensities, circRadialProfileFine, rArr = synthetic_circular_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)
        flatLineIntegratedDensities, flatRadialProfileFine, rArr = synthetic_flat_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)
        gaussianLineIntegratedDensities, gaussianRadialProfileFine, rArr = synthetic_gaussian_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)
        hollowLineIntegratedDensities, hollowRadialProfileFine, rArr = synthetic_hollow_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)

        # Interpolate the radial profiles onto impactParams for error calculation later
        circRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, circRadialProfileFine, bounds_error=False, fill_value=0)
        flatRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, flatRadialProfileFine, bounds_error=False, fill_value=0)
        gaussianRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, gaussianRadialProfileFine, bounds_error=False, fill_value=0)
        hollowRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, hollowRadialProfileFine, bounds_error=False, fill_value=0)
        circRadialProfile = circRadialProfileInterpFunc(impactParams)
        flatRadialProfile = flatRadialProfileInterpFunc(impactParams)
        gaussianRadialProfile = gaussianRadialProfileInterpFunc(impactParams)
        hollowRadialProfile = hollowRadialProfileInterpFunc(impactParams)

        # Error bars for the line integrated densities
        circErrorBars = 0.1 * circLineIntegratedDensities.reshape(-1, 1)
        flatErrorBars = 0.1 * flatLineIntegratedDensities.reshape(-1, 1)
        gaussianErrorBars = 0.1 * gaussianLineIntegratedDensities.reshape(-1, 1)
        hollowErrorBars = 0.1 * hollowLineIntegratedDensities.reshape(-1, 1)

        # Fit the data
        circBestModel, circError = best_fit(0, circLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=circErrorBars)
        flatBestModel, flatError = best_fit(0, flatLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=flatErrorBars)
        gaussianBestModel, gaussianError = best_fit(0, gaussianLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=gaussianErrorBars)
        hollowBestModel, hollowError = best_fit(0, hollowLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=hollowErrorBars)

        # Calculate the error against the radial profile itself
        fittedDensity = construct_spline_function(impactParams, 
                                                mean=circBestModel.mean.value, 
                                                limit=circBestModel.limit.value, 
                                                knot1Pos=circBestModel.knot1Pos.value, 
                                                knot2Pos=circBestModel.knot2Pos.value, 
                                                knot1Val=10**circBestModel.knot1Val.value, 
                                                knot2Val=10**circBestModel.knot2Val.value, 
                                                centralVal=10**circBestModel.centralVal.value)
        errorArr = ((fittedDensity - circRadialProfile)/circRadialProfile)**2
        circError = np.average(errorArr)

        fittedDensity = construct_spline_function(impactParams, 
                                                mean=flatBestModel.mean.value, 
                                                limit=flatBestModel.limit.value, 
                                                knot1Pos=flatBestModel.knot1Pos.value, 
                                                knot2Pos=flatBestModel.knot2Pos.value, 
                                                knot1Val=10**flatBestModel.knot1Val.value, 
                                                knot2Val=10**flatBestModel.knot2Val.value, 
                                                centralVal=10**flatBestModel.centralVal.value)
        errorArr = ((fittedDensity - flatRadialProfile)/flatRadialProfile)**2
        flatError = np.average(errorArr)

        fittedDensity = construct_spline_function(impactParams, 
                                                mean=gaussianBestModel.mean.value, 
                                                limit=gaussianBestModel.limit.value, 
                                                knot1Pos=gaussianBestModel.knot1Pos.value, 
                                                knot2Pos=gaussianBestModel.knot2Pos.value, 
                                                knot1Val=10**gaussianBestModel.knot1Val.value, 
                                                knot2Val=10**gaussianBestModel.knot2Val.value, 
                                                centralVal=10**gaussianBestModel.centralVal.value)
        errorArr = ((fittedDensity - gaussianRadialProfile)/gaussianRadialProfile)**2
        gaussianError = np.average(errorArr)

        fittedDensity = construct_spline_function(impactParams, 
                                                mean=hollowBestModel.mean.value, 
                                                limit=hollowBestModel.limit.value, 
                                                knot1Pos=hollowBestModel.knot1Pos.value,
                                                knot2Pos=hollowBestModel.knot2Pos.value,
                                                knot1Val=10**hollowBestModel.knot1Val.value, 
                                                knot2Val=10**hollowBestModel.knot2Val.value, 
                                                centralVal=10**hollowBestModel.centralVal.value)
        errorArr = ((fittedDensity - hollowRadialProfile)/hollowRadialProfile)**2
        hollowError = np.average(errorArr)

        circFittedModels.append(circBestModel)
        circErrors.append(circError)
        flatFittedModels.append(flatBestModel)
        flatErrors.append(flatError)
        gaussianFittedModels.append(gaussianBestModel)
        gaussianErrors.append(gaussianError)
        hollowFittedModels.append(hollowBestModel)
        hollowErrors.append(hollowError)

    # Convert the errors to numpy arrays for plotting
    circErrors = np.array(circErrors)
    flatErrors = np.array(flatErrors)
    gaussianErrors = np.array(gaussianErrors)
    hollowErrors = np.array(hollowErrors)

    # Plot the best fit against the input data for each profile
    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    fig.suptitle('Best fit vs input data (Global -> Local optimization)')

    # Color each noise level
    cmap = plt.get_cmap('viridis', len(noiseArr)).colors

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Circular profile
    ax1.plot(rArr, circRadialProfileFine, label='Input data', color='red', zorder=10)
    # Flat profile
    ax2.plot(rArr, flatRadialProfileFine, label='Input data', color='red', zorder=10)
    # Gaussian profile
    ax3.plot(rArr, gaussianRadialProfileFine, label='Input data', color='red', zorder=10)
    # Hollow profile
    ax4.plot(rArr, hollowRadialProfileFine, label='Input data', color='red', zorder=10)

    for i in range(len(noiseArr)):

        noiseLevel = noiseArr[i]

        # Circular profile
        bestModel = circFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value, 
                                                  knot2Val=10**bestModel.knot2Val.value, 
                                                  centralVal=10**bestModel.centralVal.value)
        ax1.plot(rArr, fittedDensity, 
                 label=f'Noise: {noiseLevel:.2f}, Error: {circErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

        # Flat profile
        bestModel = flatFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value,
                                                  knot2Val=10**bestModel.knot2Val.value,
                                                  centralVal=10**bestModel.centralVal.value)
        ax2.plot(rArr, fittedDensity, 
                 label=f'Error: {flatErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

        # Gaussian profile
        bestModel = gaussianFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value,
                                                  knot2Val=10**bestModel.knot2Val.value,
                                                  centralVal=10**bestModel.centralVal.value)
        ax3.plot(rArr, fittedDensity, 
                 label=f'Error: {gaussianErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

        # Hollow profile
        bestModel = hollowFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value,
                                                  knot2Val=10**bestModel.knot2Val.value,
                                                  centralVal=10**bestModel.centralVal.value)
        ax4.plot(rArr, fittedDensity, 
                 label=f'Error: {hollowErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

    ax1.set_xlabel('r [m]')
    ax1.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax1.set_title('Circular profile')
    ax1.legend()

    ax2.set_xlabel('r [m]')
    ax2.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax2.set_title('Flat profile')
    ax2.legend()

    ax3.set_xlabel('r [m]')
    ax3.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax3.set_title('Gaussian profile')
    ax3.legend()

    ax4.set_xlabel('r [m]')
    ax4.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax4.set_title('Hollow profile')
    ax4.legend()

    plt.show()

    return

def effect_of_smoothing():

    nMax, rMax, rOffset, impactParams = 1e20, 0.15, 0.01, get_detector_positions()

    smoothingSigmaArr = np.array([0.01, 1, 5, 10])   # sigma for Gaussian smoothing

    # Fitted models and errors for each noise level
    circFittedModels = []
    circErrors = []
    flatFittedModels = []
    flatErrors = []
    gaussianFittedModels = []
    gaussianErrors = []
    hollowFittedModels = []
    hollowErrors = []    

    # Go over each noise level and fit the data
    for i in range(len(smoothingSigmaArr)):

        smoothingSigma = smoothingSigmaArr[i]

        print(f'Fitting data with smoothing sigma: {smoothingSigma:.2f}')

        # Synthetic data
        circLineIntegratedDensities, circRadialProfileFine, rArr = synthetic_circular_profile(nMax, rMax, rOffset, impactParams, smoothingSigma=smoothingSigma)
        flatLineIntegratedDensities, flatRadialProfileFine, rArr = synthetic_flat_profile(nMax, rMax, rOffset, impactParams, smoothingSigma=smoothingSigma)
        gaussianLineIntegratedDensities, gaussianRadialProfileFine, rArr = synthetic_gaussian_profile(nMax, rMax, rOffset, impactParams, smoothingSigma=smoothingSigma)
        hollowLineIntegratedDensities, hollowRadialProfileFine, rArr = synthetic_hollow_profile(nMax, rMax, rOffset, impactParams, smoothingSigma=smoothingSigma)

        # Interpolate the radial profiles onto impactParams for error calculation later
        circRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, circRadialProfileFine, bounds_error=False, fill_value=0)
        flatRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, flatRadialProfileFine, bounds_error=False, fill_value=0)
        gaussianRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, gaussianRadialProfileFine, bounds_error=False, fill_value=0)
        hollowRadialProfileInterpFunc = sc.interpolate.interp1d(rArr, hollowRadialProfileFine, bounds_error=False, fill_value=0)
        circRadialProfile = circRadialProfileInterpFunc(impactParams)
        flatRadialProfile = flatRadialProfileInterpFunc(impactParams)
        gaussianRadialProfile = gaussianRadialProfileInterpFunc(impactParams)
        hollowRadialProfile = hollowRadialProfileInterpFunc(impactParams)

        # Error bars for the line integrated densities
        circErrorBars = 0.1 * circLineIntegratedDensities.reshape(-1, 1)
        flatErrorBars = 0.1 * flatLineIntegratedDensities.reshape(-1, 1)
        gaussianErrorBars = 0.1 * gaussianLineIntegratedDensities.reshape(-1, 1)
        hollowErrorBars = 0.1 * hollowLineIntegratedDensities.reshape(-1, 1)

        # Fit the data
        circBestModel, circError = best_fit(0, circLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=circErrorBars)
        flatBestModel, flatError = best_fit(0, flatLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=flatErrorBars)
        gaussianBestModel, gaussianError = best_fit(0, gaussianLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=gaussianErrorBars)
        hollowBestModel, hollowError = best_fit(0, hollowLineIntegratedDensities.reshape(-1, 1), impactParams, errorBars=hollowErrorBars)

        # Calculate the error against the radial profile itself
        fittedDensity = construct_spline_function(impactParams, 
                                                mean=circBestModel.mean.value, 
                                                limit=circBestModel.limit.value, 
                                                knot1Pos=circBestModel.knot1Pos.value, 
                                                knot2Pos=circBestModel.knot2Pos.value, 
                                                knot1Val=10**circBestModel.knot1Val.value, 
                                                knot2Val=10**circBestModel.knot2Val.value, 
                                                centralVal=10**circBestModel.centralVal.value)
        errorArr = ((fittedDensity - circRadialProfile)/circRadialProfile)**2
        circError = np.average(errorArr)

        fittedDensity = construct_spline_function(impactParams, 
                                                mean=flatBestModel.mean.value, 
                                                limit=flatBestModel.limit.value, 
                                                knot1Pos=flatBestModel.knot1Pos.value, 
                                                knot2Pos=flatBestModel.knot2Pos.value, 
                                                knot1Val=10**flatBestModel.knot1Val.value, 
                                                knot2Val=10**flatBestModel.knot2Val.value, 
                                                centralVal=10**flatBestModel.centralVal.value)
        errorArr = ((fittedDensity - flatRadialProfile)/flatRadialProfile)**2
        flatError = np.average(errorArr)

        fittedDensity = construct_spline_function(impactParams, 
                                                mean=gaussianBestModel.mean.value, 
                                                limit=gaussianBestModel.limit.value, 
                                                knot1Pos=gaussianBestModel.knot1Pos.value, 
                                                knot2Pos=gaussianBestModel.knot2Pos.value, 
                                                knot1Val=10**gaussianBestModel.knot1Val.value, 
                                                knot2Val=10**gaussianBestModel.knot2Val.value, 
                                                centralVal=10**gaussianBestModel.centralVal.value)
        errorArr = ((fittedDensity - gaussianRadialProfile)/gaussianRadialProfile)**2
        gaussianError = np.average(errorArr)

        fittedDensity = construct_spline_function(impactParams, 
                                                mean=hollowBestModel.mean.value, 
                                                limit=hollowBestModel.limit.value, 
                                                knot1Pos=hollowBestModel.knot1Pos.value,
                                                knot2Pos=hollowBestModel.knot2Pos.value,
                                                knot1Val=10**hollowBestModel.knot1Val.value, 
                                                knot2Val=10**hollowBestModel.knot2Val.value, 
                                                centralVal=10**hollowBestModel.centralVal.value)
        errorArr = ((fittedDensity - hollowRadialProfile)/hollowRadialProfile)**2
        hollowError = np.average(errorArr)

        circFittedModels.append(circBestModel)
        circErrors.append(circError)
        flatFittedModels.append(flatBestModel)
        flatErrors.append(flatError)
        gaussianFittedModels.append(gaussianBestModel)
        gaussianErrors.append(gaussianError)
        hollowFittedModels.append(hollowBestModel)
        hollowErrors.append(hollowError)

    # Convert the errors to numpy arrays for plotting
    circErrors = np.array(circErrors)
    flatErrors = np.array(flatErrors)
    gaussianErrors = np.array(gaussianErrors)
    hollowErrors = np.array(hollowErrors)

    # Plot the best fit against the input data for each profile
    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    fig.suptitle('Best fit vs input data (Global -> Local optimization)')

    # Color each noise level
    cmap = plt.get_cmap('viridis', len(smoothingSigmaArr)).colors

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Unsmoothed data for plotting
    _, circRadialProfileFine, rArr = synthetic_circular_profile(nMax, rMax, rOffset, impactParams)
    _, flatRadialProfileFine, rArr = synthetic_flat_profile(nMax, rMax, rOffset, impactParams)
    _, gaussianRadialProfileFine, rArr = synthetic_gaussian_profile(nMax, rMax, rOffset, impactParams)
    _, hollowRadialProfileFine, rArr = synthetic_hollow_profile(nMax, rMax, rOffset, impactParams)

    # Circular profile
    ax1.plot(rArr, circRadialProfileFine, label='Input data', color='red', zorder=10)
    # Flat profile
    ax2.plot(rArr, flatRadialProfileFine, label='Input data', color='red', zorder=10)
    # Gaussian profile
    ax3.plot(rArr, gaussianRadialProfileFine, label='Input data', color='red', zorder=10)
    # Hollow profile
    ax4.plot(rArr, hollowRadialProfileFine, label='Input data', color='red', zorder=10)

    for i in range(len(smoothingSigmaArr)):

        smoothingSigma = smoothingSigmaArr[i]

        # Circular profile
        bestModel = circFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value, 
                                                  knot2Val=10**bestModel.knot2Val.value, 
                                                  centralVal=10**bestModel.centralVal.value)
        ax1.plot(rArr, fittedDensity, 
                 label=f'Smoothing: {smoothingSigma:.2f}, Error: {circErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

        # Flat profile
        bestModel = flatFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value,
                                                  knot2Val=10**bestModel.knot2Val.value,
                                                  centralVal=10**bestModel.centralVal.value)
        ax2.plot(rArr, fittedDensity, 
                 label=f'Error: {flatErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

        # Gaussian profile
        bestModel = gaussianFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value,
                                                  knot2Val=10**bestModel.knot2Val.value,
                                                  centralVal=10**bestModel.centralVal.value)
        ax3.plot(rArr, fittedDensity, 
                 label=f'Error: {gaussianErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

        # Hollow profile
        bestModel = hollowFittedModels[i]
        fittedDensity = construct_spline_function(rArr, 
                                                  mean=bestModel.mean.value, 
                                                  limit=bestModel.limit.value, 
                                                  knot1Pos=bestModel.knot1Pos.value, 
                                                  knot2Pos=bestModel.knot2Pos.value, 
                                                  knot1Val=10**bestModel.knot1Val.value,
                                                  knot2Val=10**bestModel.knot2Val.value,
                                                  centralVal=10**bestModel.centralVal.value)
        ax4.plot(rArr, fittedDensity, 
                 label=f'Error: {hollowErrors[i]:.2e}', 
                 linewidth=3,
                 color=cmap[i])

    ax1.set_xlabel('r [m]')
    ax1.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax1.set_title('Circular profile')
    ax1.legend()

    ax2.set_xlabel('r [m]')
    ax2.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax2.set_title('Flat profile')
    ax2.legend()

    ax3.set_xlabel('r [m]')
    ax3.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax3.set_title('Gaussian profile')
    ax3.legend()

    ax4.set_xlabel('r [m]')
    ax4.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax4.set_title('Hollow profile')
    ax4.legend()

    plt.show()

    return

def noise_vs_centralVal():

    nMax, rMax, rOffset, impactParams = 1e20, 0.15, 0.01, get_detector_positions()

    # Noise array
    noiseArr = np.array([0, 0.01, 0.05, 0.1, 0.25])   # noise level as a fraction of the maximum line integrated density

    # Number of tries at each noise level
    nTries = 5

    # Array to hold the best fit centralVal for each try and noise level
    circCentralValArr = np.zeros((len(noiseArr), nTries))
    flatCentralValArr = np.zeros((len(noiseArr), nTries))
    gaussianCentralValArr = np.zeros((len(noiseArr), nTries))
    hollowCentralValArr = np.zeros((len(noiseArr), nTries))

    # Go over each noise level and try fitting the data multiple times
    for i in range(len(noiseArr)):

        noiseLevel = noiseArr[i]

        print(f'Fitting data with noise level: {noiseLevel:.2f}')

        for j in range(nTries):

            # Synthetic data
            circularLineIntegratedDensities, _, _ = synthetic_circular_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)
            flatLineIntegratedDensities, _, _ = synthetic_flat_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)
            gaussianLineIntegratedDensities, _, _ = synthetic_gaussian_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)
            hollowLineIntegratedDensities, _, _ = synthetic_hollow_profile(nMax, rMax, rOffset, impactParams, noiseLevel=noiseLevel)

            # Error bars for the line integrated densities
            circErrorBars = 0.1 * circularLineIntegratedDensities.reshape(-1, 1)
            flatErrorBars = 0.1 * flatLineIntegratedDensities.reshape(-1, 1)
            gaussianErrorBars = 0.1 * gaussianLineIntegratedDensities.reshape(-1, 1)
            hollowErrorBars = 0.1 * hollowLineIntegratedDensities.reshape(-1, 1)

            # Fit the data
            bestModelCirc, _ = best_fit(0, circularLineIntegratedDensities.reshape(-1, 1), get_detector_positions(), errorBars=circErrorBars)
            bestModelFlat, _ = best_fit(0, flatLineIntegratedDensities.reshape(-1, 1), get_detector_positions(), errorBars=flatErrorBars)
            bestModelGaussian, _ = best_fit(0, gaussianLineIntegratedDensities.reshape(-1, 1), get_detector_positions(), errorBars=gaussianErrorBars)
            bestModelHollow, _ = best_fit(0, hollowLineIntegratedDensities.reshape(-1, 1), get_detector_positions(), errorBars=hollowErrorBars)

            circCentralValArr[i, j] = 10**bestModelCirc.centralVal.value
            flatCentralValArr[i, j] = 10**bestModelFlat.centralVal.value
            gaussianCentralValArr[i, j] = 10**bestModelGaussian.centralVal.value
            hollowCentralValArr[i, j] = 10**bestModelHollow.centralVal.value

    circCentralValMean = np.mean(circCentralValArr, axis=1)
    circCentralValStd = np.std(circCentralValArr, axis=1)

    flatCentralValMean = np.mean(flatCentralValArr, axis=1)
    flatCentralValStd = np.std(flatCentralValArr, axis=1)

    gaussianCentralValMean = np.mean(gaussianCentralValArr, axis=1)
    gaussianCentralValStd = np.std(gaussianCentralValArr, axis=1)

    hollowCentralValMean = np.mean(hollowCentralValArr, axis=1)
    hollowCentralValStd = np.std(hollowCentralValArr, axis=1)

    # Plot the centralVal vs noise level
    fig = plt.figure(figsize=(10, 6), tight_layout=True)

    ax = fig.add_subplot(111)

    ax.errorbar(noiseArr*1e2, circCentralValMean, yerr=circCentralValStd, fmt='o-', capsize=5, label='Circular')
    ax.errorbar(noiseArr*1e2, flatCentralValMean, yerr=flatCentralValStd, fmt='o-', capsize=5, label='Flat')
    ax.errorbar(noiseArr*1e2, gaussianCentralValMean, yerr=gaussianCentralValStd, fmt='o-', capsize=5, label='Gaussian')
    ax.errorbar(noiseArr*1e2, hollowCentralValMean, yerr=hollowCentralValStd, fmt='o-', capsize=5, label='Hollow')
    ax.set_xlabel(r'Noise level on $\int n_e \cdot dl$ [%]')
    ax.set_ylabel('Central Value')
    ax.set_title('Effect of noise on central value')
    ax.legend()

    plt.show()

    return

def fit_shot():

    # Load the real SEE data
    impactParams, shinethruData, shinethruDataErr, timeArr = get_shinethru_data()

    # Times at which we want to fit the data
    timePointsFit = np.array([7,8,9,10,11]) * 1e-3
    # timePointsFit = np.array([8]) * 1e-3

    timeIndices = np.zeros_like(timePointsFit, dtype=int)
    for i in range(len(timePointsFit)):

        timePoint = timePointsFit[i]

        # Find the index of the closest time point in the data
        timeIndex = np.argmin(np.abs(timeArr - timePoint))
        timeIndices[i] = timeIndex

    # SEE data at those time points
    dataToFit = shinethruData[:, timeIndices]
    # Error bars for the data at those time points
    dataToFitErr = shinethruDataErr[:, timeIndices]

    # Go over each time point and fit the data
    bestModels = []
    errors = []
    maxLimit = 0

    for i in range(len(timePointsFit)):

        print(f'Fitting data at time = {timePointsFit[i]*1e3:.1f} ms...')

        bestModel, error = best_fit(i, dataToFit, impactParams, errorBars=dataToFitErr[:, i])

        print(bestModel)

        bestModels.append(bestModel)
        errors.append(error)

        currLimit = bestModel.limit.value
        if currLimit > maxLimit:
            maxLimit = currLimit

    # Plot the best fits against the data for each time point
    fig = plt.figure(figsize=(15, 10), tight_layout=True)

    # Line integrated data and fits
    ax1 = fig.add_subplot(211)
    # Radial density profiles
    ax2 = fig.add_subplot(212)

    # Color each time point on a colormap
    cmap = plt.get_cmap('viridis', len(timePointsFit)).colors

    for i in range(len(timePointsFit)):

        timePoint = timePointsFit[i]
        timeIndex = timeIndices[i]

        # Line integrated data and fits
        ax1.errorbar(impactParams, shinethruData[:, timeIndex], 
                     yerr=shinethruDataErr[:, timeIndex],
                     fmt='o',
                     ms=5,
                     color=cmap[i])

        xArr = np.linspace(-maxLimit*1.5, maxLimit*1.5, 100)

        ax1.plot(xArr, bestModels[i](xArr),
                 label=f'Time = {timePoint*1e3:.1f} ms', 
                 linewidth=3,
                 color=cmap[i])

        # Radial density profiles
        fittedModel = bestModels[i]
        fittedDensity = construct_spline_function(xArr, 
                                                  mean=fittedModel.mean.value, 
                                                  limit=fittedModel.limit.value, 
                                                  knot1Pos=fittedModel.knot1Pos.value, 
                                                  knot2Pos=fittedModel.knot2Pos.value, 
                                                  knot1Val=10**fittedModel.knot1Val.value, 
                                                  knot2Val=10**fittedModel.knot2Val.value, 
                                                  centralVal=10**fittedModel.centralVal.value)
        ax2.plot(xArr, fittedDensity, 
                 label=f'Time = {timePoint*1e3:.1f} ms', 
                 linewidth=3,
                 color=cmap[i])

        # Knot positions
        knotPosArr = np.array([0, fittedModel.knot1Pos.value, fittedModel.knot2Pos.value, fittedModel.limit.value])
        knotPosArr = np.sort(knotPosArr) + fittedModel.mean.value
        knotValArr = np.array([10**fittedModel.centralVal.value, 10**fittedModel.knot1Val.value, 10**fittedModel.knot2Val.value, 0])
        ax2.scatter(knotPosArr, knotValArr,
                    color=cmap[i], s=100, zorder=5)

    ax1.set_ylim(0, np.max(dataToFit)*1.2)
    ax1.set_xlabel('Impact parameter [m]')
    ax1.set_ylabel(r'$\int n_e \cdot dl$ [m$^{-2}$]')
    ax1.set_title('Line-integrated density profiles')
    # ax1.legend()

    ax2.set_xlabel('r [m]')
    ax2.set_ylabel(r'$n_e$ [m$^{-3}$]')
    ax2.set_title('Radial density profiles')
    ax2.legend()

    plt.show()

    return

if __name__ == '__main__':

    # Parse the command line arguments
    global args
    args = parseArgs()

    # Get the impact parameters
    impactParams = get_detector_positions()

    # Make the synthetic circular profile
    nMax = 1e20
    rMax = 0.15
    rOffset = 0.05
    # lineIntegratedDensities, radialProfile, rArr = synthetic_circular_profile(nMax, rMax, rOffset, impactParams, noiseLevel=0.1, smoothingSigma=0.01)
    # lineIntegratedDensities, radialProfile, rArr = synthetic_flat_profile(nMax, rMax, rOffset,impactParams, noiseLevel=0.1)
    # lineIntegratedDensities, radialProfile, rArr = synthetic_gaussian_profile(nMax, rMax, rOffset,impactParams, noiseLevel=0.1)
    # lineIntegratedDensities, radialProfile, rArr = synthetic_hollow_profile(nMax, rMax, rOffset,impactParams, noiseLevel=0.1)

    # spline1 = construct_spline_function(np.linspace(np.min(impactParams)*1.5, np.max(impactParams)*1.5, 100), mean=0.03, limit=0.15, knot1Pos=0.1, knot2Pos=0.14, knot1Val=1, knot2Val=0, centralVal=0.5)

    # Fit the data
    # fit_vs_input(synthetic_hollow_profile)

    # Effect of the number of fits on the error
    # effect_of_fit_count()

    # Test the global optimizer
    # global_optimizer_testing()

    # Test the effect of noise on the fitting
    # effect_of_noise()

    # Test the effect of smoothing on the fitting
    # effect_of_smoothing()

    # Test the effect of noise on the central value of the profile
    noise_vs_centralVal()

    # Load the real SEE data
    # impactParams, shinethruData, shinethruDataErr, timeArr = get_shinethru_data()

    # Fit the SEE data at different time points
    # fit_shot()