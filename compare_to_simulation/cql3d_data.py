"""
This script calculates the line-integrated density for the SEE detectors from a synthetic diagnostic implemented on a CQL3D output file.
"""

import os
import sys
import pickle
import subprocess
import scipy as sc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

# Make the font size larger
plt.rcParams.update({'font.size': 18})

global plotDest, dataDest
plotDest = '/home/sanwalka/shinethru/plots/'
dataDest = '/home/sanwalka/shinethru/data/'
dataDest = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/18.000/components/fp__cql3dm_4/'

def species_labels(filename):
    """
    This function generates an array with the labels for each 'general' species
    That is, a species whos distribution function has been explicitly 
    calculated by CQL3D.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.

    Returns
    -------
    speciesLabels : array
        Labels of all the general species.
    """
    
    # Open the dataset
    ds = xr.open_dataset(filename,
                         decode_timedelta=False)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    # Name of each species and specification (general, maxwellian etc.)
    kspeci = np.array(ds['kspeci'].values).astype('U')
    # Remove all trailing whitespace
    kspeci = np.char.rstrip(kspeci)
    
    # =========================================================================
    # Generate the species labels
    # =========================================================================
    
    # Array to store species labels
    speciesLabels=[]
    
    # Get a slice of kspeci which just has the label and type
    kspeciSlice = kspeci
    # Append correct names to the labelling array
    for i in range(len(kspeciSlice)):

        if kspeciSlice[i,1] == 'general':
            
            # Label in CQL3D
            cqlLabel = kspeciSlice[i,0]
            
            # Come up with a nicer label
            niceLabel = ''
            if cqlLabel=='d' or cqlLabel=='D' or cqlLabel=='Deuterium' or cqlLabel=='deuterium':
                niceLabel='D'
            elif cqlLabel=='t' or cqlLabel=='T' or cqlLabel=='Tritium' or cqlLabel=='tritium':
                niceLabel='T'
            elif cqlLabel=='e' or cqlLabel=='E' or cqlLabel=='Electron' or cqlLabel=='electron':
                niceLabel='e'
                
            #Add it to the array
            speciesLabels.append(niceLabel)
    
    return speciesLabels

def ion_dens(filename, makeplot=False):

    # Open the file
    ds = xr.open_dataset(filename,
                         decode_timedelta=False)
    
    # Density [Time x R x Z x Species]
    dens = ds['densz1'].values
    # [cm^-3] to [m^-3]
    dens *= 1e6

    # 1st one is ions, the 2nd is electrons.
    dens = dens[:, :, :, 0]

    # Major radius of z points (=r)
    # [cm] to [m]
    solrz = ds['solrz'].values * 1e-2
    
    # Height of z points (=z)
    # [cm] to [m]
    solzz = ds['solzz'].values * 1e-2

    # Time array
    time = ds['time'].values

    if makeplot:

        import matplotlib.animation as animation

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        # Fix the color levels across all timesteps so the colorbar is consistent
        levels = np.linspace(np.min(dens), np.max(dens), 100)

        pltObj = ax.contourf(solzz, solrz, dens[0], levels=levels, cmap='inferno')

        cbar = fig.colorbar(pltObj)
        cbar.set_label(r'Density [m$^{-3}$]')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        title = ax.set_title(f'Time = {time[0]*1e3:.4g} ms')

        def update(frame):
            ax.clear()
            ax.contourf(solzz, solrz, dens[frame], levels=levels, cmap='inferno')
            ax.set_xlabel('Z [m]')
            ax.set_ylabel('R [m]')
            ax.set_title(f'Time = {time[frame]*1e3:.4g} ms')

        anim = animation.FuncAnimation(fig, update, frames=dens.shape[0],
                                        interval=200, blit=False)

        plt.show()

    return dens, solrz, solzz, time

def generate_single_interpolation(plasmaDens, solrz, solzz, makeplot=False):
    """
    This function loads the density profile from a given .nc file and creates an interpolation function.

    Parameters
    ----------
    plasmaDens : np.array
        2D density profile
    solrz : np.array
        r-values
    solzz : np.array
        z-values
    makeplot : bool, optional
        Whether to plot the density profile. The default is False.
    
    Returns
    -------
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]    
    """

    # Extend the data to +-z
    solrz_full = np.concatenate((solrz[:, :0:-1], solrz), axis=1)
    solzz_full = np.concatenate((-solzz[:, :0:-1], solzz), axis=1)
    plasmaDens_full = np.concatenate((plasmaDens[:, :0:-1], plasmaDens), axis=1)

    nr, nz = solrz_full.shape

    def grid_triangles(nr, nz):
        ir, iz = np.meshgrid(np.arange(nr - 1), np.arange(nz - 1), indexing='ij')
        i0 = (ir * nz + iz).ravel()
        i1 = i0 + 1
        i2 = i0 + nz
        i3 = i2 + 1
        return np.concatenate((np.column_stack((i0, i1, i2)),
                                np.column_stack((i1, i3, i2))))

    rVals = solrz_full.ravel()
    zVals = solzz_full.ravel()
    densVals = plasmaDens_full.ravel()

    triangles = grid_triangles(nr, nz)

    triangulation = mtri.Triangulation(zVals, rVals, triangles)
    triInterp = mtri.LinearTriInterpolator(triangulation, densVals)

    def interpFunc(points):
        points = np.atleast_2d(points)
        densQ = triInterp(points[:, 1], points[:, 0])
        return np.ma.filled(densQ, 0.0)

    if makeplot == True:

        # Plot the density profile
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        rArr = np.linspace(0, 0.5, 100)
        zArr = np.linspace(-1.5, 1.5, 100)
        R, Z = np.meshgrid(rArr, zArr)

        points = np.array([R.flatten(), Z.flatten()]).T
        densityArr = interpFunc(points).reshape(R.shape)

        pltObj = ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')

        cbar = fig.colorbar(pltObj)
        cbar.set_label(r'Density [m$^{-3}$]')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')

        plt.show()

    return interpFunc

def generate_interpolation_functions(filename, makeplot=False):

    # Load the data
    dens, solrz, solzz, time = ion_dens(filename)

    interpFuncList = []

    # Go over each timestep and generate an interpolation function
    for i in range(len(time)):

        interpFuncList.append(generate_single_interpolation(dens[i], solrz, solzz))

    if makeplot:

        import matplotlib.animation as animation

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        # Fix the color levels across all timesteps so the colorbar is consistent
        levels = np.linspace(np.min(dens), np.max(dens), 100)

        rArr = np.linspace(0, 0.5, 100)
        zArr = np.linspace(-1.5, 1.5, 100)
        R, Z = np.meshgrid(rArr, zArr)

        points = np.array([R.flatten(), Z.flatten()]).T
        densityArr = interpFuncList[0](points).reshape(R.shape)

        pltObj = ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')

        cbar = fig.colorbar(pltObj)
        cbar.set_label(r'Density [m$^{-3}$]')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        title = ax.set_title(f'Time = {time[0]*1e3:.4g} ms')

        def update(frame):

            ax.clear()
            
            densityArr = interpFuncList[frame](points).reshape(R.shape)
            ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')
            
            ax.set_xlabel('Z [m]')
            ax.set_ylabel('R [m]')
            ax.set_title(f'Time = {time[frame]*1e3:.4g} ms')

        anim = animation.FuncAnimation(fig, update, frames=dens.shape[0],
                                        interval=200, blit=False)

        plt.show()

    return interpFuncList, time, dens, solrz, solzz

def generate_filename_list(simulationDir):

    # For each simulation, the data is stored under-
    # [SIMULATION DIR.]/simulation_results/[TIMES]/components/fp__cql3dm_4/WHAM.nc
    # Here, the times go from 1.000 to X.000 where X is the final timestep. This changes by simulation.

    simulationDir += 'simulation_results/'

    # Find all the directories
    directories = [
        d for d in os.listdir(simulationDir) if os.path.isdir(os.path.join(simulationDir, d))
    ]
    directories.remove('plasma_state')

    # Sort the directory names by time
    simTimes = np.array(directories)
    simTimesFloat = np.array(directories, dtype=float)
    sortIdx = simTimesFloat.argsort()
    simTimes = simTimes[sortIdx]
    simTimesFloat = simTimesFloat[sortIdx]

    filenameList = []
    for i in range(len(simTimes)):

        filenameCurr = simulationDir+simTimes[i]+'/components/fp__cql3dm_4/WHAM.nc'
        filenameList.append(filenameCurr)

    return filenameList

def generate_times_and_functions(simulationName):
    """
    Generate interpolation functions for the plasma density for every timestep of the simulation.

    Parameters
    ----------
    simulationName : str
        Name of the simulation (same value passed to generate_times_and_functions).

    Returns
    -------
    interpFuncs : list of functions
        Interpolation functions, one per timestep, parallel to `times`.
    times : np.array
        Time array [s].
    """

    # All simulations are stored in the same directory
    simulationDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/' + simulationName + '/'

    # Try to load the data if it has already been stored
    try:

        print('Trying to load the saved 2D density profile data')

        with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:
            saveData = pickle.load(loadFile)

            interpFuncs = [
                generate_single_interpolation(dens, saveData['solrz'], saveData['solzz'])
                for dens in saveData['dens']
            ]

            return interpFuncs, saveData['times']
    
    except:

        print('No saved 2D density profile data, loading it from the .nc files')

        # Generate a list of the directories that store the data
        filenameList = generate_filename_list(simulationDir)

        # Go over each filename and generate the interpolation functions and get the time array
        interpFuncs = []
        densList = []
        solrz = None
        solzz = None
        times = np.array([])

        endTime = 0

        for i in range(len(filenameList)):

            print(f'Loading {i+1} of {len(filenameList)}')

            try:

                currFuncs, currTimes, currDens, solrz, solzz = generate_interpolation_functions(filenameList[i])
                interpFuncs.extend(currFuncs)

                densList.append(currDens)

                times = np.concatenate([times, currTimes+endTime])
                endTime = times[-1]

            except:

                print(f'Error when loading data from the following directory- \n {filenameList[i]}')
                continue

        # Save the times and the data needed to rebuild the interpolation functions.
        saveData = {
            'dens': np.concatenate(densList, axis=0),
            'solrz': solrz,
            'solzz': solzz,
            'times': times,
        }

        savePath = simulationDir + 'density_interp_data.pkl'
        with open(savePath, 'wb') as saveFile:
            pickle.dump(saveData, saveFile)

        print(f'Saved interpolation data to- \n {savePath}')

        return interpFuncs, times

def load_detector_dictionary(pickleFilePath):

    with open(pickleFilePath, 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    return detDictList

def single_detector_see_density(detDict, interpFunc, makeplot=False):
    """
    This function calculates the line-integrated density at the detector.
    
    Parameters
    ----------
    detDict : dict
        A dictionary containing the information about the SEE detector, including its line of sight points.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
    makeplot : bool, optional
        Whether to plot the density profile and line of sight. The default is False.

    Returns
    -------
    predictedSeeDensity : float
        Line integrated density along the line-of-sight of the given detector.
    """

    # Get the line of sight points for the detector
    losPoints = detDict['los_rz'] / 1e3 # Convert from mm to m

    # Spacing between points along the line of sight
    point1 = losPoints[:, 0]
    point2 = losPoints[:, 1]
    spacing = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # Get the density at each point along the line of sight
    # Define the points in the way that the interpolation function expects (r, z)
    points = np.array([losPoints[0, :], losPoints[1, :]]).T
    densityAlongLos = interpFunc(points)

    # Integrate the line-integrated density along the line of sight to get the predicted SEE density at the detector
    predictedSeeDensity = np.trapezoid(densityAlongLos, dx=spacing)

    return predictedSeeDensity

def synthetic_see_detector(detDictList, interpFunc, makeplot=False):
    """
    Calculate the line-integrated density along the line-of-sight for each detector.

    Parameters
    ----------
    detDictList : list
        List of dictionaries defining the detector parameters.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
    makeplot : bool
        Plot the line-integrated density profile measured by the SEE detectors.

    Returns
    -------
    lineIntegratedDensArr : np.array
        Line-integrated plasma density for each detector [m^-2]
    """

    # Array to store the line-integrated densities
    lineIntegratedDensArr = np.zeros(shape=(len(detDictList)))

    # Go over each detector in the list and calculate the predicted SEE density at each detector
    for i in range(len(detDictList)):
        lineIntegratedDensArr[i] = single_detector_see_density(detDictList[i], interpFunc)

    if makeplot == True:

        # Plot the predicted line integrated density at each detector
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        impactParams = [detDict['impact_param_vertical'] for detDict in detDictList]

        ax.scatter(impactParams, lineIntegratedDensArr, c='red', s=200)

        ax.set_ylabel(r'Predicted SEE density [m$^{-2}$]')
        ax.set_xlabel('Impact parameter [mm]')

        plt.show()

    return lineIntegratedDensArr

def time_dependent_see_detector(simulationName, detDictList, makeplot=False, timeDelta=1e-3):
    """
    Run a full synthetic SEE diagnostic on the simulation.

    Parameters
    ----------
    simulationName : str
        Name of the simulation
    detDistList : list
        List of dictionaries with the detector parameters
    makeplot : bool
        Plot the synthetic diagnostic data
    timeDelta : float
        TIme interval between plots of the radial profiles. [s]

    Returns
    -------
    detDictList : list
        List of dictionaries with the synthetic simulation data added to it.
        This function adds the following keys to each dictionary
        ['simulated_signal'] = Calculated line-integrated density associated with the given detector [m^-2]
        ['simulated_signal_time'] = Corresponding time array [s]
    """

    # All simulations are stored in the same directory
    simulationDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/' + simulationName + '/'

    # Load the saved data if it already exists
    try:

        print('Trying to load the synthetic detector data.')

        dataObj = np.load(simulationDir + 'synthetic_detector_data.npz')

        syntheticSignal = dataObj['syntheticSignal']
        times = dataObj['times']

    except:

        print('No saved synthetic detector data present, generating it.')

        # Load all the interpolation functions
        interpFuncs, times = generate_times_and_functions(simulationName)

        # Array to store the time dependent SEE signals
        # [Time x Detector Number]
        syntheticSignal = np.zeros(shape=(len(times), len(detDictList)))

        for i in range(len(times)):

            # print(f'Calculating the signal for {times[i]*1e3:.2f}ms')
            syntheticSignal[i] = synthetic_see_detector(detDictList, interpFuncs[i])

        #### Save the data

        np.savez(simulationDir + 'synthetic_detector_data.npz',
                 syntheticSignal = syntheticSignal,
                 times = times)

    # Put the data into the dictionaries
    for i in range(len(detDictList)):

        detDictList[i]['simulated_signal'] = syntheticSignal[:, i]
        detDictList[i]['simulated_signal_time'] = times

    if makeplot:

        impactParams = np.array([detDict['impact_param_vertical'] for detDict in detDictList])
        sortIdx = np.argsort(impactParams)
        impactParams = impactParams[sortIdx]

        xLim = np.abs(impactParams).max()

        # Color each time point on a colormap
        cmap = plt.get_cmap('viridis', len(times)).colors

        fig = plt.figure(figsize=(10, 10), tight_layout=True)
        fig.suptitle(simulationName)

        # Synthetic detector
        ax1 = fig.add_subplot(211)

        currTime = 0
        for i in range(len(times)):

            # Only plot every timeDelta
            if times[i] - currTime <= timeDelta:
                continue

            currTime = times[i]
            
            ax1.scatter(impactParams, syntheticSignal[i][sortIdx], 
                        color = cmap[i],
                        s = 200)
            
            ax1.plot(impactParams, syntheticSignal[i][sortIdx], 
                     color = cmap[i],
                     linewidth = 2,
                     label = f'{times[i]*1e3:.2f}')
            
        ax1.legend(title='Times [ms]', ncols=2)

        ax1.set_ylabel(r'Predicted SEE density [m$^{-2}$]')
        ax1.set_xlabel('Vertical Impact Parameter [mm]')

        ax1.set_xlim(-xLim, xLim)
        ax1.set_ylim(0, None)

        # Radial density profile
        ax2 = fig.add_subplot(212)

        # Load the density profile data itself
        interpFuncs, _ = generate_times_and_functions(simulationName)

        radialValues = np.linspace(-xLim/1e3, xLim/1e3, 100)
        zValues = np.zeros_like(radialValues)
        points = np.array([np.abs(radialValues), zValues]).T

        currTime = 0
        for i in range(len(times)):

            # Only plot every 0.25ms
            if times[i] - currTime <= timeDelta:
                continue

            currTime = times[i]

            radialDensity = interpFuncs[i](points)

            ax2.plot(radialValues*1e3, radialDensity,
                     linewidth = 2,
                     color = cmap[i])
            
        ax2.set_xlabel('Vertical Impact Parameter [mm]')
        ax2.set_ylabel(r'Radial Density Profile [m$^{-3}$]')

        ax2.set_xlim(-xLim, xLim)
        ax2.set_ylim(0, None)

        plt.show()
            
    return detDictList

def load_experimental_data(shotnum, makeplot=False):
    """
    Calculates the experimental data for a given shot number.

    Parameters
    ----------
    shotnum : int
        Shot number for which we want to calculate the line-integrated plasma density.
    makeplot : bool
        Make a plot of the experimental data

    Returns
    -------
    detDictList : list
        List of dictionaries that contains the detector information AND the experimentally calculated line-integrated plasma density. [m^-2]
    """

    savename = f'/home/sanwalka/shinethru/data/{shotnum}.pkl'

    # Try to load the data if it already exists
    try:

        print(f'Trying to load the experimental data for shot {shotnum}')

        with open(savename, 'rb') as file:
            detDictList = pickle.load(file)

        if makeplot:

            # Detector impact parameters [m]
            impactParams = np.zeros(len(detDictList))
            for i in range(len(impactParams)):

                beam_pos = detDictList[i]['impact_param_vertical']
                impactParams[i] = beam_pos / 1e3

            # Get the line integrated densities and time array for each detector
            densList = []
            densErrList = []
            timeArr2D = []
            dataPresent = []
            for i in range(len(impactParams)):

                lineIntegratedDens = detDictList[i]['line_integrated_density']

                if lineIntegratedDens is not None:
                    densList.append(detDictList[i]['line_integrated_density'])
                    densErrList.append(detDictList[i]['line_integrated_density_sigma'])
                    timeArr2D.append(detDictList[i]['time_arr_slow'])
                    dataPresent.append(True)
                else:
                    dataPresent.append(False)

            # Remove the impact parameter with no data
            impactParams = impactParams[dataPresent]

            # Sort the data based on impactParams
            sortIdx = np.argsort(impactParams)
            impactParams = impactParams[sortIdx]
            timeArr2D = [timeArr2D[i] for i in sortIdx]
            densList = [densList[i] for i in sortIdx]
            densErrList = [densErrList[i] for i in sortIdx]

            # Remove the 1st detector (railed, broken)
            impactParams = impactParams[1:]
            timeArr2D = timeArr2D[1:]
            densList = densList[1:]
            densErrList = densErrList[1:]

            # Put all the data on the same time axis
            minFinalTime = 1e6
            minTimeIdx = 0
            for i in range(len(timeArr2D)):

                finalTime = timeArr2D[i][-1]
                if finalTime <= minFinalTime:
                    finalTime = minFinalTime
                    minTimeIdx = i

            timeArr = timeArr2D[minTimeIdx]

            densArr2D = np.zeros(shape=(len(densList), len(timeArr)))
            densErrArr2D = np.zeros(shape=(len(densList), len(timeArr)))

            for i in range(len(densList)):

                densArr2D[i] = np.interp(timeArr, timeArr2D[i], densList[i])
                densErrArr2D[i] = np.interp(timeArr, timeArr2D[i], densErrList[i])

            fig = plt.figure(figsize=(12, 8), tight_layout=True)
            ax = fig.add_subplot(111)

            fig.suptitle(shotnum)

            timeDelta = 1e-3

            # Color each time point on a colormap
            cmap = plt.get_cmap('viridis', len(timeArr)).colors

            # Plot each timepoint
            currTime = timeArr[0]
            for i in range(len(timeArr)):

                if timeArr[i] - currTime <= timeDelta:
                    continue

                currTime = timeArr[i]

                ax.errorbar(impactParams*1e2, densArr2D[:, i], 
                            yerr = densErrArr2D[:, i],
                            fmt = 'o',
                            ms = 10,
                            color = cmap[i],
                            elinewidth = 5,
                            label = f'{np.round(timeArr[i]*1e3, 1)}')
                ax.errorbar(impactParams*1e2, densArr2D[:, i], 
                            yerr = 3*densErrArr2D[:, i],
                            fmt = 'o',
                            ms = 10,
                            color = cmap[i])
                ax.plot(impactParams*1e2, densArr2D[:, i],
                        linewidth=2,
                        color=cmap[i])

            ax.legend(title='Time [ms]', ncols=3)
            ax.set_xlabel('Impact Parameter [cm]')
            ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')

            ax.set_ylim(0, None)
            ax.set_xlim(-np.max(np.abs(impactParams*1e2))*1.1, np.max(np.abs(impactParams*1e2))*1.1)

            plt.show()

    except Exception as e:

        print(e)

        print('Experimental data was not calculated. Computing now.')

        # Calculate with plotting
        if makeplot:
            subprocess.run([sys.executable, "experimental_data.py", "-s", f"{shotnum}", "-plot", "True"])

        else:
            subprocess.run([sys.executable, "experimental_data.py", "-s", f"{shotnum}"])

        with open(savename, 'rb') as file:
            detDictList = pickle.load(file)

    return detDictList

def single_time_comparison(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime):
    """
    Compares the simulation and experimental data and returns a single comparison metric.

    Parameters
    ----------
    expDataArr : np.array
        Line-integrated density from the experiment. [m^-2]
    expDataSigmaArr : np.array
        Error bars for the line-integrated density. [m^-2]
    simDataArr : np.array
        Line-integrated density from the simulation. [m^-2]
    impactParams : np.array
        Vertical impact parameter for each detector. [m]
    expTime : float
        Time for the experimental data. [s]
    simTime : float
        Time for the simulation data. [s]

    Returns
    -------
    comparison : float
        Metric that compares the simulation and experimental data.
    """

    # Normalize the data
    expNorm = expDataArr.max()
    expDataArr /= expNorm
    expDataSigmaArr /= expNorm

    simNorm = simDataArr.max()
    simDataArr /= simNorm

    # Difference between the simulation and experiment
    diff = expDataArr - simDataArr

    # Root-squared of the difference
    rsDiff = (diff**2)**0.5

    # Weight it by the error bars
    weights = 1/expDataSigmaArr

    comparison = np.sum(rsDiff * weights)

    return comparison

def compare_simulation_and_experiment(simulationName, shotnum, makeplot=False):
    """
    Compare the plasma density profiles between the CQL3D + KN1D simulation and the experimental result

    Paramters
    ---------
    simulationName : str
        Simulation name we want to compare against.
    shotnum : int
        Shot number we want to compare against.
    makeplot : bool
        Plot the comparison
    """

    # Load the experimental data
    detDictList = load_experimental_data(shotnum)

    # Load the simulation result
    detDictList = time_dependent_see_detector(simulationName, detDictList)

    # Detector impact parameters [m]
    impactParams = np.zeros(len(detDictList))
    for i in range(len(impactParams)):

        beam_pos = detDictList[i]['impact_param_vertical']
        impactParams[i] = beam_pos / 1e3

    #### Put the experimental data into a 2D numpy array

    # Load the experimental data
    expTimeArr = []
    expDataArr = []
    expDataSigmaArr = []
    dataPresent = []
    for i in range(len(impactParams)):

        lineIntegratedDens = detDictList[i]['line_integrated_density']

        if lineIntegratedDens is not None:
            expDataArr.append(detDictList[i]['line_integrated_density'])
            expDataSigmaArr.append(detDictList[i]['line_integrated_density_sigma'])
            expTimeArr.append(detDictList[i]['time_arr_slow'])
            dataPresent.append(True)
        else:
            dataPresent.append(False)

    # Get the final time for each detector
    finalTime = 1e5
    shortestTimeIdx = 0
    for i in range(len(expTimeArr)):
        
        currFinalTime = expTimeArr[i].max()
        
        if finalTime > currFinalTime:
            finalTime = currFinalTime
            shortestTimeIdx = i

    # Common time array for the experimental data
    expTimeArrNew = expTimeArr[shortestTimeIdx]

    # Put all the experimental data on the same timebase
    expDataArrNew = []
    expDataSigmaArrNew = []
    for i in range(len(expDataArr)):

        expDataArrNew.append(np.interp(expTimeArrNew, expTimeArr[i], expDataArr[i]))
        expDataSigmaArrNew.append(np.interp(expTimeArrNew, expTimeArr[i], expDataSigmaArr[i]))

    # Convert to numpy as rename
    expTimeArr = np.array(expTimeArrNew)
    expDataArr = np.array(expDataArrNew)
    expDataArr = np.array(expDataArrNew)
    expDataSigmaArr = np.array(expDataSigmaArrNew)

    #### Put the simulation data into a 2D numpy array
    simTimeArr = detDictList[0]['simulated_signal_time']
    
    simDataArr = np.zeros(shape=(len(detDictList), len(simTimeArr)))
    for i in range(len(detDictList)):

        simDataArr[i] = detDictList[i]['simulated_signal']

    # Remove the simulation data from the non-functioning detector
    simDataArr = simDataArr[dataPresent]

    # Remove the non-functioning impact parameter
    impactParams = impactParams[dataPresent]

    # Sort the data based on impactParams
    sortIdx = np.argsort(impactParams)
    
    impactParams = impactParams[sortIdx]
    simDataArr = simDataArr[sortIdx]
    expDataArr = expDataArr[sortIdx]
    expDataSigmaArr = expDataSigmaArr[sortIdx]

    # Remove the 1st detector (railed/broken)
    impactParams = impactParams[1:]
    simDataArr = simDataArr[1:]
    expDataArr = expDataArr[1:]
    expDataSigmaArr = expDataSigmaArr[1:]

    # Array to compare the simulation and experimental data
    # [simTime x expTime]
    comparisonArr = np.zeros(shape=(len(simTimeArr), len(expTimeArr)))

    # Go over each time point and calculate the comparison
    for i in range(len(simTimeArr)):
        for j in range(len(expTimeArr)):

            comparisonArr[i, j] = single_time_comparison(expDataArr = expDataArr[:, j],
                                                         expDataSigmaArr = expDataSigmaArr[:, j],
                                                         simDataArr = simDataArr[:, i],
                                                         impactParams = impactParams,
                                                         expTime = expTimeArr[j],
                                                         simTime = simTimeArr[i])
            
    if makeplot:

        import matplotlib.ticker as ticker

        # Plot the comparison
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        X, Y = np.meshgrid(expTimeArr*1e3, simTimeArr*1e3)

        # Define explicit log-spaced levels
        vmin = max(1e-5, comparisonArr.min())  # Avoid zeros/negatives
        vmax = comparisonArr.max()
        logLevels = np.logspace(np.log10(vmin), np.log10(vmax), 100)

        pltObj = ax.contourf(X, Y, comparisonArr, 
                            levels = logLevels,
                            locator = ticker.LogLocator(),
                            cmap = 'viridis')

        ax.set_xlabel('Experimental Time [ms]')
        ax.set_ylabel('Simulation Time [ms]')

        ax.set_title(f'{shotnum} vs {simulationName}')

        cbar = fig.colorbar(pltObj)
        cbar.locator = ticker.LogLocator(base=10.0, numticks=10)
        cbar.update_ticks()
        cbar.set_label("Comparison", rotation=90, labelpad=15)

        plt.show()

    return

if __name__ == '__main__':

    # Load the SEE detector dictionary
    with open('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl', 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    # Simulation directory
    simulationName = 'nneut_2e17_gb_2e17_NBI_800kW_ECH_0kW'
    # Shot number
    shotnum = 260426037

    # Load the experimental data for a given shot
    # detDictList = load_experimental_data(shotnum, True)

    # Load all the interpolation functions
    # interpFuncs, times = generate_times_and_functions(simulationName)

    # Check the synthetic diagnostic at a given timepoint
    # lineIntegratedDensArr = synthetic_see_detector(detDictList, interpFuncs[-1], True)

    # Generate the data for the time dependent synthetic detector
    detDictList = time_dependent_see_detector(simulationName, detDictList, True)

    # Compare simulation to experiment
    # compare_simulation_and_experiment(simulationName, shotnum, makeplot=True)