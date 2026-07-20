"""
This script calculates the line-integrated density for the SEE detectors from a synthetic diagnostic implemented on a CQL3D output file.
"""

import os
import pickle
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

def time_dependent_see_detector(simulationName, makeplot=False):

    # All simulations are stored in the same directory
    simulationDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/' + simulationName + '/'

    # Load the SEE detector dictionary
    with open('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl', 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

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

    if makeplot:

        impactParams = np.array([detDict['impact_param_vertical'] for detDict in detDictList])
        sortIdx = np.argsort(impactParams)
        impactParams = impactParams[sortIdx]

        # Color each time point on a colormap
        cmap = plt.get_cmap('viridis', len(times)).colors

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        fig.suptitle(simulationName)

        currTime = 0
        for i in range(len(times)):

            # Only plot every 0.25ms
            if times[i] - currTime <= 2.5e-4:
                continue

            currTime = times[i]
            
            ax.scatter(impactParams, syntheticSignal[i][sortIdx], 
                       color=cmap[i],
                       s=200)
            
            ax.plot(impactParams, syntheticSignal[i][sortIdx], 
                       color=cmap[i],
                       linewidth=2,
                       label=f'{times[i]*1e3:.2f}')
            
        ax.legend(title='Times [ms]', ncols=2)

        ax.set_ylabel(r'Predicted SEE density [m$^{-2}$]')
        ax.set_xlabel('Vertical Impact Parameter [mm]')

        plt.show()
            
    return syntheticSignal, times

if __name__ == '__main__':

    # Load the SEE detector dictionary
    with open('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl', 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    # Simulation directory
    simulationName = 'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW'

    # Load all the interpolation functions
    # interpFuncs, times = generate_times_and_functions(simulationName)

    # Check the synthetic diagnostic at a given timepoint
    # lineIntegratedDensArr = synthetic_see_detector(detDictList, interpFuncs[-1], True)

    # Generate the data for the time dependent synthetic detector
    syntheticSignal, times = time_dependent_see_detector(simulationName, True)