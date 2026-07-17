"""
This script calculates the line-integrated density for the SEE detectors from a synthetic diagnostic implemented on a CQL3D output file.
"""

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
dataDest = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/second_round/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/18.000/components/fp__cql3dm_4/'

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

    return interpFuncList

def load_detector_dictionary(pickleFilePath):

    with open(pickleFilePath, 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    return detDictList

def single_detector_see_density(detDict, interpFunc, makeplot=False):
    """
    This function calculates the predicted density along the line of sight of a single SEE detector and also the line-integrated density at the detector.

    It adds them both to the input dictionary and also plots the density profile and line of sight if makeplot is set to True.

    They are added to the dictionary with the keys 'see_density_along_los' and 'predicted_see_density' respectively.
    
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
    None. The input dictionary is modified to include the density along the line of sight and the predicted SEE density at the detector.
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

    # Add the density along the line of sight and the predicted SEE density to the dictionary
    detDict['see_density_along_los'] = densityAlongLos
    detDict['predicted_see_density'] = predictedSeeDensity

    if makeplot == True:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        # Plot the density profile
        rArr = np.linspace(0, 0.5, 100)
        zArr = np.linspace(-1.5, 1.5, 100)
        R, Z = np.meshgrid(rArr, zArr)

        points = np.array([R.flatten(), Z.flatten()]).T
        densityArr = interpFunc(points).reshape(R.shape)

        pltObj = ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')

        cbar = fig.colorbar(pltObj)
        cbar.set_label('Density [m^-3]')

        # Plot the line of sight of the detector
        ax.plot(losPoints[1, :], losPoints[0, :], c='white', linewidth=3, label='Line of sight')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')

        plt.show()

    return detDict

def synthetic_see_detector(detDictList, interpFunc, makeplot=False):

    # Go over each detector in the list and calculate the predicted SEE density at each detector
    for detDict in detDictList:
        single_detector_see_density(detDict, interpFunc)

    if makeplot == True:

        # Plot the predicted line integrated density at each detector
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        predictedDensities = [detDict['predicted_see_density'] for detDict in detDictList]
        impactParams = [detDict['impact_param_vertical'] for detDict in detDictList]

        ax.scatter(impactParams, predictedDensities, c='red', s=200)

        ax.set_ylabel(r'Predicted SEE density [m$^{-2}$]')
        ax.set_xlabel('Impact parameter [mm]')

        plt.show()

    return detDictList

if __name__ == '__main__':

    # Load the SEE detector dictionary
    with open('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl', 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    # Generate the density interpolation function
    filename = dataDest + '230222_newFusDiag.nc'
    filename = dataDest + 'WHAM.nc'

    # _, _, _, _ = ion_dens(filename, makeplot=True)
    interpFuncList = generate_interpolation_functions(filename, makeplot=True)

    # Add the synthetic data to the dictionary
    # detDictList = synthetic_see_detector(detDictList, interpFunc, makeplot=True)