# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:25:00 2026

@author: kunal

For a given 2D density profile and SEE detector positions, this code calculates the predicted SEE density at each detector. 
This is done by calculating the line of sight of each detector and then integrating the density along the line of sight.

It adds the predicted line-integrated density AND the density along the line of sight to the SEE detector dictionary for each detector.
The 2 keys added to the dictionary are-
1. singledetDict['see_density_along_los'] = density (m^-3) along the line of sight of the detector (defined in ['los_rz'])
2. singledetDict['predicted_see_density'] = line-integrated density (m^-2) at the detector, calculated by integrating the density along the line of sight
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import scipy as sc

def load_detector_dictionary(pickleFilePath):

    with open(pickleFilePath, 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    return detDictList

def load_density_profile(densityProfileFilePath, makeplot=False):
    """
    This function loads the density profile from a given .hdf5 file and creates an interpolation function.

    
    Parameters
    ----------
    densityProfileFilePath : str
        The file path of the .hdf5 file containing the density profile.
    makeplot : bool, optional
        Whether to plot the density profile. The default is False.
    
    Returns
    -------
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]    
    """

    # Open .hdf5 file
    with h5py.File(densityProfileFilePath, 'r') as hdfFile:

        # Load the mesh
        mesh = hdfFile['Mesh']
        rArr = mesh['R'][:]
        zArr = mesh['Z'][:]

        # Load the density
        equilibrium = hdfFile['Equilibrium']
        densityArr = equilibrium['n_e'][:].T

    # 1D arrays of the mesh points
    r1dArr = rArr[0, :]
    z1dArr = zArr[:, 0]

    # Create an interpolation function for the density profile
    interpFunc = sc.interpolate.RegularGridInterpolator((r1dArr, z1dArr), densityArr, method='linear', bounds_error=False, fill_value=0)

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
        cbar.set_label('Density [m^-3]')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')

        plt.show()

    return interpFunc

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

def radial_density_profile(rArr, interpFunc):

    # Absolute value of the radial positions (in m) where I want to calculate the density profile
    rArr = np.abs(rArr)

    # Define the points in the way that the interpolation function expects (r, z)
    zArr = np.zeros_like(rArr)
    points = np.array([rArr, zArr]).T
    densityAlongRadial = interpFunc(points)

    return densityAlongRadial

if __name__ == "__main__":

    # Load the SEE detector dictionary
    with open('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl', 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    # Load the density profile
    densityProfileFilePath = '/home/sanwalka/shinethru/pleiades_260105053.h5'
    interpFunc = load_density_profile(densityProfileFilePath, makeplot=False)

#    detDict = single_detector_see_density(detDictList[0], interpFunc, makeplot=True)

    detDictList = synthetic_see_detector(detDictList, interpFunc, makeplot=True)