"""
This script plots the radial profiles of the simulation based on when they have the most agreement with data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from cql3d_vs_data import load_detector_dictionary, time_dependent_see_detector, load_experimental_data

# Global variable to store the simulation scan directory
global simulationScanDir
simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/withRadialDiff/'

def exp_dict_to_numpy(detDictList):

    # Detector impact parameters [m]
    impactParams = np.zeros(len(detDictList))
    for i in range(len(impactParams)):

        beam_pos = detDictList[i]['impact_param_vertical']
        impactParams[i] = beam_pos / 1e3

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
    expDataSigmaArr = np.array(expDataSigmaArrNew)

    return expTimeArr, expDataArr, expDataSigmaArr, dataPresent

def sim_dict_to_numpy(detDictList):

    simTimeArr = detDictList[0]['simulated_signal_time']
            
    simDataArr = np.zeros(shape=(len(detDictList), len(simTimeArr)))
    for i in range(len(detDictList)):

        simDataArr[i] = detDictList[i]['simulated_signal']

    return simTimeArr, simDataArr

def plot_radial_and_synthetic(simulationName, shotnum, timesToPlotSim, timesToPlotExp):
    """
    This function plots the radial density profile at the midplane and the synthetic diagnostic for the given times.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    timesToPlot : np.array
        The times at which to plot the radial profiles. [s]
    """

    # All simulations are stored in the same directory
    simulationDir = simulationScanDir + simulationName + '/'

    # Load the saved data
    with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:
        saveData = pickle.load(loadFile)

        solrz = saveData['solrz']
        solzz = saveData['solzz']

        # Time not loaded as it is the same as simTimeArr (checked with np.allclose)

        # [Time x r x z]
        dens = saveData['dens']

    # Load the synthetic SEE signal
    detDictList = load_detector_dictionary('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl')
    detDictList = time_dependent_see_detector(simulationName, detDictList)
    # Convert to numpy arrays for easy plotting
    simTimeArr, simDataArr = sim_dict_to_numpy(detDictList)

    # Load the experimental data
    detDictList = load_experimental_data(shotnum)
    # Convert to numpy arrays for easy plotting
    expTimeArr, expDataArr, expDataSigmaArr, dataPresent = exp_dict_to_numpy(detDictList)

    # Impact Parameters
    impactParams = np.array([detDict['impact_param_vertical'] for detDict in detDictList]) / 1e3

    # Remove data from the broken digitizer
    impactParams = impactParams[dataPresent]
    simDataArr = simDataArr[dataPresent]

    # Sort based on the impact parameter
    sortIdx = np.argsort(impactParams)
    impactParams = impactParams[sortIdx]
    simDataArr = simDataArr[sortIdx]
    expDataArr = expDataArr[sortIdx]
    expDataSigmaArr = expDataSigmaArr[sortIdx]

    # 1st detector is broken
    impactParams = impactParams[1:]
    simDataArr = simDataArr[1:]
    expDataArr = expDataArr[1:]
    expDataSigmaArr = expDataSigmaArr[1:]

    xLim = 1.1 * np.max(np.abs(impactParams))

    # Create a figure for the radial profiles
    fig = plt.figure(figsize=(12, 10), tight_layout=True)

    # Radial profile at the midplane
    ax1 = fig.add_subplot(2, 1, 1)
    # Synthetic diagnostic
    ax2 = fig.add_subplot(2, 1, 2)

    # Plot the simulated radial profiles and synthetic diagnostics for the specified times
    for i in range(len(timesToPlotSim)):

        # Find the index of the closest time in the simulation data
        timeIdx = np.argmin(np.abs(simTimeArr - timesToPlotSim[i]))

        #### Radial profile at the midplane

        # Get the radial profile at the midplane
        radialProfile = dens[timeIdx, :, 0]

        # Plot the radial profile
        ax1.plot(solrz[:, 0], radialProfile, 
                 color=f'C{i}', 
                 linewidth=3)
        ax1.plot(-solrz[:, 0], radialProfile, 
                 color=f'C{i}', 
                 linewidth=3)

        #### Synthetic SEE signal
        ax2.plot(impactParams, simDataArr[:, timeIdx], 
                 label=f'(sim) {timesToPlotSim[i]*1e3:.2f}', 
                 color=f'C{i}', 
                 linewidth=3)

    # Plot the experimental data
    for i in range(len(timesToPlotExp)):

        # Find the index of the closest time in the experimental data
        timeIdx = np.argmin(np.abs(expTimeArr - timesToPlotExp[i]))

        ax2.plot(impactParams, expDataArr[:, timeIdx], 
                 label=f'(exp) {timesToPlotExp[i]*1e3:.2f}', 
                 color=f'C{i}', 
                 linewidth=3, 
                 linestyle='dashed')

        ax2.errorbar(impactParams, expDataArr[:, timeIdx], 
                     yerr=expDataSigmaArr[:, timeIdx],
                     color=f'C{i}',
                     fmt='o',
                     ms=10,
                     elinewidth=7)
        
        ax2.errorbar(impactParams, expDataArr[:, timeIdx], 
                     yerr=3*expDataSigmaArr[:, timeIdx],
                     color=f'C{i}',
                     fmt='o',
                     ms=10,
                     elinewidth=3)

    ax1.set_ylabel(r'n$_i$ [m$^{-3}$]')
    ax1.set_ylim(0, None)

    ax1.set_xticks([])
    ax1.set_xlim(-xLim, xLim)

    
    ax2.set_ylabel(r'$\int n_i \cdot dl$ [m$^{-2}$]')
    ax2.set_ylim(0, None)

    ax2.set_xlim(-xLim, xLim)
    ax2.set_xlabel('Radius [m]')

    ax2.legend(title='Time [ms]')

    plt.show()

    return

def synthetic_interferometer(simulationName, makeplot=False):

    # All simulations are stored in the same directory
    simulationDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/' + simulationName + '/'

    with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:

        saveData = pickle.load(loadFile)

        # [Time x radius x axial]
        dens = saveData['dens']

        solrz = saveData['solrz']
        solzz = saveData['solzz']
        times = saveData['times']

    # Radial profile at the midplane
    densRad = dens[:, :, 0]
    # Radial points at the midplane
    r1D = solrz[:, 0]

    # Average over the radial points to get the 'synthetic interferometer'
    synInf = np.trapezoid(densRad, x=r1D, axis=1)

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(times*1e3, synInf, linewidth=3)

        ax.set_title(simulationName)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-3}$]')

        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        plt.show()

    return times, synInf

if __name__ == "__main__":

    simName = 'nneut_5e18_gb_5e18_NBI_800kW_ECH_0kW_ionDrrOn'

    # Times to plot (in seconds)
    timesToPlotSim = np.array([3.5]) * 1e-3
    timesToPlotExp = np.array([7.6]) * 1e-3

    shotnum = 260426037

    plot_radial_and_synthetic(simName, shotnum, timesToPlotSim, timesToPlotExp)
    # times, synInf = synthetic_interferometer(simName, makeplot=True)