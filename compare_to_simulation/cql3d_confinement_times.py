"""
This script plots the radial confinement times for a given cql3d simulation at a given time.
"""

import os
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from kn1dc_parser import generate_filename_list, time_dep_source_rate, disambiguate_source_rate
import extend_cql3d_with_axuv as eca
from cql3d_vs_data import parse_simulation_name, density_label

# Global variable to store the simulation scan directory
simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/withRadialDiff/'

def flux_tube_vol(filenameListCQL3D):

    fluxTubeVol = []
    rArr = []

    for i in range(len(filenameListCQL3D)):

        # Open the file
        ds = xr.open_dataset(filenameListCQL3D[i],
                             decode_timedelta=False)

        # [cm^3] to [m^3]
        fluxTubeVolTemp = ds['dvol'].values / 1e6
        fluxTubeVol.append(fluxTubeVolTemp)

        # [cm] to [m]
        solrz = ds['solrz'].values / 1e2
        rArr.append(solrz[:, 0])

    fluxTubeVol = np.array(fluxTubeVol)
    rArr = np.array(rArr)

    return fluxTubeVol, rArr

def taup_cql3d(filenameListCQL3D):

    # [Time x Radius]
    tauPCQL3D = []
    timeArr = np.array([])
    currTime = 0

    for i in range(len(filenameListCQL3D)):
    
        # Open the file
        ds = xr.open_dataset(filenameListCQL3D[i],
                             decode_timedelta=False)

        tauPTemp = ds['tauc_code'].values

        # 1st is ions, 2nd is electrons
        tauPTemp = tauPTemp[:, :, 0]

        time = ds['time'].values

        timeArr = np.concatenate([timeArr, time+currTime])
        tauPCQL3D.append(tauPTemp)

        currTime = timeArr[-1]

    tauPCQL3D = np.concatenate(tauPCQL3D)
    timeArr = np.array(timeArr)

    return tauPCQL3D, timeArr

def density_profile(simName, timeArrIPS, makeplot=False):

    # All simulations are stored in the same directory
    simulationDir = simulationScanDir + simName + '/'

    # Load the saved data
    with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:
        saveData = pickle.load(loadFile)

        solrz = saveData['solrz']
        solzz = saveData['solzz']

        times = saveData['times']

        # [Time x r x z]
        dens = saveData['dens']

    # Get the midplane density profile
    dens = dens[:, :, 0]

    densIPS = np.zeros(shape=(len(timeArrIPS), len(dens[0])))

    for i in range(len(timeArrIPS)):

        timeIdx = np.argmin(np.abs(times - timeArrIPS[i]))

        densIPS[i] = dens[timeIdx]

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        cmap = plt.get_cmap('viridis', len(timeArrIPS)).colors

        for i in range(len(timeArrIPS)):

            ax.plot(solrz[:, 0], densIPS[i], color=cmap[i], label=f'{timeArrIPS[i]*1e3:.1f}')

        ax.set_xlabel('Radius [m]')
        ax.set_ylabel(r'n$_e$ [m$^{-3}$]')

        ax.legend(title='Time [ms]', ncols=3)
        ax.set_title(simName)

        plt.show()

    return densIPS

def radial_taup_profiles(simName, timesToPlotSim, tStart=None, tStop=None, legendSpacing=0.5e-3,
                         makeplot=False, saveplot=False):
    """
    Plot the radial confinement time profiles (tauc_code) from CQL3D between tStart and tStop.

    Profiles at timesToPlotSim are highlighted in red and set the y-axis limit.

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timesToPlotSim : np.array
        The simulation times to highlight. [s]
        The closest available simulated timestep is used.
    tStart : float
        Earliest time to plot. [s]
        Default is None, which starts at the beginning of the simulation.
    tStop : float
        Latest time to plot. [s]
        Default is None, which stops at the end of the simulation.
    legendSpacing : float
        Time spacing between the profiles labelled in the legend. [s]
        Default is 0.5e-3.
    makeplot : bool
        Show the plot.
        Default is False.
    saveplot : bool
        Save the plot in /home/sanwalka/shinethru/plots/ as f'{simName}_tauc.png'.
        Default is False.
    """

    # Filename list for the simulation
    filenameListMain, filenameListGasBox, filenameListCQL3D, filenameListEQDSK = generate_filename_list(simName)

    # Time dependent source rate
    rhoH, Sion2D, _, _, timeArrIPS = time_dep_source_rate(simName)

    # Volume of each flux tube
    # fluxTubeVol = [Time x Radius]
    # rArr2D = [Time x Radius]
    fluxTubeVol, rArr2D = flux_tube_vol(filenameListCQL3D)

    # Go over each timepoint and remap Sion2D onto rArr2D
    remapSion2D = np.zeros_like(fluxTubeVol)

    for i in range(len(timeArrIPS)):
        remapSion2D[i] = np.interp(rArr2D[i], rhoH, Sion2D[i])
        
    # Multiply by the volume to get the source rate
    sourceRate2D = fluxTubeVol * remapSion2D

    # Get the radial density profile
    densIPS = density_profile(simName, timeArrIPS)

    # Confinement time
    tauP = densIPS / sourceRate2D

    # Confinement time calculated directly in CQL3D
    tauPCQL3D, timeArrCQL3D = taup_cql3d(filenameListCQL3D)

    if False:

        fig = plt.figure(figsize=(10, 10), tight_layout=True)
        fig.suptitle(simName)

        # Manual Calculation
        ax1 = fig.add_subplot(211)
        # CQL3D direct output
        ax2 = fig.add_subplot(212)

        cmap1 = plt.get_cmap('viridis', len(timeArrIPS)).colors
        cmap2 = plt.get_cmap('viridis', len(timeArrCQL3D)).colors

        for i in range(len(timeArrIPS)):

            ax1.plot(rArr2D[i], tauP[i]*1e3, color=cmap1[i], label=f'{timeArrIPS[i]*1e3:.1f}')

        ax1.set_ylim(0, 25)
        ax1.set_xticks([])
        ax1.set_ylabel('Confinement Time [ms]')
        ax1.legend(title='Time [ms]', ncols=3)
        ax1.set_title('Density (CQL3D) / Source Rate (KN1DC)')

        for i in range(len(timeArrCQL3D)):

            if i % 50 == 0:
                ax2.plot(rArr2D[0], tauPCQL3D[i]*1e3, color=cmap2[i], label=f'{timeArrCQL3D[i]*1e3:.1f}')
            else:
                ax2.plot(rArr2D[0], tauPCQL3D[i]*1e3, color=cmap2[i])

        ax2.set_ylim(0, 5)
        ax2.legend(title='Time [ms]', ncols=3)
        ax2.set_ylabel('Confinement Time [ms]')
        ax2.set_xlabel('Radius [m]')
        ax2.set_title('tauc_code')

        plt.show()

    if makeplot or saveplot:

        # Default to the full simulation time range
        if tStart is None:
            tStart = timeArrCQL3D[0]
        if tStop is None:
            tStop = timeArrCQL3D[-1]

        # Indices of the timesteps within [tStart, tStop]
        plotIdxArr = np.where((timeArrCQL3D >= tStart) & (timeArrCQL3D <= tStop))[0]

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        fig.suptitle(simName)

        # CQL3D direct output
        ax = fig.add_subplot(111)

        cmap = plt.get_cmap('viridis', len(plotIdxArr)).colors

        simIdxArr = np.zeros(len(timesToPlotSim), dtype=int)
        for i in range(len(timesToPlotSim)):
            simIdxArr[i] = np.argmin(np.abs(timeArrCQL3D-timesToPlotSim[i]))

        # Timesteps closest to each multiple of legendSpacing get a legend label
        legendTimes = np.arange(tStart, tStop + 1e-3*legendSpacing, legendSpacing)
        legendIdxArr = plotIdxArr[[np.argmin(np.abs(timeArrCQL3D[plotIdxArr]-t)) for t in legendTimes]]

        tauPImportant = []
        for j, i in enumerate(plotIdxArr):

            if i in simIdxArr:
                ax.plot(rArr2D[0], tauPCQL3D[i]*1e3, color='red', label=f'{timeArrCQL3D[i]*1e3:.1f}', linewidth=5, zorder=10)
                tauPImportant.append(tauPCQL3D[i]*1e3)
            elif i in legendIdxArr:
                ax.plot(rArr2D[0], tauPCQL3D[i]*1e3, color=cmap[j], label=f'{timeArrCQL3D[i]*1e3:.1f}')
            else:
                ax.plot(rArr2D[0], tauPCQL3D[i]*1e3, color=cmap[j])
        tauPImportant = np.array(tauPImportant)

        # Scale to the highlighted profiles if any are in the time window
        if len(tauPImportant) > 0:
            ax.set_ylim(0, 1.1*tauPImportant.max())
        else:
            ax.set_ylim(0, None)
        ax.legend(title='Time [ms]', ncols=3, framealpha=1)
        ax.set_ylabel('Confinement Time [ms]')
        ax.set_xlabel('Radius [m]')
        ax.set_title('tauc_code')

        if saveplot:
            savePath = f'/home/sanwalka/shinethru/plots/{simName}_tauc.png'
            plt.savefig(savePath, dpi=600)
            print(f'Saved plot to {savePath}')

        if makeplot:
            plt.show()
        else:
            plt.close(fig)

    return

def plot_taup_density_source(simName, shotnum, timeToPlot, timeToPlotExp, diodeArrayNum=1,
                             rateKHz=10.0, tanhSteepness=4.0, axuvTMin=None, axuvTMax=None,
                             redoAnalysis=False, xMax=0.2, makeplot=True, saveplot=False):
    """
    Plot the confinement time, midplane density and disambiguated source rates vs. radius at a given time.

    Top- Confinement time calculated directly in CQL3D (tauc_code).
    Middle- Ion density at the midplane, extended out to the plasma radius measured by AXUV
            (see build_density_extension in extend_cql3d_with_axuv.py).
    Bottom- Gas box and main vessel source rates at the midplane (see disambiguate_source_rate in kn1dc_parser.py).

    Parameters
    ----------
    simName : str
        The name of the simulation.
    shotnum : int
        The shot number the AXUV plasma radius of the extension is taken from.
    timeToPlot : float
        The simulation time at which to plot all the quantities. [s]
        The closest available timestep of each quantity is used. The source rates are only
        saved once per IPS timestep, so their time can differ from that of the other two.
    timeToPlotExp : float
        The experimental time the AXUV plasma radius of the extension is taken at. [s]
        The closest available AXUV timestep is used.
    diodeArrayNum : int
        The diode array the AXUV plasma radius is taken from, i.e. 1 for DIODEARRAY1.
        Default is 1.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Default is 4.
    axuvTMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    axuvTMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.
    redoAnalysis : bool
        Force the extension to be rebuilt instead of loading the saved one.
        Default is False.
    xMax : float
        Largest radius to plot. [m]
        Default is 0.2.
    makeplot : bool
        Show the plot.
        Default is True.
    saveplot : bool
        Save the plot in /home/sanwalka/shinethru/plots/ as f'{simName}_taup_dens_source.png'.
        Default is False.
    """

    # Filename list for the simulation
    _, _, filenameListCQL3D, _ = generate_filename_list(simName)

    #### Confinement time

    # Confinement time calculated directly in CQL3D [Time x Radius]
    tauPCQL3D, timeArrCQL3D = taup_cql3d(filenameListCQL3D)

    # Radial grid of the CQL3D flux surfaces [Time x Radius]
    _, rArr2D = flux_tube_vol(filenameListCQL3D)

    tauPIdx = np.argmin(np.abs(timeArrCQL3D - timeToPlot))

    #### Midplane density

    # That module keeps its own copy of the scan directory, so point it at the one used here
    eca.simulationScanDir = simulationScanDir

    extData = eca.build_density_extension(simName, shotnum, diodeArrayNum,
                                          rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                                          tMin=axuvTMin, tMax=axuvTMax,
                                          redoAnalysis=redoAnalysis)

    # [simTime x r x z], with the restart slices already dropped and the density clipped
    times = extData['times']
    dens = extData['densSim']

    densIdx = np.argmin(np.abs(times - timeToPlot))
    axuvIdx = np.argmin(np.abs(extData['axuvTimes'] - timeToPlotExp))

    # Midplane density as CQL3D solves it
    rSim = extData['solrz'][:, 0]
    densSimMid = dens[densIdx, :, 0]

    # The extension is its shape scaled by the midplane density at the edge of the simulation grid.
    # The edge point of the simulation is prepended so that the two curves join up.
    rExt = np.concatenate([[rSim[-1]], extData['rExtGrid']])
    densExtMid = densSimMid[-1] * np.concatenate([[1], extData['shapeExt'][axuvIdx]])

    #### Disambiguated source rates

    rhoH, SionMV, SionGBMid, timeSource = disambiguate_source_rate(simName, timeToPlot, makeplot=False)

    fig = plt.figure(figsize=(10, 14), tight_layout=True)

    # Confinement time
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(rArr2D[0], tauPCQL3D[tauPIdx]*1e3, color='k', linewidth=3)
    ax1.set_ylabel(r'$\tau_p$ [ms]')
    ax1.set_ylim(0, None)
    ax1.set_title(f'Shot #{shotnum}', loc='right')

    # Label the top panel with the times and neutral densities of the simulation
    paramText = (r'$t_{Exp}$' + f' = {extData["axuvTimes"][axuvIdx]*1e3:.2f} ms\n'
                 r'$t_{Sim}$' + f' = {timeArrCQL3D[tauPIdx]*1e3:.2f} ms')
    mainVesselDens, gasBoxDens = parse_simulation_name(simName)
    if mainVesselDens is not None:
        paramText += (f'\nGas Box '+r'$n_n$'+f': {density_label(gasBoxDens)}\n'
                      f'Main Vessel '+r'$n_n$'+f': {density_label(mainVesselDens)}')
    ax1.text(0.02, 0.05, paramText,
             transform=ax1.transAxes,
             ha='left', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Midplane density, colored the same as in extend_2d_density() in extend_cql3d_with_axuv.py
    simColor = mpl.colormaps['inferno'](0.6)
    extColor = mpl.colormaps['viridis'](0.6)

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(rSim, densSimMid, label='CQL3D', color=simColor, linewidth=3)
    ax2.plot(rExt, densExtMid, label='AXUV extension',
             color=extColor, linewidth=3, linestyle='--')
    ax2.set_ylabel(r'Midplane n$_i$ [m$^{-3}$]')
    ax2.set_ylim(0, None)
    ax2.legend()

    # Disambiguated source rates
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    ax3.plot(rhoH, SionMV, label='Main Vessel', color='tab:blue', linewidth=3)
    ax3.plot(rhoH, SionGBMid, label='Gas Box', color='tab:orange', linewidth=3)
    ax3.set_ylabel('Source Rate [#/s]')
    ax3.set_yscale('log')
    ax3.legend()

    ax1.set_xlim(0, xMax)

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.set_xlabel('Radius [m]')

    if saveplot:
        savePath = f'/home/sanwalka/shinethru/plots/{simName}_taup_dens_source.png'
        plt.savefig(savePath, dpi=600)
        print(f'Saved plot to {savePath}')

    if makeplot:
        plt.show()
    else:
        plt.close(fig)

    return

if __name__ == '__main__':

    simName = 'nneut_2e17_gb_2e17_NBI_800kW_ECH_0kW_ionDrrOn'
    
    # Times to plot (in seconds)
    timesToPlotSim = np.array([3]) * 1e-3

    # radial_taup_profiles(simName, timesToPlotSim, makeplot=True, saveplot=True)
    # radial_taup_profiles(simName, timesToPlotSim, tStart=0.5e-3, tStop=4.5e-3, makeplot=True, saveplot=True)

    # Confinement time, midplane density and disambiguated source rates at a single time
    plot_taup_density_source(simName, shotnum=260709061, timeToPlot=3e-3, timeToPlotExp=7.83e-3, makeplot=True, saveplot=True)