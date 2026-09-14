"""
This script plots the radial confinement times for a given cql3d simulation at a given time.
"""

import os
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from kn1dc_parser import generate_filename_list, time_dep_source_rate

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
    simulationDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/' + simName + '/'

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

def radial_taup_profiles(simName, timesToPlotSim, makeplot=False):

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

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        fig.suptitle(simName)

        # CQL3D direct output
        ax = fig.add_subplot(111)

        cmap = plt.get_cmap('viridis', len(timeArrCQL3D)).colors

        simIdxArr = np.zeros_like(timesToPlotSim)
        for i in range(len(timesToPlotSim)):
            simIdxArr[i] = np.argmin(np.abs(timeArrCQL3D-timesToPlotSim[i]))

        tauPImportant = []
        for i in range(len(timeArrCQL3D)):

            if i in simIdxArr:
                ax.plot(rArr2D[0], tauPCQL3D[i]*1e3, color='red', label=f'{timeArrCQL3D[i]*1e3:.1f}', linewidth=5, zorder=10)
                tauPImportant.append(tauPCQL3D[i]*1e3)
            elif i % 50 == 0:
                ax.plot(rArr2D[0], tauPCQL3D[i]*1e3, color=cmap[i], label=f'{timeArrCQL3D[i]*1e3:.1f}')
            else:
                ax.plot(rArr2D[0], tauPCQL3D[i]*1e3, color=cmap[i])
        tauPImportant = np.array(tauPImportant)

        ax.set_ylim(0, 1.1*tauPImportant.max())
        ax.legend(title='Time [ms]', ncols=3, framealpha=1)
        ax.set_ylabel('Confinement Time [ms]')
        ax.set_xlabel('Radius [m]')
        ax.set_title('tauc_code')

        plt.show()

    return

if __name__ == '__main__':

    simName = 'nneut_1e18_gb_1e18_NBI_800kW_ECH_0kW'
    
    # Times to plot (in seconds)
    timesToPlotSim = np.array([3.5]) * 1e-3

    radial_taup_profiles(simName, timesToPlotSim, makeplot=True)