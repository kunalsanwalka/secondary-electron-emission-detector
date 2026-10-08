"""
This script returns the source rate for the main vessel and gas box vs. normalized midplane radius 
"""

import os
import subprocess
import scipy as sc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from cql3d_radial_profiles import load_eqdsk_flux

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

# Make the font size larger
plt.rcParams.update({'font.size': 18})

# Global variable to store the simulation scan directory
simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/withRadialDiff/'

def generate_filename_list(simulationName):

    # For each simulation, the data is stored under-
    # /mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/[SIMULATION NAME]/simulation_results/[TIMES]/components/neut__kn1dc_5/kn1dc.nc
    # /mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/[SIMULATION NAME]/simulation_results/[TIMES]/components/neut__kn1dc_5/kn1dc_gasbox.nc
    # Here, the times go from 1.000 to X.000 where X is the final timestep. This changes by simulation.

    simulationDir = simulationScanDir + simulationName + '/simulation_results/'

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

    filenameListMain = []
    filenameListGasBox = []
    filenameListCQL3D = []
    filenameListEQDSK = []
    for i in range(len(simTimes)):

        # Main vessel
        filenameCurr = simulationDir+simTimes[i]+'/components/neut__kn1dc_5/kn1dc_lite.nc'
        filenameListMain.append(filenameCurr)

        # Gas box
        filenameCurr = simulationDir+simTimes[i]+'/components/neut__kn1dc_5/kn1dc_gasbox_lite.nc'
        filenameListGasBox.append(filenameCurr)

        # CQL3D
        filenameCurr = simulationDir+simTimes[i]+'/components/fp__cql3dm_4/WHAM.nc'
        filenameListCQL3D.append(filenameCurr)

        # EQDSK
        filenameCurr = simulationDir+simTimes[i]+'/components/eq__pleiades_3/eqdsk'
        filenameListEQDSK.append(filenameCurr)

    filenameListMain = np.array(filenameListMain)
    filenameListGasBox = np.array(filenameListGasBox)
    filenameListCQL3D = np.array(filenameListCQL3D)
    filenameListEQDSK = np.array(filenameListEQDSK)

    return filenameListMain, filenameListGasBox, filenameListCQL3D, filenameListEQDSK

def source_rate_profile(filename, makeplot=False):

    # Open the file
    ds = xr.open_dataset(filename,
                         decode_timedelta=False)

    # Particle source rate for each velocity point
    # [radius x vperp x vpar]
    net_ion_source_vperpvpar = ds['net_ion_source_vperpvpar'].values

    # Radial positon [m]
    rhoH = ds['rhoH_lowres'].values

    # Velocity volume elements
    vol_2v = ds['vol_2v'].values

    # Calculate the source rate for each radial position
    source_integrand = net_ion_source_vperpvpar * vol_2v[np.newaxis, :, :]
    Sion = np.sum(source_integrand, axis=(1, 2))

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(rhoH, Sion)

        plt.show()

    return Sion, rhoH

def sim_times(filenameListCQL3D):

    timeArr = np.array([])
    currTime = 0

    for i in range(len(filenameListCQL3D)):

        # Open the file
        ds = xr.open_dataset(filenameListCQL3D[i],
                             decode_timedelta=False)

        # Get the times
        time = ds['time'].values

        # Append the last timestep to the array
        timeArr = np.concatenate([timeArr, [time[-1]+currTime]])
        currTime = timeArr[-1]

    return timeArr

def time_dep_source_rate(simName, makeplot=False):

    # Get the filename lists
    filenameListMain, filenameListGasBox, filenameListCQL3D, _ = generate_filename_list(simName)

    # Time array [s]
    timeArr = sim_times(filenameListCQL3D)

    # Time dependent main vessel source rate
    Sion2D = []
    rhoH = None
    for i in range(len(filenameListMain)):

        Sion, rhoH = source_rate_profile(filenameListMain[i])
        Sion2D.append(Sion)

    Sion2D = np.array(Sion2D)

    # Time dependent gas box source rate
    Sion2DGB = []
    rhoHGB = None
    for i in range(len(filenameListGasBox)):
    
        Sion, rhoHGB = source_rate_profile(filenameListGasBox[i])
        Sion2DGB.append(Sion)

    Sion2DGB = np.array(Sion2DGB)

    if makeplot:

        fig = plt.figure(figsize=(10, 10), tight_layout=True)

        xMin = 0
        xMax = 0.2

        # Gax Box
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title('Gas Box')
        # Main Vessel
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title('Main Vessel')

        # Color each time point on a colormap
        cmap = plt.get_cmap('viridis', len(timeArr)).colors

        for i in range(len(timeArr)):

            # Gas box source rate
            ax1.plot(rhoHGB, Sion2DGB[i], 
                     label=f'{np.round(timeArr[i]*1e3, 1)}',
                     color=cmap[i])

            # Main vessel source rate
            ax2.plot(rhoH, Sion2D[i],
                     color=cmap[i])

        ax1.set_xlim(xMin, xMax)
        ax2.set_xlim(xMin, xMax)

        ax1.set_xticks([])
        ax2.set_xlabel('Radius [m]')

        ax1.legend(title='Time [ms]', ncols=3)

        ax1.set_ylabel('Source Rate [#/s]')
        ax2.set_ylabel('Source Rate [#/s]')

        plt.show()

    return rhoH, Sion2D, rhoHGB, Sion2DGB, timeArr

def b_field_interpolation(filenameEQDSK, makeplot=False):
    """
    This function defines the magnetic field interpolation global variables
    used by a bunch of other functions in this script.

    Parameters
    ----------
    filenameEQDSK : str
        Location of the eqdsk file being used to generate the magnetic field
        interpolation variables.
    """

    # Save the .npz files in the same directory as the eqdsk itself for easy parsing
    filenameArr = filenameEQDSK.split('/')
    filenameArr = filenameArr[:-1]
    filenameArr.append('eqdsk_npz_data')

    savename = '/'.join(filenameArr)

    # Start the script to create an .npz file with the quantities of interest
    # in the shared pleiades_env
    eqdskScriptPath = '/home/sanwalka/synthetic_proton_detector/eqdsk_analysis_functions.py'
    subprocess.run(['/share/envs/pleiades_env/bin/python',
                    eqdskScriptPath,
                    '-filename', filenameEQDSK,
                    '-savename', savename], check=True)

    # Load the data from the .npz file made by this function
    npzFilename = np.load(savename+'.npz', allow_pickle=True)

    # These are all 2D arrays with regular spacing for r and z.
    Rmesh = npzFilename['Rmesh']
    Zmesh = npzFilename['Zmesh']
    Br = npzFilename['Br']
    Bz = npzFilename['Bz']
    Bmag = npzFilename['Bmag']
    eqdsk_psi = npzFilename['magneticFlux']
    psilim = npzFilename['psilim']

    r1D = Rmesh[0]
    z1D = Zmesh[:, 0]

    # Interpolation function for the flux
    fluxInterpFunc = sc.interpolate.RectBivariateSpline(r1D, z1D, eqdsk_psi)

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.contour(Zmesh, Rmesh, eqdsk_psi)

        plt.show()

    return

def source_rate_flux(simName, makeplot=False):

    # Load the source rate data vs. real coordinates
    # rhoH, rhoGB are 1D [m]
    # timeArr is 1D [s]
    # Sion2D, Sion2DGB have shape [Time x radius]
    rhoH, Sion2D, rhoHGB, Sion2DGB, timeArr = time_dep_source_rate(simName)

    # Load the eqdsk files for each IPS timestep
    _, _, _, filenameListEQDSK = generate_filename_list(simName)

    # Load the EQDSK file for each timestep

def disambiguate_source_rate(simName, timeToPlot, zGasBox=0.8, makeplot=True):
    """
    Separate the gas box and main vessel contributions to the main vessel source rate.

    The gas box source rate is calculated at z = zGasBox and the main vessel source rate
    at the midplane (z = 0). The main vessel source rate includes the contribution from
    the gas box, so the gas box source rate is mapped to the midplane along lines of
    constant flux and subtracted from the main vessel source rate.

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timeToPlot : float
        The time at which to disambiguate the source rates. [s]
        The closest available simulation timestep is used.
    zGasBox : float
        Axial location of the gas box source rate. [m]
        Default is 0.8.
    makeplot : bool
        Plot the original, remapped and disambiguated source rates.
        Default is True.

    Returns
    -------
    rhoH : np.array
        Midplane radius of the main vessel source rate. [m]
    SionMV : np.array
        Main vessel source rate with the gas box contribution removed. [#/s]
    SionGBMid : np.array
        Gas box source rate mapped to the midplane on the rhoH grid. [#/s]
    timeVal : float
        The simulation time actually used. [s]
    """

    # rhoH, rhoHGB are 1D [m]
    # Sion2D, Sion2DGB have shape [Time x radius]
    rhoH, Sion2D, rhoHGB, Sion2DGB, timeArr = time_dep_source_rate(simName)

    # Find the index of the closest time in the simulation data
    timeIdx = np.argmin(np.abs(timeArr - timeToPlot))
    timeVal = timeArr[timeIdx]

    # Clip the unphysical negative source rates before remapping and subtracting
    SionMain = np.clip(Sion2D[timeIdx], a_min=0, a_max=None)
    SionGB = np.clip(Sion2DGB[timeIdx], a_min=0, a_max=None)

    # Poloidal flux from the eqdsk of the last IPS timestep [z x r]
    Rmesh, Zmesh, psi = load_eqdsk_flux(simName)
    fluxInterpFunc = sc.interpolate.RectBivariateSpline(Zmesh[:, 0], Rmesh[0], psi)

    # Flux of each gas box radial point
    psiGB = fluxInterpFunc(zGasBox, rhoHGB).ravel()

    # Invert psi(r) at the midplane to find the radius of each gas box flux surface.
    # psi increases monotonically with r, so it can be used directly as the interpolation abscissa.
    rMidFine = np.linspace(0, Rmesh.max(), 2000)
    psiMidFine = fluxInterpFunc(0, rMidFine).ravel()
    rhoHGBMid = np.interp(psiGB, psiMidFine, rMidFine, right=np.nan)

    # Put the remapped gas box source rate on the main vessel grid.
    # Inside the innermost remapped point the profile is flat, outside the gas box grid there is no source.
    valid = ~np.isnan(rhoHGBMid)
    SionGBMid = np.interp(rhoH, rhoHGBMid[valid], SionGB[valid], left=SionGB[valid][0], right=0)

    # Remove the gas box contribution from the main vessel
    SionMV = SionMain - SionGBMid

    if makeplot:

        fig = plt.figure(figsize=(10, 14), tight_layout=True)
        fig.suptitle(f'{simName}\nTime = {np.round(timeVal*1e3, 2)} ms', fontsize=16)

        xMin = 0
        xMax = 0.2

        # Gas box as calculated
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_title(f'Gas Box (z = {zGasBox} m)')

        ax1.plot(rhoHGB, SionGB, color='tab:orange')

        # Main vessel and the remapped gas box
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax2.set_title('Main Vessel (z = 0 m)')

        ax2.plot(rhoH, SionMain, label='Main Vessel (total)', color='tab:blue')
        ax2.plot(rhoH, SionGBMid, label='Gas Box (remapped)', color='tab:orange', linestyle='--')

        # Disambiguated contributions at the midplane
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        ax3.set_title('Disambiguated (z = 0 m)')

        ax3.plot(rhoH, SionMV, label='Main Vessel', color='tab:blue')
        ax3.plot(rhoH, SionGBMid, label='Gas Box', color='tab:orange')

        ax1.set_xlim(xMin, xMax)

        ax1.tick_params(labelbottom=False)
        ax2.tick_params(labelbottom=False)
        ax3.set_xlabel('Radius [m]')

        ax2.legend()
        ax3.legend()

        ax1.set_ylabel('Source Rate [#/s]')
        ax2.set_ylabel('Source Rate [#/s]')
        ax3.set_ylabel('Source Rate [#/s]')

        # Source rates span multiple decades
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')

        plt.show()

    return rhoH, SionMV, SionGBMid, timeVal

if __name__ == '__main__':

    simName = 'nneut_2e17_gb_2e17_NBI_800kW_ECH_0kW_ionDrrOn'

    # Get the filename lists
    filenameListMain, filenameListGasBox, filenameListCQL3D, filenameListEQDSK = generate_filename_list(simName)

    # Radial source rate
    # Sion, rhoH = source_rate_profile(filenameListMain[0], makeplot=True)

    # IPS time array
    # timeArr = sim_times(filenameListCQL3D)

    # Time dependent source rates
    # rhoH, Sion2D, rhoHGB, Sion2DGB, timeArr = time_dep_source_rate(simName, makeplot=True)

    # Separate the gas box and main vessel source rates at the midplane
    disambiguate_source_rate(simName, timeToPlot=3e-3)

    # Source rates vs. flux
    # source_rate_flux(simName, True)

    # Plot flux surfaces
    # b_field_interpolation(filenameListEQDSK[0], makeplot=True)
