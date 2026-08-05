"""
This script returns the source rate for the main vessel and gas box vs. normalized midplane radius 
"""

import os
import subprocess
import scipy as sc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

# Make the font size larger
plt.rcParams.update({'font.size': 18})

def generate_filename_list(simulationName):

    # For each simulation, the data is stored under-
    # /mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/[SIMULATION NAME]/simulation_results/[TIMES]/components/neut__kn1dc_5/kn1dc.nc
    # /mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/[SIMULATION NAME]/simulation_results/[TIMES]/components/neut__kn1dc_5/kn1dc_gasbox.nc
    # Here, the times go from 1.000 to X.000 where X is the final timestep. This changes by simulation.

    simulationDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/' + simulationName + '/simulation_results/'

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

if __name__ == '__main__':

    simName = 'nneut_1e18_gb_2e18_NBI_800kW_ECH_0kW'

    # Get the filename lists
    filenameListMain, filenameListGasBox, filenameListCQL3D, filenameListEQDSK = generate_filename_list(simName)

    # Radial source rate
    # Sion, rhoH = source_rate_profile(filenameListMain[0], makeplot=True)

    # IPS time array
    # timeArr = sim_times(filenameListCQL3D)

    # Time dependent source rates
    # rhoH, Sion2D, rhoHGB, Sion2DGB, timeArr = time_dep_source_rate(simName, makeplot=True)

    # Source rates vs. flux
    # source_rate_flux(simName)

    # Plot flux surfaces
    # b_field_interpolation(filenameListEQDSK[0], makeplot=True)
