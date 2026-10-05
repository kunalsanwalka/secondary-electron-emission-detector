"""
This script extends the CQL3D midplane density profile out to the plasma radius measured by AXUV.

CQL3D only solves out to the edge of its own radial grid, which sits well inside the plasma
radius seen by AXUV. The density between the two is filled in with a tanh fall-off that goes
from the simulated edge density to 0 at the AXUV radius.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

from pathlib import Path
from scipy.stats import binned_statistic

import cql3d_radial_profiles

from cql3d_radial_profiles import load_eqdsk_flux
from cql3d_vs_data import find_valid_time_slices, maxIonDensity

# Global variable to store the simulation scan directory
global simulationScanDir
simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/withRadialDiff/'

# load_eqdsk_flux() looks for the eqdsk relative to the scan directory defined in its own
# module, so point that at the one defined above
cql3d_radial_profiles.simulationScanDir = simulationScanDir

# The AXUV results are written to the data directory at the top level of the repository
axuvDataDir = Path(__file__).resolve().parent.parent / 'data'

def load_simulation_density(simulationName):
    """
    Load the midplane radial density profile vs. time from the cached simulation data.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.

    Returns
    -------
    times : np.array
        1D array of the simulation times. [s]
    rSim : np.array
        1D array of the midplane radial positions of the simulation grid. [m]
    densSim : np.array
        2D array [Time x r] of the midplane density. [m^-3]
    """

    # All simulations are stored in the same directory
    simulationDir = simulationScanDir + simulationName + '/'

    # Load the saved data
    with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:
        saveData = pickle.load(loadFile)

        solrz = saveData['solrz']
        times = saveData['times']

        # [Time x r x z]
        dens = saveData['dens']

    # Drop the CQL3D restart slices and clip the unphysical density spikes
    keep = find_valid_time_slices(times)
    times = times[keep]
    dens = np.clip(dens[keep], a_min=0, a_max=maxIonDensity)

    # The midplane is the first axial index
    rSim = solrz[:, 0]
    densSim = dens[:, :, 0]

    return times, rSim, densSim

def average_down(timeArr, values, rateKHz):
    """
    Average the values into windows of width 1/rateKHz.

    This is the same averaging as is done in axuv_scripts/plot_plasma_radius.py.

    Parameters
    ----------
    timeArr : np.array
        1D array of the times of the values. [s]
    values : np.array
        1D array of the values being averaged down.
    rateKHz : float
        Width of the averaging window. [kHz]

    Returns
    -------
    centres : np.array
        1D array of the centres of the averaging windows. [s]
    means : np.array
        1D array of the values averaged within each window.
    """

    windowWidth = 1 / (rateKHz * 1e3)
    bins = np.arange(timeArr[0], timeArr[-1] + windowWidth, windowWidth)

    means, edges, _ = binned_statistic(timeArr, values, statistic='mean', bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Empty bins come back as NaN, which would break the interpolation
    keep = np.isfinite(means)

    return centres[keep], means[keep]

def load_axuv_radius(shotnum, diodeArrayNum, rateKHz=10.0, tMin=None, tMax=None):
    """
    Load the plasma radius vs. time measured by AXUV for the given shot.

    The radius is the RMS radius stored under the `radius_cm` key of the AXUV NPZ files
    (see axuv_scripts/plot_plasma_radius.py). It is averaged down from the native (~1 MHz)
    rate to suppress the noise.

    Parameters
    ----------
    shotnum : int
        The shot number.
    diodeArrayNum : int
        The diode array the radius is taken from, i.e. 1 for DIODEARRAY1.
    rateKHz : float
        Rate the native data is averaged down to. [kHz]
        Default is 10.
    tMin : float
        Start of the experimental time window the data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    tMax : float
        End of the experimental time window the data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.

    Returns
    -------
    axuvTimes : np.array
        1D array of the times of the averaged down radius. [s]
    axuvRadius : np.array
        1D array of the plasma radius. [m]
    """

    # The data directory can hold duplicates of the same diode (e.g. ' copy' files), which
    # sort after the original they copy
    filenameList = sorted(axuvDataDir.glob(f'AXUV_results*{shotnum}*DA{diodeArrayNum}*.npz'),
                          key=lambda p: ('copy' in p.stem, p.name))

    if len(filenameList) == 0:
        raise FileNotFoundError(f'No AXUV data for shot {shotnum}, DA{diodeArrayNum} in {axuvDataDir}')

    with np.load(filenameList[0], allow_pickle=True) as data:
        axuvTimes = data[f's{shotnum}_time_s']
        # [cm]
        axuvRadius = data[f's{shotnum}_radius_cm']

    axuvTimes, axuvRadius = average_down(axuvTimes, axuvRadius, rateKHz)

    # Convert to m to match the simulation
    axuvRadius = axuvRadius / 1e2

    # Trim to the part of the shot the radius is well behaved in
    keep = np.ones(len(axuvTimes), dtype=bool)
    if tMin is not None:
        keep &= axuvTimes >= tMin
    if tMax is not None:
        keep &= axuvTimes <= tMax

    if not np.any(keep):
        raise ValueError(f'No AXUV data between tMin={tMin} s and tMax={tMax} s for shot {shotnum}')

    axuvTimes = axuvTimes[keep]
    axuvRadius = axuvRadius[keep]

    return axuvTimes, axuvRadius

def midplane_eqdsk_grid(simulationName):
    """
    Get the midplane radial grid of the eqdsk of the last IPS timestep of the given simulation.

    The density extension is evaluated on this grid so that it lives on the same radial grid
    as the equilibrium rather than on an arbitrary one.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.

    Returns
    -------
    rEQDSK : np.array
        1D array of the radial positions of the eqdsk grid. [m]
    """

    Rmesh, Zmesh, _ = load_eqdsk_flux(simulationName)

    # The eqdsk grid is regular, so the radial positions are the same at every axial position
    rEQDSK = Rmesh[0]

    return rEQDSK

def extend_profile(rSim, densSim, plasmaRadius, rEQDSK, tanhSteepness=4.0):
    """
    Extend a midplane density profile from the edge of the simulation grid out to the plasma radius.

    The density falls from the edge density of the simulation to 0 at the plasma radius as a
    tanh, rescaled so that it hits both end points exactly.

    Parameters
    ----------
    rSim : np.array
        1D array of the midplane radial positions of the simulation grid. [m]
    densSim : np.array
        1D array of the midplane density of the simulation. [m^-3]
    plasmaRadius : float
        Radius at which the density goes to 0. [m]
    rEQDSK : np.array
        1D array of the radial positions of the eqdsk grid, used as the grid of the extension. [m]
    tanhSteepness : float
        Steepness of the tanh fall-off. Larger values keep the density flat for longer before
        dropping sharply at the plasma radius.
        Default is 4.

    Returns
    -------
    rExt : np.array
        1D array of the radial positions of the extension. [m]
        This starts at the edge of the simulation grid so the extension joins onto the profile.
    densExt : np.array
        1D array of the density of the extension. [m^-3]
    """

    rEdge = rSim[-1]
    densEdge = densSim[-1]

    # Nothing to extend if AXUV sees a plasma smaller than the simulation grid
    if plasmaRadius <= rEdge:
        return np.array([]), np.array([])

    # The eqdsk points between the two radii, with the end points added so the extension
    # joins the simulation exactly and reaches 0 exactly at the plasma radius
    rExt = rEQDSK[(rEQDSK > rEdge) & (rEQDSK < plasmaRadius)]
    rExt = np.concatenate([[rEdge], rExt, [plasmaRadius]])

    # Normalized coordinate, 0 at the edge of the simulation and 1 at the plasma radius
    rNorm = (rExt - rEdge) / (plasmaRadius - rEdge)

    tanhProfile = 0.5 * (1 - np.tanh(tanhSteepness * (rNorm - 0.5)))

    # Rescale so the fall-off goes from exactly densEdge to exactly 0
    tanhStart = 0.5 * (1 + np.tanh(tanhSteepness / 2))
    tanhEnd = 0.5 * (1 - np.tanh(tanhSteepness / 2))

    densExt = densEdge * (tanhProfile - tanhEnd) / (tanhStart - tanhEnd)

    return rExt, densExt

def plot_extended_radial_profiles(simulationName, shotnum, diodeArrayNum, simTimeToPlot,
                                  cmap='viridis', rateKHz=10.0, tanhSteepness=4.0,
                                  tMin=None, tMax=None):
    """
    Plot the midplane density profile of the simulation at one timestep, extended out to the
    AXUV plasma radius at every experimental timestep.

    The simulated profile is drawn in black and each extension is coloured by the experimental
    time its plasma radius comes from.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the AXUV plasma radius is taken from.
    diodeArrayNum : int
        The diode array the radius is taken from, i.e. 1 for DIODEARRAY1.
    simTimeToPlot : float
        The simulation time whose profile is extended. [s]
        The closest available simulation timestep is used.
    cmap : str
        Colormap used to color the extensions by experimental time.
        Default is 'viridis'.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Default is 4.
    tMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    tMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.
    """

    # Midplane density of the simulation vs. time
    times, rSim, densSim = load_simulation_density(simulationName)

    # Plasma radius measured by AXUV vs. time
    axuvTimes, axuvRadius = load_axuv_radius(shotnum, diodeArrayNum, rateKHz=rateKHz,
                                             tMin=tMin, tMax=tMax)

    # Radial grid the extension is evaluated on
    rEQDSK = midplane_eqdsk_grid(simulationName)

    # The one simulated profile that gets extended
    simTimeIdx = np.argmin(np.abs(times - simTimeToPlot))
    radialProfile = densSim[simTimeIdx]

    # Color each extension by the experimental time of its plasma radius
    norm = mpl.colors.Normalize(vmin=axuvTimes[0]*1e3, vmax=axuvTimes[-1]*1e3)
    colormap = mpl.colormaps[cmap]

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Extend the simulated profile to the plasma radius at every experimental timestep
    for i in range(len(axuvTimes)):

        rExt, densExt = extend_profile(rSim, radialProfile, axuvRadius[i],
                                       rEQDSK, tanhSteepness=tanhSteepness)

        color = colormap(norm(axuvTimes[i]*1e3))

        ax.plot(rExt, densExt,
                color=color,
                linewidth=1)
        ax.plot(-rExt, densExt,
                color=color,
                linewidth=1)

    # The simulated profile is the same for every extension, so it only gets drawn once
    ax.plot(rSim, radialProfile,
            color='black',
            linewidth=3,
            label=f'Simulation, t = {times[simTimeIdx]*1e3:.4g} ms')
    ax.plot(-rSim, radialProfile,
            color='black',
            linewidth=3)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
    cbar.set_label(f'Shot {shotnum} DA{diodeArrayNum} Time [ms]')

    ax.legend(loc='upper center')

    ax.set_title(simulationName)
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel(r'n$_i$ [m$^{-3}$]')

    xLim = 1.1 * np.max(axuvRadius)
    ax.set_xlim(-xLim, xLim)
    ax.set_ylim(0, None)

    plt.show()

    return

def map_along_flux_surfaces(rMidplane, zGrid, Rmesh, Zmesh, psi, rSimOuter):
    """
    Map a set of midplane radii along z by following the eqdsk flux surface each one sits on.

    The flux surface through a midplane radius is the contour of constant poloidal flux through
    that point, so the radius of that surface at an axial position z is the radius at which the
    flux at z equals the flux at the midplane.

    Parameters
    ----------
    rMidplane : np.array
        1D array of the midplane radii of the flux surfaces being mapped. [m]
    zGrid : np.array
        1D array of the axial positions the surfaces are mapped onto. [m]
    Rmesh : np.array
        2D array [z x r] of the radial coordinates of the eqdsk grid. [m]
    Zmesh : np.array
        2D array [z x r] of the axial coordinates of the eqdsk grid. [m]
    psi : np.array
        2D array [z x r] of the poloidal flux on the eqdsk grid. [Wb]
    rSimOuter : np.array
        1D array of the radius of the outermost simulated flux surface at each point of zGrid. [m]
        This is used to continue the mapping where the eqdsk cannot (see the note below).

    Returns
    -------
    rMapped : np.array
        2D array [rMidplane x zGrid] of the radius of each flux surface at each axial position. [m]
    """

    # The eqdsk grid is regular, so the same radial (axial) positions are used at every axial
    # (radial) position
    r1D = Rmesh[0]
    z1D = Zmesh[:, 0]

    def monotonic_prefix(psiSlice):
        """
        Number of points of an axial slice of the flux over which it increases with radius.

        The flux only rises monotonically from the machine axis out to the point where the
        eqdsk grid runs past the mirror coils, beyond which it turns over. Only the rising
        part can be inverted to get a radius from a flux.
        """

        turnover = np.nonzero(np.diff(psiSlice) <= 0)[0]

        if len(turnover) == 0:
            return len(psiSlice)

        return turnover[0] + 1

    # Flux on each of the surfaces being mapped
    izMid = np.argmin(np.abs(z1D))
    nMid = monotonic_prefix(psi[izMid])
    psiTarget = np.interp(rMidplane, r1D[:nMid], psi[izMid, :nMid])

    # Radius of each surface at every axial position of the eqdsk
    rTrace = np.full((len(rMidplane), len(z1D)), np.nan)
    for i in range(len(z1D)):

        nRise = monotonic_prefix(psi[i])

        if nRise < 2:
            continue

        # Surfaces whose flux is above the largest invertible flux of this slice have left
        # the eqdsk grid and are filled in further down instead
        onGrid = psiTarget <= psi[i, nRise-1]

        rTrace[onGrid, i] = np.interp(psiTarget[onGrid], psi[i, :nRise], r1D[:nRise])

    # Put the traced surfaces onto the axial grid of the simulation
    rMapped = np.full((len(rMidplane), len(zGrid)), np.nan)
    for k in range(len(rMidplane)):

        onGrid = np.isfinite(rTrace[k])

        # A surface that never lands on the eqdsk grid is left as NaN for the fill in below
        if np.sum(onGrid) < 2:
            continue

        rMapped[k] = np.interp(zGrid, z1D[onGrid], rTrace[k, onGrid],
                               left=np.nan, right=np.nan)

    # Past the mirror throat the surfaces expand off the edge of the eqdsk grid. There they are
    # continued with the flux expansion of the outermost simulated surface, anchored to the last
    # radius that could be traced so that the mapping stays continuous.
    for k in range(len(rMidplane)):

        onGrid = np.isfinite(rMapped[k])

        if np.all(onGrid):
            continue

        if not np.any(onGrid):
            rMapped[k] = rMidplane[k] * rSimOuter / rSimOuter[0]
            continue

        idxOnGrid = np.nonzero(onGrid)[0]

        offGridLow = np.arange(0, idxOnGrid[0])
        offGridHigh = np.arange(idxOnGrid[-1] + 1, len(zGrid))

        for edgeIdx, offGrid in ((idxOnGrid[0], offGridLow), (idxOnGrid[-1], offGridHigh)):

            if len(offGrid) == 0:
                continue

            rMapped[k, offGrid] = rMapped[k, edgeIdx] * rSimOuter[offGrid] / rSimOuter[edgeIdx]

    return rMapped

def build_density_extension(simulationName, shotnum, diodeArrayNum,
                            rateKHz=10.0, tanhSteepness=4.0, tMin=None, tMax=None,
                            redoAnalysis=False):
    """
    Build everything the density extension needs that does not depend on the simulation time.

    The radial grid of the extension, the flux surfaces it is mapped along and the shape of its
    tanh fall-off are all the same at every simulated timestep, so they are worked out once here
    and scaled by the edge density of whichever timestep is being extended.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the AXUV plasma radius is taken from.
    diodeArrayNum : int
        The diode array the radius is taken from, i.e. 1 for DIODEARRAY1.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Default is 4.
    tMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    tMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.
    redoAnalysis : bool
        Force the extension to be rebuilt instead of loading the saved one.
        Default is False.

    Returns
    -------
    extData : dict
        Dictionary holding-
            'times' : 1D array of the simulation times. [s]
            'densSim' : 3D array [simTime x r x z] of the simulated density. [m^-3]
            'solrz' : 2D array [r x z] of the r values of the simulation grid. [m]
            'solzz' : 2D array [r x z] of the z values of the simulation grid. [m]
            'axuvTimes' : 1D array of the experimental times of the AXUV plasma radius. [s]
            'axuvRadius' : 1D array of the AXUV plasma radius. [m]
            'rExtGrid' : 1D array of the midplane radii of the extension. [m]
            'rExtMapped' : 2D array [rExt x z] of the radii of the flux surfaces of the
                           extension at every axial position. [m]
            'zGrid' : 1D array of the axial positions of the extension. [m]
            'shapeExt' : 2D array [expTime x rExt] of the shape of the extension, normalized
                         to 1 at the edge of the simulation grid.
            'Rmesh', 'Zmesh', 'psi' : the eqdsk grid and poloidal flux on it. [m, m, Wb]

    Notes
    -----
    The parts of the extension that do not come out of density_interp_data.pkl are cached in-
    simulationScanDir + f'{simulationName}/density_interp_data_{shotnum}_extension.pkl'
    which is kept separate from the density_interp_data.pkl of the unextended simulation. The
    simulated density is not written to it, so it stays small and cannot go out of sync with
    the unextended data. The cache records the settings it was built with and is rebuilt
    automatically if any of them, or the simulation grid, has changed since.
    """

    # All simulations are stored in the same directory
    simulationDir = simulationScanDir + simulationName + '/'

    # Kept separate from the unextended density_interp_data.pkl
    savePath = simulationDir + f'density_interp_data_{shotnum}_extension.pkl'

    # Load the saved data
    with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:
        saveData = pickle.load(loadFile)

        solrz = saveData['solrz']
        solzz = saveData['solzz']
        times = saveData['times']

        # [Time x r x z]
        dens = saveData['dens']

    # Drop the CQL3D restart slices and clip the unphysical density spikes
    keep = find_valid_time_slices(times)
    times = times[keep]
    dens = np.clip(dens[keep], a_min=0, a_max=maxIonDensity)

    # The settings the cached extension has to have been built with for it to be reused. The
    # shot number is in the filename, so it does not need to be checked here.
    settings = {'diodeArrayNum': diodeArrayNum,
                'rateKHz': rateKHz,
                'tanhSteepness': tanhSteepness,
                'tMin': tMin,
                'tMax': tMax}

    # The extension is mapped along the flux surfaces starting from the outermost surface of
    # the simulation grid, so a cache built on a different grid cannot be reused
    gridFingerprint = {'rSimOuter': solrz[-1], 'zGrid': solzz[-1]}

    # Load the saved extension if it already exists
    try:

        if redoAnalysis:
            raise Exception('Forcing redo of the density extension')

        print(f'Trying to load the saved density extension for {simulationName}')

        with open(savePath, 'rb') as loadFile:
            extData = pickle.load(loadFile)

        if extData['settings'] != settings:
            raise Exception('The saved density extension was built with different settings, '
                            'rebuilding it.')

        if not all(np.array_equal(extData['gridFingerprint'][key], gridFingerprint[key])
                   for key in gridFingerprint):
            raise Exception('The saved density extension was built on a different simulation '
                            'grid, rebuilding it.')

        # The simulated density is not cached, so that it cannot go out of sync with
        # density_interp_data.pkl, and is put back on here
        extData['times'] = times
        extData['densSim'] = dens
        extData['solrz'] = solrz
        extData['solzz'] = solzz

        return extData

    except Exception as e:

        print(e)
        print('No saved density extension present, generating it.')

    # Plasma radius measured by AXUV vs. time
    axuvTimes, axuvRadius = load_axuv_radius(shotnum, diodeArrayNum, rateKHz=rateKHz,
                                             tMin=tMin, tMax=tMax)

    # Equilibrium the extension is mapped along
    Rmesh, Zmesh, psi = load_eqdsk_flux(simulationName)

    # The eqdsk grid is regular, so the radial positions are the same at every axial position
    rEQDSK = Rmesh[0]

    # Midplane radii of the simulation grid, and the edge it is extended from
    rSim = solrz[:, 0]
    rEdge = rSim[-1]

    # The extension has to be on one radial grid for every experimental time, so it is put on
    # the eqdsk points out to the largest plasma radius of the shot. The density at each time
    # then falls to 0 at that time's plasma radius and stays 0 beyond it.
    # The largest plasma radius is added as the outermost point so the extension ends exactly
    # there, and the eqdsk points within half a cell of it are dropped so that the last two
    # surfaces do not sit on top of each other
    rMaxExt = np.max(axuvRadius)
    halfCell = 0.5 * np.mean(np.diff(rEQDSK))

    rExtGrid = rEQDSK[(rEQDSK > rEdge) & (rEQDSK < rMaxExt - halfCell)]
    rExtGrid = np.concatenate([rExtGrid, [rMaxExt]])

    # Axial grid of the extension, taken from the outermost simulated surface as that is the
    # one the extension joins onto
    zGrid = solzz[-1]
    rSimOuter = solrz[-1]

    # Radius of each point of the extension at every axial position of its flux surface
    rExtMapped = map_along_flux_surfaces(rExtGrid, zGrid, Rmesh, Zmesh, psi, rSimOuter)

    # The tanh fall-off is proportional to the density at the edge of the simulation grid, so
    # its shape is worked out once with an edge density of 1 and scaled by the edge density of
    # whichever timestep is being extended. This is the same extension extend_profile() makes,
    # without having to call it once per simulated timestep.
    shapeExt = np.zeros((len(axuvTimes), len(rExtGrid)))
    for i in range(len(axuvTimes)):

        rExt, shapeProfile = extend_profile(rSim, np.ones_like(rSim), axuvRadius[i],
                                            rEQDSK, tanhSteepness=tanhSteepness)

        # AXUV can see a plasma smaller than the simulation grid, in which case there is
        # nothing to extend and the shape stays 0
        if len(rExt) == 0:
            continue

        # Onto the common radial grid, with nothing outside this time's plasma radius
        shapeExt[i] = np.interp(rExtGrid, rExt, shapeProfile,
                                left=shapeProfile[0], right=0.0)

    extData = {'times': times,
               'densSim': dens,
               'solrz': solrz,
               'solzz': solzz,
               'axuvTimes': axuvTimes,
               'axuvRadius': axuvRadius,
               'rExtGrid': rExtGrid,
               'rExtMapped': rExtMapped,
               'zGrid': zGrid,
               'shapeExt': shapeExt,
               'Rmesh': Rmesh,
               'Zmesh': Zmesh,
               'psi': psi,
               'settings': settings,
               'gridFingerprint': gridFingerprint}

    #### Save the data

    # Everything except the simulated density, which is left in density_interp_data.pkl so
    # that the two cannot go out of sync and so that this file stays small
    saveData = {key: value for key, value in extData.items()
                if key not in ('times', 'densSim', 'solrz', 'solzz')}

    with open(savePath, 'wb') as saveFile:
        pickle.dump(saveData, saveFile)

    print(f'Saved the density extension to- \n {savePath}')

    return extData

def plot_flux_surfaces(ax, Rmesh, Zmesh, psi, rEdgeOfPlot, fluxColor='white', numFluxSurfaces=15):
    """
    Draw the eqdsk flux surfaces on top of a (Z, R) density plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis the surfaces are drawn on.
    Rmesh : np.array
        2D array [z x r] of the radial coordinates of the eqdsk grid. [m]
    Zmesh : np.array
        2D array [z x r] of the axial coordinates of the eqdsk grid. [m]
    psi : np.array
        2D array [z x r] of the poloidal flux on the eqdsk grid. [Wb]
    rEdgeOfPlot : float
        Largest radius shown on the plot, which sets the spacing of the surfaces. [m]
    fluxColor : str
        Color of the flux surfaces.
        Default is 'white'.
    numFluxSurfaces : int
        Number of flux surfaces to draw.
        Default is 15.
    """

    # The flux at the mirror throat is far above the flux at the midplane, so the spacing of
    # the surfaces is set by the midplane flux out to the edge of the plot. Spacing it by the
    # largest flux anywhere in the plot would push almost every surface outside it.
    izMid = np.argmin(np.abs(Zmesh[:, 0]))
    psiEdge = np.interp(rEdgeOfPlot, Rmesh[0], psi[izMid])

    # Evenly spaced in flux, dropping the 0 level as it is the machine axis
    fluxLevels = np.linspace(0, psiEdge*1e6, numFluxSurfaces + 1)[1:]

    ax.contour(Zmesh, Rmesh, psi*1e6, levels=fluxLevels, colors=fluxColor, linewidths=1.5)

    return

def extend_2d_density(simulationName, shotnum, diodeArrayNum, simTimeToPlot,
                      makeplot=False, expTimeToPlot=None,
                      rateKHz=10.0, tanhSteepness=4.0, tMin=None, tMax=None,
                      redoAnalysis=False,
                      densCmap='inferno', extCmap='viridis', fluxColor='white',
                      numFluxSurfaces=15):
    """
    Build a 2D (R, Z) density profile that keeps the simulated density and adds the extension
    out to the AXUV plasma radius onto it.

    The midplane extension is the one made by `extend_profile()`, i.e. the same profile that
    `plot_extended_radial_profiles()` draws. Each radial point of that extension sits on a flux
    surface, and the extension is mapped along z by following that surface out of the midplane.
    The density of the extension is constant along its flux surface, so the whole extension is
    set by its midplane value.

    The simulated part of the profile is the single timestep at `simTimeToPlot`, held fixed. The
    extension changes with time because the AXUV plasma radius does, so the returned profile is
    on the experimental timebase rather than the simulated one. Use `extend_2d_density_all_times()`
    to get the same thing at every simulated timestep instead.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the AXUV plasma radius is taken from.
    diodeArrayNum : int
        The diode array the radius is taken from, i.e. 1 for DIODEARRAY1.
    simTimeToPlot : float
        The simulation time whose density profile is extended. [s]
        The closest available simulation timestep is used.
    makeplot : bool
        Whether to plot the extended 2D density profile.
        Default is False.
    expTimeToPlot : float
        The experimental time whose extension is plotted. [s]
        The closest available experimental timestep is used.
        Default is None, which plots the first timestep of the experimental time window.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Default is 4.
    tMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    tMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.
    redoAnalysis : bool
        Force the density extension to be rebuilt instead of loading the saved one.
        Default is False.
    densCmap : str
        Colormap used for the simulated density.
        Default is 'inferno'.
    extCmap : str
        Colormap used for the density of the extension.
        Default is 'viridis'.
    fluxColor : str
        Color of the eqdsk flux surfaces drawn over the density.
        Default is 'white'.
    numFluxSurfaces : int
        Number of flux surfaces to plot.
        Default is 15.

    Returns
    -------
    dens : np.array
        3D array [Time x R x Z] of the extended ion density profile. [m^-3]
    solrz : np.array
        2D array [R x Z] of the r values. [m]
    solzz : np.array
        2D array [R x Z] of the z values. [m]
    time : np.array
        1D array of the experimental times of the AXUV plasma radius. [s]
    """

    extData = build_density_extension(simulationName, shotnum, diodeArrayNum,
                                      rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                                      tMin=tMin, tMax=tMax, redoAnalysis=redoAnalysis)

    times = extData['times']
    solrz = extData['solrz']
    solzz = extData['solzz']
    axuvTimes = extData['axuvTimes']
    axuvRadius = extData['axuvRadius']
    rExtGrid = extData['rExtGrid']
    rExtMapped = extData['rExtMapped']
    zGrid = extData['zGrid']
    shapeExt = extData['shapeExt']

    # The one simulated timestep that gets extended
    simTimeIdx = np.argmin(np.abs(times - simTimeToPlot))
    densSim = extData['densSim'][simTimeIdx]

    # The extension is the shape of the tanh fall-off scaled by the midplane density at the
    # edge of the simulation grid
    # [Time x r]
    densExtMid = densSim[-1, 0] * shapeExt

    # The density is constant along each flux surface of the extension
    # [Time x r x z]
    densExtFull = np.repeat(densExtMid[:, :, np.newaxis], len(zGrid), axis=2)

    # The simulated density does not change with the experimental time, so it is the same
    # timestep repeated onto the experimental timebase
    densSimFull = np.broadcast_to(densSim, (len(axuvTimes),) + densSim.shape)

    # Stack the extension onto the outside of the simulation grid
    densNew = np.concatenate([densSimFull, densExtFull], axis=1)
    solrzNew = np.concatenate([solrz, rExtMapped], axis=0)
    solzzNew = np.concatenate([solzz, np.tile(zGrid, (len(rExtGrid), 1))], axis=0)

    if makeplot:

        if expTimeToPlot is None:
            expTimeIdx = 0
        else:
            expTimeIdx = np.argmin(np.abs(axuvTimes - expTimeToPlot))

        fig = plt.figure(figsize=(14, 6), tight_layout=True)
        fig.suptitle(simulationName)
        ax = fig.add_subplot(1, 1, 1)

        # Axial and radial extent of the plot
        zLim = (0, 0.8)
        rLim = (0, 1.05 * np.max(axuvRadius))

        #### Simulated density

        simLevels = np.linspace(0, np.max(densSim), 100)

        simObj = ax.contourf(solzz, solrz, densSim, levels=simLevels, cmap=densCmap)

        simCbar = fig.colorbar(simObj, ax=ax)
        simCbar.set_label(r'Simulation n$_i$ [m$^{-3}$]')

        #### Extension

        # The outermost simulated surface is prepended so that the extension joins onto the
        # simulated density with no gap between the two contour plots
        rExtPlot = np.concatenate([[solrz[-1]], rExtMapped], axis=0)
        zExtPlot = np.tile(zGrid, (len(rExtGrid) + 1, 1))
        densExtPlot = np.concatenate([[densSim[-1]], densExtFull[expTimeIdx]], axis=0)

        # Nothing is drawn outside the plasma radius of this timestep
        densExtPlot = np.ma.masked_less_equal(densExtPlot, 0)

        extLevels = np.linspace(0, np.max(densExtMid[expTimeIdx]), 100)

        extObj = ax.contourf(zExtPlot, rExtPlot, densExtPlot, levels=extLevels, cmap=extCmap)

        extCbar = fig.colorbar(extObj, ax=ax)
        extCbar.set_label(r'AXUV extension n$_i$ [m$^{-3}$]')

        #### Flux surfaces

        plot_flux_surfaces(ax, extData['Rmesh'], extData['Zmesh'], extData['psi'], rLim[1],
                           fluxColor=fluxColor, numFluxSurfaces=numFluxSurfaces)

        ax.set_xlim(zLim)
        ax.set_ylim(rLim)
        ax.set_aspect('equal')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title(f'Simulation t = {times[simTimeIdx]*1e3:.4g} ms\n'
                     f'Shot {shotnum} DA{diodeArrayNum} t = {axuvTimes[expTimeIdx]*1e3:.4g} ms, '
                     f'a = {axuvRadius[expTimeIdx]*1e2:.4g} cm')

        plt.show()

    return densNew, solrzNew, solzzNew, axuvTimes

def extend_2d_density_all_times(simulationName, shotnum, diodeArrayNum,
                                rateKHz=10.0, tanhSteepness=4.0, tMin=None, tMax=None,
                                redoAnalysis=False,
                                makeplot=False, simTimeToPlot=None, expTimeToPlot=None,
                                densCmap='inferno', fluxColor='white', numFluxSurfaces=15):
    """
    Build the extended 2D (R, Z) density profile of `extend_2d_density()` at every simulated
    timestep.

    The extension is the tanh fall-off of `extend_profile()` mapped along the eqdsk flux
    surfaces, held constant along each surface. Its shape is set by the AXUV plasma radius, so
    it depends on the experimental time, and its size is set by the density at the edge of the
    simulation grid, so it also depends on the simulated time. The result is therefore a
    profile per pair of times.

    Note that the returned density is a large array- it holds a full (R, Z) profile for every
    pair of simulated and experimental timesteps, which for a typical run and a 9 ms window
    averaged down to 10 kHz is around 0.6 GB. Narrow the window with `tMin` and `tMax`, or drop
    `rateKHz`, to make it smaller.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the AXUV plasma radius is taken from.
    diodeArrayNum : int
        The diode array the radius is taken from, i.e. 1 for DIODEARRAY1.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        This sets the number of experimental timesteps.
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Default is 4.
    tMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    tMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.
    redoAnalysis : bool
        Force the density extension to be rebuilt instead of loading the saved one.
        Default is False.
    makeplot : bool
        Whether to plot one of the extended 2D density profiles.
        Default is False.
    simTimeToPlot : float
        The simulation time of the profile that is plotted. [s]
        The closest available simulated timestep is used.
        Default is None, which plots the first simulated timestep.
    expTimeToPlot : float
        The experimental time of the profile that is plotted. [s]
        The closest available experimental timestep is used.
        Default is None, which plots the first timestep of the experimental time window.
    densCmap : str
        Colormap used for the density.
        Default is 'inferno'.
    fluxColor : str
        Color of the eqdsk flux surfaces drawn over the density.
        Default is 'white'.
    numFluxSurfaces : int
        Number of flux surfaces to plot.
        Default is 15.

    Returns
    -------
    dens : np.array
        4D array [simTime x expTime x R x Z] of the extended ion density profile. [m^-3]
    solrz : np.array
        2D array [R x Z] of the r values. [m]
    solzz : np.array
        2D array [R x Z] of the z values. [m]
    expTime : np.array
        1D array of the experimental times of the AXUV plasma radius. [s]
    simTime : np.array
        1D array of the simulation times. [s]
    """

    extData = build_density_extension(simulationName, shotnum, diodeArrayNum,
                                      rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                                      tMin=tMin, tMax=tMax, redoAnalysis=redoAnalysis)

    simTime = extData['times']
    densSim = extData['densSim']
    solrz = extData['solrz']
    solzz = extData['solzz']
    expTime = extData['axuvTimes']
    axuvRadius = extData['axuvRadius']
    rExtGrid = extData['rExtGrid']
    rExtMapped = extData['rExtMapped']
    zGrid = extData['zGrid']
    shapeExt = extData['shapeExt']

    numSim = len(simTime)
    numExp = len(expTime)
    numRSim, numZ = solrz.shape
    numRExt = len(rExtGrid)

    # Midplane density at the edge of the simulation grid at every simulated timestep, which is
    # what the extension is scaled by
    densEdge = densSim[:, -1, 0]

    # [simTime x expTime x rExt]
    densExtMid = densEdge[:, np.newaxis, np.newaxis] * shapeExt[np.newaxis, :, :]

    # The simulated part does not depend on the experimental time and the extension does not
    # depend on z, so the full profile is filled in by broadcasting rather than by looping
    dens = np.empty((numSim, numExp, numRSim + numRExt, numZ))
    dens[:, :, :numRSim, :] = densSim[:, np.newaxis, :, :]
    dens[:, :, numRSim:, :] = densExtMid[:, :, :, np.newaxis]

    # Stack the flux surfaces of the extension onto the outside of the simulation grid
    solrzNew = np.concatenate([solrz, rExtMapped], axis=0)
    solzzNew = np.concatenate([solzz, np.tile(zGrid, (numRExt, 1))], axis=0)

    if makeplot:

        if simTimeToPlot is None:
            simTimeIdx = 0
        else:
            simTimeIdx = np.argmin(np.abs(simTime - simTimeToPlot))

        if expTimeToPlot is None:
            expTimeIdx = 0
        else:
            expTimeIdx = np.argmin(np.abs(expTime - expTimeToPlot))

        densToPlot = dens[simTimeIdx, expTimeIdx]

        fig = plt.figure(figsize=(12, 10), tight_layout=True)
        fig.suptitle(f'{simulationName}\n'
                     f'Simulation t = {simTime[simTimeIdx]*1e3:.4g} ms, '
                     f'shot {shotnum} DA{diodeArrayNum} t = {expTime[expTimeIdx]*1e3:.4g} ms, '
                     f'a = {axuvRadius[expTimeIdx]*1e2:.4g} cm')

        # 2D density
        ax1 = fig.add_subplot(2, 1, 1)
        # Midplane radial profile of the same density
        ax2 = fig.add_subplot(2, 1, 2)

        # Axial and radial extent of the 2D plot
        zLim = (0, 0.8)
        rLim = (0, 1.05 * np.max(axuvRadius))

        #### 2D density

        # The simulation and the extension are one array on one grid, so they share a colorbar
        levels = np.linspace(0, np.max(densToPlot), 100)

        pltObj = ax1.contourf(solzzNew, solrzNew, densToPlot, levels=levels, cmap=densCmap)

        cbar = fig.colorbar(pltObj, ax=ax1)
        cbar.set_label(r'n$_i$ [m$^{-3}$]')

        plot_flux_surfaces(ax1, extData['Rmesh'], extData['Zmesh'], extData['psi'], rLim[1],
                           fluxColor=fluxColor, numFluxSurfaces=numFluxSurfaces)

        ax1.set_xlim(zLim)
        ax1.set_ylim(rLim)
        ax1.set_aspect('equal')

        ax1.set_xlabel('Z [m]')
        ax1.set_ylabel('R [m]')

        #### Midplane radial profile

        # The simulated part and the extension are drawn separately so that the join between
        # the two is visible, with the last simulated point repeated to close the gap
        ax2.plot(solrzNew[:numRSim, 0], densToPlot[:numRSim, 0],
                 color='black',
                 linewidth=3,
                 label='Simulation')

        ax2.plot(solrzNew[numRSim-1:, 0], densToPlot[numRSim-1:, 0],
                 color='tab:red',
                 linewidth=3,
                 marker='o',
                 ms=5,
                 label='AXUV extension')

        ax2.axvline(axuvRadius[expTimeIdx],
                    color='tab:blue',
                    linewidth=2,
                    linestyle='dashed',
                    label=f'AXUV plasma radius = {axuvRadius[expTimeIdx]*1e2:.4g} cm')

        # The same radial range as the vertical axis of the 2D plot above
        ax2.set_xlim(rLim)
        ax2.set_ylim(0, None)

        ax2.set_xlabel('R [m]')
        ax2.set_ylabel(r'n$_i$ [m$^{-3}$]')

        ax2.legend()

        plt.show()

    return dens, solrzNew, solzzNew, expTime, simTime


if __name__ == '__main__':

    simName = 'nneut_1e18_gb_2e17_NBI_800kW_ECH_0kW_ionDrrOn'
    shotnum = 260426037
    diodeArrayNum = 1

    # Simulation time whose radial profile gets extended [s]
    simTimeToPlot = 0.62e-3

    # Experimental time window the AXUV plasma radius is taken from [s]
    tMin = 3e-3
    tMax = 12e-3

    # Experimental time whose extension gets plotted [s]
    expTimeToPlot = 10e-3

    # plot_extended_radial_profiles(simName, shotnum, diodeArrayNum, simTimeToPlot,
    #                               cmap='viridis',
    #                               tMin=tMin, tMax=tMax)

    # dens, solrz, solzz, time = extend_2d_density(simName, shotnum, diodeArrayNum, simTimeToPlot,
    #                                              makeplot=True,
    #                                              expTimeToPlot=expTimeToPlot,
    #                                              tMin=tMin, tMax=tMax)

    dens, solrz, solzz, expTime, simTime = extend_2d_density_all_times(simName, shotnum, diodeArrayNum,
                                                                       tMin=tMin, tMax=tMax,
                                                                       makeplot=True,
                                                                       simTimeToPlot=simTimeToPlot,
                                                                       expTimeToPlot=expTimeToPlot)
