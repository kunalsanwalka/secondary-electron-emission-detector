"""
This script plots the radial confinement time, density and ion temperature profiles of a given cql3d simulation at a given time.

All three are read directly from the CQL3D output (WHAM.nc) of each IPS timestep. The confinement
time is compared to the GDT and collisionless scalings calculated from the density and temperature.
"""

import numpy as np
import scipy as sc
import xarray as xr
import matplotlib.pyplot as plt

import kn1dc_parser
from kn1dc_parser import generate_filename_list
from cql3d_vs_data import find_valid_time_slices, parse_simulation_name, density_label
from cql3d_radial_profiles import read_eqdsk_flux

# Set after the imports, as kn1dc_parser and cql3d_vs_data set their own font size when imported
plt.rcParams.update({'font.size': 20})

# Global variable to store the simulation scan directory
simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/withRadialDiff/'

# Deuterium ion mass [kg]
massD = sc.constants.physical_constants['deuteron mass'][0]

def load_cql3d_profiles(simName):
    """
    Load the time dependent radial profiles of the ions from the CQL3D output of every IPS timestep.

    The temp variable in the CQL3D output is only updated for the Maxwellian species, so for the
    general (ion) species it stays at its input value. The ion temperature is instead taken as the
    effective temperature of the distribution function at the midplane, T = (2/3) * <E>.

    Parameters
    ----------
    simName : str
        The name of the simulation.

    Returns
    -------
    profiles : dict
        'time'    : [Time], simulation time. [s]
        'radius'  : [Radius], midplane radius of each flux surface. [m]
        'tauP'    : [Time x Radius], ion confinement time calculated in CQL3D (tauc_code). [s]
        'dens'    : [Time x Radius], ion density. [m^-3]
        'Ti'      : [Time x Radius], ion temperature from the energy per particle at the midplane (energym). [keV]
        'eqdsk'   : [Time], eqdsk file of the IPS timestep each time slice belongs to.
        The CQL3D restart slices are dropped.
    """

    # That module keeps its own copy of the scan directory, so point it at the one used here
    kn1dc_parser.simulationScanDir = simulationScanDir

    # Filename list for the simulation
    _, _, filenameListCQL3D, filenameListEQDSK = generate_filename_list(simName)

    timeArr = []
    tauP = []
    dens = []
    Ti = []
    eqdsk = []
    currTime = 0

    for filename, filenameEQDSK in zip(filenameListCQL3D, filenameListEQDSK):

        # Open the file
        ds = xr.open_dataset(filename,
                             decode_timedelta=False)

        # Each file starts at t=0, so offset it by the end of the previous one
        time = ds['time'].values + currTime
        timeArr.append(time)
        currTime = time[-1]

        # For tauc_code, the 1st general species is ions, the 2nd is electrons
        tauP.append(ds['tauc_code'].values[:, :, 0])

        # For density, the species are [general D, general e, maxwell D, maxwell W]
        # [cm^-3] to [m^-3]
        dens.append(ds['density'].values[:, :, 0] * 1e6)

        # [Time x Species x Radius], ions are the 1st general species
        Ti.append(2/3 * ds['energym'].values[:, 0, :])

        eqdsk.append(np.full(len(time), filenameEQDSK))

        # Midplane radius of each flux surface, the same for every file
        # [cm] to [m]
        radius = ds['solrz'].values[:, 0] / 1e2

        ds.close()

    timeArr = np.concatenate(timeArr)

    # Drop the CQL3D restart slices
    keep = find_valid_time_slices(timeArr)

    profiles = {'time': timeArr[keep],
                'radius': radius,
                'tauP': np.concatenate(tauP)[keep],
                'dens': np.concatenate(dens)[keep],
                'Ti': np.concatenate(Ti)[keep],
                'eqdsk': np.concatenate(eqdsk)[keep]}

    return profiles

def mirror_geometry(filenameEQDSK):
    """
    Distance from the midplane (z = 0) to the mirror throat and the mirror ratio on axis.

    The mirror throat is taken as the peak of |B| on axis. On axis B = Bz = (1/R) * dpsi/dR,
    which is evaluated on the first grid column off axis.

    Parameters
    ----------
    filenameEQDSK : str
        Location of the eqdsk file.

    Returns
    -------
    L : float
        Distance from the midplane to the mirror throat. [m]
        Limited to the axial resolution of the eqdsk grid.
    mirrorRatio : float
        |B| at the mirror throat over |B| at the midplane, on axis.
    """

    Rmesh, Zmesh, psi = read_eqdsk_flux(filenameEQDSK)

    r1D = Rmesh[0]
    z1D = Zmesh[:, 0]

    # First grid column off axis, as Bz is 0/0 at R = 0
    rIdx = np.argmax(r1D > 0)
    BAxis = np.abs(np.gradient(psi, r1D, axis=1)[:, rIdx] / r1D[rIdx])

    # The field is symmetric about the midplane, so only look at one side of it
    pos = z1D > 0
    throatIdx = np.argmax(BAxis[pos])

    L = z1D[pos][throatIdx]
    mirrorRatio = BAxis[pos][throatIdx] / BAxis[np.argmin(np.abs(z1D))]

    return L, mirrorRatio

def tau_gdt(Ti, L, mirrorRatio):
    """
    GDT particle confinement time, tau = R_m * L * sqrt(m_i / (k_B * T_i)).

    Parameters
    ----------
    Ti : np.array
        Ion temperature. [keV]
    L : float
        Distance from the midplane to the mirror throat. [m]
    mirrorRatio : float
        Mirror ratio, R_m.

    Returns
    -------
    tau : np.array
        Confinement time. [s]
    """

    # [keV] to [J]
    TiJoules = Ti * 1e3 * sc.constants.e

    return mirrorRatio * L * np.sqrt(massD / TiJoules) / 2

def tau_collisionless(Ti, dens, mirrorRatio):
    """
    Collisionless particle confinement time, tau = tau_ii * log10(R_m).

    tau_ii = 2e12 * T_i^(3/2) / n_i is the ion-ion collision time for deuterium with lnLambda ~ 15.

    Parameters
    ----------
    Ti : np.array
        Ion temperature. [keV]
    dens : np.array
        Ion density. [m^-3]
    mirrorRatio : float
        Mirror ratio, R_m.

    Returns
    -------
    tau : np.array
        Confinement time. [s]
    """

    # [keV] to [eV]
    TieV = Ti * 1e3

    return 2e12 * TieV**1.5 / dens * np.log10(mirrorRatio)

def plot_taup_dens_ti(simName, timeToPlot, coldFrac=None, linestyle='-', label=None, axes=None,
                      makeplot=True, saveplot=False):
    """
    Plot the radial confinement time, ion density and ion temperature profiles at a given time.

    Top- Ion confinement time calculated directly in CQL3D (tauc_code, black), along with the
         GDT (blue) and collisionless (red) scalings. See tau_gdt() and tau_collisionless().
         The length and mirror ratio used in the GDT scaling are taken from the eqdsk of the
         IPS timestep, see mirror_geometry().
    Middle- Ion density (black).
    Bottom- Ion temperature (black), from the energy per particle at the midplane. See
            load_cql3d_profiles().

    If coldFrac is given, the distribution function of each flux tube is also split into a cold
    and a hot part, see split_temp_profiles(), and the following are added-
    Top- Effective confinement time (purple) of the cold part with the GDT scaling and the hot
         part with the collisionless scaling, see calculate_tau_eff().
    Middle- Density of the cold (blue) and hot (red) parts.
    Bottom- Temperature of the cold (blue) and hot (red) parts.

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timeToPlot : float
        The simulation time at which to plot the profiles. [s]
        The closest available simulated timestep is used. If coldFrac is given, the closest
        end of an IPS timestep is used instead, as the distribution function is only saved then.
    coldFrac : float
        Fraction of the density in the cold part, between 0 and 1.
        Default is None, which does not split the distribution function.
    linestyle : str
        Line style of the profiles, to tell apart several simulations or times on the same axes.
        Default is '-'.
    label : str
        Legend label of the profiles.
        Default is None, which labels them with the simulated time.
    axes : tuple of matplotlib.axes.Axes
        (confinement time axis, density axis, temperature axis) to draw into instead of making
        a new figure, e.g. to overlay several simulations or times for comparison.
        Default is None, which makes a new figure.
    makeplot : bool
        Show the plot.
        Default is True.
    saveplot : bool
        Save the plot in /home/sanwalka/shinethru/plots/ as f'{simName}_taup_dens_ti.png'.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : tuple of matplotlib.axes.Axes
        (confinement time axis, density axis, temperature axis)
    """

    profiles = load_cql3d_profiles(simName)
    radius = profiles['radius']

    if coldFrac is not None:
        # The distribution function is only saved at the end of each IPS timestep, so plot all
        # the profiles at that time
        splitProfiles = split_temp_profiles(simName, timeToPlot, coldFrac, len(radius))
        timeToPlot = splitProfiles['time']

    # Find the index of the closest time in the simulation data
    timeIdx = np.argmin(np.abs(profiles['time'] - timeToPlot))
    simTime = profiles['time'][timeIdx]

    dens = profiles['dens'][timeIdx]
    Ti = profiles['Ti'][timeIdx]

    # Confinement time scalings
    L, mirrorRatio = mirror_geometry(profiles['eqdsk'][timeIdx])
    tauGDT = tau_gdt(Ti, L, mirrorRatio)
    tauCollisionless = tau_collisionless(Ti, dens, mirrorRatio)

    if coldFrac is not None:
        # The hot ions scatter off all the ions, so the total density is used for them
        tauEff = calculate_tau_eff(coldFrac,
                                   tau_gdt(splitProfiles['TCold'], L, mirrorRatio),
                                   tau_collisionless(splitProfiles['THot'], dens, mirrorRatio))

    if label is None:
        label = r'$t_{Sim}$' + f' = {simTime*1e3:.2f} ms'

    if axes is None:

        fig = plt.figure(figsize=(10, 14), tight_layout=True)

        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

        # Label the bottom panel with the cold fraction and the neutral densities of the
        # simulation, as the top panel is taken up by its legends
        textLines = []
        if coldFrac is not None:
            textLines.append(r'$n_{cold}/n_{tot}$' + f' = {coldFrac}')
        mainVesselDens, gasBoxDens = parse_simulation_name(simName)
        if mainVesselDens is not None:
            textLines.append(f'Gas Box '+r'$n_n$'+f': {density_label(gasBoxDens)}')
            textLines.append(f'Main Vessel '+r'$n_n$'+f': {density_label(mainVesselDens)}')
        if textLines:
            ax3.text(0.02, 0.05, '\n'.join(textLines),
                     transform=ax3.transAxes,
                     ha='left', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Legend for the colors of the top panel. It is added as an artist so that it is kept
        # when the legend for the line styles is redrawn by later calls with these axes.
        colorHandles = [plt.Line2D([], [], color='k', linewidth=3, label='CQL3D'),
                        plt.Line2D([], [], color='tab:blue', linewidth=3, label='GDT'),
                        plt.Line2D([], [], color='tab:red', linewidth=3, label='Collisionless')]
        if coldFrac is not None:
            colorHandles.append(plt.Line2D([], [], color='tab:purple', linewidth=3, label='Effective'))
        colorLegend = ax1.legend(handles=colorHandles, loc='upper left')
        ax1.add_artist(colorLegend)

        # Legend for the colors of the cold and hot parts, shared by the lower panels
        if coldFrac is not None:
            ax2.legend(handles=[plt.Line2D([], [], color='k', linewidth=3, label='Total'),
                                plt.Line2D([], [], color='tab:blue', linewidth=3, label='Cold'),
                                plt.Line2D([], [], color='tab:red', linewidth=3, label='Hot')],
                       loc='upper right')

    else:

        ax1, ax2, ax3 = axes
        fig = ax1.figure

        # The y limits were fixed by the profiles already on the axes, so rescale to fit the new ones
        for ax in axes:
            ax.set_autoscaley_on(True)

    # Confinement time
    ax1.plot(radius, profiles['tauP'][timeIdx]*1e3, color='k', label=label, linewidth=3, linestyle=linestyle)
    ax1.plot(radius, tauGDT*1e3, color='tab:blue', linewidth=3, linestyle=linestyle)
    ax1.plot(radius, tauCollisionless*1e3, color='tab:red', linewidth=3, linestyle=linestyle)
    if coldFrac is not None:
        ax1.plot(radius, tauEff*1e3, color='tab:purple', linewidth=3, linestyle=linestyle)
    ax1.set_ylabel(r'$\tau_p$ [ms]')
    # The scalings are orders of magnitude apart
    ax1.set_yscale('log')
    # Leave room above all the profiles on the axis for the legends
    ax1.set_ylim(ax1.dataLim.y0/3, ax1.dataLim.y1*100)

    # Legend for the line styles, which tell apart the simulations or times on the axes
    ax1.legend(loc='upper right')

    # Density
    ax2.plot(radius, dens, color='k', linewidth=3, linestyle=linestyle)
    if coldFrac is not None:
        ax2.plot(radius, coldFrac*dens, color='tab:blue', linewidth=3, linestyle=linestyle)
        ax2.plot(radius, (1 - coldFrac)*dens, color='tab:red', linewidth=3, linestyle=linestyle)
    ax2.set_ylabel(r'n$_i$ [m$^{-3}$]')
    ax2.set_ylim(0, None)

    # Leave room above all the profiles on the axis for the cold and hot legend
    if coldFrac is not None:
        ax2.set_ylim(0, ax2.dataLim.y1*1.6)

    # Ion temperature
    ax3.plot(radius, Ti, color='k', linewidth=3, linestyle=linestyle)
    if coldFrac is not None:
        ax3.plot(radius, splitProfiles['TCold'], color='tab:blue', linewidth=3, linestyle=linestyle)
        ax3.plot(radius, splitProfiles['THot'], color='tab:red', linewidth=3, linestyle=linestyle)
    ax3.set_ylabel(r'T$_i$ = $\frac{2}{3}\langle E \rangle$ [keV]')
    ax3.set_ylim(0, None)

    ax1.set_xlim(0, None)

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.set_xlabel('Radius [m]')

    if saveplot:
        savePath = f'/home/sanwalka/shinethru/plots/{simName}_taup_dens_ti.png'
        fig.savefig(savePath, dpi=600)
        print(f'Saved plot to {savePath}')

    if makeplot:
        plt.show()

    return fig, (ax1, ax2, ax3)

def load_distribution_function(simName, timeToPlot, radIdx=0):
    """
    Load the ion distribution function at the midplane of a flux tube.

    CQL3D only saves the distribution function at the end of each IPS timestep, so the IPS
    timestep that ends closest to timeToPlot is used.

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timeToPlot : float
        The simulation time at which to load the distribution function. [s]
        The closest end of an IPS timestep is used.
    radIdx : int
        Index of the flux tube, from the innermost one outwards.
        Default is 0, the innermost flux tube.

    Returns
    -------
    distFunc : dict
        'vPar'   : [Momentum x Pitch angle], parallel velocity. [m/s]
        'vPerp'  : [Momentum x Pitch angle], perpendicular velocity. [m/s]
        'f'      : [Momentum x Pitch angle], ion distribution function. [m^-3 / (m/s)^3]
        'pitch'  : [Pitch angle], pitch angle at the midplane. [rad]
        'itl'    : int, lower trapped-passing boundary on the pitch angle grid (1-indexed).
        'itu'    : int, upper trapped-passing boundary on the pitch angle grid (1-indexed).
        'time'   : float, simulation time at the end of the IPS timestep. [s]
        'radius' : float, midplane radius of the flux tube. [m]
    """

    # That module keeps its own copy of the scan directory, so point it at the one used here
    kn1dc_parser.simulationScanDir = simulationScanDir

    # Filename list for the simulation
    _, _, filenameListCQL3D, _ = generate_filename_list(simName)

    # Time at the end of each IPS timestep. Each file starts at t=0, so they are summed up.
    endTimes = np.zeros(len(filenameListCQL3D))
    currTime = 0
    for i, filename in enumerate(filenameListCQL3D):

        with xr.open_dataset(filename, decode_timedelta=False) as ds:
            currTime += ds['time'].values[-1]
        endTimes[i] = currTime

    fileIdx = np.argmin(np.abs(endTimes - timeToPlot))

    with xr.open_dataset(filenameListCQL3D[fileIdx], decode_timedelta=False) as ds:

        # [Species x Radius x Momentum x Pitch angle], ions are the 1st general species
        f = ds['f'].values[0, radIdx]

        # Normalized momentum-per-mass, which is the velocity for the (non-relativistic) ions
        x = ds['x'].values
        # Pitch angle at the midplane [rad], which differs between flux tubes
        y = ds['y'].values[radIdx]

        # [cm/s] to [m/s]
        vnorm = ds['vnorm'].values / 1e2

        # Trapped-passing boundaries (1-indexed)
        itl = int(ds['itl'].values[radIdx])
        itu = int(ds['itu'].values[radIdx])

        # Midplane radius of the flux tube
        # [cm] to [m]
        radius = ds['solrz'].values[int(ds['lrindx'].values[radIdx]) - 1, 0] / 1e2

    # Integrating f * x^2 dx * 2*pi*sin(y) dy gives the density in [cm^-3], so f is in
    # [cm^-3 / vnorm^3]. Convert to [m^-3 / (m/s)^3].
    f = f * 1e6 / vnorm**3

    # [Momentum x Pitch angle] velocity grid [m/s]
    vel = (x * vnorm)[:, None]
    vPar = vel * np.cos(y)[None, :]
    vPerp = vel * np.sin(y)[None, :]

    distFunc = {'vPar': vPar,
                'vPerp': vPerp,
                'f': f,
                'pitch': y,
                'itl': itl,
                'itu': itu,
                'time': endTimes[fileIdx],
                'radius': radius}

    return distFunc

def plot_distribution_function(simName, timeToPlot, eMax=30.0, numDecades=8, cmap='viridis',
                               makeplot=True, saveplot=False):
    """
    Plot the ion distribution function at the midplane of the innermost flux tube vs. (v_par, v_perp).

    CQL3D only saves the distribution function at the end of each IPS timestep, so the IPS
    timestep that ends closest to timeToPlot is used. See load_distribution_function(). The
    trapped-passing (loss cone) boundary is overlaid in white.

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timeToPlot : float
        The simulation time at which to plot the distribution function. [s]
        The closest end of an IPS timestep is used.
    eMax : float
        Largest ion energy to plot. [keV]
        Default is 30.
    numDecades : int
        Number of decades below the peak of the distribution function shown by the colorbar.
        Default is 8.
    cmap : str
        Colormap used for the distribution function.
        Default is 'viridis'.
    makeplot : bool
        Show the plot.
        Default is True.
    saveplot : bool
        Save the plot in /home/sanwalka/shinethru/plots/ as f'{simName}_ion_distribution.png'.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axis.
    """

    distFunc = load_distribution_function(simName, timeToPlot)
    y = distFunc['pitch']

    # Only keep the velocities up to eMax
    vMax = np.sqrt(2 * eMax * 1e3 * sc.constants.e / massD)
    vel = np.hypot(distFunc['vPar'], distFunc['vPerp'])[:, 0]
    xIdx = vel <= vMax

    vPar = distFunc['vPar'][xIdx]
    vPerp = distFunc['vPerp'][xIdx]

    logF = np.log10(np.clip(distFunc['f'][xIdx], a_min=1e-300, a_max=None))
    levels = np.linspace(logF.max() - numDecades, logF.max(), 100)

    fig = plt.figure(figsize=(12, 7), tight_layout=True)
    fig.suptitle(simName)
    ax = fig.add_subplot(1, 1, 1)

    pltObj = ax.contourf(vPar/1e6, vPerp/1e6, logF, levels=levels, cmap=cmap, extend='min')

    cbar = fig.colorbar(pltObj, ax=ax)
    cbar.set_label(r'log$_{10}$ f$_i$ [m$^{-6}$ s$^3$]')
    # Ticks on whole decades
    cbar.set_ticks(np.arange(np.ceil(levels[0]), np.floor(levels[-1]) + 1))

    # Loss cone boundaries
    for idx in (distFunc['itl'], distFunc['itu']):
        ax.plot([0, vMax*np.cos(y[idx-1])/1e6], [0, vMax*np.sin(y[idx-1])/1e6],
                color='white', linewidth=2, linestyle='--')

    ax.set_xlim(-vMax/1e6, vMax/1e6)
    ax.set_ylim(0, vMax/1e6)
    ax.set_aspect('equal')

    ax.set_xlabel(r'v$_\parallel$ [10$^6$ m/s]')
    ax.set_ylabel(r'v$_\perp$ [10$^6$ m/s]')
    ax.set_title(f'Time = {distFunc["time"]*1e3:.2f} ms, R = {distFunc["radius"]*1e2:.2f} cm')

    if saveplot:
        savePath = f'/home/sanwalka/shinethru/plots/{simName}_ion_distribution.png'
        fig.savefig(savePath, dpi=600)
        print(f'Saved plot to {savePath}')

    if makeplot:
        plt.show()

    return fig, ax

def velocity_space_grid(vPerp, vPar):
    """
    Speed and pitch angle grids, and the velocity space volume of each cell, for integrating over
    the distribution function.

    Integrals over velocity space are sum(f * weights), e.g. the density is
    n = sum(f * v^2 dv * 2*pi*sin(theta) dtheta). The cell widths are halfway between grid
    points, and halved at the ends of the grid, as done in CQL3D.

    Parameters
    ----------
    vPerp : np.array
        [Momentum x Pitch angle], perpendicular velocity. [m/s]
    vPar : np.array
        [Momentum x Pitch angle], parallel velocity. [m/s]

    Returns
    -------
    v : np.array
        [Momentum], speed. [m/s]
    theta : np.array
        [Pitch angle], pitch angle. [rad]
    vEdges : np.array
        [Momentum + 1], edges of the velocity shells. [m/s]
    weights : np.array
        [Momentum x Pitch angle], velocity space volume of each cell. [(m/s)^3]
    """

    # The pitch angle is taken at the largest speed, as it is undefined at v = 0
    v = np.hypot(vPar, vPerp)[:, 0]
    theta = np.arctan2(vPerp, vPar)[-1]

    # Cell edges halfway between grid points, with the ends of the grid as the outermost edges
    vEdges = np.concatenate(([v[0]], (v[1:] + v[:-1]) / 2, [v[-1]]))
    thetaEdges = np.concatenate(([theta[0]], (theta[1:] + theta[:-1]) / 2, [theta[-1]]))

    weights = (v**2 * np.diff(vEdges))[:, None] * (2 * np.pi * np.sin(theta) * np.diff(thetaEdges))[None, :]

    return v, theta, vEdges, weights

def split_f(vPerp, vPar, f, coldFrac, eMax=30.0, numDecades=8, cmap='viridis', simName=None,
            makeplot=False, saveplot=False):
    """
    Split the distribution function into a cold, low energy part and a hot, high energy part.

    The cut is a line of constant energy in (v_par, v_perp). It is placed such that the density
    of fCold is coldFrac times the density of f. The velocity shell that the cut falls in is
    shared between the two parts, so that the density fraction is met exactly and fCold + fHot = f.
    The density is calculated as in velocity_space_grid().

    Parameters
    ----------
    vPerp : np.array
        [Momentum x Pitch angle], perpendicular velocity. [m/s]
    vPar : np.array
        [Momentum x Pitch angle], parallel velocity. [m/s]
    f : np.array
        [Momentum x Pitch angle], distribution function. [m^-3 / (m/s)^3]
    coldFrac : float
        Fraction of the total density in fCold, between 0 and 1.
    eMax : float
        Largest ion energy to plot. [keV]
        Default is 30.
    numDecades : int
        Number of decades below the peak of the distribution function shown by the colorbar.
        Default is 8.
    cmap : str
        Colormap used for the distribution function.
        Default is 'viridis'.
    simName : str
        The name of the simulation, used for the title and filename of the plot.
        Default is None.
    makeplot : bool
        Show the plot of the split.
        Default is False.
    saveplot : bool
        Save the plot in /home/sanwalka/shinethru/plots/ as f'{simName}_f_split.png'.
        Default is False.

    Returns
    -------
    fCold : np.array
        [Momentum x Pitch angle], cold part of the distribution function. [m^-3 / (m/s)^3]
    fHot : np.array
        [Momentum x Pitch angle], hot part of the distribution function. [m^-3 / (m/s)^3]
    """

    if not 0 <= coldFrac <= 1:
        raise ValueError(f'coldFrac must be between 0 and 1, got {coldFrac}')

    v, theta, vEdges, weights = velocity_space_grid(vPerp, vPar)

    # Density in each velocity shell, and the cumulative density up to the top of each shell
    shellDens = np.sum(f * weights, axis=1)
    cumDens = np.cumsum(shellDens)
    totalDens = cumDens[-1]
    targetDens = coldFrac * totalDens

    # Shell that the cut falls in, and the fraction of it that goes into fCold
    cutIdx = min(np.searchsorted(cumDens, targetDens), len(v) - 1)
    densBelow = cumDens[cutIdx - 1] if cutIdx > 0 else 0
    shellFrac = np.clip((targetDens - densBelow) / shellDens[cutIdx], 0, 1) if shellDens[cutIdx] > 0 else 0

    fCold = np.zeros_like(f)
    fCold[:cutIdx] = f[:cutIdx]
    fCold[cutIdx] = shellFrac * f[cutIdx]

    fHot = np.zeros_like(f)
    fHot[cutIdx] = (1 - shellFrac) * f[cutIdx]
    fHot[cutIdx+1:] = f[cutIdx+1:]

    if makeplot or saveplot:

        # Energy of the cut, taking the density to be spread evenly across the cut shell
        vCut = vEdges[cutIdx] + shellFrac * (vEdges[cutIdx+1] - vEdges[cutIdx])
        eCut = 0.5 * massD * vCut**2 / sc.constants.e / 1e3

        coldDens = np.sum(fCold * weights)
        hotDens = np.sum(fHot * weights)

        # Only plot the velocities up to eMax
        vMax = np.sqrt(2 * eMax * 1e3 * sc.constants.e / massD)
        vIdx = v <= vMax

        logF = np.log10(np.clip(f[vIdx], a_min=1e-300, a_max=None))
        levels = np.linspace(logF.max() - numDecades, logF.max(), 100)

        fig = plt.figure(figsize=(16, 10), tight_layout=True)
        if simName is not None:
            fig.suptitle(simName)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)

        # Distribution functions, on the same color scale. The zeros of the split parts are
        # left blank so that the cut is visible.
        for ax, fToPlot, title in ((ax1, f, f'Total, n = {density_label(totalDens)}'),
                                   (ax3, fCold, f'Cold, n = {density_label(coldDens)}\n({coldDens/totalDens:.3f} of total)'),
                                   (ax4, fHot, f'Hot, n = {density_label(hotDens)}\n({hotDens/totalDens:.3f} of total)')):

            fPlot = np.ma.masked_less_equal(fToPlot[vIdx], 0)
            pltObj = ax.contourf(vPar[vIdx]/1e6, vPerp[vIdx]/1e6, np.ma.log10(fPlot),
                                 levels=levels, cmap=cmap, extend='min')

            cbar = fig.colorbar(pltObj, ax=ax)
            cbar.set_label(r'log$_{10}$ f$_i$ [m$^{-6}$ s$^3$]')
            # Ticks on whole decades
            cbar.set_ticks(np.arange(np.ceil(levels[0]), np.floor(levels[-1]) + 1))

            # Cut line
            ax.plot(vCut*np.cos(theta)/1e6, vCut*np.sin(theta)/1e6,
                    color='red', linewidth=2, linestyle='--')

            ax.set_aspect('equal')
            ax.set_xlabel(r'v$_\parallel$ [10$^6$ m/s]')
            ax.set_ylabel(r'v$_\perp$ [10$^6$ m/s]')
            ax.set_title(title)

        ax1.set_xlim(-vMax/1e6, vMax/1e6)
        ax1.set_ylim(0, vMax/1e6)

        # Fraction of the density below each energy, at the top of each velocity shell
        eEdges = 0.5 * massD * vEdges[1:]**2 / sc.constants.e / 1e3
        ax2.plot(eEdges, cumDens / totalDens, color='k', linewidth=3)
        ax2.axhline(coldFrac, color='red', linewidth=2, linestyle='--', label=f'coldFrac = {coldFrac}')
        ax2.axvline(eCut, color='red', linewidth=2, linestyle=':', label=r'E$_{cut}$' + f' = {eCut:.2f} keV')
        ax2.set_xscale('log')
        ax2.set_xlim(eEdges[0], eEdges[-1])
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('Energy [keV]')
        ax2.set_ylabel('Fraction of density below energy')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        if saveplot:
            savePath = f'/home/sanwalka/shinethru/plots/{simName}_f_split.png'
            fig.savefig(savePath, dpi=600)
            print(f'Saved plot to {savePath}')

        if makeplot:
            plt.show()

    return fCold, fHot

def calculate_temp(vPerp, vPar, f):
    """
    Effective temperature of the distribution function, T = (2/3) * <E>.

    <E> is the average ion energy, sum(f * 0.5 * m * v^2 * weights) / sum(f * weights). See
    velocity_space_grid() for the integration.

    Parameters
    ----------
    vPerp : np.array
        [Momentum x Pitch angle], perpendicular velocity. [m/s]
    vPar : np.array
        [Momentum x Pitch angle], parallel velocity. [m/s]
    f : np.array
        [Momentum x Pitch angle], distribution function. [m^-3 / (m/s)^3]

    Returns
    -------
    T : float
        Effective temperature. [keV]
        NaN if the distribution function is empty.
    """

    v, _, _, weights = velocity_space_grid(vPerp, vPar)

    dens = np.sum(f * weights)
    if dens <= 0:
        return np.nan

    # [keV]
    energy = 0.5 * massD * v**2 / sc.constants.e / 1e3
    avgEnergy = np.sum(f * weights * energy[:, None]) / dens

    return 2/3 * avgEnergy

def scan_cold_frac(vPerp, vPar, f, coldFracArr):
    """
    Split the distribution function at each coldFrac with split_f(), and calculate the
    temperature of each part with calculate_temp().

    Parameters
    ----------
    vPerp : np.array
        [Momentum x Pitch angle], perpendicular velocity. [m/s]
    vPar : np.array
        [Momentum x Pitch angle], parallel velocity. [m/s]
    f : np.array
        [Momentum x Pitch angle], distribution function. [m^-3 / (m/s)^3]
    coldFracArr : np.array
        Fractions of the density in the cold part, between 0 and 1.

    Returns
    -------
    scan : dict
        'TCold'    : [coldFrac], temperature of the cold part. [keV]
                     NaN at coldFrac = 0, where the cold part is empty.
        'THot'     : [coldFrac], temperature of the hot part. [keV]
                     NaN at coldFrac = 1, where the hot part is empty.
        'densTotal': float, density of the full distribution function. [m^-3]
        'TTotal'   : float, temperature of the full distribution function. [keV]
    """

    _, _, _, weights = velocity_space_grid(vPerp, vPar)
    densTotal = np.sum(f * weights)

    TCold = np.zeros(len(coldFracArr))
    THot = np.zeros(len(coldFracArr))

    for i, coldFrac in enumerate(coldFracArr):

        fCold, fHot = split_f(vPerp, vPar, f, coldFrac)
        TCold[i] = calculate_temp(vPerp, vPar, fCold)
        THot[i] = calculate_temp(vPerp, vPar, fHot)

    scan = {'TCold': TCold,
            'THot': THot,
            'densTotal': densTotal,
            'TTotal': calculate_temp(vPerp, vPar, f)}

    return scan

def plot_temp_vs_cold_frac(simName, timeToPlot, numPoints=101, makeplot=True, saveplot=False):
    """
    Plot the effective temperatures of the cold and hot parts of the ion distribution function
    vs. the fraction of the density in the cold part.

    See scan_cold_frac() for the split. The temperature of the full distribution function is
    shown for reference. At coldFrac = 0 (1) the cold (hot) part is empty, so it has no
    temperature.

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timeToPlot : float
        The simulation time at which to split the distribution function. [s]
        The closest end of an IPS timestep is used, see load_distribution_function().
    numPoints : int
        Number of coldFrac values between 0 and 1.
        Default is 101.
    makeplot : bool
        Show the plot.
        Default is True.
    saveplot : bool
        Save the plot in /home/sanwalka/shinethru/plots/ as f'{simName}_temp_vs_cold_frac.png'.
        Default is False.

    Returns
    -------
    coldFracArr : np.array
        [numPoints], fraction of the density in the cold part.
    TCold : np.array
        [numPoints], temperature of the cold part. [keV]
    THot : np.array
        [numPoints], temperature of the hot part. [keV]
    """

    distFunc = load_distribution_function(simName, timeToPlot)

    coldFracArr = np.linspace(0, 1, numPoints)
    scan = scan_cold_frac(distFunc['vPerp'], distFunc['vPar'], distFunc['f'], coldFracArr)
    TCold = scan['TCold']
    THot = scan['THot']
    TTotal = scan['TTotal']

    fig = plt.figure(figsize=(10, 7), tight_layout=True)
    fig.suptitle(simName)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(coldFracArr, TCold, color='tab:blue', linewidth=3, label='Cold')
    ax.plot(coldFracArr, THot, color='tab:red', linewidth=3, label='Hot')
    ax.axhline(TTotal, color='k', linewidth=2, linestyle='--', label=f'Total, {TTotal:.2f} keV')

    # The cold and hot temperatures are orders of magnitude apart
    ax.set_yscale('log')
    ax.set_xlim(0, 1)

    ax.set_xlabel(r'$n_{cold}/n_{total}$')
    ax.set_ylabel(r'T = $\frac{2}{3}\langle E \rangle$ [keV]')
    ax.set_title(f'Time = {distFunc["time"]*1e3:.2f} ms, R = {distFunc["radius"]*1e2:.2f} cm')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    if saveplot:
        savePath = f'/home/sanwalka/shinethru/plots/{simName}_temp_vs_cold_frac.png'
        fig.savefig(savePath, dpi=600)
        print(f'Saved plot to {savePath}')

    if makeplot:
        plt.show()

    return coldFracArr, TCold, THot

def calculate_tau_eff(coldFrac, tauCold, tauHot):
    """
    Effective particle confinement time of a plasma made up of a cold and a hot part.

    The total loss rate is the sum of the loss rates of each part,
    n / tau_eff = n_cold / tau_cold + n_hot / tau_hot, which gives
    1 / tau_eff = coldFrac / tau_cold + (1 - coldFrac) / tau_hot.

    Parameters
    ----------
    coldFrac : float or np.array
        Fraction of the density in the cold part, between 0 and 1.
    tauCold : float or np.array
        Confinement time of the cold part. [s]
        Ignored where coldFrac = 0, so it may be NaN there.
    tauHot : float or np.array
        Confinement time of the hot part. [s]
        Ignored where coldFrac = 1, so it may be NaN there.

    Returns
    -------
    tauEff : float or np.array
        Effective confinement time. [s]
    """

    coldFrac = np.asarray(coldFrac)

    # An empty part does not lose any particles, even though its confinement time is undefined
    lossRateCold = np.where(coldFrac > 0, coldFrac / tauCold, 0)
    lossRateHot = np.where(coldFrac < 1, (1 - coldFrac) / tauHot, 0)

    return 1 / (lossRateCold + lossRateHot)

def split_temp_profiles(simName, timeToPlot, coldFrac, numRadii):
    """
    Radial profiles of the temperatures of the cold and hot parts of the ion distribution function.

    The distribution function of each flux tube is split with split_f(), such that coldFrac of
    the density of every flux tube is in the cold part, and the temperature of each part is
    calculated with calculate_temp().

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timeToPlot : float
        The simulation time at which to split the distribution function. [s]
        The closest end of an IPS timestep is used, see load_distribution_function().
    coldFrac : float
        Fraction of the density in the cold part, between 0 and 1.
    numRadii : int
        Number of flux tubes in the simulation.

    Returns
    -------
    splitProfiles : dict
        'time'   : float, simulation time at the end of the IPS timestep. [s]
        'radius' : [Radius], midplane radius of each flux tube. [m]
        'TCold'  : [Radius], temperature of the cold part. [keV]
                   NaN if coldFrac = 0, where the cold part is empty.
        'THot'   : [Radius], temperature of the hot part. [keV]
                   NaN if coldFrac = 1, where the hot part is empty.
    """

    radius = np.zeros(numRadii)
    TCold = np.zeros(numRadii)
    THot = np.zeros(numRadii)

    for radIdx in range(numRadii):

        distFunc = load_distribution_function(simName, timeToPlot, radIdx=radIdx)

        fCold, fHot = split_f(distFunc['vPerp'], distFunc['vPar'], distFunc['f'], coldFrac)
        TCold[radIdx] = calculate_temp(distFunc['vPerp'], distFunc['vPar'], fCold)
        THot[radIdx] = calculate_temp(distFunc['vPerp'], distFunc['vPar'], fHot)
        radius[radIdx] = distFunc['radius']

    splitProfiles = {'time': distFunc['time'],
                     'radius': radius,
                     'TCold': TCold,
                     'THot': THot}

    return splitProfiles

def plot_taup_vs_cold_frac(simName, timeToPlot, numPoints=101, makeplot=True, saveplot=False):
    """
    Plot the GDT confinement time of the cold part, the collisionless confinement time of the
    hot part, and the effective confinement time of the two together, vs. the fraction of the
    density in the cold part.

    The distribution function is split as in scan_cold_frac(). The confinement times are from
    tau_gdt() with the cold temperature, and tau_collisionless() with the hot temperature and the
    total density, as the hot ions scatter off all the ions, not just the other hot ions. The
    length and mirror ratio are taken from the eqdsk of the
    IPS timestep, see mirror_geometry(). The effective confinement time is from
    calculate_tau_eff(). The confinement time calculated in CQL3D (tauc_code) on the same flux
    surface is shown for reference.

    Parameters
    ----------
    simName : str
        The name of the simulation.
    timeToPlot : float
        The simulation time at which to split the distribution function. [s]
        The closest end of an IPS timestep is used, see load_distribution_function().
    numPoints : int
        Number of coldFrac values between 0 and 1.
        Default is 101.
    makeplot : bool
        Show the plot.
        Default is True.
    saveplot : bool
        Save the plot in /home/sanwalka/shinethru/plots/ as f'{simName}_taup_vs_cold_frac.png'.
        Default is False.

    Returns
    -------
    coldFracArr : np.array
        [numPoints], fraction of the density in the cold part.
    tauGDTCold : np.array
        [numPoints], GDT confinement time of the cold part. [s]
    tauCollisionlessHot : np.array
        [numPoints], collisionless confinement time of the hot part. [s]
    tauEff : np.array
        [numPoints], effective confinement time of the cold and hot parts together. [s]
    """

    distFunc = load_distribution_function(simName, timeToPlot)

    coldFracArr = np.linspace(0, 1, numPoints)
    scan = scan_cold_frac(distFunc['vPerp'], distFunc['vPar'], distFunc['f'], coldFracArr)

    # CQL3D profiles at the time and flux surface of the distribution function
    profiles = load_cql3d_profiles(simName)
    timeIdx = np.argmin(np.abs(profiles['time'] - distFunc['time']))
    radIdx = np.argmin(np.abs(profiles['radius'] - distFunc['radius']))
    tauPCQL3D = profiles['tauP'][timeIdx, radIdx]

    L, mirrorRatio = mirror_geometry(profiles['eqdsk'][timeIdx])

    tauGDTCold = tau_gdt(scan['TCold'], L, mirrorRatio)
    tauCollisionlessHot = tau_collisionless(scan['THot'], scan['densTotal'], mirrorRatio)
    tauEff = calculate_tau_eff(coldFracArr, tauGDTCold, tauCollisionlessHot)

    fig = plt.figure(figsize=(10, 7), tight_layout=True)
    fig.suptitle(simName)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(coldFracArr, tauGDTCold*1e3, color='tab:blue', linewidth=3,
            label=f'GDT (cold), L = {L:.3f} m, ' + r'R$_m$' + f' = {mirrorRatio:.1f}')
    ax.plot(coldFracArr, tauCollisionlessHot*1e3, color='tab:red', linewidth=3,
            label='Collisionless (hot)')
    ax.axhline(tauPCQL3D*1e3, color='k', linewidth=2, linestyle='--',
               label=f'CQL3D (tauc_code), {tauPCQL3D*1e3:.3f} ms')
    ax.plot(coldFracArr, tauEff*1e3, color='tab:purple', linewidth=3, label='Effective')

    # The scalings are orders of magnitude apart
    ax.set_yscale('log')
    ax.set_xlim(0, 1)

    ax.set_xlabel(r'$n_{cold}/n_{total}$')
    ax.set_ylabel(r'$\tau_p$ [ms]')
    ax.set_title(f'Time = {distFunc["time"]*1e3:.2f} ms, R = {distFunc["radius"]*1e2:.2f} cm')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    if saveplot:
        savePath = f'/home/sanwalka/shinethru/plots/{simName}_taup_vs_cold_frac.png'
        fig.savefig(savePath, dpi=600)
        print(f'Saved plot to {savePath}')

    if makeplot:
        plt.show()

    return coldFracArr, tauGDTCold, tauCollisionlessHot, tauEff

if __name__ == '__main__':

    simName = 'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'

    # Time to plot (in seconds)
    timeToPlot = 3e-3
    coldFrac = 0.26

    # plot_taup_dens_ti(simName, timeToPlot, coldFrac=coldFrac, makeplot=True, saveplot=True)

    # plot_distribution_function(simName, timeToPlot, makeplot=True, saveplot=False)

    distFunc = load_distribution_function(simName, timeToPlot)
    fCold, fHot = split_f(distFunc['vPerp'], distFunc['vPar'], distFunc['f'], coldFrac=0.26,
                          simName=simName, makeplot=True, saveplot=True)

    # plot_temp_vs_cold_frac(simName, timeToPlot, makeplot=True, saveplot=False)

    # plot_taup_vs_cold_frac(simName, timeToPlot, makeplot=True, saveplot=False)
