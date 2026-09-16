"""
Metrics that compare the synthetic SEE diagnostic (CQL3D + KN1D) against the experimental
line-integrated density at pairs of simulation and experimental times.

Every metric has the same call signature, so any of them can be passed to
compare_simulation_and_experiment(), plot_simulation_scan_panel() or compare_all_simulations()
in cql3d_vs_data.py-

    import comparison_metrics as cm

    compare_simulation_and_experiment(simulationName, shotnum, metric=cm.log_ratio_rms)

Metrics with tunable parameters take them as extra keyword arguments. Use functools.partial
to fix them and keep the same signature-

    from functools import partial
    metric = partial(cm.reduced_chi_squared, relSysErr=0.1)

Shared parameters
-----------------
expDataArr : np.array
    Line-integrated density from the experiment. [m^-2]
expDataSigmaArr : np.array
    Error bars for the line-integrated density. [m^-2]
simDataArr : np.array
    Line-integrated density from the simulation. [m^-2]
impactParams : np.array
    Vertical impact parameter for each detector, sorted in increasing order (always 1D). [m]
expTime : float or np.array
    Time for the experimental data. [s]
simTime : float or np.array
    Time for the simulation data. [s]

Array shapes
------------
The first axis of the data arrays is the detector. Every metric reduces over that axis only
and broadcasts over any others. For a single pair of times, pass 1D arrays and get a float.
compare_simulation_and_experiment() computes the whole map in one call by passing expDataArr
as [detector x 1 x expTime] and simDataArr as [detector x simTime x 1], which returns a
[simTime x expTime] array.

Conventions
-----------
- Lower is better and 0 is a perfect match. The one exception is best_fit_scale_factor(),
  where 1 is a perfect match. All values are >= 0, so they work with the log color scale in
  plot_simulation_scan_panel().
- The input arrays are views into the full [detector x time] arrays, so nothing here
  modifies them in place.
- Saved comparison data and plots are labeled with metric_name(), which includes every
  setting of the metric, so each metric and each choice of settings gets its own files.
"""

import inspect
import numpy as np
from functools import partial

def _along_detectors(impactParams, ndim):
    """
    Reshape the 1D impact parameters to broadcast along the first (detector) axis of an ndim array.
    """

    return np.reshape(impactParams, (-1,) + (1,)*(ndim - 1))

def _total_sigma(expDataArr, expDataSigmaArr, relSysErr):
    """
    Experimental error bars with a relative systematic error added in quadrature. [m^-2]

    The error bars from experimental_data.py are 9% of the signal for the in-vessel chords but
    only 0.1% for the beam dump chords. Without a systematic floor, each beam dump chord gets
    ~8000x the weight of an in-vessel chord. relSysErr accounts for errors the error bars do not
    include (calibration, chord geometry, the synthetic diagnostic itself).
    """

    return np.sqrt(expDataSigmaArr**2 + (relSysErr*expDataArr)**2)

def _scale_fit(expDataArr, expDataSigmaArr, simDataArr, relSysErr):
    """
    Error-weighted least squares fit of expDataArr = scale * simDataArr.

    Returns
    -------
    scale : float or np.array
        Best-fit scale factor.
    sigma : np.array
        Total error bars used in the fit. [m^-2]
    """

    sigma = _total_sigma(expDataArr, expDataSigmaArr, relSysErr)
    weights = 1/sigma**2

    scale = np.sum(weights*expDataArr*simDataArr, axis=0) / np.sum(weights*simDataArr**2, axis=0)

    return scale, sigma

def _profile_width(lineDensArr, impactParams):
    """
    RMS width of a line-integrated density profile about its centroid. [m]
    """

    impactParams = _along_detectors(impactParams, np.ndim(lineDensArr))

    total = np.trapezoid(lineDensArr, impactParams, axis=0)
    centroid = np.trapezoid(impactParams*lineDensArr, impactParams, axis=0) / total

    return np.sqrt(np.trapezoid((impactParams - centroid)**2 * lineDensArr, impactParams, axis=0) / total)

def normalized_mean_abs_error(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime):
    """
    Mean absolute difference, normalized by the mean of both data sets.

    This was the original metric in cql3d_vs_data.py, and is the default there.

    It can never exceed 2. When the simulation is much larger than the experiment, the
    simulation dominates both the numerator and the denominator, so a simulation that is 5x
    too dense scores almost the same as one that is 50x too dense.

    Returns
    -------
    comparison : float or np.array
        Normalized mean absolute error, between 0 and 2.
    """

    absDiff = np.mean(np.abs(expDataArr - simDataArr), axis=0)

    # Mean of both data sets together (they have the same number of detectors)
    avgDens = (np.mean(expDataArr, axis=0) + np.mean(simDataArr, axis=0)) / 2

    return absDiff/avgDens

def log_ratio_rms(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime, densFloor=1e16):
    """
    Root-mean-square of ln(sim/exp) over the chords.

    Over- and under-prediction count the same (a factor of 2 too high costs the same as a
    factor of 2 too low), and there is no upper bound. exp(comparison) is roughly the typical
    factor by which each chord is off, e.g. 0.69 is a factor of 2.

    Parameters
    ----------
    densFloor : float
        Floor applied to both data sets before taking the log, to avoid log(0). [m^-2]
        Default is 1e16.

    Returns
    -------
    comparison : float or np.array
        RMS of the log ratio.
    """

    logRatio = np.log(np.maximum(simDataArr, densFloor) / np.maximum(expDataArr, densFloor))

    return np.sqrt(np.mean(logRatio**2, axis=0))

def reduced_chi_squared(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime, relSysErr=0.05):
    """
    Chi-squared per chord, weighted by the experimental error bars.

    Unlike the other metrics this has an absolute scale: ~1 means the simulation agrees with
    the experiment within the error bars, and >>1 means it does not.

    Parameters
    ----------
    relSysErr : float
        Relative systematic error added in quadrature to the error bars (see _total_sigma).
        Default is 0.05, a placeholder. Set it to your best estimate.

    Returns
    -------
    comparison : float or np.array
        Reduced chi-squared.
    """

    sigma = _total_sigma(expDataArr, expDataSigmaArr, relSysErr)

    return np.mean(((expDataArr - simDataArr)/sigma)**2, axis=0)

def best_fit_scale_factor(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime, relSysErr=0.05):
    """
    Factor the simulation has to be multiplied by to best match the experiment.

    This is the only function here where 1 (not 0) is a perfect match. It shows the direction
    of the mismatch: < 1 means the simulation is too dense, > 1 means it is not dense enough.

    Parameters
    ----------
    relSysErr : float
        Relative systematic error added in quadrature to the error bars (see _total_sigma).
        Default is 0.05.

    Returns
    -------
    scale : float or np.array
        Best-fit scale factor.
    """

    scale, _ = _scale_fit(expDataArr, expDataSigmaArr, simDataArr, relSysErr)

    return scale

def amplitude_mismatch(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime, relSysErr=0.05):
    """
    How far off the overall density level is, ignoring the profile shape.

    Equal to |ln(best_fit_scale_factor)|, so 0.69 means the simulation is a factor of 2 too
    high or too low. Use it together with shape_mismatch() to tell whether a simulation gets
    the profile shape right but the density wrong, or vice versa.

    Parameters
    ----------
    relSysErr : float
        Relative systematic error added in quadrature to the error bars (see _total_sigma).
        Default is 0.05.

    Returns
    -------
    comparison : float or np.array
        Absolute log of the best-fit scale factor.
    """

    scale, _ = _scale_fit(expDataArr, expDataSigmaArr, simDataArr, relSysErr)

    return np.abs(np.log(scale))

def shape_mismatch(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime, relSysErr=0.05):
    """
    Reduced chi-squared after scaling the simulation by best_fit_scale_factor().

    This is insensitive to the overall density level and only measures the profile shape. It
    is on the same scale as reduced_chi_squared(), so the gap between the two is how much of
    the mismatch comes from the density level alone.

    Parameters
    ----------
    relSysErr : float
        Relative systematic error added in quadrature to the error bars (see _total_sigma).
        Default is 0.05.

    Returns
    -------
    comparison : float or np.array
        Reduced chi-squared of the scaled simulation, with one degree of freedom used by the fit.
    """

    scale, sigma = _scale_fit(expDataArr, expDataSigmaArr, simDataArr, relSysErr)

    return np.sum(((expDataArr - scale*simDataArr)/sigma)**2, axis=0) / (np.shape(expDataArr)[0] - 1)

def width_mismatch(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime):
    """
    Mismatch in the RMS width of the line-integrated density profile across the chords.

    The width is the second moment of the profile about its centroid. It only covers the range
    of impact parameters the chords span. This is a single number for how peaked or broad the
    profile is, which should respond to the neutral density at the edge.

    Returns
    -------
    comparison : float or np.array
        |ln(simulated width / experimental width)|.
    """

    expWidth = _profile_width(expDataArr, impactParams)
    simWidth = _profile_width(simDataArr, impactParams)

    return np.abs(np.log(simWidth/expWidth))

def inventory_mismatch(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime):
    """
    Mismatch in the line-integrated density integrated over the impact parameter.

    For parallel chords this integral is the density integrated over the area the chords
    cover, i.e. the total plasma content seen by the diagnostic. Unlike amplitude_mismatch(),
    this weights each chord by its spacing, not its error bar, so the densely packed beam dump
    chords do not dominate.

    Returns
    -------
    comparison : float or np.array
        |ln(simulated content / experimental content)|.
    """

    expInventory = np.trapezoid(expDataArr, impactParams, axis=0)
    simInventory = np.trapezoid(simDataArr, impactParams, axis=0)

    return np.abs(np.log(simInventory/expInventory))

def restrict_to_chords(metric, absImpactMin=0, absImpactMax=np.inf):
    """
    Make a version of a metric that only uses chords with absImpactMin <= |impact param| <= absImpactMax.

    Comparing the core and edge separately can help separate the two scan parameters, since the
    main vessel and gas box neutral densities should affect different parts of the profile. For
    shot 260426037, 0.09 m separates the 14 beam dump chords from the 5 in-vessel chords-

        coreMetric = cm.restrict_to_chords(cm.log_ratio_rms, absImpactMax=0.09)
        edgeMetric = cm.restrict_to_chords(cm.log_ratio_rms, absImpactMin=0.09)

    width_mismatch() and inventory_mismatch() integrate across any gap in the chords, so only
    use them with a contiguous range (absImpactMin=0).

    Fix a metric's settings with functools.partial before restricting it, so that they are
    included in its name-

        cm.restrict_to_chords(partial(cm.reduced_chi_squared, relSysErr=0.1), absImpactMax=0.09)

    Parameters
    ----------
    metric : function
        Any metric in this module.
    absImpactMin : float
        Smallest |impact parameter| to keep. [m]
        Default is 0.
    absImpactMax : float
        Largest |impact parameter| to keep. [m]
        Default is np.inf.

    Returns
    -------
    restrictedMetric : function
        Metric with the same signature that only uses the selected chords.
    """

    def restrictedMetric(expDataArr, expDataSigmaArr, simDataArr, impactParams, expTime, simTime):

        # Select along the first (detector) axis
        keep = (np.abs(impactParams) >= absImpactMin) & (np.abs(impactParams) <= absImpactMax)

        return metric(expDataArr = expDataArr[keep],
                      expDataSigmaArr = expDataSigmaArr[keep],
                      simDataArr = simDataArr[keep],
                      impactParams = impactParams[keep],
                      expTime = expTime,
                      simTime = simTime)

    # Unique name for the saved data and plots
    restrictedMetric.__name__ = f'{metric_name(metric)}_absImpact={absImpactMin}-{absImpactMax}'

    return restrictedMetric

def metric_name(metric):
    """
    Unique name for a comparison metric, used to label the saved data and plots.

    The name is the function name followed by every setting it uses, including defaults, e.g.
    'reduced_chi_squared_relSysErr=0.05'. Changing a setting (with functools.partial or by
    editing a default in this file) changes the name, so old results are never reused.

    Parameters
    ----------
    metric : function
        Any metric in this module, optionally wrapped with functools.partial or restrict_to_chords.

    Returns
    -------
    name : str
        Name of the metric.
    """

    # Settings fixed with functools.partial
    settings = {}
    if isinstance(metric, partial):
        settings = metric.keywords
        metric = metric.func

    # Only the settings have defaults, the shared parameters do not
    name = metric.__name__
    for param in inspect.signature(metric).parameters.values():
        if param.default is not param.empty:
            name += f'_{param.name}={settings.get(param.name, param.default)}'

    return name

# Every metric in this module. compare_all_simulations() in cql3d_vs_data.py uses these by default.
METRICS = [normalized_mean_abs_error,
           log_ratio_rms,
           reduced_chi_squared,
           best_fit_scale_factor,
           amplitude_mismatch,
           shape_mismatch,
           width_mismatch,
           inventory_mismatch]
