import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

def build_inverse_interpolator(lineDensScanArr, nbiVoltageScanArr, shineThruTable):
    """
    Returns a function: (shineThru, nbiVoltage) -> lineDens
    Pre-builds one inverse spline per nbiVoltage column.
    """
    inverse_interps = []

    for j, voltage in enumerate(nbiVoltageScanArr):
        shine_col = shineThruTable[:, j]  # shineThru vs lineDens at this voltage

        # shineThru must be monotonic in lineDens for inversion to work
        # If it's decreasing, flip it
        if shine_col[0] > shine_col[-1]:
            inverse_interps.append(
                sc.interpolate.interp1d(shine_col[::-1], lineDensScanArr[::-1],
                         kind='linear', bounds_error=False, fill_value='extrapolate')
            )
        else:
            inverse_interps.append(
                sc.interpolate.interp1d(shine_col, lineDensScanArr,
                         kind='linear', bounds_error=False, fill_value='extrapolate')
            )

    # inverse_interps[j] gives lineDens(shineThru) at nbiVoltageScanArr[j]

    # Now build a 2D lookup: interpolate across voltage dimension too
    def get_lineDens(shineThru_vals, nbiVoltage_vals):
        shineThru_vals = np.asarray(shineThru_vals)
        nbiVoltage_vals = np.asarray(nbiVoltage_vals)

        # For each voltage column, get lineDens estimates
        lineDens_grid = np.array([f(shineThru_vals) for f in inverse_interps])
        # Shape: (32, N) — one row per voltage column

        # Interpolate across the voltage axis for each sample
        results = np.zeros(len(shineThru_vals))
        for i in range(len(shineThru_vals)):
            v_interp = sc.interpolate.interp1d(nbiVoltageScanArr, lineDens_grid[:, i],
                                kind='linear', bounds_error=False, fill_value='extrapolate')
            results[i] = v_interp(nbiVoltage_vals[i])

        return results

    return get_lineDens

def plot_lookup_table():

    # Load the lookup table data
    lookupTable = np.load('/home/sanwalka/shinethru/lookup_tables/shine_thru_table_d.npz')

    # Extract the data from the lookup table
    shineThruTable = lookupTable['shineThruTable']
    lineDensScanArr = lookupTable['lineDensScanArr'] # [m^-2]
    nbiVoltageScanArr = lookupTable['nbiVoltageScanArr'] / 1e3 # [kV]

    # Inverse lookup function
    shineThruTableFunc = build_inverse_interpolator(lineDensScanArr, nbiVoltageScanArr, shineThruTable)

    # Example usage: get line density for given shine-through and NBI voltage
    shineThruArr = np.linspace(0, 1, 1200)
    nbiVoltageArr = np.zeros_like(shineThruArr) + 20

    lineDensArr = shineThruTableFunc(shineThruArr, nbiVoltageArr)

    # Plot the lookup table
    fig = plt.figure(figsize=(10, 10), tight_layout=True)

    # Plot line density vs shine-through for a fixed NBI voltage
    ax1 = fig.add_subplot(211)

    # % difference in line density between each data point
    ax2 = fig.add_subplot(212)

    ax1.plot(shineThruArr, lineDensArr)

    ax1.set_xlabel(r'$n_{ref}/n_{plasma}$')
    ax1.set_ylabel(r'$\int n_e \cdot dl$ (m$^{-2}$)')
    ax1.set_title('20kV NBI Voltage')
    ax1.set_yscale('log')

    # Compute the % difference in line density between each data point
    diff = np.diff(lineDensArr)
    percent_diff = np.abs((diff / lineDensArr[:-1]) * 100)
    ax2.plot(shineThruArr[:-1], percent_diff)

    ax2.set_xlabel(r'$n_{ref}/n_{plasma}$')
    ax2.set_ylabel('% Difference')
    ax2.set_title('Variation in Line Density')
    ax2.set_yscale('log')

    plt.show()

def lookup_table_coarseness():

    # Load the lookup table data
    lookupTable = np.load('/home/sanwalka/shinethru/lookup_tables/shine_thru_table_d.npz')

    # Extract the data from the lookup table
    shineThruTable = lookupTable['shineThruTable']
    lineDensScanArr = lookupTable['lineDensScanArr'] # [m^-2]
    nbiVoltageScanArr = lookupTable['nbiVoltageScanArr'] / 1e3 # [kV]

    # Gradient of the lookup table
    gradDens, gradVolt = np.gradient(shineThruTable, lineDensScanArr, nbiVoltageScanArr, edge_order=2)
    gradientArr = np.sqrt(gradDens**2 + gradVolt**2)
    # Normalize to the shinethru array
    normGradient = np.abs(gradientArr / shineThruTable)
    percentGradient = normGradient * 100

    # Ignore values below 5kV
    mask = nbiVoltageScanArr > 5
    nbiVoltageScanArr = nbiVoltageScanArr[mask]
    shineThruTable = shineThruTable[:, mask]
    percentGradient = percentGradient[:, mask]

    # Plot the lookup table
    fig = plt.figure(figsize=(10, 10), tight_layout=True)

    # Lookup table contour plot
    ax1 = fig.add_subplot(211)
    # Gradient contour plot
    ax2 = fig.add_subplot(212)

    # Create a contour plot of the lookup table
    contour = ax1.contourf(nbiVoltageScanArr, lineDensScanArr, np.log(shineThruTable), levels=50, cmap='viridis')
    plt.colorbar(contour, label=r'ln$\left(\frac{n_{ref}}{n_{plasma}}\right)$')

    ax1.set_xlabel('NBI Voltage (kV)')
    ax1.set_ylabel(r'$\int n_e \cdot dl$ (m$^{-2}$)')
    ax1.set_title('Shine-Through Lookup Table')

    # Create a contour plot of the gradient
    contour2 = ax2.contourf(nbiVoltageScanArr, lineDensScanArr, percentGradient, levels=50, cmap='inferno')
    plt.colorbar(contour2, label='Gradient (%)')

    ax2.set_xlabel('NBI Voltage (kV)')
    ax2.set_ylabel(r'$\int n_e \cdot dl$ (m$^{-2}$)')
    ax2.set_title('Gradient of Shine-Through Lookup Table')

    plt.show()

if __name__ == "__main__":

    plot_lookup_table()

    # lookup_table_coarseness()