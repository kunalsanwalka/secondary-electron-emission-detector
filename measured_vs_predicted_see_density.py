# -*- coding: utf-8 -*-
"""

This code compares the measured and predicted SEE densities.

"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import scipy as sc
from predicted_see_density import load_detector_dictionary, load_density_profile, synthetic_see_detector, radial_density_profile
from measured_see_density import parseArgs, calculate_line_densities, model_error_envelope

plt.rcParams.update({'font.size': 20})

if __name__ == "__main__":
    
    # Pleaides file path
    pleiadesFilePath = '/home/sanwalka/shinethru/pleiades_260105053.h5'

    # Time of the pleiades simulation
    pleiadesTime = 6 # in ms

    # Get the arguments (mostly used to set the shot number)
    args = parseArgs()

    # Load the SEE detector dictionary
    pickleFilePath = '/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl'
    detDictList = load_detector_dictionary(pickleFilePath)

    # Calculate the prediction from pleiades
    interpFunc = load_density_profile(pleiadesFilePath, makeplot=False)
    detDictList = synthetic_see_detector(detDictList, interpFunc, makeplot=False)

    # Calculate the measured line-integrated densities
    _, _, _ = calculate_line_densities(args, detDictList)

    """
    Sam: If all you want is detDictList to contain the measured and synthetic line integrated densities, you can stop here.

    The rest of the code is plotting and calculating 100 forward models of the radial density profile.
    """

    # Plot the comparison between the measured and prediced values
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # Predicted density array
    predicedDensities = np.array([detDict['predicted_see_density'] for detDict in detDictList])

    for i in range(len(detDictList)):

        detDict = detDictList[i]

        # Plot the pleiades prediction
        if i == 0:
            ax.scatter(detDict['impact_param_vertical'], detDict['predicted_see_density'], c='blue', s=200, label='Pleiades Prediction')
        else:
            ax.scatter(detDict['impact_param_vertical'], detDict['predicted_see_density'], c='blue', s=200)

        # Plot the measured value
        timeArr = detDict['time']
        timeIdx = np.argmin(np.abs(timeArr - pleiadesTime))
        if i == 0:
            ax.errorbar(detDict['impact_param_vertical'], 
                        detDict['line_integrated_density'][timeIdx], 
                        yerr=detDict['line_integrated_density_sigma'][timeIdx], 
                        fmt='o', 
                        c='red',
                        label='Measured Value')
        else:
            try:
                ax.errorbar(detDict['impact_param_vertical'], 
                            detDict['line_integrated_density'][timeIdx], 
                            yerr=detDict['line_integrated_density_sigma'][timeIdx], 
                            fmt='o', 
                            c='red')
            except:
                pass
    
    ax.set_ylim(0, np.max(predicedDensities)*1.2)
    ax.set_xlabel('Impact parameter [mm]')
    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
    ax.set_title(f'{args.shotnum} t={pleiadesTime} ms')
    ax.legend()
    # plt.show()

    # Get the density profile reconstrction from the SEE arrays
    radialFits, lineIntegratedFits, radialPosArr = model_error_envelope(args, detDictList)

    # Radial density profile from pleiades
    predictedRadialProfile = radial_density_profile(radialPosArr, interpFunc)

    # Plot the predicted and reconstructed radial density profiles
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap('viridis', np.shape(radialFits)[1])
    for i in range(np.shape(radialFits)[1]):

        if i == 0:
            ax.plot(radialPosArr, radialFits[:, i], c=cmap(i), label='Reconstructed Profiles')
        else:
            ax.plot(radialPosArr, radialFits[:, i], c=cmap(i))

    ax.plot(radialPosArr, predictedRadialProfile, c='black', label='Pleiades Prediction', linewidth=3, zorder=10)

    ax.set_xlabel('Radial position [m]')
    ax.set_ylabel(r'Density [m$^{-3}$]')
    ax.set_title(f'{args.shotnum} t={pleiadesTime} ms')
    ax.legend()
    plt.show()