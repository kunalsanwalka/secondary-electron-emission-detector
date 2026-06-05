"""
Compare the effect of correcting for the difference in the NBI current between the plasma and reference shot.

Based on previous analysis, this is the single largest source of error in the line-integrated density calculations by a factor of 3x.
"""

import pickle
import subprocess
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

# Make the font size larger
plt.rcParams.update({'font.size': 18})

def load_data(shotnum):

    try:
        # .pkl file for no current correction
        filename = f'/home/sanwalka/shinethru/data/{shotnum}_no_current_correction.pkl'
        with open(filename, "rb") as file:
            noCorrection = pickle.load(file)

        # .pkl file with current correction
        filename = f'/home/sanwalka/shinethru/data/{shotnum}_with_current_correction.pkl'
        with open(filename, "rb") as file:
            withCorrection = pickle.load(file)

    except Exception as e:
        print(e)
        print('The data for this shot does not exist, running the analysis to generate the data')

        #### Run the analysis

        # Without current correction
        commandStr = f'python3 postproc_shinethru.py -s {shotnum}'
        subprocess.run(commandStr, shell=True, check=True)

        # With current correction
        commandStr = f'python3 postproc_shinethru_nbi_current_corrected.py -s {shotnum}'
        subprocess.run(commandStr, shell=True, check=True)

        #### Load the .pkl files

        # .pkl file for no current correction
        filename = f'/home/sanwalka/shinethru/data/{shotnum}_no_current_correction.pkl'
        with open(filename, "rb") as file:
            noCorrection = pickle.load(file)

        # .pkl file with current correction
        filename = f'/home/sanwalka/shinethru/data/{shotnum}_with_current_correction.pkl'
        with open(filename, "rb") as file:
            withCorrection = pickle.load(file)

    # Detector impact parameters
    impactParams = np.zeros(len(noCorrection))
    for i in range(len(impactParams)):

        beam_pos = noCorrection[i]['beam_pos']
        impactParams[i] = beam_pos[1] / 1e3

    # Get the line integrated densities and time array for each detector
    noCorrectionList = []
    withCorrectionList = []
    timeArr2D = []
    dataPresent = []
    for i in range(len(impactParams)):

        lineIntegratedDens = noCorrection[i]['line_integrated_density']

        if lineIntegratedDens is not None:
            noCorrectionList.append(noCorrection[i]['line_integrated_density'])
            withCorrectionList.append(withCorrection[i]['line_integrated_density'])
            timeArr2D.append(withCorrection[i]['line_integrated_density_timeArr'])
            dataPresent.append(True)
        else:
            dataPresent.append(False)

    # Remove the impact parameter with no data
    impactParams = impactParams[dataPresent]

    return impactParams, timeArr2D, noCorrectionList, withCorrectionList

def effect_of_current_correction(shotnum, timeToPlot):

    impactParams, timeArr2D, noCorrectionList, withCorrectionList = load_data(shotnum)

    # Data to plot
    noCorrectionArr = []
    withCorrectionArr = []
    for i in range(len(impactParams)):

        timeIdx = np.abs(timeArr2D[i] - timeToPlot).argmin()
        noCorrectionArr.append(noCorrectionList[i][timeIdx])
        withCorrectionArr.append(withCorrectionList[i][timeIdx])

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    fig.suptitle(f'Effect of NBI Current Correction \n {shotnum}; t={timeToPlot*1e3}ms')

    ax.scatter(impactParams*1e2, noCorrectionArr, s=200, label='No Correction')
    ax.scatter(impactParams*1e2, withCorrectionArr, s=200, label='With Correction')

    ax.set_xlabel('Impact Parameter [cm]')
    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')

    ax.set_ylim(0, None)

    ax.legend()

    plt.show()

    return

if __name__ == '__main__':

    shotnum = 260309040

    timeToPlot = 7 * 1e-3
    effect_of_current_correction(shotnum, timeToPlot)