"""
Function to copy plots from one location to another so they can all be downloaded.
"""

import os
import numpy as np

if __name__ == '__main__':

    # Shot number we are comparing against
    shotnum = 260426037

    # Directory with all the simulations
    simulationDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/'

    # Find all the simulation names
    rawSimNameList = [d for d in os.listdir(simulationDir) if os.path.isdir(os.path.join(simulationDir, d))]
    rawSimNameList = np.array(rawSimNameList)

    # Add the directory to the simulation names
    simNameList = np.array([os.path.join(simulationDir, d) for d in rawSimNameList])

    # Add the plot directory to the simulation names
    plotDirList = np.array([os.path.join(d, 'plots/') for d in simNameList])

    # Names of all the plots we want to copy
    plotNameList = np.array([os.path.join(d, f'{shotnum}_vs_{rawSimNameList[i]}.png') for i, d in enumerate(plotDirList)])

    # Directory to copy the plots to
    targetDir = '/home/sanwalka/shinethru/plots/'

    # Copy the plots to the target directory
    for plotName in plotNameList:

        if os.path.exists(plotName):
            os.system(f'cp {plotName} {targetDir}')
        else:
            print(f'Plot {plotName} does not exist.')