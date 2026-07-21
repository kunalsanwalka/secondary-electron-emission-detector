# -*- coding: utf-8 -*-
"""
This code plots the line-integrated density for a given shot number from the SEE data.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import argparse
import MDSplus as mds

plt.rcParams.update({'font.size': 18})

def parseArgs():

    parser = argparse.ArgumentParser(description='post processing script arguments')
    
    # Shot number
    parser.add_argument('-s','--shotnum', metavar = 'shot number', type=int, default=0,
                        help = 'Shot number')

    # Reference shot number
    parser.add_argument('-r','--ref_shotnum', metavar = 'reference shot number', type=int, default=0,
                        help = 'NBI reference shot number')
    
    args = parser.parse_args()
    
    # If the reference shot number is 0, check if it is in the MDSplus tree for that shot
    if args.ref_shotnum == 0:

        try:
            tree = mds.Tree('wham', args.shotnum)
            ref_shotnum = tree.getNode('diag.shinethru.ref_shotnum').getData().data()
            args.ref_shotnum = ref_shotnum
            tree.close()
        except Exception as e:
            print(e)
    
    return args

def plot_data(timesToPlot):

    ###########################################################################
    # Load the densities from each detector
    ###########################################################################

    timeArr = []
    densityArr = []
    densityErrArr = []
    impactParams = np.zeros(21)

    tree = mds.Tree('wham', args.shotnum)

    for i in range(1, 22):

        try:
            # Load the line integrated density and time for each detector
            node = tree.getNode(f'diag.shinethru.det_{i:02d}.linedens')
            dens = node.getData().data()
            time = node.dim_of().data()
            timeArr.append(time)
            densityArr.append(dens)

            # Load the error in the line integrated density for each detector
            node_err = tree.getNode(f'diag.shinethru.det_{i:02d}.linedens_err')
            dens_err = node_err.getData().data()
            densityErrArr.append(dens_err)

            # Load the vertical impact parameter for each detector
            node_impact = tree.getNode(f'diag.shinethru.det_{i:02d}.v_impact')
            impactParams[i-1] = node_impact.getData().data()

            print(f'Detector {i} data loaded successfully.')

        except Exception as e:
            print(f'Detector {i} data not found: {e}')
            timeArr.append(None)
            densityArr.append(None)

    tree.close()

    ###########################################################################
    # Put all the data on the same time axis by interpolating the data to a 
    # common time array
    ###########################################################################

    common_time = timeArr[0]  # Use the time array from the first detector as the common time array
    densityArr_interp = []
    densityErrArr_interp = []
    for i in range(len(densityArr)):
        if densityArr[i] is not None:
            dens_interp = np.interp(common_time, timeArr[i], densityArr[i])
            densityArr_interp.append(dens_interp)
            dens_err_interp = np.interp(common_time, timeArr[i], densityErrArr[i])
            densityErrArr_interp.append(dens_err_interp)
        else:
            densityArr_interp.append(np.zeros_like(common_time))
            densityErrArr_interp.append(np.zeros_like(common_time))

    timeArr = common_time

    # Make everything a numpy array
    densityArr_interp = np.array(densityArr_interp)
    densityErrArr_interp = np.array(densityErrArr_interp)

    ###########################################################################
    # Remove the 6th and 16th detectors (railed, missing)
    ###########################################################################

    impactParams = np.delete(impactParams, [5, 15])
    densityArr_interp = np.delete(densityArr_interp, [5, 15], axis=0)
    densityErrArr_interp = np.delete(densityErrArr_interp, [5, 15], axis=0)

    ###########################################################################
    # Plot the data
    ###########################################################################

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    # Color each time point on a colormap
    cmap = plt.get_cmap('viridis', len(timesToPlot)).colors

    for i in range(len(timesToPlot)):

        timeIdx = np.argmin(np.abs(timeArr - timesToPlot[i]))
        densAtTime = densityArr_interp[:, timeIdx]
        densErrAtTime = densityErrArr_interp[:, timeIdx]

        # Sort the data based on impactParams
        sortIdx = np.argsort(impactParams)
        impactParams_sorted = impactParams[sortIdx]
        densAtTime_sorted = densAtTime[sortIdx]
        densErrAtTime_sorted = densErrAtTime[sortIdx]

        # Scatter with error bars
        ax.errorbar(impactParams_sorted, densAtTime_sorted,
                    yerr=densErrAtTime_sorted,
                    fmt='o',
                    color=cmap[i],
                    label=f'{np.round(timesToPlot[i]*1e3, 1)}')

        # Line plot
        ax.plot(impactParams_sorted, densAtTime_sorted, color=cmap[i], linewidth=2)

    ax.set_xlabel('Impact Parameter [m]')
    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
    ax.set_title(args.shotnum)
    
    ax.legend(title='Time [ms]', ncols=2)
    plt.show()

    return

if __name__ == "__main__":

    # Parse the command line arguments
    global args
    args = parseArgs()

    # Plot the data
    timesToPlot = np.array([1,3,5,7,9,11]) * 1e-3  # [s]
    timesToPlot = np.arange(2, 8.1, 0.5) * 1e-3 # [s]
    timesToPlot = np.arange(4.5, 5.6, 0.1) * 1e-3 # [s]
    
    plot_data(timesToPlot)
