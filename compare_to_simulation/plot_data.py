"""
Compare the effect of correcting for the difference in the NBI current between the plasma and reference shot.

Based on previous analysis, this is the single largest source of error in the line-integrated density calculations by a factor of 3x.
"""

import pickle
import subprocess
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

# Make the font size larger
plt.rcParams.update({'font.size': 18})

def load_data(shotnum):

    try:
        # .pkl file with the data
        filename = f'/home/sanwalka/shinethru/data/{shotnum}.pkl'
        with open(filename, "rb") as file:
            detDictList = pickle.load(file)

    except Exception as e:
        print(e)
        print('The data for this shot does not exist, running the analysis to generate the data')

        #### Run the analysis

        commandStr = f'python3 experimental_data.py -s {shotnum}'
        subprocess.run(commandStr, shell=True, check=True)

        #### Load the .pkl files

        # .pkl file for no current correction
        filename = f'/home/sanwalka/shinethru/data/{shotnum}.pkl'
        with open(filename, "rb") as file:
            detDictList = pickle.load(file)

    # Detector impact parameters [m]
    impactParams = np.zeros(len(detDictList))
    for i in range(len(impactParams)):

        beam_pos = detDictList[i]['impact_param_vertical']
        impactParams[i] = beam_pos / 1e3

    # Get the line integrated densities and time array for each detector
    densList = []
    densErrList = []
    timeArr2D = []
    dataPresent = []
    for i in range(len(impactParams)):

        lineIntegratedDens = detDictList[i]['line_integrated_density']

        if lineIntegratedDens is not None:
            densList.append(detDictList[i]['line_integrated_density'])
            densErrList.append(detDictList[i]['line_integrated_density_sigma'])
            timeArr2D.append(detDictList[i]['time_arr_slow'])
            dataPresent.append(True)
        else:
            dataPresent.append(False)

    # Remove the impact parameter with no data
    impactParams = impactParams[dataPresent]

    # Sort the data based on impactParams
    sortIdx = np.argsort(impactParams)
    impactParams = impactParams[sortIdx]
    timeArr2D = [timeArr2D[i] for i in sortIdx]
    densList = [densList[i] for i in sortIdx]
    densErrList = [densErrList[i] for i in sortIdx]

    # Remove the 1st detector (railed, broken)
    impactParams = impactParams[1:]
    timeArr2D = timeArr2D[1:]
    densList = densList[1:]
    densErrList = densErrList[1:]

    return impactParams, timeArr2D, densList, densErrList

def plot_data(shotnum, timesToPlot):

    # Load the experimental data
    impactParams, timeArr2D, densList, densErrList = load_data(shotnum)

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    fig.suptitle(shotnum)

    # Color each time point on a colormap
    cmap = plt.get_cmap('viridis', len(timesToPlot)).colors

    # Plot each timepoint
    for i in range(len(timesToPlot)):

        # Data to plot
        densAtTime = []
        densErrAtTime = []
        for j in range(len(impactParams)):

            timeIdx = np.abs(timeArr2D[j] - timesToPlot[i]).argmin()
            densAtTime.append(densList[j][timeIdx])
            densErrAtTime.append(densErrList[j][timeIdx])

        densAtTime = np.array(densAtTime)
        densErrAtTime = np.array(densErrAtTime)

        ax.errorbar(impactParams*1e2, densAtTime, 
                    yerr=densErrAtTime,
                    fmt='o',
                    ms=10,
                    color=cmap[i],
                    elinewidth=5,
                    label=f'{np.round(timesToPlot[i]*1e3, 1)}')
        
        ax.errorbar(impactParams*1e2, densAtTime, 
                    yerr=3*densErrAtTime,
                    fmt='o',
                    ms=10,
                    color=cmap[i])
        
        ax.plot(impactParams*1e2, densAtTime,
                linewidth=2,
                color=cmap[i])


    ax.legend(title='Time [ms]')
    ax.set_xlabel('Impact Parameter [cm]')
    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')

    ax.set_ylim(0, None)
    ax.set_xlim(-np.max(np.abs(impactParams*1e2))*1.1, np.max(np.abs(impactParams*1e2))*1.1)

    plt.show()

    return

def animate_plot(shotnum, savePath=None):

    # Load the experimental data
    impactParams, timeArr2D, densList, densErrList = load_data(shotnum)

    # Convert to cm
    impactParams_cm = impactParams * 1e2

    #### timeArr2D, densList, densErrList are jagged lists, convert them to a numpy array by trimming the data
    
    # Length of the shortest array
    shortLen = np.min(np.array([len(timeArr) for timeArr in timeArr2D]))

    # Trim all data to that length
    trimmedTime = np.array([timeArr[:shortLen] for timeArr in timeArr2D])
    trimmedDens = np.array([densArr[:shortLen] for densArr in densList])
    trimmedDensErr = np.array([densErrArr[:shortLen] for densErrArr in densErrList])

    # Now all detectors share the same timebase
    timeArr = trimmedTime[0]

    # Convert the error bars and data into numpy arrays
    # 1st index - impact parameter
    # 2nd index - time
    densList = np.array(trimmedDens)
    densErrList = np.array(trimmedDensErr)

    # Maximum density
    maxDens = np.max(densList)

    # Define the figure
    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)

    ax.set_xlabel('Impact Parameter [cm]')
    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')

    def update(frame):

        # Remove all plots, labels etc. from the subplot
        ax.clear()

        # Change the plot title
        timeInMS = np.round(timeArr[frame]*1e3, 2)
        ax.set_title(f'{shotnum} \n {timeInMS}ms')

        # Keep labels after clearing
        ax.set_xlabel('Impact Parameter [cm]')
        ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')

        # Make sure the plot limits don't change over the course of the animation
        ax.set_xlim(-np.max(np.abs(impactParams_cm)) * 1.1, np.max(np.abs(impactParams_cm)) * 1.1)
        ax.set_ylim(0, maxDens * 1.1)

        # Get the values to plot
        densAtTime = densList[:, frame]
        densErrAtTime = densErrList[:, frame]

        # Plot +-1 sigma
        ax.errorbar(impactParams_cm, densAtTime, 
                    yerr=densErrAtTime,
                    fmt='o',
                    ms=10,
                    elinewidth=5,
                    color='C0')
        
        ax.errorbar(impactParams_cm, densAtTime, 
                    yerr=3*densErrAtTime,
                    fmt='o',
                    ms=10,
                    color='C0')
        
        ax.plot(impactParams_cm, densAtTime,
                linewidth=2,
                color='C0')


    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(timeArr),
        interval=1,
        blit=False
    )

    if savePath is not None:
        anim.save(savePath, 
                  writer='ffmpeg', 
                  fps=10,
                  dpi=150)

    else:
        plt.show()

    plt.close()

    return

if __name__ == '__main__':

    shotnum = 260507094

    # timeToPlot = np.array([7, 8, 9, 10, 11]) * 1e-3
    # timesToPlot = np.array([2, 4, 6, 8, 10, 12]) * 1e-3
    timesToPlot = np.array([3, 4, 5, 6, 7, 8]) * 1e-3
    timesToPlot = np.array([4, 6, 8, 10, 12]) * 1e-3
    plot_data(shotnum, timesToPlot)

    # _, _ = calculate_chi2_param(shotnum, 1e3, True)

    # animate_plot(shotnum, savePath=f'/home/sanwalka/shinethru/compare_to_simulation/{shotnum}_animation.mp4')