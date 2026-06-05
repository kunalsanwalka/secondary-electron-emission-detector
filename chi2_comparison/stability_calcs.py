"""
Compare various plasma stability parameters like-
1. chi^2
2. 2nd derivative
For may shots. 

The goal is to see if the wiggles spatially in the shinethrough data are real or processing artifacts.
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

    # Detector impact parameters
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

def calculate_chi2_param(shotnum, freq, makeplot=False):

    # Load the experimental data
    impactParams, timeArr2D, densList, densErrList = load_data(shotnum)

    # Indices of the in-vessel detectors
    inVesselIdx = np.argwhere(np.abs(impactParams)*1e2 > 10).T.flatten()

    # Find the longest timeArr in the list
    timesLenArr = np.array([len(timeArr2D[i]) for i in range(len(timeArr2D))])

    # Index of the longest time array
    maxIdx = timesLenArr.argmax()

    # 1D time array corresponding to the maximum length time array
    timeArr = timeArr2D[maxIdx]

    # Go over each timepoint and calculate the centroid and RMS value
    centroidArr = []
    rmsArr = []
    for i in range(len(timeArr)):

        # Get the line-integrated densities at the given time-point
        currDens = []
        # Corresponding impact parameters
        currImpactParams = []

        for j in range(len(densList)):

            try:
                currDens.append(densList[j][i])
                currImpactParams = impactParams
            except:
                if j not in inVesselIdx:
                    currDens.append(densList[j][i])
                currImpactParams = np.delete(np.copy(impactParams), inVesselIdx)

        currDens = np.array(currDens)
        currImpactParams = np.array(currImpactParams)

        # Normalize
        currDens /= np.max(currDens)
        currImpactParams /= np.max(currImpactParams)

        # Centroid
        centroid = np.sum(currDens*currImpactParams) / np.sum(currDens)

        # RMS radius
        rms = 2 * np.sqrt( np.sum(currDens * ((currImpactParams - centroid)**2)) / np.sum(currDens))

        centroidArr.append(centroid)
        rmsArr.append(rms)

    centroidArr = np.array(centroidArr)
    rmsArr = np.array(rmsArr)

    #### Calculate the covariance matrix at the given frequency
    
    chi2 = np.zeros_like(timeArr)

    currFreq = 1/(timeArr[1] - timeArr[0])
    window_size = int(currFreq/freq)

    half_window = window_size // 2

    for i in range(half_window, len(timeArr) - half_window):

        # Extract local window
        x_win = centroidArr[i-half_window:i+half_window]
        y_win = rmsArr[i-half_window:i+half_window]

        # Stack into 2xM array
        data = np.vstack((x_win, y_win))

        # Covariance matrix
        cov_matrix = np.cov(data)

        # chi^2 parameter
        chi2[i] = np.sqrt(np.abs(np.linalg.det(cov_matrix)))

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        fig.suptitle(f'{shotnum} \n' + r'$\chi^2$ at ' + f'{freq/1e3}kHz')

        ax.plot(timeArr*1e3, chi2, linewidth=3)

        ax.set_yscale('log')

        ax.set_ylabel(r'$\chi^2$')
        ax.set_xlabel('Time [ms]')

        plt.show()

    return chi2, timeArr

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

def compare_chi2(shotnumArr, freq):

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    fig.suptitle(r'$\chi^2$ at ' + f'{freq/1e3}kHz')

    for i in range(len(shotnumArr)):

        shotnum = shotnumArr[i]
        
        # Load chi2 from shinethru data
        chi2, timeArr = calculate_chi2_param(shotnum, 1e3)
        # Normalize
        chi2 /= np.max(chi2)
        # Remove values where it is 0
        mask = chi2 > 1e-7
        chi2 = chi2[mask]
        timeArr = timeArr[mask]

        ax.plot(timeArr*1e3, chi2, 
                linewidth=3, 
                color=f'C{i}',
                label=f'{shotnum}')

        # Load chi2 from AXUV data
        timeArr, chi2_da1_data, chi2_da2_data, chi2_da3_data = load_axuv_chi2(shotnum)
        # Normalize
        chi2_da1_data /= np.max(chi2_da1_data)
        chi2_da2_data /= np.max(chi2_da2_data)
        chi2_da3_data /= np.max(chi2_da3_data)

        ax.plot(timeArr*1e3, chi2_da1_data,
                linewidth=2,
                linestyle='dashed',
                color=f'C{i}')
        ax.plot(timeArr*1e3, chi2_da2_data,
                linewidth=2,
                linestyle='dotted',
                color=f'C{i}')
        ax.plot(timeArr*1e3, chi2_da3_data,
                linewidth=2,
                linestyle='dashdot',
                color=f'C{i}')


    # Add labels to the legend
    ax.plot([], [],
            linewidth=3,
            color='k',
            label='SEE')
    ax.plot([], [],
            linewidth=2,
            color='k',
            linestyle='dashed',
            label='Broadband')
    ax.plot([], [],
            linewidth=2,
            color='k',
            linestyle='dotted',
            label=r'H$_{\alpha}$')
    ax.plot([], [],
            linewidth=2,
            color='k',
            linestyle='dashdot',
            label='SXR')

    ax.set_yscale('log')

    ax.set_ylabel(r'Normalized $\chi^2$')
    ax.set_xlabel('Time [ms]')

    ax.legend()

    plt.show()

    return

def effect_of_decimation(shotnum):

    # Array of decimation frequencies which we want to scan
    decimationArr = np.array([1e5, 5e4, 2e4, 1e4], dtype=int)

    timeArr_2d = []
    chi2_2d = []

    # Run the analysis for each decimation level
    for i in range(len(decimationArr)):

        print(f'Working on {decimationArr[i]/1e3}kHz')

        commandStr = f'python3 experimental_data.py -s {shotnum} -dec {decimationArr[i]}'
        subprocess.run(commandStr, shell=True, check=True)

        chi2, timeArr = calculate_chi2_param(shotnum, 1e3) 
        chi2_2d.append(chi2)
        timeArr_2d.append(timeArr)

    # Plot chi2 at different decimation levels
    fig = plt.figure(figsize=(12, 8), tight_layout=True)    
    ax = fig.add_subplot(111)

    ax.set_title(r'Effect of raw data rate on $\chi^2$')

    for i in range(len(decimationArr)):

        ax.plot(timeArr_2d[i]*1e3, chi2_2d[i],
                label=f'{decimationArr[i]/1e3}kHz')
        
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'$\chi^2$')

    ax.set_yscale('log')

    ax.legend()

    plt.show()

    return

def load_axuv_chi2(shotnum):

    # Open the 3 .npz files
    da1_file = np.load(f'AXUV_results{shotnum}_{shotnum}_DA1.npz',
                       allow_pickle=True)
    da2_file = np.load(f'AXUV_results{shotnum}_{shotnum}_DA2.npz',
                       allow_pickle=True)
    da3_file = np.load(f'AXUV_results{shotnum}_{shotnum}_DA3.npz',
                       allow_pickle=True)

    # chi2 data
    chi2_da1 = da1_file[f's{shotnum}_macro_stability_chi_and_time_cm2_s'].item()
    chi2_da2 = da2_file[f's{shotnum}_macro_stability_chi_and_time_cm2_s'].item()
    chi2_da3 = da3_file[f's{shotnum}_macro_stability_chi_and_time_cm2_s'].item()

    # 1kHz data
    chi2_da1_data = chi2_da1['1_dev']
    chi2_da2_data = chi2_da2['1_dev']
    chi2_da3_data = chi2_da3['1_dev']

    timeArr = chi2_da1['1_t']

    return timeArr, chi2_da1_data, chi2_da2_data, chi2_da3_data

if __name__ == '__main__':

    shotnum = 260426037
    # shotnum = 260302079

    effect_of_decimation(shotnum)

    # timeArr, chi2_da1, chi2_da2, chi2_da3 = load_axuv_chi2(shotnum)

    # timeToPlot = np.array([7, 8, 9, 10, 11]) * 1e-3
    # timesToPlot = np.array([2, 4, 6, 8, 10, 12]) * 1e-3
    # plot_data(shotnum, timesToPlot)

    # _, _ = calculate_chi2_param(shotnum, 1e3, True)

    shotnumArr = np.array([260302079, 260426037])
    shotnumArr = np.array([260426037])
    compare_chi2(shotnumArr, 1e3)