"""
Effect of changing magnetic fields on the the SEE detector signals.

The goal is to characterize the sensitivity of the SEE detector signals to changes in the magnetic field. 
This is an important step in uncertainity quantification,.
"""

import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

# Make the font size larger
plt.rcParams.update({'font.size': 18})

global minTime, maxTime
minTime = -0.06
maxTime = 0.06

def transrex_current(tree, makeplot=False):

    TRANSREX_CURRENT_NODE = 'transrex.isum_lem.filtered'
    currentNode = tree.getNode(TRANSREX_CURRENT_NODE)

    current = currentNode.data()
    timeArr = currentNode.dim_of().data()

    # We only want the data around the pulse, so we will only keep the data from -0.2 to 0.2 seconds
    mask = (timeArr >= minTime) & (timeArr <= maxTime)
    timeArr = timeArr[mask]
    current = current[mask]

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(timeArr, current, label='Transrex Current')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')

        ax.legend()
        plt.show()

    return timeArr, current

def pps_current(tree, makeplot=False):

    #### South Coil

    # Raw signal
    rawNode = tree.getNode('raw.scope_1074.ch_01.signal')
    # Offset
    offset = tree.getNode('raw.scope_1074.ch_01.offset').data()
    # Frequency
    frequency = tree.getNode('raw.scope_1074.ch_01.freq').data()
    # Delay
    delay = tree.getNode('raw.scope_1074.ch_01.delay').data()

    # Current
    southCurrent = (-rawNode.data()+offset) * 5e3 # Amps
    # Time
    timeArr = rawNode.dim_of().data()/frequency + delay -0.065 # seconds

    # Get the mean after the pulse to use as a baseline
    baseline = np.mean(southCurrent[timeArr > 0.1])
    southCurrent -= baseline

    #### North Coil

    # Raw signal
    rawNode = tree.getNode('raw.scope_1074.ch_02.signal')
    # Offset
    offset = tree.getNode('raw.scope_1074.ch_02.offset').data()
    # Frequency
    frequency = tree.getNode('raw.scope_1074.ch_02.freq').data()
    # Delay
    delay = tree.getNode('raw.scope_1074.ch_02.delay').data()

    # Current
    northCurrent = (-rawNode.data()+offset) * 5e3 # Amps
    # Time
    timeArr = rawNode.dim_of().data()/frequency + delay -0.065 # seconds

    # Get the mean after the pulse to use as a baseline
    baseline = np.mean(northCurrent[timeArr > 0.1])
    northCurrent -= baseline

    # Only keep the data where the current is above 0.5kA
    mask = northCurrent > 0.5e3
    timeArr = timeArr[mask]
    southCurrent = southCurrent[mask]
    northCurrent = northCurrent[mask]

    # Plot the data
    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(timeArr, northCurrent, label='North Coil Current')
        ax.plot(timeArr, southCurrent, label='South Coil Current')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')

        ax.legend()
        plt.show()

    return timeArr, northCurrent, southCurrent

def plot_currents(tree):

    # Load the transrex current
    timeArrTransrex, currentTransrex = transrex_current(tree)

    # Load the PPS currents
    timeArrPPS, northCurrent, southCurrent = pps_current(tree)

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(timeArrTransrex, currentTransrex/1e3, label='Transrex Current')
    ax.plot(timeArrPPS, northCurrent/1e3, label='North Coil Current')
    ax.plot(timeArrPPS, southCurrent/1e3, label='South Coil Current')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Current (kA)')

    ax.legend()
    plt.show()

    return

def average_data(timeArr, data, freq, makeplot=False):

    # We will average the data in bins of 1/freq seconds
    bin_size = 1/freq
    bins = np.arange(timeArr[0], timeArr[-1], bin_size)

    # Use scipy to average the data in the bins
    newData, _, _ = sc.stats.binned_statistic(timeArr, data, statistic='mean', bins=bins)
    newTimeArr = (bins[:-1] + bins[1:]) / 2

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(timeArr, data, label='Original Data')
        ax.plot(newTimeArr, newData, label='Averaged Data')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Data')

        ax.legend()
        plt.show()

    return newTimeArr, newData

def load_currents(tree, makeplot=False):

    # Load the transrex current
    timeArrTransrex, currentTransrex = transrex_current(tree)
    # Load the PPS currents
    timeArrPPS, northCurrent, southCurrent = pps_current(tree)

    # Average the data to 10kHz
    timeArrTransrex, currentTransrex = average_data(timeArrTransrex, currentTransrex, 1e4)
    timeArrPPSNew, northCurrent = average_data(timeArrPPS, northCurrent, 1e4)
    timeArrPPSNew, southCurrent = average_data(timeArrPPS, southCurrent, 1e4)

    # Put the data on the same time base by interpolating the PPS data to the transrex time base
    northCurrent = np.interp(timeArrTransrex, timeArrPPSNew, northCurrent)
    southCurrent = np.interp(timeArrTransrex, timeArrPPSNew, southCurrent)

    timeArr = timeArrTransrex

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(timeArr, currentTransrex/1e3, label='Transrex Current')
        ax.plot(timeArr, northCurrent/1e3, label='North Coil Current')
        ax.plot(timeArr, southCurrent/1e3, label='South Coil Current')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (kA)')

        ax.legend()
        plt.show()

    return timeArr, currentTransrex, northCurrent, southCurrent

def nbi_parameters(tree, makeplot=False):

    currNode = tree.getNode('nbi.i_beam')
    voltNode = tree.getNode('nbi.v_beam')

    nbiCurrent = currNode.data()
    nbiVoltage = voltNode.data()
    timeArr = currNode.dim_of().data()

    # Bin down to 10kHz
    timeArrNew, nbiCurrent = average_data(timeArr, nbiCurrent, 1e4)
    timeArrNew, nbiVoltage = average_data(timeArr, nbiVoltage, 1e4)

    timeArr = timeArrNew

    # Oonly keep the data between the min and max time
    mask = (timeArr >= minTime) & (timeArr <= maxTime)
    timeArr = timeArr[mask]
    nbiCurrent = nbiCurrent[mask]
    nbiVoltage = nbiVoltage[mask]

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(timeArr, nbiCurrent, label='Current [A]')
        ax.plot(timeArr, nbiVoltage/1e3, label='Voltage [kV]')

        ax.set_xlabel('Time (s)')
        
        ax.legend()

        plt.show()

    return timeArr, nbiCurrent, nbiVoltage

def see_signals(tree, makeplot=False):

    # In-vessel detector
    invNode = tree.getNode('diag.shinethru.det_17.raw_signal')
    # Beam dump detector
    dumpNode = tree.getNode('diag.shinethru.det_01.raw_signal')

    invSignal = invNode.data()
    invTimeArr = invNode.dim_of().data()

    dumpSignal = dumpNode.data()
    dumpTimeArr = dumpNode.dim_of().data()

    # Bin down to 10kHz
    invTimeArr, invSignal = average_data(invTimeArr, invSignal, 1e4)
    dumpTimeArr, dumpSignal = average_data(dumpTimeArr, dumpSignal, 1e4)

    # Put the data on the same time base by interpolating the dump signal to the in-vessel time base
    dumpSignal = np.interp(invTimeArr, dumpTimeArr, dumpSignal)

    timeArr = invTimeArr

    # Only keep the data between the min and max time
    mask = (timeArr >= minTime) & (timeArr <= maxTime)
    timeArr = timeArr[mask]
    invSignal = invSignal[mask]
    dumpSignal = dumpSignal[mask]

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(timeArr*1e3, invSignal* 1e3, label='In-vessel Detector')
        ax.plot(timeArr*1e3, dumpSignal* 1e3, label='Beam Dump Detector')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Signal [mA]')

        ax.legend()
        plt.show()

    return timeArr, invSignal, dumpSignal

def shot_data(shotnum):

    tree = mds.Tree('wham', shotnum)

    timeArrMag, currentTransrex, northCurrent, southCurrent = load_currents(tree)
    timeArrNBI, nbiCurrent, nbiVoltage = nbi_parameters(tree)
    timeArrSEE, invSignal, dumpSignal = see_signals(tree)

    # Put all the data on the same time base by interpolating the NBI and SEE data to the magnetic field time base
    nbiCurrent = np.interp(timeArrMag, timeArrNBI, nbiCurrent)
    nbiVoltage = np.interp(timeArrMag, timeArrNBI, nbiVoltage)
    invSignal = np.interp(timeArrMag, timeArrSEE, invSignal)
    dumpSignal = np.interp(timeArrMag, timeArrSEE, dumpSignal)

    timeArr = timeArrMag

    return timeArr, currentTransrex, northCurrent, southCurrent, nbiCurrent, nbiVoltage, invSignal, dumpSignal

def nominal_shot_params():
    """
    Characterize the shot-to-shot variation in the the SEE signals.
    """

    shotnumArr = 260430000 + np.array([13, 14, 15, 18, 19, 20, 21, 22, 23, 24])

    # Load the data for each shot
    timeArr = []
    currentTransrex2D = []
    northCurrent2D = []
    southCurrent2D = []
    nbiCurrent2D = []
    nbiVoltage2D = []
    invSignal2D = []
    dumpSignal2D = []

    for shotnum in shotnumArr:
        timeArr, currentTransrex, northCurrent, southCurrent, nbiCurrent, nbiVoltage, invSignal, dumpSignal = shot_data(shotnum)

        currentTransrex2D.append(currentTransrex)
        northCurrent2D.append(northCurrent)
        southCurrent2D.append(southCurrent)
        nbiCurrent2D.append(nbiCurrent)
        nbiVoltage2D.append(nbiVoltage)
        invSignal2D.append(invSignal)
        dumpSignal2D.append(dumpSignal)

    # Convert the lists to numpy arrays
    currentTransrex2D = np.array(currentTransrex2D)
    northCurrent2D = np.array(northCurrent2D)
    southCurrent2D = np.array(southCurrent2D)
    nbiCurrent2D = np.array(nbiCurrent2D)
    nbiVoltage2D = np.array(nbiVoltage2D)
    invSignal2D = np.array(invSignal2D)
    dumpSignal2D = np.array(dumpSignal2D)

    # Calculate the following across the shots:
    # - Mean
    # - Standard deviation
    # - % error for each shot compared to the mean
    currentTransrexMean = np.mean(currentTransrex2D, axis=0)
    currentTransrexStd = np.std(currentTransrex2D, axis=0)
    currentTransrexError = (currentTransrex2D - currentTransrexMean) / currentTransrexMean * 100

    northCurrentMean = np.mean(northCurrent2D, axis=0)
    northCurrentStd = np.std(northCurrent2D, axis=0)
    northCurrentError = (northCurrent2D - northCurrentMean) / northCurrentMean * 100

    southCurrentMean = np.mean(southCurrent2D, axis=0)
    southCurrentStd = np.std(southCurrent2D, axis=0)
    southCurrentError = (southCurrent2D - southCurrentMean) / southCurrentMean * 100

    nbiCurrentMean = np.mean(nbiCurrent2D, axis=0)
    nbiCurrentStd = np.std(nbiCurrent2D, axis=0)
    nbiCurrentError = (nbiCurrent2D - nbiCurrentMean) / nbiCurrentMean * 100

    nbiVoltageMean = np.mean(nbiVoltage2D, axis=0)
    nbiVoltageStd = np.std(nbiVoltage2D, axis=0)
    nbiVoltageError = (nbiVoltage2D - nbiVoltageMean) / nbiVoltageMean * 100

    invSignalMean = np.mean(invSignal2D, axis=0)
    invSignalStd = np.std(invSignal2D, axis=0)
    invSignalError = (invSignal2D - invSignalMean) / invSignalMean * 100

    dumpSignalMean = np.mean(dumpSignal2D, axis=0)
    dumpSignalStd = np.std(dumpSignal2D, axis=0)
    dumpSignalError = (dumpSignal2D - dumpSignalMean) / dumpSignalMean * 100

    #### Plot the mean and standard deviation of the signals
    
    # Transrex
    fig = plt.figure(figsize=(12, 8), tight_layout=True)

    fig.suptitle('Transrex Current')

    # Mean with standard deviation shaded
    ax1 = fig.add_subplot(121)
    # % error
    ax2 = fig.add_subplot(122)

    ax1.plot(timeArr*1e3, currentTransrexMean/1e3, 
             linewidth=3, label='Mean')
    ax1.fill_between(timeArr*1e3, (currentTransrexMean - currentTransrexStd)/1e3, (currentTransrexMean + currentTransrexStd)/1e3, 
                     alpha=0.5)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (kA)')

    ax2.plot(timeArr*1e3, currentTransrexError.T)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('% Error')

    # plt.show()

    # North PPS
    fig = plt.figure(figsize=(12, 8), tight_layout=True)

    fig.suptitle('North PPS Current')

    # Only plot when the PPS currents are above 5.5kA
    mask = northCurrentMean > 5.5e3
    timeArrPlotting = timeArr[mask]
    northCurrentMean = northCurrentMean[mask]
    northCurrentStd = northCurrentStd[mask]
    northCurrentError = northCurrentError[:, mask]

    # Mean with standard deviation shaded
    ax1 = fig.add_subplot(121)
    # % error
    ax2 = fig.add_subplot(122)

    ax1.plot(timeArrPlotting*1e3, northCurrentMean/1e3, 
             linewidth=3, label='Mean')
    ax1.fill_between(timeArrPlotting*1e3, (northCurrentMean - northCurrentStd)/1e3, (northCurrentMean + northCurrentStd)/1e3, 
                     alpha=0.5)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (kA)')

    ax2.plot(timeArrPlotting*1e3, northCurrentError.T)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('% Error')

    # plt.show()

    # South PPS
    fig = plt.figure(figsize=(12, 8), tight_layout=True)

    fig.suptitle('South PPS Current')

    # Only plot when the PPS currents are above 5.5kA
    mask = southCurrentMean > 5.5e3
    timeArrPlotting = timeArr[mask]
    southCurrentMean = southCurrentMean[mask]
    southCurrentStd = southCurrentStd[mask]
    southCurrentError = southCurrentError[:, mask]

    # Mean with standard deviation shaded
    ax1 = fig.add_subplot(121)
    # % error
    ax2 = fig.add_subplot(122)

    ax1.plot(timeArrPlotting*1e3, southCurrentMean/1e3, 
             linewidth=3, label='Mean')
    ax1.fill_between(timeArrPlotting*1e3, (southCurrentMean - southCurrentStd)/1e3, (southCurrentMean + southCurrentStd)/1e3, 
                     alpha=0.5)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (kA)')

    ax2.plot(timeArrPlotting*1e3, southCurrentError.T)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('% Error')

    # plt.show()

    # NBI Current
    fig = plt.figure(figsize=(12, 8), tight_layout=True)

    fig.suptitle('NBI Current')

    # Only plot when the NBI current is above 45A
    mask = nbiCurrentMean > 45
    timeArrPlotting = timeArr[mask]
    nbiCurrentMean = nbiCurrentMean[mask]
    nbiCurrentStd = nbiCurrentStd[mask]
    nbiCurrentError = nbiCurrentError[:, mask]

    # Mean with standard deviation shaded
    ax1 = fig.add_subplot(121)
    # % error
    ax2 = fig.add_subplot(122)

    ax1.plot(timeArrPlotting*1e3, nbiCurrentMean, 
             linewidth=3, label='Mean')
    ax1.fill_between(timeArrPlotting*1e3, (nbiCurrentMean - nbiCurrentStd), (nbiCurrentMean + nbiCurrentStd), 
                     alpha=0.5)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (A)')

    ax2.plot(timeArrPlotting*1e3, nbiCurrentError.T)

    ax2.set_ylim(-5, 5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('% Error')

    # plt.show()

    # In-vessel detector
    fig = plt.figure(figsize=(12, 8), tight_layout=True)

    fig.suptitle('In-vessel SEE detector')

    # Only plot when the signal is above 1mA
    mask = invSignalMean > 1e-3
    timeArrPlotting = timeArr[mask]
    invSignalMean = invSignalMean[mask]
    invSignalStd = invSignalStd[mask]
    invSignalError = invSignalError[:, mask]

    # Mean with standard deviation shaded
    ax1 = fig.add_subplot(121)
    # % error
    ax2 = fig.add_subplot(122)

    ax1.plot(timeArrPlotting*1e3, invSignalMean*1e3, 
             linewidth=3, label='Mean')
    ax1.fill_between(timeArrPlotting*1e3, (invSignalMean - invSignalStd)*1e3, (invSignalMean + invSignalStd)*1e3, 
                     alpha=0.5)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (mA)')

    ax2.plot(timeArrPlotting*1e3, invSignalError.T)

    ax2.set_ylim(-5, 5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('% Error')

    # plt.show()

    # Beam dump detector
    fig = plt.figure(figsize=(12, 8), tight_layout=True)

    fig.suptitle('Beam dump detector')

    # Only plot when the signal is above 1mA
    mask = dumpSignalMean > 1e-3
    timeArrPlotting = timeArr[mask]
    dumpSignalMean = dumpSignalMean[mask]
    dumpSignalStd = dumpSignalStd[mask]
    dumpSignalError = dumpSignalError[:, mask]

    # Mean with standard deviation shaded
    ax1 = fig.add_subplot(121)
    # % error
    ax2 = fig.add_subplot(122)

    ax1.plot(timeArrPlotting*1e3, dumpSignalMean*1e3, 
             linewidth=3, label='Mean')
    ax1.fill_between(timeArrPlotting*1e3, (dumpSignalMean - dumpSignalStd)*1e3, (dumpSignalMean + dumpSignalStd)*1e3, 
                     alpha=0.5)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (mA)')

    ax2.plot(timeArrPlotting*1e3, dumpSignalError.T)

    ax2.set_ylim(-5, 5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('% Error')

    plt.show()

    return

def in_vessel_average(shotnumArr):

    # Flatten (this makes the code agnostic to the shape of shotnumArr)
    flatShotnumArr = shotnumArr.flatten()

    # Arrays to store the average
    invDetAvg = np.zeros_like(flatShotnumArr, dtype=float)
    dumpDetAvg = np.zeros_like(flatShotnumArr, dtype=float)

    # Go over each shot number and get the see signals
    for i in range(len(flatShotnumArr)):

        print(f'{i+1} of {len(flatShotnumArr)}')

        shotnum = flatShotnumArr[i]
        print(shotnum)

        tree = mds.Tree('wham', shotnum)
        timeArr, invSignal, dumpSignal = see_signals(tree)
        tree.close()

        # Convert to ms and mA
        timeArr *= 1e3
        invSignal *= 1e3
        dumpSignal *= 1e3

        # Only look at the data between t=3ms and t=12.5ms (NBI flattop)
        mask = (timeArr >= 3) & (timeArr <= 12.5)

        invSignal = invSignal[mask]
        dumpSignal = dumpSignal[mask]

        # Get the average and add it to the arrays 
        invDetAvg[i] = np.average(invSignal)
        dumpDetAvg[i] = np.average(dumpSignal)

    # Reshape the data back to shotnumArr
    invDetAvg = invDetAvg.reshape(shotnumArr.shape)    
    dumpDetAvg = dumpDetAvg.reshape(shotnumArr.shape)

    return invDetAvg, dumpDetAvg

def magnet_variation():

    # Check if the saved data exists. If not, run the code and make the plot.
    try:
        dataObj = np.load('/home/sanwalka/shinethru/magnet_effect_data.npz')

        invPercentDiff = dataObj['invPercentDiff']
        dumpPercentDiff = dataObj['dumpPercentDiff']

        transrexDemand = dataObj['transrexDemand']
        ppsDemand = dataObj['ppsDemand']

        # In vessel detectors
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        fig.suptitle('In-vessel detector')

        pltObj = ax.imshow(invPercentDiff,
                            origin='lower',
                            interpolation='none')
        
        cbar = fig.colorbar(pltObj, label='% difference')
        
        ax.set_xticks(np.arange(len(transrexDemand)))
        ax.set_yticks(np.arange(len(ppsDemand)))

        ax.set_xticklabels(transrexDemand)
        ax.set_yticklabels(ppsDemand)

        ax.set_xlabel('Transrex Demand')
        ax.set_ylabel('PPS Demand')

        plt.show()

        # Beam dump detectors
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        fig.suptitle('Beam dump detector')

        pltObj = ax.imshow(dumpPercentDiff,
                            origin='lower',
                            interpolation='none')
        
        cbar = fig.colorbar(pltObj, label='% difference')

        ax.set_xticks(np.arange(len(transrexDemand)))
        ax.set_yticks(np.arange(len(ppsDemand)))

        ax.set_xticklabels(transrexDemand)
        ax.set_yticklabels(ppsDemand)

        ax.set_xlabel('Transrex Demand')
        ax.set_ylabel('PPS Demand')

        plt.show()

        return

    except Exception as e:
        print(e)
        print('No data saved, running analysis from scratch.')

    # Transrex Demand
    transrexDemand = np.array([4.05, 4.04, 4.03, 4.02, 4.01, 4.00, 3.95])
    # PPS Demand
    ppsDemand = np.array([6000, 5990, 5980, 5970, 5960, 5950, 5900])

    # Shots that correspond to the demand values
    # 1st index = Transrex demand
    # 2nd index = PPS demand
    # 3rd index = shot number
    shotnumArr = np.array([[[ 23, 135, 101,  79, 131,  45, 109],
                            [107, 125, 129,  53, 113, 119,  63],
                            [ 61,  94, 115,  65, 117,  49,  83],
                            [ 90,  81,  55,  35,  99,  86,  43],
                            [133,  41,  88, 139,  39,  97,  72],
                            [137, 103, 127, 123,  57, 121,  75],
                            [105,  59, 111,  37,  51,  47,  77]],
                           [[ 24, 136, 102,  80, 132,  46, 110],
                            [108, 126, 130,  54, 114, 120,  64],
                            [ 62,  95, 116,  66, 118,  50,  84],
                            [ 91,  82,  56,  36, 100,  87,  44],
                            [134,  42,  89, 140,  40,  98,  73],
                            [138, 104, 128, 124,  58, 122,  76],
                            [106,  60, 112,  38,  52,  48,  78]]])

    # Add the date
    shotnumArr += 260430000

    # Get the average current on the in-vessel and beam dump SEE detectors for each shot
    invDetAvg, dumpDetAvg = in_vessel_average(shotnumArr)

    # Average over the repeated shots
    invDetAvg = np.average(invDetAvg, axis=0)
    dumpDetAvg = np.average(dumpDetAvg, axis=0)

    # Subtract the average of the nominal shot (Transrex=4.05, PPS=6000)
    nominalInv = invDetAvg[0, 0]
    nominalDump = dumpDetAvg[0, 0]

    invDiff = invDetAvg - nominalInv
    dumpDiff = dumpDetAvg - nominalDump

    # Calculate the % difference from the nominal shot
    invPercentDiff = 100 * (np.abs(invDiff)/nominalInv)
    dumpPercentDiff = 100 * (np.abs(dumpDiff)/nominalDump)

    #### Save the data
    np.savez('/home/sanwalka/shinethru/magnet_effect_data.npz',
             
             transrexDemand=transrexDemand,
             ppsDemand=ppsDemand,

             invPercentDiff=invPercentDiff,
             dumpPercentDiff=dumpPercentDiff,
             
             invDetAvg=invDetAvg,
             dumpDetAvg=dumpDetAvg)

    #### Plot the data

    # In vessel detectors
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.imshow(invPercentDiff,
              origin='lower',
              interpolation='none')
    
    ax.set_xticks(np.arange(len(transrexDemand)))
    ax.set_yticks(np.arange(len(ppsDemand)))

    ax.set_xticklabels(transrexDemand)
    ax.set_yticklabels(ppsDemand)

    plt.show()

    # Beam dump detectors
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.imshow(dumpPercentDiff,
              origin='lower',
              interpolation='none')
    
    ax.set_xticks(np.arange(len(transrexDemand)))
    ax.set_yticks(np.arange(len(ppsDemand)))

    ax.set_xticklabels(transrexDemand)
    ax.set_yticklabels(ppsDemand)

    plt.show()

    return

if __name__ == "__main__":

    shotnum = 260430112
    tree = mds.Tree('wham', shotnum)

    # PPS currents
    # timeArr, northCurrent, southCurrent = pps_current(tree, makeplot=True)

    # Load the currents from the shot
    # timeArr, currentTransrex, northCurrent, southCurrent = load_currents(tree)

    # Load the NBI parameters
    # timeArr, nbiCurrent, nbiVoltage = nbi_parameters(tree, makeplot=True)

    # Load the SEE signals
    # timeArr, invSignal, dumpSignal = see_signals(tree, makeplot=True)

    # Shot-to-shot variation in the signals
    # nominal_shot_params()

    # Effect of magnet variation
    magnet_variation()