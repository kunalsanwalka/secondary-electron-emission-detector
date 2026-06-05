import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

def edge_detector():

    shotnum = 260409006
    # Connect to MDSplus server and load data
    tree = mds.Tree('wham', shotnum)

    nodeName = 'raw.acq1001_633.ch_04'
    plasmaCurrentNode = tree.getNode(nodeName)

    # Current for each detector
    rawData = plasmaCurrentNode.getData().data() / 100

    savGol1 = sc.signal.savgol_filter(rawData, 51, 2)
    savGol2 = sc.signal.savgol_filter(rawData, 101, 2)
    savGol3 = sc.signal.savgol_filter(rawData, 201, 2)

    # Average before plasma
    plasmaStart = 0
    plasmaStop = 5e-3

    # Time array
    tempTime = plasmaCurrentNode.dim_of().data()
    # Fix the time array
    startTime = tempTime[0]
    tempTime -= startTime
    tempTime *= 1e3
    tempTime += startTime
    tempTime /= 1e3
    timeArr = tempTime

    # Average before plasma
    prePlasmaMask = (timeArr < plasmaStart)
    prePlasmaAvgRaw = np.mean(rawData[prePlasmaMask])
    prePlasmaAvg51 = np.mean(savGol1[prePlasmaMask])
    prePlasmaAvg101 = np.mean(savGol2[prePlasmaMask])
    prePlasmaAvg201 = np.mean(savGol3[prePlasmaMask])

    # Average during plasma
    plasmaMask = (timeArr >= plasmaStart) & (timeArr <= plasmaStop)
    plasmaAvgRaw = np.mean(rawData[plasmaMask])
    plasmaAvg51 = np.mean(savGol1[plasmaMask])
    plasmaAvg101 = np.mean(savGol2[plasmaMask])
    plasmaAvg201 = np.mean(savGol3[plasmaMask])

    # Average after plasma
    postPlasmaMask = (timeArr > plasmaStop)
    postPlasmaAvgRaw = np.mean(rawData[postPlasmaMask])
    postPlasmaAvg51 = np.mean(savGol1[postPlasmaMask])
    postPlasmaAvg101 = np.mean(savGol2[postPlasmaMask])
    postPlasmaAvg201 = np.mean(savGol3[postPlasmaMask])

    print(f"Pre-plasma average (raw): {prePlasmaAvgRaw*1e3} mA")
    print(f"Pre-plasma average (51-point SavGol): {prePlasmaAvg51*1e3} mA")
    print(f"Pre-plasma average (101-point SavGol): {prePlasmaAvg101*1e3} mA")
    print(f"Pre-plasma average (201-point SavGol): {prePlasmaAvg201*1e3} mA")
    print("--------------------------------------------------")

    print(f"Plasma average (raw): {plasmaAvgRaw*1e3} mA")
    print(f"Plasma average (51-point SavGol): {plasmaAvg51*1e3} mA")
    print(f"Plasma average (101-point SavGol): {plasmaAvg101*1e3} mA")
    print(f"Plasma average (201-point SavGol): {plasmaAvg201*1e3} mA")
    print("--------------------------------------------------")

    print(f"Post-plasma average (raw): {postPlasmaAvgRaw*1e3} mA")
    print(f"Post-plasma average (51-point SavGol): {postPlasmaAvg51*1e3} mA")
    print(f"Post-plasma average (101-point SavGol): {postPlasmaAvg101*1e3} mA")
    print(f"Post-plasma average (201-point SavGol): {postPlasmaAvg201*1e3} mA")

    if True:

        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        ax1 = fig.add_subplot(111)

        ax1.plot(timeArr*1e3, rawData*1e3, label='Raw Data')
        ax1.plot(timeArr*1e3, savGol1*1e3, label='51-point SavGol')
        ax1.plot(timeArr*1e3, savGol2*1e3, label='101-point SavGol')
        ax1.plot(timeArr*1e3, savGol3*1e3, label='201-point SavGol')

        ax1.set_title(f'Current vs Time for Shot {shotnum}')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (mA)')
        ax1.legend()

        plt.show()

    return timeArr, rawData

def dump_detector():

    shotnum = 260409006
    # Connect to MDSplus server and load data
    tree = mds.Tree('wham', shotnum)

    nodeName = 'raw.acq1001_635.ch_05'
    plasmaCurrentNode = tree.getNode(nodeName)

    # Current for each detector
    rawData = plasmaCurrentNode.getData().data() / 1e3

    savGol1 = sc.signal.savgol_filter(rawData, 51, 2)
    savGol2 = sc.signal.savgol_filter(rawData, 101, 2)
    savGol3 = sc.signal.savgol_filter(rawData, 201, 2)

    # Average before plasma
    plasmaStart = 0
    plasmaStop = 5e-3

    # Delay for the digitizer
    delayInSeconds = -tree.getNode('raw.acq1001_632.trig_time').getData().data()
    # Get the time array for the raw signal
    timeArr = -delayInSeconds + np.arange(len(rawData)) / 1e6


    # Average before plasma
    prePlasmaMask = (timeArr < plasmaStart)
    prePlasmaAvgRaw = np.mean(rawData[prePlasmaMask])
    prePlasmaAvg51 = np.mean(savGol1[prePlasmaMask])
    prePlasmaAvg101 = np.mean(savGol2[prePlasmaMask])
    prePlasmaAvg201 = np.mean(savGol3[prePlasmaMask])

    # Average during plasma
    plasmaMask = (timeArr >= plasmaStart) & (timeArr <= plasmaStop)
    plasmaAvgRaw = np.mean(rawData[plasmaMask])
    plasmaAvg51 = np.mean(savGol1[plasmaMask])
    plasmaAvg101 = np.mean(savGol2[plasmaMask])
    plasmaAvg201 = np.mean(savGol3[plasmaMask])

    # Average after plasma
    postPlasmaMask = (timeArr > plasmaStop)
    postPlasmaAvgRaw = np.mean(rawData[postPlasmaMask])
    postPlasmaAvg51 = np.mean(savGol1[postPlasmaMask])
    postPlasmaAvg101 = np.mean(savGol2[postPlasmaMask])
    postPlasmaAvg201 = np.mean(savGol3[postPlasmaMask])

    print(f"Pre-plasma average (raw): {prePlasmaAvgRaw*1e3} mA")
    print(f"Pre-plasma average (51-point SavGol): {prePlasmaAvg51*1e3} mA")
    print(f"Pre-plasma average (101-point SavGol): {prePlasmaAvg101*1e3} mA")
    print(f"Pre-plasma average (201-point SavGol): {prePlasmaAvg201*1e3} mA")
    print("--------------------------------------------------")

    print(f"Plasma average (raw): {plasmaAvgRaw*1e3} mA")
    print(f"Plasma average (51-point SavGol): {plasmaAvg51*1e3} mA")
    print(f"Plasma average (101-point SavGol): {plasmaAvg101*1e3} mA")
    print(f"Plasma average (201-point SavGol): {plasmaAvg201*1e3} mA")
    print("--------------------------------------------------")

    print(f"Post-plasma average (raw): {postPlasmaAvgRaw*1e3} mA")
    print(f"Post-plasma average (51-point SavGol): {postPlasmaAvg51*1e3} mA")
    print(f"Post-plasma average (101-point SavGol): {postPlasmaAvg101*1e3} mA")
    print(f"Post-plasma average (201-point SavGol): {postPlasmaAvg201*1e3} mA")

    if True:

        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        ax1 = fig.add_subplot(111)

        ax1.plot(timeArr*1e3, rawData*1e3, label='Raw Data')
        ax1.plot(timeArr*1e3, savGol1*1e3, label='51-point SavGol')
        ax1.plot(timeArr*1e3, savGol2*1e3, label='101-point SavGol')
        ax1.plot(timeArr*1e3, savGol3*1e3, label='201-point SavGol')

        ax1.set_title(f'Current vs Time for Shot {shotnum}')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (mA)')
        ax1.legend()

        plt.show()

    return timeArr, rawData

if __name__ == "__main__":

    timeArr, rawData = dump_detector()