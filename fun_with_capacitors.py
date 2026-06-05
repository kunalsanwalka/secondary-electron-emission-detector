import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Change the backend to TkAgg for interactive plotting
plt.switch_backend('TkAgg')

global tree, refTree, capacitance, voltage
capacitance = 10e-6  # Farads
voltage = 150  # Volts

def bin_down_data(dataArr, decimation):

    # Trim the data to be a multiple of the decimation factor
    nPoints = len(dataArr)
    nTrim = (nPoints // decimation) * decimation

    dataArr = dataArr[:nTrim]

    # Reshape the data for binning
    dataArr = dataArr.reshape(-1, decimation)

    # Average the data within each bin
    newDataArr = dataArr.mean(axis=1)

    return newDataArr

def load_data_from_mdsplus(shotnum, makeplot=False):

    # Connect to MDSplus server and load data
    tree = mds.Tree('wham', shotnum)

    ref_shotnum = tree.getNode('diag.shinethru.ref_shotnum').data()
    refTree = mds.Tree('wham', ref_shotnum)

    nodeName = 'raw.acq1001_633.ch_04'
    plasmaCurrentNode = tree.getNode(nodeName)
    refCurrentNode = refTree.getNode(nodeName)

    # Current for each detector
    plasmaCurrentArr = plasmaCurrentNode.getData().data() / 100
    refCurrentArr = refCurrentNode.getData().data() / 100

    # Time array
    tempTime = plasmaCurrentNode.dim_of().data()
    # Fix the time array
    startTime = tempTime[0]
    tempTime -= startTime
    tempTime *= 1e3
    tempTime += startTime
    tempTime /= 1e3
    timeArr = tempTime

    # Bin down the data to reduce noise
    decimation = 80 * 100 # bin down to 10kHz
    newPlasmaCurrentArr = bin_down_data(plasmaCurrentArr, decimation)
    newRefCurrentArr = bin_down_data(refCurrentArr, decimation)
    newTimeArr = bin_down_data(timeArr, decimation)

    if makeplot:

        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        ax1 = fig.add_subplot(111)

        ax1.plot(newTimeArr*1e3, newRefCurrentArr*1e3, label='Reference Shot', linewidth=3, color='blue')
        ax1.plot(newTimeArr*1e3, newPlasmaCurrentArr*1e3, label='Plasma Shot', linewidth=3, color='red')

        ax1.plot(timeArr*1e3, refCurrentArr*1e3, alpha=0.3, color='blue')
        ax1.plot(timeArr*1e3, plasmaCurrentArr*1e3, alpha=0.3, color='red')

        ax1.set_title(f'{shotnum}')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (mA)')
        ax1.legend()

        plt.show()

    return newTimeArr, newRefCurrentArr, newPlasmaCurrentArr

def voltage_vs_attenuation(appliedVoltage):

    # Calculate the attenuation factor based on the applied voltage
    attenuationFactor = appliedVoltage / voltage

    return attenuationFactor

def run_synthetic_test():

    # Create synthetic data
    timeArr = np.linspace(0, 50, 1000) * 1e-3

    # Reference shot (IDEAL)
    idealRefCurrentArr = np.zeros_like(timeArr)
    idealRefCurrentArr[timeArr >= 12.5e-3] = 5e-3 # Amps
    idealRefCurrentArr[timeArr > 37.5e-3] = 0

    # Reference shot (REAL)
    chargeArr = np.cumsum(idealRefCurrentArr) * (timeArr[1] - timeArr[0])  # Q = I * dt
    voltageDropArr = chargeArr / capacitance
    appliedVoltageArr = voltage - voltageDropArr
    attnFactor = voltage_vs_attenuation(appliedVoltageArr)
    realRefCurrentArr = idealRefCurrentArr * attnFactor

    # Plasma shot (IDEAL)
    idealPlasmaCurrentArr = np.zeros_like(timeArr)
    idealPlasmaCurrentArr[timeArr >= 12.5e-3] = 2.5e-3 # Amps
    idealPlasmaCurrentArr[timeArr > 37.5e-3] = 0

    # Plasma shot (REAL)
    chargeArr = np.cumsum(idealPlasmaCurrentArr) * (timeArr[1] - timeArr[0])  # Q = I * dt
    voltageDropArr = chargeArr / capacitance
    appliedVoltageArr = voltage - voltageDropArr
    attnFactor = voltage_vs_attenuation(appliedVoltageArr)
    realPlasmaCurrentArr = idealPlasmaCurrentArr * attnFactor

    # 'Density' (IDEAL)
    idealDensityArr = np.log(idealRefCurrentArr/idealPlasmaCurrentArr)
    realDensityArr = np.log(realRefCurrentArr/realPlasmaCurrentArr)
    idealDensityArr[np.isnan(idealDensityArr)] = 0
    realDensityArr[np.isnan(realDensityArr)] = 0

    # % error between ideal and real density
    errorArr = 100 * (np.abs(idealDensityArr - realDensityArr) / np.abs(idealDensityArr))
    errorArr[np.isnan(errorArr)] = 0

    # Plot the data
    fig = plt.figure(figsize=(10, 8), tight_layout=True)
    
    # Reference shot
    ax1 = fig.add_subplot(221)
    # Plasma shot
    ax2 = fig.add_subplot(222)
    # 'Density'
    ax3 = fig.add_subplot(223)
    # Error
    ax4 = fig.add_subplot(224)
    
    ax1.plot(timeArr*1e3, idealRefCurrentArr*1e3, label='Ideal')
    ax1.plot(timeArr*1e3, realRefCurrentArr*1e3, label='Real')

    ax2.plot(timeArr*1e3, idealPlasmaCurrentArr*1e3, label='Ideal')
    ax2.plot(timeArr*1e3, realPlasmaCurrentArr*1e3, label='Real')

    ax3.plot(timeArr*1e3, idealDensityArr, label='Ideal')
    ax3.plot(timeArr*1e3, realDensityArr, label='Real')

    ax4.plot(timeArr*1e3, errorArr, label='Error', color='green')

    ax1.set_title('Reference Shot')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (mA)')
    ax1.legend()

    ax2.set_title('Plasma Shot')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (mA)')

    ax3.set_title('ln(Reference/Plasma)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Density (a.u.)')

    ax4.set_title('Error (%)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Error (%)')

    plt.show()

    return

def calculate_density(timeArr, refCurrentArr, plasmaCurrentArr):

    # Calculate the 'density' as ln(Reference/Plasma)
    densityArr = np.log(refCurrentArr/plasmaCurrentArr)
    densityArr[np.isnan(densityArr)] = 0

    # Calculate the total charge vs. time for the plasma and reference shots
    refChargeArr = np.cumsum(refCurrentArr) * (timeArr[1] - timeArr[0])  # Q = I * dt
    plasmaChargeArr = np.cumsum(plasmaCurrentArr) * (timeArr[1] - timeArr[0])  # Q = I * dt

    # Drop in voltage of the capacitor
    refVoltageDropArr = refChargeArr / capacitance
    plasmaVoltageDropArr = plasmaChargeArr / capacitance

    # Effective voltage applied to the detector in the plasma and reference shots
    refVoltageArr = voltage - refVoltageDropArr
    plasmaVoltageArr = voltage - plasmaVoltageDropArr

    # Attenuation factor for the plasma and reference shots
    refAttnFactorArr = voltage_vs_attenuation(refVoltageArr)
    plasmaAttnFactorArr = voltage_vs_attenuation(plasmaVoltageArr)

    # Corrected currents for the plasma and reference shots
    correctedRefCurrentArr = refCurrentArr / refAttnFactorArr
    correctedPlasmaCurrentArr = plasmaCurrentArr / plasmaAttnFactorArr

    # Recalculate the 'density' using the corrected currents
    correctedDensityArr = np.log(correctedRefCurrentArr/correctedPlasmaCurrentArr)
    correctedDensityArr[np.isnan(correctedDensityArr)] = 0

    # % error between original and corrected density
    errorArr = 100 * (np.abs(densityArr - correctedDensityArr) / np.abs(densityArr))
    errorArr[np.isnan(errorArr)] = 0

    fig = plt.figure(figsize=(10, 10), tight_layout=True)

    xMin, xMax = 6.5, 12.5

    # Plot the original and corrected density vs. time
    ax1 = fig.add_subplot(211)
    # Error vs. time
    ax2 = fig.add_subplot(212)

    ax1.plot(timeArr*1e3, densityArr, label='No droop correction')
    ax1.plot(timeArr*1e3, correctedDensityArr, label='With droop correction')

    ax1.set_ylim(0, 1)
    ax1.set_xlim(xMin, xMax)
    ax1.set_title('Density vs Time')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Density (a.u.)')
    ax1.legend()

    ax2.plot(timeArr*1e3, errorArr, label='Error', color='green')

    ax2.set_ylim(0, 2)
    ax2.set_xlim(xMin, xMax)
    ax2.set_title('Error vs Time')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Error (%)')

    plt.show()

    return densityArr

if __name__ == "__main__":

    # Run synthetic test
    # run_synthetic_test()

    # Load data from MDSplus and plot
    timeArr, refCurrentArr, plasmaCurrentArr = load_data_from_mdsplus(260302079, makeplot=True)

    # Calculate density and plot
    densityArr = calculate_density(timeArr, refCurrentArr, plasmaCurrentArr)