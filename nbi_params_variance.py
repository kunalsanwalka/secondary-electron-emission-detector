import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Change the backend to TkAgg for interactive plotting
plt.switch_backend('TkAgg')

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

    node = 'nbi.i_beam'
    nbiCurrentNode = tree.getNode(node)
    nbiCurrent = nbiCurrentNode.getData().data()
    refnbiCurrent = refTree.getNode(node).getData().data()
    timeArr = nbiCurrentNode.dim_of().data()

    node = 'nbi.v_beam'
    nbiVoltageNode = tree.getNode(node)
    nbiVoltage = nbiVoltageNode.getData().data()
    refnbiVoltage = refTree.getNode(node).getData().data()

    # Optionally make a plot of the data
    if makeplot:

        fig = plt.figure(figsize=(12, 12), tight_layout=True)
        
        # NBI voltage
        ax1 = fig.add_subplot(221)
        # NBI current
        ax2 = fig.add_subplot(222)
        # NBI voltage difference
        ax3 = fig.add_subplot(223)
        # NBI current difference
        ax4 = fig.add_subplot(224)

        fig.suptitle(f'Shot {shotnum} vs Reference Shot {ref_shotnum}')

        # Only plot when the current is above 1A and the voltage is above 1kV
        voltageMask = nbiVoltage > 1e3
        currentMask = nbiCurrent > 1

        ax1.plot(timeArr[voltageMask]*1e3, nbiVoltage[voltageMask]/1e3, label='Plasma')
        ax1.plot(timeArr[voltageMask]*1e3, refnbiVoltage[voltageMask]/1e3, label='Reference')
        ax1.set_title('NBI Voltage')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (kV)')

        ax2.plot(timeArr[currentMask]*1e3, nbiCurrent[currentMask])
        ax2.plot(timeArr[currentMask]*1e3, refnbiCurrent[currentMask])
        ax2.set_title('NBI Current')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (A)')

        # % difference in voltage
        voltageDiff = np.abs((nbiVoltage - refnbiVoltage) / refnbiVoltage) * 100
        ax3.plot(timeArr[voltageMask]*1e3, voltageDiff[voltageMask])
        ax3.set_title('NBI Voltage % Difference')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Voltage Difference (%)')
        ax3.set_ylim(0, 10)

        # % difference in current
        currentDiff = np.abs((nbiCurrent - refnbiCurrent) / refnbiCurrent) * 100
        ax4.plot(timeArr[currentMask]*1e3, currentDiff[currentMask])
        ax4.set_title('NBI Current % Difference')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Current Difference (%)')
        ax4.set_ylim(0, 2)

        ax1.legend()

        plt.show()

    # Bin down the data to 10kHz
    nbiCurrent = bin_down_data(nbiCurrent, 100)
    nbiVoltage = bin_down_data(nbiVoltage, 100)
    refnbiCurrent = bin_down_data(refnbiCurrent, 100)
    refnbiVoltage = bin_down_data(refnbiVoltage, 100)
    timeArr = bin_down_data(timeArr, 100)

    if makeplot:

        fig = plt.figure(figsize=(12, 12), tight_layout=True)
        
        # NBI voltage
        ax1 = fig.add_subplot(221)
        # NBI current
        ax2 = fig.add_subplot(222)
        # NBI voltage difference
        ax3 = fig.add_subplot(223)
        # NBI current difference
        ax4 = fig.add_subplot(224)

        fig.suptitle(f'Shot {shotnum} vs Reference Shot {ref_shotnum}')

        # Only plot when the current is above 1A and the voltage is above 1kV
        voltageMask = nbiVoltage > 1e3
        currentMask = nbiCurrent > 1

        ax1.plot(timeArr[voltageMask]*1e3, nbiVoltage[voltageMask]/1e3, label='Plasma')
        ax1.plot(timeArr[voltageMask]*1e3, refnbiVoltage[voltageMask]/1e3, label='Reference')
        ax1.set_title('NBI Voltage')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (kV)')

        ax2.plot(timeArr[currentMask]*1e3, nbiCurrent[currentMask])
        ax2.plot(timeArr[currentMask]*1e3, refnbiCurrent[currentMask])
        ax2.set_title('NBI Current')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (A)')

        # % difference in voltage
        voltageDiff = np.abs((nbiVoltage - refnbiVoltage) / refnbiVoltage) * 100
        ax3.plot(timeArr[voltageMask]*1e3, voltageDiff[voltageMask])
        ax3.set_title('NBI Voltage % Difference')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Voltage Difference (%)')
        ax3.set_ylim(0, 10)

        # % difference in current
        currentDiff = np.abs((nbiCurrent - refnbiCurrent) / refnbiCurrent) * 100
        ax4.plot(timeArr[currentMask]*1e3, currentDiff[currentMask])
        ax4.set_title('NBI Current % Difference')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Current Difference (%)')
        ax4.set_ylim(0, 2)

        ax1.legend()

        plt.show()

    tree.close()
    refTree.close()

    return timeArr, nbiCurrent, refnbiCurrent, nbiVoltage, refnbiVoltage

def multiple_shots(shotnums):

    timeArr = []
    nbiCurrents = []
    refnbiCurrents = []
    nbiVoltages = []
    refnbiVoltages = []

    for shotnum in shotnums:

        print(f"Loading data for shot {shotnum}...")

        try:
            timeArr, nbiCurrentShot, refnbiCurrentShot, nbiVoltageShot, refnbiVoltageShot = load_data_from_mdsplus(shotnum, makeplot=False)
            nbiCurrents.append(nbiCurrentShot)
            refnbiCurrents.append(refnbiCurrentShot)
            nbiVoltages.append(nbiVoltageShot)
            refnbiVoltages.append(refnbiVoltageShot)
        except Exception as e:
            print(f"Error loading data for shot {shotnum}: {e}")

    # Convert lists to numpy arrays
    nbiCurrents = np.array(nbiCurrents)
    refnbiCurrents = np.array(refnbiCurrents)
    nbiVoltages = np.array(nbiVoltages)
    refnbiVoltages = np.array(refnbiVoltages)

    numShots = nbiCurrents.shape[0]

    # Make a plot of the % difference in voltage and current for each shot
    fig = plt.figure(figsize=(12, 12), tight_layout=True)

    # NBI voltage
    ax1 = fig.add_subplot(221)
    # NBI current
    ax2 = fig.add_subplot(222)
    # NBI voltage difference
    ax3 = fig.add_subplot(223)
    # NBI current difference
    ax4 = fig.add_subplot(224)

    for i in range(numShots):

        # Only plot when the current is above 1A and the voltage is above 1kV
        voltageMask = nbiVoltages[i] > 1e3
        currentMask = nbiCurrents[i] > 1

        ax1.plot(timeArr[voltageMask]*1e3, nbiVoltages[i][voltageMask]/1e3, label='Plasma')
        ax1.plot(timeArr[voltageMask]*1e3, refnbiVoltages[i][voltageMask]/1e3, label='Reference')
        ax1.set_title('NBI Voltage')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (kV)')

        ax2.plot(timeArr[currentMask]*1e3, nbiCurrents[i][currentMask])
        ax2.plot(timeArr[currentMask]*1e3, refnbiCurrents[i][currentMask])
        ax2.set_title('NBI Current')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (A)')

        # % difference in voltage
        voltageDiff = np.abs((nbiVoltages[i] - refnbiVoltages[i]) / refnbiVoltages[i]) * 100
        ax3.plot(timeArr[voltageMask]*1e3, voltageDiff[voltageMask])
        ax3.set_title('NBI Voltage % Difference')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Voltage Difference (%)')
        ax3.set_ylim(0, 10)

        # % difference in current
        currentDiff = np.abs((nbiCurrents[i] - refnbiCurrents[i]) / refnbiCurrents[i]) * 100
        ax4.plot(timeArr[currentMask]*1e3, currentDiff[currentMask])
        ax4.set_title('NBI Current % Difference')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Current Difference (%)')
        ax4.set_ylim(0, 10)

    plt.show()

    # Print the shot number with the biggest difference in NBI current between the plasma and reference shot
    timeOfDiff = 7 * 1e-3
    timeIdx = np.abs(timeArr - timeOfDiff).argmin()
    
    # % difference in the NBI currents between the plasma and reference shot
    currentDiff = np.abs((nbiCurrents - refnbiCurrents) / refnbiCurrents) * 100

    # Shot number with the maximum difference
    diffsAtTime = currentDiff[:, timeIdx]
    maxDiffIdx = diffsAtTime.argmax()
    print(shotnums[maxDiffIdx])

    return

if __name__ == "__main__":

    # Run synthetic test
    # run_synthetic_test()

    SHOTDAY = 260309000
    SHOT_RANGE = range(22, 51)
    # SHOT_RANGE = range(22, 28)
    shotnums = [SHOTDAY + i for i in SHOT_RANGE]

    # Load data from MDSplus and plot
    # timeArr, nbiCurrent, refnbiCurrent, nbiVoltage, refnbiVoltage = load_data_from_mdsplus(260309022, makeplot=True)

    # Compare multiple shots
    multiple_shots(shotnums)