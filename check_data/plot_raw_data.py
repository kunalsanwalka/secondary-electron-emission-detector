import argparse
import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt

# Make the font size larger
plt.rcParams.update({'font.size': 18})

def parseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('-shotnum',
                        type=int)

    args = parser.parse_args()
    
    return args

def plot_data(shotnum):

    detCount = 21

    timeArr2D = []
    sigArr2D = []
    impactParams = []

    tree = mds.Tree('wham', shotnum)

    for i in range(1, detCount+1):

        nodeName = f'diag.shinethru.det_{i:02d}.raw_signal'
        node = tree.getNode(nodeName)

        try:

            sigArr = node.getData().data()
            timeArr = node.dim_of().data()

            if sigArr is not None:

                sigArr2D.append(sigArr)
                timeArr2D.append(timeArr)

                impactNode = tree.getNode(f'diag.shinethru.det_{i:02d}.v_impact')
                impactParams.append(impactNode.getData().data())

        except:
            print(f'No data for node {nodeName}')

    tree.close()

    # Plot the data
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    fig.suptitle(f'{shotnum}')

    xMin, xMax = -5, 35

    # Beam dump detectors
    ax1 = fig.add_subplot(211)
    # In-vessel detectors
    ax2 = fig.add_subplot(212)

    for i in range(len(impactParams)):

        if np.abs(impactParams[i]) > 0.1:
            ax2.plot(timeArr2D[i]*1e3, sigArr2D[i]*1e3, alpha=0.5)
        else:
            ax1.plot(timeArr2D[i]*1e3, sigArr2D[i]*1e3)

    ax1.set_title('Beam Dump')
    ax1.set_xticks([])
    ax1.set_xlim(xMin, xMax)
    ax1.set_ylabel('Current [mA]')

    ax2.set_title('In Vessel')
    ax2.set_xlim(xMin, xMax)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Current [mA]')

    plt.show()

    return

def average_data(time, data):

    decFreq = 1e4
    bin_size = 1/decFreq
    bins = np.arange(time[0], time[-1], bin_size)

    newData, _, _ = sc.stats.binned_statistic(time, data, statistic='mean', bins=bins)

    newTimeArr = (bins[:-1] + bins[1:]) / 2

    return newTimeArr, newData

def plot_in_vessel_details(shotnum):

    tree = mds.Tree('wham', shotnum)

    timeArr = None
    sigArr2D = []
    impactParams = []

    for i in range(16, 22):

        nodeName = f'diag.shinethru.det_{i:02d}.raw_signal'
        node = tree.getNode(nodeName)

        try:

            sigArr = node.getData().data()
            timeArr = node.dim_of().data()

            if sigArr is not None:

                sigArr2D.append(sigArr)

                impactNode = tree.getNode(f'diag.shinethru.det_{i:02d}.v_impact')
                impactParams.append(impactNode.getData().data())

        except:
            print(f'No data for node {nodeName}')

    tree.close()

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), tight_layout=True)
    fig.suptitle(shotnum)

    for i, ax in enumerate(axs.T.flatten()):

        ax.plot(timeArr, sigArr2D[i], label=f'{np.round(impactParams[i]*1e2, 1)} cm')

        newTime, newData = average_data(timeArr, sigArr2D[i])
        ax.plot(newTime, newData, linewidth=3)

        ax.legend()

    plt.show()

    return

def plot_vs_interferometer(shotnum):

    tree = mds.Tree('wham', shotnum)

    # Beam dump data

    detCount = 15
    
    timeArr2D = []
    sigArr2D = []
    impactParams = []

    tree = mds.Tree('wham', shotnum)

    for i in range(1, detCount+1):

        nodeName = f'diag.shinethru.det_{i:02d}.linedens'
        node = tree.getNode(nodeName)

        try:

            sigArr = node.getData().data()
            timeArr = node.dim_of().data()

            if sigArr is not None:

                sigArr2D.append(sigArr)
                timeArr2D.append(timeArr)

                impactNode = tree.getNode(f'diag.shinethru.det_{i:02d}.v_impact')
                impactParams.append(impactNode.getData().data())

        except:
            print(f'No data for node {nodeName}')

    # Interferometer data
    node = tree.getNode('diag.interferomtr.dec_linedens')
    infDens = node.getData().data()
    infTime = node.dim_of().data()
    
    tree.close()

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    for i in range(len(impactParams)):

        ax.plot(timeArr2D[i]*1e3, sigArr2D[i]/2)

    ax.plot(infTime*1e3, infDens, color='black')

    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
    ax.set_xlabel('Time [ms]')

    plt.show()

    return

def plot_vs_interferometer_invessel(shotnum):

    tree = mds.Tree('wham', shotnum)
    
    # Beam dump data

    detCount = 21
    
    timeArr2D = []
    sigArr2D = []
    impactParams = []

    tree = mds.Tree('wham', shotnum)

    for i in range(1, detCount+1):

        nodeName = f'diag.shinethru.det_{i:02d}.linedens'
        node = tree.getNode(nodeName)

        try:

            sigArr = node.getData().data()
            timeArr = node.dim_of().data()

            if sigArr is not None:

                sigArr2D.append(sigArr)
                timeArr2D.append(timeArr)

                impactNode = tree.getNode(f'diag.shinethru.det_{i:02d}.v_impact')
                impactParams.append(impactNode.getData().data())

        except:
            print(f'No data for node {nodeName}')

    # Interferometer data
    node = tree.getNode('diag.interferomtr.dec_linedens')
    infDens = node.getData().data()
    infTime = node.dim_of().data()
    
    tree.close()

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    for i in range(len(impactParams)):

        ax.plot(timeArr2D[i]*1e3, sigArr2D[i]/1.414)

    ax.plot(infTime*1e3, infDens, color='black')

    ax.set_ylabel(r'$\int n_p \cdot dl$ [m$^{-2}$]')
    ax.set_xlabel('Time [ms]')

    plt.show()

    return

if __name__ == '__main__':

    args = parseArgs()

    plot_vs_interferometer(args.shotnum)
    # plot_vs_interferometer_invessel(args.shotnum)
    # plot_data(args.shotnum)
    # plot_in_vessel_details(args.shotnum)