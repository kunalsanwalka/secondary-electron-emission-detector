import numpy as np
import matplotlib.pyplot as plt
import MDSplus as mds
import argparse
import matplotlib
import scipy as sc
matplotlib.use('TkAgg')

def parseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('-shotnum',
                        type=int)

    args = parser.parse_args()
    
    return args

def get_data(shotnum):

    tree = mds.Tree('wham', shotnum)

    labelArr = []
    dataArr = []

    # Top-Top
    node = tree.getNode('raw.acq1001_633.ch_01')
    dataArr.append(node.getData().data())
    timeArr = node.dim_of().data()
    labelArr.append('Top-Top')

    # Top-Middle
    node = tree.getNode('raw.acq1001_633.ch_02')
    dataArr.append(node.getData().data())
    labelArr.append('Top-Middle')
    
    # Top-Bottom
    node = tree.getNode('raw.acq1001_633.ch_03')
    dataArr.append(node.getData().data())
    labelArr.append('Top-Bottom')

    # Bottom-Top
    node = tree.getNode('raw.acq1001_633.ch_04')
    dataArr.append(node.getData().data())
    labelArr.append('Bottom-Top')
    
    # Bottom-Middle
    node = tree.getNode('raw.acq1001_633.ch_05')
    dataArr.append(node.getData().data())
    labelArr.append('Bottom-Middle')

    # Bottom-Bottom
    node = tree.getNode('raw.acq1001_633.ch_06')
    dataArr.append(node.getData().data())
    labelArr.append('Bottom-Bottom')

    dataArr = np.array(dataArr)

    tree.close()

    # Fix the time array
    startTime = timeArr[0]
    timeArr -= startTime
    timeArr *= 1e3
    timeArr += startTime
    
    return dataArr, timeArr, labelArr

def plot_data(shotnum):

    dataArr, timeArr, labelArr = get_data(shotnum)

    fig = plt.figure(figsize=(12, 12), tight_layout=True)

    ax = fig.add_subplot(211)

    tree = mds.Tree('wham', shotnum)
    node = tree.getNode('diag.shinethru.detector_01')

    bdumpData = node.getData().data()
    bdumpTime = node.dim_of().data() * 1e3
    
    ax.plot(bdumpTime, bdumpData)
    ax.set_xlim(-5, 25)
    
    ax = fig.add_subplot(212)

    for i in range(len(labelArr)):

        filteredData = sc.signal.savgol_filter(dataArr[i], 500, 2)

        # Re-baseline the data to the beam dump time array
        interpData = np.interp(bdumpTime, timeArr, filteredData, right=0, left=0)

        ax.plot(bdumpTime, interpData,
                label=labelArr[i])
        
#        ax.plot(timeArr, filteredData,
#                label=labelArr[i])

    ax.legend()
    ax.set_xlim(-5, 25)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Signal [V]')

    ax.set_title(shotnum)

    plt.show()

    return

if __name__ == '__main__':

    args = parseArgs()

    plot_data(args.shotnum)
