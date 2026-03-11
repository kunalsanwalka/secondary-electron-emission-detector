# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:36:15 2025

@author: kunal

This file contains the (x, y, z) locations of the SEE detectors in WHAM.
If more detectors are added, add the positions to the detPosArr variable.

The detector positions are recorded in the following format-
1st index = x-position (west is +ve, east is -ve)
2nd index = y-position (up is +ve, down is -ve)
3rd index = z-position (north is +ve, south is -ve)

Units = mm
"""

import numpy as np

detPosArr = np.array([[ -944.6,  103.7,  -986.1],   # 1st 15 are in the dump
                      [ -940.9,   89.7,  -947.7],
                      [ -906.9,   68.1,  -939.5],
                      [ -903.4,   54.2,  -901.3],
                      [ -869.9,	  32.6,	 -893.6],
                      [ -856.3,	   0.0,	 -856.3],
                      [ -867.4,	 -27.5,	 -876.1],
                      [ -904.9,	 -41.5,	 -879.0],
                      [ -912.8,	 -63.0,	 -912.7],
                      [ -951.1,	 -77.0,	 -916.4],
                      [ -959.3,	 -98.5,	 -950.3],
                      [ -881.9,	 -15.7,	-1007.4],
                      [ -860.2,	  -7.8,	 -923.0],
                      [ -923.0,	   7.8,	 -860.2],
                      [-1007.4,   15.7,	 -881.9],
                      [ -310.8, -156.1,  -310.8],   # the last 6 are in the CC
                      [ -317.5, -139.6,  -317.5],
                      [ -324.3, -123.1,  -324.3],
                      [ -310.8,  156.1,  -310.8],
                      [ -317.5,  139.6,  -317.5],
                      [ -324.3,  123.1,  -324.3]])

# Convert to a dictionary
detPosDict = {}
for i in range(len(detPosArr)):
    detPosDict[f'det_{i}_machine_x'] = detPosArr[i, 0]
    detPosDict[f'det_{i}_machine_y'] = detPosArr[i, 1]
    detPosDict[f'det_{i}_machine_z'] = detPosArr[i, 2]

#### Convert machine aligned (x,y,z) to beam aligned (x,y,z)

# Position of the NBI ground grid center (machine aligned coordinates)
beamPos = np.array([2474.8, 0, 2474.8])
beamAngle = 3*np.pi/4

for i in range(len(detPosArr)):
    
    # Convert the x and z positions to NBI aligned coordinates
    beamAlignedZ = np.cos(beamAngle)*detPosDict[f'det_{i}_machine_x'] - np.sin(beamAngle)*detPosDict[f'det_{i}_machine_z'] + np.sqrt(beamPos[0]**2 + beamPos[2]**2)
    beamAlignedX = np.sin(beamAngle)*detPosDict[f'det_{i}_machine_x'] + np.cos(beamAngle)*detPosDict[f'det_{i}_machine_z']

    detPosDict[f'det_{i}_beamAligned_x'] = beamAlignedX
    detPosDict[f'det_{i}_beamAligned_y'] = detPosDict[f'det_{i}_machine_y']
    detPosDict[f'det_{i}_beamAligned_z'] = beamAlignedZ
    
    # Here r is signed with respect to the y-axis. This is done so that the
    # impact parameter calculated in other scripts is done properly.
    detPosDict[f'det_{i}_beamAligned_r'] = np.sign(detPosDict[f'det_{i}_beamAligned_y']) * (((beamAlignedX)**2 + (detPosDict[f'det_{i}_beamAligned_y'])**2)**0.5)

detPosDict['numDetectors'] = len(detPosArr)

# Make a 3D plot of the data if this function is called seperately
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size' : 32})
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('Machine aligned coordinates')
    
    for i in range(len(detPosArr)):
        
        ax.scatter([detPosDict[f'det_{i}_machine_x']], [detPosDict[f'det_{i}_machine_y']], [detPosDict[f'det_{i}_machine_z']], 
                   c='red', s=200)
    
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    
    plt.show()
    
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('Beam aligned coordinates')
    
    for i in range(len(detPosArr)):
        
        ax.scatter([detPosDict[f'det_{i}_beamAligned_x']], [detPosDict[f'det_{i}_beamAligned_y']], [detPosDict[f'det_{i}_beamAligned_z']], 
                   c='red', s=200)
    
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    
    plt.show()
    
        





