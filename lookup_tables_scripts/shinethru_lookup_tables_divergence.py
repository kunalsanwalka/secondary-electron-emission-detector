# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:28:03 2025

@author: kunal

This codes generates lookup tables therefore it only needs to be run if there
are changes in the SEE diagnostic hardware. It does NOT need to be run every
shot.

The data is stored in the following format for each detector-
1. singleDetDict['machine_pos'] = [x, y, z] # machine aligned coordinates of the detector [mm, mm, mm]
2. singleDetDict['beam_pos'] = [x, y, z] # beam aligned coordinates of the detector [mm, mm, mm]
3. singleDetDict['beam_pos_polar'] = [r, phi, z] # beam aligned polar coordinates of the detector [mm, radians, mm]
4. singleDetDict['impact_param_vertical'] = float # vertical impact parameter for the given beamlet divergence. Used in density profile reconstructions. [mm]
5. singleDetDict['impact_param_radial'] = float # radial impact parameter for the given beamlet divergence. Used to get the 2nd point to make a LOS. [mm]
6. singleDetDict['beam_sight_point'] = [x, y, z] # point in beam coordinates that defines the line of sight of the detector. This is calculated using the radial impact parameter. [mm, mm, mm]
7. singleDetDict['machine_sight_point'] = [x, y, z] # point in machine coordinates that defines the line of sight of the detector. This is calculated by adding the origin of the beam coordinates expressed in machine coordinates to the beam sight point. [mm, mm, mm]

Each dictionary can be accessed from the main detPosList. Each dictionary is an entry in the list. This list is stored under-
/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl
"""

import matplotlib
matplotlib.use('TkAgg')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from nbi_radial_dist import radial_weight
from astropy.modeling import models, fitting
plt.rcParams.update({'font.size' : 32})

# Import the positions of the shinethru detectors in both machine and beam 
# aligned coordinates
from shinethru_detector_positions import detPosDict

dataDest = '/home/sanwalka/shinethru/lookup_tables/'

def gaussian_func(x, a, x0, sigma): 
    return a*np.exp(-(x-x0)**2/(2*(sigma**2)))

def position_tracker(x0, y0, angle, xLocArr):
    
    #Slope of the line
    slope=np.tan(angle)
    
    #y = m*(x - x_1) + y_1
    yLocArr=slope*(xLocArr-x0)+y0
    
    return xLocArr, yLocArr

def dict_to_arrays():
    """
    Convert the dictionary to numpy arrays to make going over each detectors
    positions easier in other functions.

    Returns
    -------
    machineXPosArr : np.array
        Machine aligned x-position.
    machineYPosArr : np.array
        Machine aligned y-position.
    machineZPosArr : np.array
        Machine aligned z-position.
    beamXPosArr : np.array
        Beam aligned x-position.
    beamYPosArr : np.array
        Beam aligned y-position.
    beamZPosArr : np.array
        Beam aligned z-position.
    beamRPosArr : np.array
        Beam aligned r-position.
    beamThetaPosArr : np.array
        Beam aligned theta position.
    """
    
    machineXPosArr, machineYPosArr, machineZPosArr, beamXPosArr, beamYPosArr, beamZPosArr, beamRPosArr = [], [], [], [], [], [], []
    
    for i in range(detPosDict['numDetectors']):
        machineXPosArr.append(detPosDict[f'det_{i}_machine_x'])
        machineYPosArr.append(detPosDict[f'det_{i}_machine_y'])
        machineZPosArr.append(detPosDict[f'det_{i}_machine_z'])
        beamXPosArr.append(detPosDict[f'det_{i}_beamAligned_x'])
        beamYPosArr.append(detPosDict[f'det_{i}_beamAligned_y'])
        beamZPosArr.append(detPosDict[f'det_{i}_beamAligned_z'])
        beamRPosArr.append(detPosDict[f'det_{i}_beamAligned_r'])
        
    machineXPosArr = np.array(machineXPosArr)
    machineYPosArr = np.array(machineYPosArr)
    machineZPosArr = np.array(machineZPosArr)
    beamXPosArr = np.array(beamXPosArr)
    beamYPosArr = np.array(beamYPosArr)
    beamZPosArr = np.array(beamZPosArr)
    beamRPosArr = np.array(beamRPosArr)

    # Theta position in beam alined coordinates
    beamThetaPosArr = np.arctan2(beamYPosArr, beamXPosArr)
    
    return machineXPosArr, machineYPosArr, machineZPosArr, beamXPosArr, beamYPosArr, beamZPosArr, beamRPosArr, beamThetaPosArr

def create_detector_dict_list():
    """
    Generates a list of dictionaries where each dictionary contains the position and impact parameter information for a single detector. This is the main data structure that is used in the density profile reconstructions and other data analysis.

    Each dictionary has the following format-
    singleDetDict['machine_pos'] = [x, y, z] # machine aligned coordinates of the detector [mm, mm, mm]
    singleDetDict['beam_pos'] = [x, y, z] # beam aligned coordinates of the detector [mm, mm, mm]
    singleDetDict['beam_pos_polar'] = [r, phi, z] # beam aligned polar coordinates of the detector [mm, radians, mm]
    """

    # Load the positions of each detector
    machineXPosArr, machineYPosArr, machineZPosArr, beamXPosArr, beamYPosArr, beamZPosArr, beamRPosArr, beamThetaPosArr = dict_to_arrays()

    # Go over each detector and create a dictionary for it
    detDictList = []
    for i in range(len(machineXPosArr)):
        
        singleDetDict = {}
        singleDetDict['machine_pos'] = np.array([machineXPosArr[i], machineYPosArr[i], machineZPosArr[i]])
        singleDetDict['beam_pos'] = np.array([beamXPosArr[i], beamYPosArr[i], beamZPosArr[i]])
        singleDetDict['beam_pos_polar'] = np.array([beamRPosArr[i], beamThetaPosArr[i], beamZPosArr[i]])
        
        detDictList.append(singleDetDict)

    return detDictList

def generate_particle_tracks(gridCurve, gridRad, beamletDiv, beamDumpLoc, numXPos):
    """
    This function generates a set of particle tracks given the NBI parameters.
    This is an axisymmetrized function. i.e. the beam is modelled as a 1D
    emitter of neutral particles and their positions are tracked in 2D.

    Parameters
    ----------
    gridCurve : float
        Curvature of the NBI grid.
    gridRad : float
        Radius of the NBI grid.
    beamletDiv : float
        Divergence of the beamlet.
        Divergence is defined in the documentation for np.random.normal.
    beamDumpLoc : float
        Location of the beam dump (how far is it from the beam grid). 
        This sets the end point for the particle tracking.
    numXPos : int
        Number of positions along the beam path where we want to track the beam particle positions.

    Returns
    -------
    trackingXPosArr : np.array
        1D array that contains all the x positions for the particles.
    trackingYPosArr : np.array
        2D array that contains all the y positions for the particles.
        1st index- particle number
        2nd index- y position
    """
    
    #### Computational parameters

    # JV: Number of discrete beamlets radii (must be odd the preserve rotational symmetry, r = 0 case)
    # JV: Includes +/- r so number of radii = numBeamlets / 2 (for even) or (numBeamlets - 1) / 2 (for odd) 
    numBeamlets = 20 #JV: NO LONGER # OF BEAMLETS, WANTED TO PRESERVE VARIABLE NAME

    # JV: Square rooted number of beamlet points on a square beamlet grid
    numTotal = 50 #JV: NEW NUMBER OF BEAMLETS IN GRID, SQUARE TO FIND NUMBER OF BEAMLET LOCATIONS IN SQUARE GRID

    # Number of particles per beamlet
    particlesBeamlet = 150

    # x values at which we want the particle tracks
    trackingXPosArr = np.linspace(0, beamDumpLoc, numXPos)

    # Angle subtended by the outermost grid point
    gridAngle = np.arctan(gridRad/np.max(gridCurve)) #radians

    # Array of constantly spaced angles
    angleArr = np.linspace(-gridAngle, gridAngle, numBeamlets) #radians

    # Get the launch point positions in cartesian coordinates
    gridXValsArr = gridCurve*np.cos(angleArr)
    gridYValsArr = gridCurve*np.sin(angleArr)

    ############# JV: Weight launch positions by radii ###########################
    weights = radial_weight(numTotal, gridYValsArr).counter(cart=False, plots=0)
    gridXValsArr = np.repeat(gridXValsArr, weights)
    gridYValsArr = np.repeat(gridYValsArr, weights)
    angleArr = np.repeat(angleArr, weights)
    numBeamlets = len(gridYValsArr) #update number of beamlets

    # Flip the x values so the beam launches from left to right
    gridXValsArr = -gridXValsArr

    # Move the x values by the curvature radius so the middle of the grid is at (0,0)
    gridXValsArr += gridCurve

    # Angles at which 0 divergence particles launch from the beamlet
    launchAngleArr = (np.pi-angleArr) % np.pi

    #### Launch and track all the particles

    # Array to store the y values of the particle tracks. It is a 2D array
    # 1st index- particle number
    # 2nd index- x position
    trackingYPosArr = np.zeros(shape=(numBeamlets*particlesBeamlet, len(trackingXPosArr)))

    # Index of the current particle being tracked
    currParticleNumber = 0

    # Go over each beamlet
    for k in range(len(gridXValsArr)):
        
        # x and y positions
        xCurr = gridXValsArr[k]
        yCurr = gridYValsArr[k]
        
        # Get the slopes of the particles
        slopesArr = np.random.normal(loc=launchAngleArr[k], scale=beamletDiv, size=(particlesBeamlet))

        # Go over each slope and track the particle positions
        for j in range(len(slopesArr)):
            
            # Get the particle track
            xArr, yArr = position_tracker(xCurr, yCurr, slopesArr[j], trackingXPosArr)
            
            # Append yArr to trackingYPosArr
            trackingYPosArr[currParticleNumber] = yArr
            
            currParticleNumber+=1
    
    return trackingXPosArr, trackingYPosArr

def detector_impact_params_single(beamletDiv, makeplot=False):
    """
    This function calculates the impact parameters for each SEE detector for a
    given beamlet divergence. While this function can be run stand-alone for
    debugging, it is normally called from detector_impact_params when a beamlet
    divergence array is passed to it
    
    Parameters
    ----------
    beamletDiv : float
        Divergence of the beamlets being modelled.
        Units - radians
        
    Returns
    -------
    impactParamsVertical : np.array
        1D array with the plasma impact parameters calculated for the given beamlet divergence. This is used in the density profile reconstructions to calculate the line of sight of the detectors in plasma coordinates.
    impactParamsRadial : np.array
        1D array with the radial impact parameters calculated for the given beamlet divergence. This is used to calculate the line of sight of the detectors in machine coordinates.
    """
    
    # Get the detector positions. This function uses an old set of coordinates
    # where x is the position along the beam and y is the impact parameter
    _, _, _, beamXPosArr, beamYPosArr, beamZPosArr, beamRPosArr, _ = dict_to_arrays()

    # Convert mm to m
    beamXPosArr /= 1e3
    beamYPosArr /= 1e3
    beamZPosArr /= 1e3
    beamRPosArr /= 1e3
    
    # Detector radius
    detectorRadius = 0.094 * 2.54/1e2 # [m]
    
    # Plasma position from the grids
    plasmaPos = 3.5 # [m]
    
    # Get the beam particle tracks
    trackingXPosArr, trackingYPosArr = generate_particle_tracks(gridCurve = 3.4, 
                                                                gridRad = 4/39.37, 
                                                                beamletDiv = beamletDiv,
                                                                beamDumpLoc = np.max(beamZPosArr), 
                                                                numXPos = 500)
    
    # Find the plasma impact parameter for each detector
    impactParamsVertical = np.zeros_like(beamYPosArr)
    impactParamsRadial = np.zeros_like(beamYPosArr)
    
    # Vertical impact parameter calculation
    for i in range(len(beamYPosArr)):
        
        # Vertical centerpoint of the detector
        detectorRadialPos = beamYPosArr[i]
        
        # Upper limit of the detector
        detectorUpper = detectorRadialPos + detectorRadius
        detectorLower = detectorRadialPos - detectorRadius
        
        # Get the particle positions in y for the detector x position
        xPosIndex = np.abs(trackingXPosArr - beamZPosArr[i]).argmin()
        particlePosAtDet = trackingYPosArr [:, xPosIndex]
            
        # Indices of the particles that hit the SEE detector
        indices = []
        for j in range(len(particlePosAtDet)):
            
            finalPos = particlePosAtDet[j]
            
            if finalPos<=detectorUpper and finalPos>=detectorLower:
                indices.append(j)
        
        # Extract the tracks that hit the detector
        particlesHittingDetector = []
        for index in indices:
            particlesHittingDetector.append(trackingYPosArr[index])
        particlesHittingDetector = np.array(particlesHittingDetector)
        
        # Index of x position closest to the plasma position
        plasmaIndex = (np.abs(trackingXPosArr - plasmaPos)).argmin()
        
        # Y positions of the particles at the plasma position
        detectorPlasmaPosArr = particlesHittingDetector[:, plasmaIndex]
        
        # Average of the positions
        rValinPlasma = np.average(detectorPlasmaPosArr)
        
        impactParamsVertical[i] = rValinPlasma
    
    # Horizontal impact parameter calculation
    for i in range(len(beamYPosArr)):
        
        # Vertical centerpoint of the detector
        detectorRadialPos = beamRPosArr[i]
        
        # Upper limit of the detector
        detectorUpper = detectorRadialPos + detectorRadius
        detectorLower = detectorRadialPos - detectorRadius
        
        # Get the particle positions in y for the detector x position
        xPosIndex = np.abs(trackingXPosArr - beamZPosArr[i]).argmin()
        particlePosAtDet = trackingYPosArr [:, xPosIndex]
            
        # Indices of the particles that hit the SEE detector
        indices = []
        for j in range(len(particlePosAtDet)):
            
            finalPos = particlePosAtDet[j]
            
            if finalPos<=detectorUpper and finalPos>=detectorLower:
                indices.append(j)
        
        # Extract the tracks that hit the detector
        particlesHittingDetector = []
        for index in indices:
            particlesHittingDetector.append(trackingYPosArr[index])
        particlesHittingDetector = np.array(particlesHittingDetector)
        
        # Index of x position closest to the plasma position
        plasmaIndex = (np.abs(trackingXPosArr - plasmaPos)).argmin()
        
        # Y positions of the particles at the plasma position
        detectorPlasmaPosArr = particlesHittingDetector[:, plasmaIndex]
        
        # Average of the positions
        rValinPlasma = np.average(detectorPlasmaPosArr)
        
        impactParamsRadial[i] = rValinPlasma

    if makeplot == True:
        
        fig = plt.figure(figsize=(20, 10), tight_layout='True')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # Plot the particle tracks that hit the SEE detector
        for i in range(len(particlesHittingDetector)):
            ax1.plot(trackingXPosArr, particlesHittingDetector[i], color='black')
                
        # Plot the detector positions
        ax1.scatter(beamZPosArr, beamYPosArr, color='blue', s=100, label='Detector locations')
        ax2.scatter(beamZPosArr, beamRPosArr, color='blue', s=100, label='Detector locations')

        # Plot the plasma impact parameters
        ax1.scatter(np.full(detPosDict['numDetectors'], plasmaPos), impactParamsVertical, color='green', s=100, label='Impact Parameters')
        ax2.scatter(np.full(detPosDict['numDetectors'], plasmaPos), impactParamsRadial, color='green', s=100, label='Impact Parameters')
        
        # Draw lines connecting the impact parameter to the detector position
        for i in range(detPosDict['numDetectors']):
            ax1.plot([plasmaPos, beamZPosArr[i]], [impactParamsVertical[i], beamYPosArr[i]],
                    color='green')
            ax2.plot([plasmaPos, beamZPosArr[i]], [impactParamsRadial[i], beamRPosArr[i]],
                    color='green')

        # Draw a circle for the WHAM plasmas
        plasmaRad = 0.16 # meters
        circle = plt.Circle((plasmaPos, 0), plasmaRad, color='red', fill=False, linewidth=3, label='WHAM0.3')
        ax1.add_patch(circle)
        circle = plt.Circle((plasmaPos, 0), plasmaRad, color='red', fill=False, linewidth=3, label='WHAM0.3')
        ax2.add_patch(circle)
        plasmaRad = 0.11 # meters
        circle = plt.Circle((plasmaPos, 0), plasmaRad, color='red', linestyle='dashed', fill=False, linewidth=3, label='WHAM0.8')
        ax1.add_patch(circle)
        circle = plt.Circle((plasmaPos, 0), plasmaRad, color='red', linestyle='dashed', fill=False, linewidth=3, label='WHAM0.8')
        ax2.add_patch(circle)

        # ax.text(4.6, 0.1, r'r$_{Physical}$ = '+'{}cm'.format(np.round(detectorRadialPosArr[-1]*100, 2))+r'; r$_{det. in plasma}$ = '+'{}cm'.format(np.round(rValinPlasma*100, 2)), color='black', 
        #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        ax1.legend(loc=(1.01, 0))

        ax1.set_xlim(plasmaPos-0.2, np.max(beamZPosArr)+0.2)
        ax2.set_xlim(plasmaPos-0.2, np.max(beamZPosArr)+0.2)

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        
        ax1.set_xlabel('Z [m]')
        ax1.set_ylabel('X [m]')
        ax2.set_xlabel('Z [m]')
        ax2.set_ylabel('R [m]')

        ax1.set_title(f'Vertical impact parameters for {beamletDiv*1e3} mrad')
        ax2.set_title(f'Radial impact parameters')
        
        plt.show()
    
    # Convert back to mm for other functions
    impactParamsVertical *= 1e3
    impactParamsRadial *= 1e3

    return impactParamsVertical, impactParamsRadial

def add_impact_params_to_dict(detDictList, beamletDiv=0.025, makeplot=False):
    """
    Adds the impact parameters for a given beamlet divergence to the list of detector dictionaries. This is used to generate the main data structure that is used in the density profile reconstructions and other data analysis.

    The keys added to each element in the dictionary here are-
    singleDetDict['impact_param_vertical'] = float # vertical impact parameter for the given beamlet divergence. [mm]
    singleDetDict['impact_param_radial'] = float # radial impact parameter for the given beamlet divergence. [mm]
    singleDetDict['beam_sight_point'] = [x, y, z] # point in beam coordinates that defines the line of sight of the detector. This is calculated using the radial impact parameter. [mm, mm, mm]
    singleDetDict['machine_sight_point'] = [x, y, z] # point in machine coordinates that defines the line of sight of the detector. This is calculated by adding the origin of the beam coordinates expressed in machine coordinates to the beam sight point. [mm, mm, mm]
    
    Parameters
    ----------
    detDictList : list
        List of dictionaries that define the detector positions. This is generated from create_detector_dict_list.
    beamletDiv : float, optional
        Beamlet divergence for which the impact parameters are calculated. The default is 0.025 radians (25 mrad).
    makeplot : bool, optional
        Make a plot of the detector positions and the sight points. The default is False.

    Returns
    -------
    detDictList : list
        List of dictionaries that define the detector positions and impact parameters for the given beamlet divergence.
    """

    # Get the impact parameters for the given beamlet divergence
    impactParamsVertical, impactParamsRadial = detector_impact_params_single(beamletDiv, makeplot=False)

    # Add the impact parameters to each detector dictionary in detDictList
    for i, singleDetDict in enumerate(detDictList):
        singleDetDict['impact_param_vertical'] = impactParamsVertical[i]
        singleDetDict['impact_param_radial'] = impactParamsRadial[i]

    # Create a 2nd point in beam and machine coordinates that defines the line of sight of the detector using the radial impact parameter
    for i, singleDetDict in enumerate(detDictList):

        # Convert polar to cartesian coordinates
        impactX = np.abs(singleDetDict['impact_param_radial']) * np.cos(singleDetDict['beam_pos_polar'][1])
        impactY = np.abs(singleDetDict['impact_param_radial']) * np.sin(singleDetDict['beam_pos_polar'][1])

        # 3.5m is the distance of the plasma from the beam grid. This is used to calculate the z position of the 2nd point in beam coordinates.
        singleDetDict['beam_sight_point'] = np.array([impactX, impactY, 3500])

        # Go from beam coordinates to machine coordinates for the sight point
        beamX, beamZ = singleDetDict['beam_sight_point'][0], singleDetDict['beam_sight_point'][2]
        beamPos = np.array([beamX, beamZ])
        beamAngle = 3*np.pi/4
        rotationMatrix = np.array([[np.cos(beamAngle), -np.sin(beamAngle)],
                                    [np.sin(beamAngle), np.cos(beamAngle)]])

        # Apply the rotation matrix to the beam sight point
        machineSightPoint = np.dot(rotationMatrix, beamPos)

        # Apply the translation
        machineSightPoint += np.array([2474.8, 2474.8])
        singleDetDict['machine_sight_point'] = np.array([machineSightPoint[0], singleDetDict['beam_sight_point'][1], machineSightPoint[1]])

    if makeplot:

        # Projections in beam coordinates
        fig = plt.figure(figsize=(20, 10), tight_layout=True)
        fig.suptitle('Projections in beam aligned coordinates')

        # XZ projection
        ax1 = fig.add_subplot(211)
        # YZ projection
        ax2 = fig.add_subplot(212)

        # Plot the detector positions and the sight points
        for i, singleDetDict in enumerate(detDictList):

            # Plot the detector positions
            if i == 0:
                ax1.scatter(singleDetDict['beam_pos'][2], singleDetDict['beam_pos'][0], color='blue', s=100, label=f'Detector Location')
                ax2.scatter(singleDetDict['beam_pos'][2], singleDetDict['beam_pos'][1], color='blue', s=100)
            else:
                ax1.scatter(singleDetDict['beam_pos'][2], singleDetDict['beam_pos'][0], color='blue', s=100)
                ax2.scatter(singleDetDict['beam_pos'][2], singleDetDict['beam_pos'][1], color='blue', s=100)

            # Plot the sight points
            if i == 0:
                ax1.scatter(singleDetDict['beam_sight_point'][2], singleDetDict['beam_sight_point'][0], color='green', s=100, label=f'Detector sight point')
                ax2.scatter(singleDetDict['beam_sight_point'][2], singleDetDict['beam_sight_point'][1], color='green', s=100)
            else:
                ax1.scatter(singleDetDict['beam_sight_point'][2], singleDetDict['beam_sight_point'][0], color='green', s=100)
                ax2.scatter(singleDetDict['beam_sight_point'][2], singleDetDict['beam_sight_point'][1], color='green', s=100)
            
            # Plot the line of sight
            ax1.plot([singleDetDict['beam_pos'][2], singleDetDict['beam_sight_point'][2]], 
                    [singleDetDict['beam_pos'][0], singleDetDict['beam_sight_point'][0]], 
                    color='black')
            ax2.plot([singleDetDict['beam_pos'][2], singleDetDict['beam_sight_point'][2]], 
                    [singleDetDict['beam_pos'][1], singleDetDict['beam_sight_point'][1]],
                    color='black')
            
        ax1.set_xlabel('Z [m]')
        ax1.set_ylabel('X [m]')
        ax1.set_title('XZ projection')
        ax1.set_aspect('equal')
        ax1.legend(loc=(1.01, 0))

        ax2.set_xlabel('Z [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_title('YZ projection')
        ax2.set_aspect('equal')

        plt.show()

        # Projections in machine coordinates
        fig = plt.figure(figsize=(20, 10), tight_layout=True)
        fig.suptitle('Projections in machine coordinates')

        # XY projection
        ax1 = fig.add_subplot(211)
        # XZ projection
        ax2 = fig.add_subplot(212)

        # Plot the detector positions and the sight points
        for i, singleDetDict in enumerate(detDictList):

            # Plot the detector positions
            if i == 0:
                ax1.scatter(singleDetDict['machine_pos'][0], singleDetDict['machine_pos'][1], color='blue', s=100, label=f'Detector Location')
                ax2.scatter(singleDetDict['machine_pos'][0], singleDetDict['machine_pos'][2], color='blue', s=100)
            else:
                ax1.scatter(singleDetDict['machine_pos'][0], singleDetDict['machine_pos'][1], color='blue', s=100)
                ax2.scatter(singleDetDict['machine_pos'][0], singleDetDict['machine_pos'][2], color='blue', s=100)

            # Plot the sight points
            if i == 0:
                ax1.scatter(singleDetDict['machine_sight_point'][0], singleDetDict['machine_sight_point'][1], color='green', s=100, label=f'Detector sight point')
                ax2.scatter(singleDetDict['machine_sight_point'][0], singleDetDict['machine_sight_point'][2], color='green', s=100)
            else:
                ax1.scatter(singleDetDict['machine_sight_point'][0], singleDetDict['machine_sight_point'][1], color='green', s=100)
                ax2.scatter(singleDetDict['machine_sight_point'][0], singleDetDict['machine_sight_point'][2], color='green', s=100)
            
            # Plot the line of sight
            ax1.plot([singleDetDict['machine_pos'][0], singleDetDict['machine_sight_point'][0]], 
                    [singleDetDict['machine_pos'][1], singleDetDict['machine_sight_point'][1]], 
                    color='black')
            ax2.plot([singleDetDict['machine_pos'][0], singleDetDict['machine_sight_point'][0]], 
                    [singleDetDict['machine_pos'][2], singleDetDict['machine_sight_point'][2]],
                    color='black')
            
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('XY projection')
        ax1.set_aspect('equal')
        ax1.legend(loc=(1.01, 0))

        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Z [m]')
        ax2.set_title('XZ projection')
        ax2.set_aspect('equal')

        plt.show()

    return detDictList

def detector_impact_params(beamletDivergenceArr, savedata=False, makeplot=False):
    """
    This function adds the impact parameter for each detector to the detPosDict
    dictionary for each beamlet divergence in the array.
    
    The impact parameters are added to the dictionary in the following form-
    detPosDict['det_[DETNUM]_impact_param']
    
    Parameters
    ----------
    beamletDivergenceArr : np.array
        Array with the beamlet divergences to scan.
        Unit - radians
    savedata : bool, optional
        Save the data and fit to a .npz file. 
        The default is False.
    makeplot : bool, optional
        Make a plot of the data and line of best fit. 
        The default is False.
        
    Returns
    -------
    detPosDict: dictionary
        Dictionary that contains all the information for the SEE detectors
    """
    
    print('Calculating the detector impact parameters')
    
    # Array to store the plasma impact parameters for each detector for each
    # beamlet divergence.
    # 1st index = detector number
    # 2nd index = detector impact parameter
    impactParams2D = np.zeros(shape=(detPosDict['numDetectors'], len(beamletDivergenceArr)))
    
    # Go over each beamlet divergence and calculate the plasma impact parameter
    for i in range(len(beamletDivergenceArr)):
        
        print(f'{i+1} of {len(beamletDivergenceArr)}')
        impactParams2D[:, i] = detector_impact_params_single(beamletDivergenceArr[i])
        
    # Add this information to the dictionary
    for i in range(detPosDict['numDetectors']):
        detPosDict[f'det_{i}_impact_param'] = impactParams2D[i, :]
        
    if savedata == True:
        
        with open(dataDest+'see_detector_dictionary.pkl', 'wb') as pickleFile:
            pickle.dump(detPosDict, pickleFile)
        
    if makeplot == True:
        
        fig = plt.figure(figsize=(22, 8), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Go over each detector and plot its impact parameter vs the beamlet
        # divergence
        for i in range(detPosDict['numDetectors']):
            
            ax.plot(beamletDivergenceArr*1e3, detPosDict[f'det_{i}_impact_param']*1e2,
                    label=f'Det. {i}', linewidth=3)
            
        ax.set_xlabel('Beamlet Divergence [urad]')
        ax.set_ylabel('Impact Parameter [cm]')
        ax.legend(loc=(1.01, 0), ncol=2)
        
        plt.show()
    
    return detPosDict

def divergence_to_gaussian_single(beamletDiv, makeplot=False):
    """
    This function-
    1. Calculates a synthetic SEE detector response given an input NBI beamlet divergence.
    2. Fits a gaussian to the synthetic profile.
    3. Returns the standard deviation of the fitted gaussian.
    
    This is used to generate a lookup table than can convert the measured
    standard deviation at the SEE array in the beam dump to the beamlet
    divergence.
    
    It takes into account the fact that some SEE detectors are further upstream
    than others which causes a more peaked gaussian than one would normally
    expect with aligned SEE detectors.
    
    While it can be called standalone, normally this function is called by
    divergence_to_gaussian to generate the lookup table.

    Parameters
    ----------
    beamletDiv : float
        Divergence of a beamlet.
        Unit - radians
    makeplot : bool, optional
        Make a plot of the fitted gaussian. Useful when debugging. 
        The default is False.

    Returns
    -------
    gaussianWidth : float
        Width of the gaussian in the beam dump.
        Unit - meters
    """
    
    #### Calculate the detector response
    
    # Get the detector r and z positions in beam aligned coordinates
    _, _, _, _, _, zPosArr, rPosArr = dict_to_arrays()
    # Convert to m
    zPosArr /= 1e3
    rPosArr /= 1e3
    
    # Remove the in-vessel detectors as they are not used for this analysis
    rPosArr = rPosArr[zPosArr>4.5]
    zPosArr = zPosArr[zPosArr>4.5]
    
    # Generate the particle tracks
    trackingXPosArr, trackingYPosArr = generate_particle_tracks(gridCurve = 3.4, 
                                                                gridRad = 4/39.37, 
                                                                beamletDiv = beamletDiv, 
                                                                beamDumpLoc = np.max(zPosArr), 
                                                                numXPos = 500)
    
    # Calculate the number of particles that hit each detector
    detResponse = np.zeros(len(zPosArr))
    
    # Detector radius
    detectorRadius = 0.094 * 2.54/1e2 # [m]
    
    for i in range(len(detResponse)):
        
        # Upper and lower limit of the detector
        detectorUpper = rPosArr[i] + detectorRadius
        detectorLower = rPosArr[i] - detectorRadius
        
        # Get the particle positions in r for the detector z position
        xPosIndex = np.abs(trackingXPosArr - zPosArr[i]).argmin()
        particlePosAtDet = trackingYPosArr[:, xPosIndex]
        
        # Indices of the particles that hit the SEE detector
        for j in range(len(particlePosAtDet)):
            
            finalPos = particlePosAtDet[j]
            
            if finalPos<=detectorUpper and finalPos>=detectorLower:
                detResponse[i] += 1
                
    detResponse /= np.max(detResponse)
    
    # Location of the half max
    halfMaxLoc = np.abs(detResponse - 0.5).argmin()
    
    # Calculate the FWHM
    fwhm = np.abs(2 * rPosArr[halfMaxLoc])
    
    # Get a crude stddev
    stddevGuess = fwhm / 2.355
    
    # Sometimes the fitter breaks
    try:
        # Get the beam waist at the beam dump via a gaussian fit
        fitter = fitting.LevMarLSQFitter()
        model = models.Gaussian1D(mean = 0, stddev = stddevGuess, amplitude=1)
        
        # Fix the mean of the gaussian at 0
        model.mean.fixed = True
        model.amplitude.fixed = True
        
        # Make sure the standard deviation doesn't change too much from a crude calculation
        model.stddev.min = stddevGuess * 0.9
        model.stddev.max = stddevGuess * 2
        
        # Fit the model to the data
        fittedModel = fitter(model, rPosArr, detResponse)
        
        gaussianWidth = fittedModel.stddev[0]
        
        goodFit = True
    
    except:
        print('Fit failed')
        goodFit = False
        gaussianWidth = 0
    
    if makeplot == True and goodFit == True:
        
        fig = plt.figure(figsize=(12, 8), tight_layout='True')
        ax = fig.add_subplot(111)
        
        yPosPlotting = np.linspace(np.min(rPosArr), np.max(rPosArr), num=100)
        ax.scatter(rPosArr, detResponse, s=200, color='C0', label='Data/Tracking')
        ax.plot(yPosPlotting, fittedModel(yPosPlotting), linewidth=5, color='C1', label='Fit')
        
        ax.set_xlabel('Impact Parameter [m]')
        ax.set_ylabel('Signal Strength [arb. u.]')
        ax.set_title(f'Beamlet Divergence = {beamletDiv*1e3}urad')
        
        ax.legend()
        
        plt.show()
    
    return gaussianWidth

def divergence_to_gaussian(beamletDivergenceArr, savedata=False, makeplot=False):
    """
    This function generates a lookup table that converts the measured width of
    the NBI at the beam dump to the beamlet divergence.
    
    The lookup table is stored as an .npz file and it also contains the
    coefficients for a line of best fit to the data for fast conversion of the
    fitted gaussian to the beamlet divergence.
    
    If the data is saved in an .npz file, the format is-
    1. Destination- dataDest+'dump_width_to_divergence.npz'
    2. beamletDivergenceArr - input beamlet divergences. [mrad]
    3. gaussianWidthArr - gaussian widths at the beam dump [m]
    4. fitSlope - slope of the linear fit between x(=gaussianWidthArr) and 
                  y(=beamletDivergenceArr)
    5. fitIntercept - intercept of the linear fit between x(=gaussianWidthArr) 
                      and y(=beamletDivergenceArr)

    Parameters
    ----------
    beamletDivergenceArr : np.array
        Array with the beamlet divergences to scan.
        Unit - radians
    savedata : bool, optional
        Save the data and fit to a .npz file. 
        The default is False.
    makeplot : bool, optional
        Make a plot of the data and line of best fit. 
        The default is False.

    Returns
    -------
    gaussianWidthArr : np.array
        Width of the NBI at beam dump.
    """
    
    print('Generating the width-divergence lookup table')
    
    gaussianWidthArr = np.zeros(len(beamletDivergenceArr))
    
    for i in range(len(gaussianWidthArr)):
        
        print(f'{i+1} of {len(gaussianWidthArr)}')
        
        gaussianWidthArr[i] = divergence_to_gaussian_single(beamletDivergenceArr[i])
        
    # Fit a linear function to the data
    slope, intercept= np.polyfit(gaussianWidthArr, beamletDivergenceArr, 1)
    
    if savedata == True:
        np.savez(dataDest+'dump_width_to_divergence.npz', 
                 beamletDivergenceArr = beamletDivergenceArr,
                 gaussianWidthArr = gaussianWidthArr,
                 fitSlope = slope,
                 fitIntercept = intercept)
        
    if makeplot == True:
        
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Plot the data
        ax.plot(gaussianWidthArr*1e2, beamletDivergenceArr*1e3, linewidth=3,
                label='Data')
        
        # Plot the line of best fit
        xPosPlotting = np.linspace(0, 0.2, 100)
        ax.plot(xPosPlotting*1e2, ((slope*xPosPlotting)+intercept)*1e3,
                linewidth = 3, color='k', label='Fit')
        
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 40)
        
        ax.legend(loc='lower right')
        
        ax.set_xlabel('Width at beam dump [cm]')
        ax.set_ylabel('Beamlet divergence [urad]')
        
        plt.show()
    
    return gaussianWidthArr

def los_points(detDict):
    """
    Make a set of 100 points that interpolate along the line of sight of the detector. 
    This is used to calculate the density along the line of sight for each detector.

    Parameters
    ----------
    detDict : dict
        Dictionary containing the machine and coordinates of the detector and the sight point.

    Returns
    -------
    xArr : np.array
        x-coordinates of the points along the line of sight in machine coordinates [mm]
    yArr : np.array
        y-coordinates of the points along the line of sight in machine coordinates [mm]
    zArr : np.array
        z-coordinates of the points along the line of sight in machine coordinates [mm]
    rArr : np.array
        r-coordinates of the points along the line of sight in machine coordinates [mm]
    """

    # Get the machine coordinates of the detector and the sight point
    detMachinePos = np.array(detDict['machine_pos'])
    sightMachinePos = np.array(detDict['machine_sight_point'])

    xArr = np.linspace(-200, 200, 100)

    # Define the intercept and slope of the line of sight in machine coordinates
    ySlope = (sightMachinePos[1] - detMachinePos[1]) / (sightMachinePos[0] - detMachinePos[0])
    yIntercept = detMachinePos[1] - ySlope * detMachinePos[0]

    zSlope = (sightMachinePos[2] - detMachinePos[2]) / (sightMachinePos[0] - detMachinePos[0])
    zIntercept = detMachinePos[2] - zSlope * detMachinePos[0]

    yArr = ySlope * xArr + yIntercept
    zArr = zSlope * xArr + zIntercept

    rArr = np.sqrt(xArr**2 + yArr**2)

    return xArr, yArr, zArr, rArr

def add_los_points_to_dict(detDictList, makeplot=False):
    """
    Add the points that define the detector line of sight to the dictionary for each detector. 
    This is used to calculate the line-integrated density along the line of sight for each detector.
    
    :param detDictList: Description
    :param makeplot: Description
    """

    for detDict in detDictList:
        xArr, yArr, zArr, rArr = los_points(detDict)
        detDict['los_points'] = np.array([xArr, yArr, zArr])
        detDict['los_rz'] = np.array([rArr, zArr])

    if makeplot == True:

        # Plot the line of sight of each detector
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')

        for detDict in detDictList:

            xArr, yArr, zArr, rArr = los_points(detDict)

            # Plot the line of sight of the detector
            ax.plot(xArr, yArr, zArr)

            # Plot the position of the detector
            ax.scatter(detDict['machine_pos'][0], detDict['machine_pos'][1], detDict['machine_pos'][2], c='red', s=200)

            # Plot the sight point of the detector
            ax.scatter(detDict['machine_sight_point'][0], detDict['machine_sight_point'][1], detDict['machine_sight_point'][2], c='blue', s=200)

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal')

        plt.show()

    return detDictList

if __name__ == '__main__':
    
    # Creates a list with the dictionaries for each detector
    detDictList = create_detector_dict_list()

    # Add the impact parameters for each detector to the dictionary
    detDictList = add_impact_params_to_dict(detDictList, makeplot=True)

    # Add the line of sight points for each detector to the dictionary
    detDictList = add_los_points_to_dict(detDictList, makeplot=True)

    # Save the list of dictionaries as a pickle file
    with open(dataDest+'see_detector_dictionary.pkl', 'wb') as pickleFile:
        pickle.dump(detDictList, pickleFile)

    # Beamlet divergence array [mrad]
    beamletDivergenceArr = np.linspace(10, 35, 100) * 1e-3
    
    # Test case for a single gaussian fit
    # _ = divergence_to_gaussian_single(0.02, True)
    
    # Calculate the gaussian width as measured by the beam dump detectors
    # gaussianWidthArr = divergence_to_gaussian(beamletDivergenceArr, 
    #                                           savedata=True, 
    #                                           makeplot=True)
    
    # Test case for a single impact parameter calculation
    # _, _ = detector_impact_params_single(0.025, True)
    
    # Add the detector impact parameters to the dictionary
    # detPosDict = detector_impact_params(beamletDivergenceArr, 
    #                                     savedata=True, 
    #                                     makeplot=True)