"""
This script calculates the line-integrated density for the SEE detectors from a synthetic diagnostic implemented on a CQL3D output file.
"""

import pickle
import scipy as sc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

# Make the font size larger
plt.rcParams.update({'font.size': 18})

global plotDest, dataDest
plotDest = '/home/sanwalka/shinethru/plots/'
dataDest = '/home/sanwalka/shinethru/data/'

def species_labels(filename):
    """
    This function generates an array with the labels for each 'general' species
    That is, a species whos distribution function has been explicitly 
    calculated by CQL3D.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.

    Returns
    -------
    speciesLabels : array
        Labels of all the general species.
    """
    
    # Open the dataset
    ds = xr.open_dataset(filename,
                         decode_timedelta=False)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    # Name of each species and specification (general, maxwellian etc.)
    kspeci = np.array(ds['kspeci'].values).astype('U')
    # Remove all trailing whitespace
    kspeci = np.char.rstrip(kspeci)
    
    # =========================================================================
    # Generate the species labels
    # =========================================================================
    
    # Array to store species labels
    speciesLabels=[]
    
    # Get a slice of kspeci which just has the label and type
    kspeciSlice = kspeci
    # Append correct names to the labelling array
    for i in range(len(kspeciSlice)):

        if kspeciSlice[i,1] == 'general':
            
            # Label in CQL3D
            cqlLabel = kspeciSlice[i,0]
            
            # Come up with a nicer label
            niceLabel = ''
            if cqlLabel=='d' or cqlLabel=='D' or cqlLabel=='Deuterium' or cqlLabel=='deuterium':
                niceLabel='D'
            elif cqlLabel=='t' or cqlLabel=='T' or cqlLabel=='Tritium' or cqlLabel=='tritium':
                niceLabel='T'
                
            #Add it to the array
            speciesLabels.append(niceLabel)
    
    return speciesLabels

def ion_dens(filename,makeplot=False,saveplot=False,savedata=False,efastd=6,species=0):
    """
    This function returns the ion densities along with the associated 
    coordinate arrays.
    
    CQL3D does not output the ion densities directly. They are calculated by
    taking an integral of the distribution function over velocity space.
    
    Here, the ion densities are defined by-
    
    Fast ions = >6keV
    Warm ions = <6keV
    Total ions = Warm ions + Fast ions
    
    This threshold can be changed by altering the 'efastd' variable.
    
    NOTE: This function was originally written in Fortran, then converted
          to IDL and is finally in Python. A lot of optimizations can be made 
          to this code that are Python specific.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    savedata : boolean
        Save the data.
    efastd : float
        Boundary between warm and fast ions (keV).
    species : int
        Index of species.

    Returns
    -------
    ndwarmz : np.array
        Warm ion density function.
        It has the form- ndwarmz(radial position,z position)
        Units - m^-3
    ndfz : np.array
        Fast ion density function.
        It has the form- ndfz(radial position,z position)
        Units - m^-3
    ndtotz : np.array
        Total ion density function.
        It has the form- ndtotz(radial position,z position)
        Units - m^-3
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
        Units - m
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
        Units - m
    """
    
    # Open the file
    ds = xr.open_dataset(filename,
                         decode_timedelta=False)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    # Major radius of z points (=r)
    solrz = ds['solrz'].values
    
    # Height of z points (=z)
    solzz = ds['solzz'].values
    
    # Dimension of z-grid along B
    lz = int(ds['lz'].values)
    
    # Number of radial surface bins (=rdim)
    lrz = int(ds['lrz'].values)
    
    # Distribution function
    f = ds['f'].values
    
    # Pitch angle array
    y = ds['y'].values
    
    # Maximum pitch angle dimension (=ydim)
    iy = int(ds['iy'].values)
    
    # Normalized momentum-per-mass array
    x = ds['x'].values
    
    # Momentum-per-mass dimension (=xdim)
    jx = int(ds['jx'].values)
    
    # dx centered on x-mesh points
    dx = ds['dx'].values
    
    # Velocity normalization factor
    vnorm = float(ds['vnorm'].values)
    
    # Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi = ds['bbpsi'].values
    
    # Number of general (whose distribution functions are evaluated) species
    ngen = int(ds['ngen'].values)
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Clean up distribution function
    # =========================================================================
    
    #Remove all nan values (set them to 0)
    f=np.nan_to_num(f)
    
    #Set all values at or below 0 to 1e-5 (helps with taking the log)
    f[f<=0]=1e-5
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get the f for the right species (else f has the wrong shape)
        f=f[species]
    
    # =========================================================================
    # Create the ion density arrays
    # =========================================================================
    
    # Pitch angles from one central radial point
    pitchAngleArr = y[0,:]
    
    # Pitch angle step size
    dtheta = np.max(pitchAngleArr)/len(pitchAngleArr)
     
    # Define the values of the theta arrays
    theta0 = (dtheta / 2) + np.arange(iy) * dtheta
    stheta = np.sin(theta0)
    ctheta = np.cos(theta0)
    theta1 = 2 * np.pi * stheta * dtheta
    theta2 = theta1 * (ctheta**2)
    theta3 = np.pi * (stheta**3) * dtheta
        
    # Create the x location arrays
    xloc1 = (x**2) * dx
    xloc2 = (vnorm**2) * (x**2) * xloc1
        
    # Create the cosz and sinz arrays
    cosz = np.zeros((iy,lz,lrz))
    bsinz = np.zeros((iy,lz,lrz))
    
    # Define cosz and sinz
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for i in range(0,iy): #pitch angles
                sign=-1    
                if y[ilr,i]<=(np.pi/2):
                    sign=1.0
                else:
                    sign=-1.0
                if (1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)>0:
                    cosz[i,ilz,ilr]=sign*np.sqrt(1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)
                bsinz[i,ilz,ilr]=np.sqrt(1-cosz[i,ilz,ilr]**2)
    
    #Create itheta
    #TODO- What is itheta?
    itheta=np.zeros((iy,lz,lrz))
    
    #Define itheta
    for lr in range(0,lrz):
        for l in range(0,lz):
            for i in range(0,int(iy/2)):
                tempvalArr=np.where(bsinz[0:int(iy/2),l,lr]>=stheta[i])
                if np.size(tempvalArr)==0:
                    tempval=0
                else:
                    tempval=np.min(tempvalArr)
                #Check if tempval is larger than iy/2
                if tempval>(iy/2):
                    itheta[i,l,lr]=int(iy/2)
                else:
                    itheta[i,l,lr]=tempval
                #Make itheta symmetric
                itheta[iy-i-1,l,lr]=itheta[i,l,lr]
    
    #Create the ion density arrays
    ndfz=np.zeros((lz,lrz)) #Fast ions
    ndwarmz=np.zeros((lz,lrz)) #Warm ions
    ndtotz=np.zeros((lz,lrz)) #Total ions
    
    #Species atomic number (assume D for single species)
    anumd=0
    if len(speciesLabels)==1:
        anumd=2
    else: #multi-ion cql3d simulation
        if speciesLabels[species]=='D':
            anumd=2
        elif speciesLabels[species]=='T':
            anumd=3
        
    #Velocity of the fast ions
    vfastd=np.sqrt(2*efastd*1000/(anumd*938e6))*3e10
    #Array with indices where velocity is greater than vfastd
    fastArr=np.where(vnorm*x>=vfastd)
    #Minimum index
    jfast_mind=np.min(fastArr)
    
    #Calculate the ion densities
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for ij in range(0,jx): #energy bins
                ithetahere=itheta[:,ilz,ilr].astype(int)
                
                #Total ion density
                ndtotz[ilz,ilr]+=np.sum(theta1*xloc1[ij]*f[ilr,ij,ithetahere])
                
                #Fast ion density
                if ij>=jfast_mind:
                    ndfz[ilz,ilr]+=np.sum(theta1*xloc1[ij]*f[ilr,ij,ithetahere])
                    
                #Warm ion density
                else:
                    ndwarmz[ilz,ilr]+=np.sum(theta1*xloc1[ij]*f[ilr,ij,ithetahere])
    
    #Transpose the density arrays to match the indexing convention of Python. 
    #ndfz,ndwarmz and ndtotz use the same indexing convention as IDL. Since 
    #solrz and solzz follow the Python convention, we need to make sure the 
    #density arrays are consistent with solrz and solzz
    ndfz=np.transpose(ndfz)
    ndtotz=np.transpose(ndtotz)
    ndwarmz=np.transpose(ndwarmz)
    
    #Convert solrz and solzz from cm to m
    solrz /= 100
    solzz /= 100
    
    #Convert density from 1/cm^3 to 1/m^3
    ndwarmz *= 1e6
    ndfz *= 1e6
    ndtotz *= 1e6
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot == True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName = filename.split('/')[-1]
        #Remove the .nc part
        ncName = ncName[0:-3]
        #Add suffix for all the plots
        savenameFast = ncName+'_fast_ion_dens.png'
        savenameTot = ncName+'_total_ion_dens.png'
        savenameWarm = ncName+'_warm_ion_dens.png'
        if multiSpecies:
            savenameFast = ncName+'_fast_ion_dens_species_'+speciesLabels[species]+'.png'
            savenameTot = ncName+'_total_ion_dens_species_'+speciesLabels[species]+'.png'
            savenameWarm = ncName+'_warm_ion_dens_species_'+speciesLabels[species]+'.png'
        
        #Normalize all plots with respect to each other
        maxDens = np.max(ndtotz) #m^-3
        #Round maxDens to the nearest 5e19 for plot colorbar
        maxDensRounded = np.round(maxDens/5e19) * 5e19
        
        # =====================================================================
        # Fast ion density
        # =====================================================================
        
        fig1 = plt.figure(figsize=(20, 8))
        ax1 = fig1.add_subplot(111)
        
        pltobj = ax1.contourf(solzz, solrz, ndfz,
                              levels=np.linspace(0, maxDens, 200))
        
        # ax1.contour(pltobj,colors='black')
        cbar1 = fig1.colorbar(pltobj)
        cbar1.set_label(r'Density [m$^{-3}$]')
        cbar1.set_ticks(np.linspace(0, maxDensRounded, 6))
        
        ax1.set_xlabel('Z [m]')
        ax1.set_ylabel('R [m]')
        ax1.set_title('Fast Ion Density (>'+str(efastd)+'keV); Species - '+speciesLabels[species])
        
        ax1.grid(True)
        
        if saveplot == True:
            plt.savefig(plotDest+savenameFast, bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Warm ion density
        # =====================================================================
        
        fig2 = plt.figure(figsize=(20, 8))
        ax2 = fig2.add_subplot(111)
        
        pltobj = ax2.contourf(solzz, solrz, ndwarmz,
                              levels=np.linspace(0, maxDens, 200))
        
        # ax2.contour(pltobj,colors='black')
        cbar2 = fig2.colorbar(pltobj)
        cbar2.set_label(r'Density [m$^{-3}$]')
        cbar2.set_ticks(np.linspace(0, maxDensRounded, 6))
        
        ax2.set_xlabel('Z [m]')
        ax2.set_ylabel('R [m]')
        ax2.set_title('Warm Ion Density (<'+str(efastd)+'keV); Species - '+speciesLabels[species])
        
        ax2.grid(True)
        
        if saveplot == True:
            plt.savefig(plotDest+savenameWarm, bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Total ion density
        # =====================================================================
        
        fig3 = plt.figure(figsize=(20,8))
        ax3 = fig3.add_subplot(111)
        
        pltobj = ax3.contourf(solzz, solrz, ndtotz,
                              levels=np.linspace(0, maxDens, 200))
        
        # ax3.contour(pltobj,colors='black')
        cbar3 = fig3.colorbar(pltobj)
        cbar3.set_label(r'Density [m$^{-3}$]')
        cbar3.set_ticks(np.linspace(0, maxDensRounded, 6))
        
        ax3.set_xlabel('Z [m]')
        ax3.set_ylabel('R [m]')
        ax3.set_title('Total Ion Density; Species - '+speciesLabels[species])
        
        ax3.grid(True)
        
        if saveplot == True:
            plt.savefig(plotDest+savenameTot, bbox_inches='tight')
        plt.show()
        
    # =========================================================================
    # Save the data
    # =========================================================================
    
    if savedata == True:
        
        #Generate the savename of the data
        #Get the name of the .nc file
        ncName = filename.split('/')[-1]
        #Remove the .nc part
        ncName = ncName[0:-3]
        savenameWarm = ncName+'_warm_ion_density.npy'
        savenameFast = ncName+'_fast_ion_density.npy'
        savenameTot = ncName+'_total_ion_density.npy'
        savenameR = ncName+'_r_grid.npy'
        savenameZ = ncName+'_z_grid.npy'
        
        np.save(dataDest+savenameWarm, ndwarmz)
        np.save(dataDest+savenameFast, ndfz)
        np.save(dataDest+savenameTot, ndtotz)
        np.save(dataDest+savenameR, solrz)
        np.save(dataDest+savenameZ, solzz)
    
    return ndwarmz, ndfz, ndtotz, solrz, solzz

def load_density_profile_cql3d(filename, makeplot=False):
    """
    This function loads the density profile from a given .nc file and creates an interpolation function.

    Parameters
    ----------
    densityProfileFilePath : str
        The file path of the .nc file containing the CQL3D output.
    makeplot : bool, optional
        Whether to plot the density profile. The default is False.
    
    Returns
    -------
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]    
    """

    # Load the arrays from the cql3d output file
    _, _, plasmaDens, solrz, solzz = ion_dens(filename)

    # Flatten the arrays to make an interpolation function
    rVals = solrz.ravel()
    zVals = solzz.ravel()
    densVals = plasmaDens.ravel()

    # Mirrored negative z values
    rVals_mirr = rVals.copy()
    zVals_mirr = -zVals
    densVals_mirr = densVals.copy()

    # Combine both positive and negative values
    rVals = np.concatenate((rVals, rVals_mirr))
    zVals = np.concatenate((zVals, zVals_mirr))
    densVals = np.concatenate((densVals, densVals_mirr))

    # Points for the interpolator
    points = np.column_stack((rVals, zVals))

    interpFunc = sc.interpolate.LinearNDInterpolator(points, densVals,
                                                     fill_value=0.0)
    
    if makeplot == True:

        # Plot the density profile
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        rArr = np.linspace(0, 0.5, 100)
        zArr = np.linspace(-1.5, 1.5, 100)
        R, Z = np.meshgrid(rArr, zArr)

        points = np.array([R.flatten(), Z.flatten()]).T
        densityArr = interpFunc(points).reshape(R.shape)

        pltObj = ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')

        cbar = fig.colorbar(pltObj)
        cbar.set_label('Density [m^-3]')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')

        plt.show()    

    return interpFunc

def load_detector_dictionary(pickleFilePath):

    with open(pickleFilePath, 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    return detDictList

def single_detector_see_density(detDict, interpFunc, makeplot=False):
    """
    This function calculates the predicted density along the line of sight of a single SEE detector and also the line-integrated density at the detector.

    It adds them both to the input dictionary and also plots the density profile and line of sight if makeplot is set to True.

    They are added to the dictionary with the keys 'see_density_along_los' and 'predicted_see_density' respectively.
    
    Parameters
    ----------
    detDict : dict
        A dictionary containing the information about the SEE detector, including its line of sight points.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]
    makeplot : bool, optional
        Whether to plot the density profile and line of sight. The default is False.

    Returns
    -------
    None. The input dictionary is modified to include the density along the line of sight and the predicted SEE density at the detector.
    """

    # Get the line of sight points for the detector
    losPoints = detDict['los_rz'] / 1e3 # Convert from mm to m

    # Spacing between points along the line of sight
    point1 = losPoints[:, 0]
    point2 = losPoints[:, 1]
    spacing = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # Get the density at each point along the line of sight
    # Define the points in the way that the interpolation function expects (r, z)
    points = np.array([losPoints[0, :], losPoints[1, :]]).T
    densityAlongLos = interpFunc(points)

    # Integrate the line-integrated density along the line of sight to get the predicted SEE density at the detector
    predictedSeeDensity = np.trapezoid(densityAlongLos, dx=spacing)

    # Add the density along the line of sight and the predicted SEE density to the dictionary
    detDict['see_density_along_los'] = densityAlongLos
    detDict['predicted_see_density'] = predictedSeeDensity

    if makeplot == True:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        # Plot the density profile
        rArr = np.linspace(0, 0.5, 100)
        zArr = np.linspace(-1.5, 1.5, 100)
        R, Z = np.meshgrid(rArr, zArr)

        points = np.array([R.flatten(), Z.flatten()]).T
        densityArr = interpFunc(points).reshape(R.shape)

        pltObj = ax.contourf(Z, R, densityArr, levels=100, cmap='inferno')

        cbar = fig.colorbar(pltObj)
        cbar.set_label('Density [m^-3]')

        # Plot the line of sight of the detector
        ax.plot(losPoints[1, :], losPoints[0, :], c='white', linewidth=3, label='Line of sight')

        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')

        plt.show()

    return detDict

def synthetic_see_detector(detDictList, interpFunc, makeplot=False):

    # Go over each detector in the list and calculate the predicted SEE density at each detector
    for detDict in detDictList:
        single_detector_see_density(detDict, interpFunc)

    if makeplot == True:

        # Plot the predicted line integrated density at each detector
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        predictedDensities = [detDict['predicted_see_density'] for detDict in detDictList]
        impactParams = [detDict['impact_param_vertical'] for detDict in detDictList]

        ax.scatter(impactParams, predictedDensities, c='red', s=200)

        ax.set_ylabel(r'Predicted SEE density [m$^{-2}$]')
        ax.set_xlabel('Impact parameter [mm]')

        plt.show()

    return detDictList

if __name__ == '__main__':

    # Load the SEE detector dictionary
    with open('/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl', 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    # Generate the density interpolation function
    filename = dataDest + '230222_newFusDiag.nc'

    # _, _, plasmaDens, solrz, solzz = ion_dens(filename, makeplot=True)
    interpFunc = load_density_profile_cql3d(filename, makeplot=True)

    # Add the synthetic data to the dictionary
    detDictList = synthetic_see_detector(detDictList, interpFunc, makeplot=True)