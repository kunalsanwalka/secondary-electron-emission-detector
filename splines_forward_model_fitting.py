# -*- coding: utf-8 -*-

"""
This script contains the functions needed to calculate the radial density profile using a basis set of splines.

Normally, only radial_profile_splines_model would be imported into the parent script.

The output of the script is the radial density values at the specified radial positions.
"""

import numpy as np
import scipy as sc
import MDSplus as mds
import matplotlib.pyplot as plt
import argparse
from astropy import modeling
from astropy.modeling import Fittable1DModel, Parameter, models, fitting
from astropy.modeling.models import custom_model
plt.rcParams.update({'font.size':22})

import matplotlib
matplotlib.use('TkAgg')

def construct_spline_functions(x, mean=0, limit=0.15,
                               knot1Pos=0.1, knot2Pos=0.5, knot3Pos=0.6,
                               knot1Val=1, knot2Val=1, knot3Val=1, centralVal=1, debug=False):
    
    # Array with the knot positions
    knotPosArr = np.array([0, knot1Pos, knot2Pos, knot3Pos, 1]) * limit

    # If explicit knot values are being passed, then centralVal should not be 1
    if knot1Val != 1 and centralVal == 1:
        centralVal = knot1Val

    if debug == True:
        
        print('in the spline constructor')
        print(mean)
        print(limit)
        print(knot1Pos)
        print(knot2Pos)
        print(knot3Pos)
        print(knotPosArr)
        print(np.abs(x-mean))
    
    # Spline 1
    spline1Func = sc.interpolate.pchip(knotPosArr, np.array([centralVal, knot1Val, 0, 0, 0]), extrapolate=False)
    spline1 = spline1Func(np.abs(x - mean))    

    # Spline 2
    spline2Func = sc.interpolate.pchip(knotPosArr, np.array([0, 0, knot2Val, 0, 0]), extrapolate=False)
    spline2 = spline2Func(np.abs(x - mean))

    # Spline 3
    spline3Func = sc.interpolate.pchip(knotPosArr, np.array([0, 0, 0, knot3Val, 0]), extrapolate=False)
    spline3 = spline3Func(np.abs(x - mean))

    splinesArr = np.array([spline1, spline2, spline3])

    # Remove nan values
    splinesArr = np.nan_to_num(splinesArr)
    # Ensure that none of them have negative values
    splinesArr[splinesArr < 0] = 0

    if debug == True:

        fig = plt.figure(figsize=(12, 8), tight_layout='True')
        ax = fig.add_subplot(111)

        ax.set_title(f'mean = {mean}, limit={limit}')

        for i in range(len(splinesArr)):
            ax.plot(x, splinesArr[i], linewidth=4, color=f'C{i}')
            ax.scatter([knotPosArr[i+1]+mean], [1], s=200, color=f'C{i}')
#            ax.scatter([-knotPosArr[i+1]+mean], [1], s=200, color=f'C{i}')

        radialProf = sc.signal.savgol_filter(np.sum(splinesArr, axis=0), window_length=5, polyorder=2)
        ax.plot(x, radialProf, linewidth=2, linestyle='dashed', color='k')

#        ax.axvline(mean, label='mean', linewidth=3, color='k')
#        ax.axvline(limit+mean, label='limit', linewidth=3, linestyle='dashed', color='k')
#        ax.axvline(-limit+mean, linewidth=3, linestyle='dashed', color='k')

        ax.set_xlim(0, None)
        ax.set_xlabel('Radius')
        ax.set_ylabel('Density')
        ax.legend()
        
        plt.show()
    
    return splinesArr

@custom_model
def integrated_spline_functions(x, mean=0, limit=0.15,
                                knot1Pos=0.1, knot2Delta=0.5, knot3Delta=0.6,
                                knot1Val=1, knot2Val=1, knot3Val=1, centralVal=1):

    try:
        
        # print(x)
        # print(np.shape(x))
        # print(mean)
        # print(limit)
        # print(knot1Pos)
        # print(knot2Delta)
        # print(knot3Delta)
        # print(knot1Val)
        # print(knot2Val)
        # print(knot3Val)
        # print(centralVal)

        _, lineIntegratedDens = radial_and_integrated(x, mean, limit, knot1Pos, knot2Delta, knot3Delta, knot1Val, knot2Val, knot3Val, centralVal)
        return lineIntegratedDens
    
    except Exception as e:

        print(e)
        raise

def radial_and_integrated(x, mean=0, limit=0.15,
                          knot1Pos=0.1, knot2Delta=1e-3, knot3Delta=1e-3,
                          knot1Val=1, knot2Val=1, knot3Val=1, centralVal=1,
                          debug=False):
    
    # Get the values from the 1 element arrays passed to this function by the fitter
    try:
        mean = mean[0]
        limit = limit[0]
        knot1Pos = knot1Pos[0]
        knot2Delta = knot2Delta[0]
        knot3Delta = knot3Delta[0]
        knot1Val = knot1Val[0]
        knot2Val = knot2Val[0]
        knot3Val = knot3Val[0]
        centralVal = centralVal[0]
    except:
        pass

    # Calculate knot2Pos and knot3Pos
    # This way of doing it ensures that knots cannot switch positions
    # This is important as scipy.interpolate.pchip fails if the knot positions are not sorted
    knot2Pos = knot1Pos + knot2Delta
    knot3Pos = knot2Pos + knot3Delta

    # If knot3Pos is >= 1, set it to 0.99
    if knot3Pos >= 1:
        knot3Pos = 0.99

    try:

        # Force sort x as astropy is funky and can throw errors when these arrays are unsorted
        sortIdx = np.argsort(x)
        sortedX  = x[sortIdx]

        # Finer x array to get the radially integrated values
        hardLim = limit*1.2 + 0.1
        xFiner = np.linspace(-hardLim, hardLim, 51)
        if not np.all(np.diff(xFiner)>0):
            raise ValueError('xFiner is not strictly increasing')
        
        # Get the UNSHIFTED spline functions
        splinesArr = construct_spline_functions(xFiner,
                                                mean = 0,
                                                limit = limit,
                                                knot1Pos = knot1Pos,
                                                knot2Pos = knot2Pos,
                                                knot3Pos = knot3Pos,
                                                knot1Val = knot1Val,
                                                knot2Val = knot2Val,
                                                knot3Val = knot3Val,
                                                centralVal = centralVal)

        # Get the SHIFTED spline functions
        splinesArrShifted = construct_spline_functions(x,
                                                       mean = mean,
                                                       limit = limit,
                                                       knot1Pos = knot1Pos,
                                                       knot2Pos = knot2Pos,
                                                       knot3Pos = knot3Pos,
                                                       knot1Val = knot1Val,
                                                       knot2Val = knot2Val,
                                                       knot3Val = knot3Val,
                                                       centralVal = centralVal)
        
        # Multiply the spline functions by the scaling
        scaledSplines = splinesArr
        scaledShiftedSplines = splinesArrShifted

        # Sum the different splines to get the total density profile
        sumSpline = np.sum(scaledSplines, axis=0)
        sumShiftedSpline = np.sum(scaledShiftedSplines, axis=0)
        
        # Smooth the splines to avoid 0 derivative spots
        kernel = np.ones(5)/5
        smoothSpline = np.convolve(sumSpline, kernel, mode='same')
        radialDens = np.convolve(sumShiftedSpline, kernel, mode='same')

        # Remove negative values
        smoothSpline[smoothSpline < 0] = 0
        radialDens[radialDens < 0] = 0

        if debug == True:

            print('***********************************')
            print(f'limit = {limit}')
            print(f'mean = {mean}')
            print(f'knot1Pos = {knot1Pos}')
            print(f'knot2Pos = {knot2Pos}')
            print(f'knot3Pos = {knot3Pos}')
            print(f'knot1Val = {knot1Val}')
            print(f'knot2Val = {knot2Val}')
            print(f'knot3Val = {knot3Val}')
            print(f'centralVal = {centralVal}')
        
        # Map the 1D density profile to a 2D map
        y = np.copy(xFiner)
        XARR, YARR = np.meshgrid(xFiner, y)
        RARR = (XARR**2 + YARR**2)**0.5
        rArr = np.ravel(RARR)

        twoDMap = np.interp(rArr, xFiner, smoothSpline, left=0, right=0).reshape(RARR.shape)

        # Interpolate the function to the lab frame
        XARRlab = np.copy(XARR)
        YARRlab = np.copy(YARR)

        YARRnew = YARR - mean
        newPoints = np.array([XARR.flatten(), YARRnew.flatten()]).T

        interpFunc = sc.interpolate.RegularGridInterpolator((xFiner, y),
                                                            twoDMap,
                                                            method='linear',
                                                            fill_value=0,
                                                            bounds_error=False)
        
        twoDMapNew = interpFunc(newPoints).reshape(XARR.shape)

        # Calculate the line integrated densities for the shifted functions
        lineIntegratedDens = np.trapz(twoDMapNew, xFiner, axis=1)
        
        # Interpolation function to get the data back on the input x array
        lineIntegratedDens = np.interp(sortedX, xFiner, lineIntegratedDens)

        # Make sure the density outside the limit is 0
        # lineIntegratedDens[np.abs(sortedX) > limit] = 0
        
        # Go back to the same ordering as before
        unsortedDens = np.zeros_like(lineIntegratedDens)
        unsortedDens[sortIdx] = lineIntegratedDens

        if debug == True and np.max(smoothSpline)>0:

            fig = plt.figure(figsize=(20, 8), tight_layout=True)
            ax = fig.add_subplot(121)

            ax.plot(x, radialDens, linewidth=4, label='shifted')
            ax.plot(xFiner, smoothSpline, linewidth=4, label='unshifted')
            ax.legend()
            ax.set_ylabel('Density')

            ax = fig.add_subplot(122)

            ax.plot(x, unsortedDens, linewidth=4)
            ax.set_ylabel('Line Integrated')

            plt.show()
        
        return radialDens, unsortedDens
    
    except Exception as e:

        print('exception in the forward model')
        print('in radial_and_integrated')
        print(f'{knot1Pos}, {knot2Pos}, {knot3Pos}')
        print(f'{knot2Delta}, {knot3Delta}')
        print(e)
        
        raise

def chi2_with_reg(model, x, y):
    """
    Regularization is applied on the radial profile rather than the line-integrated profile as
    the radial profile is often more noisy than the line-integrated.
    """
    
    # Calculate the normal chi2 value
    chi2 = np.sum((y - model(x))**2)

    # Load the radial profile from the model
    mean = [model.mean[0]]
    limit = [model.limit[0]]
    knot1Pos = [model.knot1Pos[0]]
    knot2Delta = [model.knot2Delta[0]]
    knot3Delta = [model.knot3Delta[0]]
    knot1Val = [model.knot1Val[0]]
    knot2Val = [model.knot2Val[0]]
    knot3Val = [model.knot3Val[0]]
    centralVal = [model.centralVal[0]]

    radialProfile, _ = radial_and_integrated(np.linspace(-limit[0], limit[0], 50),
                                             mean=[0],
                                             limit=limit,
                                             knot1Pos=knot1Pos,
                                             knot2Delta=knot2Delta,
                                             knot3Delta=knot3Delta,
                                             knot1Val=knot1Val,
                                             knot2Val=knot2Val,
                                             knot3Val=knot3Val,
                                             centralVal=centralVal)
    
    smoothPenalty = 1e-1 * np.sum(np.diff(radialProfile,2)**2)

    regChi2 = chi2 + smoothPenalty
    
    return regChi2
    
def first_guess(normDensArr, detectorPos):

    # Sort
    sortIdx = np.argsort(detectorPos)
    sortedDetectorPositions = detectorPos[sortIdx]
    sortedLineIntegratedDensArr = normDensArr[sortIdx]

    #### Load the splines and their radial integrals
    splinesArr = construct_spline_functions(sortedDetectorPositions)
    
    # 2D mesh to integrate the splines
    XARR, YARR = np.meshgrid(detectorPos, detectorPos)
    RARR = np.sqrt(XARR**2 + YARR**2)
    
    # Calculate the integrated splines
    integratedSplines = np.zeros_like(splinesArr)
    
    for i in range(len(splinesArr)):
        
        interpFunc = sc.interpolate.interp1d(detectorPos, splinesArr[i], kind='linear', fill_value='extrapolate')
        twoDSpline = interpFunc(RARR)
        twoDSpline[twoDSpline<0] = 0
        
        # Integrate along x
        integratedSplines[i] = np.trapz(twoDSpline, XARR[0], axis=1)

    # M matrix for calculating the first guess
    M = integratedSplines.T
    Minv = np.linalg.pinv(M)

    # Multiply Minv with the line integrated data
    coeffs = np.matmul(Minv, sortedLineIntegratedDensArr)

    knot1Val = coeffs[0]
    knot2Val = coeffs[1]
    knot3Val = coeffs[2]

    return knot1Val, knot2Val, knot3Val

def radial_profile_splines_forward_model(timeArr, lineIntegratedDensArr, sigmaArr, detectorVerticalPositions, radialPosArr, args):

    # Sort the detectors and the line integrated densities by the detector vertical positions
    sortIdx = np.argsort(detectorVerticalPositions)
    sortedDetectorPositions = detectorVerticalPositions[sortIdx]
    sortedLineIntegratedDensArr = lineIntegratedDensArr[sortIdx]

    # Decimation
    dec = args.dec
    
    #### Fit the splines forward model to the data
    fittedMeanArr = np.zeros(int(len(timeArr)/dec))
    fittedLimitArr = np.zeros(len(fittedMeanArr))
    knot1PosArr = np.zeros(len(fittedMeanArr))
    knot2PosArr = np.zeros(len(fittedMeanArr))
    knot3PosArr = np.zeros(len(fittedMeanArr))
    knot2DeltaArr = np.zeros(len(fittedMeanArr))
    knot3DeltaArr = np.zeros(len(fittedMeanArr))
    knot1ValArr = np.zeros(len(fittedMeanArr))
    knot2ValArr = np.zeros(len(fittedMeanArr))
    knot3ValArr = np.zeros(len(fittedMeanArr))
    centralValArr = np.zeros(len(fittedMeanArr))
    fittedTimeArr = np.zeros(len(fittedMeanArr))

    for i in range(len(timeArr)):

        print(f'{i+1} of {len(timeArr)}')

        if i%dec == 0:

            j = int(i/dec)

            currRadialProfile = np.copy(sortedLineIntegratedDensArr[:, i])
            currUncertainty = np.copy(sigmaArr[:, i])
            fittedTimeArr[j] = timeArr[i]

            # Skip for no plasma
            if np.max(currRadialProfile) < 1e10:

                print(f'skipping t={np.round(timeArr[i], 3)}ms; no plasma')
                
                fittedMeanArr[j] = 0
                fittedLimitArr[j] = 0.15
                knot1PosArr[j] = 0.1
                knot2PosArr[j] = 0.2
                knot3PosArr[j] = 0.3
                knot2DeltaArr[j] = 0.1
                knot3DeltaArr[j] = 0.1
                knot1ValArr[j] = 0
                knot2ValArr[j] = 0
                knot3ValArr[j] = 0
                centralValArr[j] = 0

                continue

            else:

                # print('**************************************')
                # print(f'fitting t={np.round(timeArr[i], 3)}ms')
                
                # Normalize the radial profile (makes fitting more numerically stable)
                normVal = np.max(currRadialProfile)
                currRadialProfile /= normVal
                currUncertainty /= normVal

                try:

                    #### Run the fitter many times and pick the model with the best fit to the data
                    bestChi2 = np.inf
                    bestModel = None
                    
                    numTries = args.tries
                    for k in range(numTries):

                        model = integrated_spline_functions()
                        fitter = fitting.LevMarLSQFitter()

                        # Some reasonable bounds on the model
                        model.knot1Val.min = 0.2
                        model.knot2Val.min = 0.2
                        model.knot3Val.min = 0.2
                        model.centralVal.min = 0.2
                        model.knot1Val.max = 10
                        model.knot2Val.max = 10
                        model.knot3Val.max = 10
                        model.centralVal.max = 10

                        model.mean.min = -0.05
                        model.mean.max = 0.05

                        model.limit.min = 0.05
                        model.limit.max = 0.2

                        # First guesses
                        model.knot1Val, model.knot2Val, model.knot3Val = first_guess(currRadialProfile, sortedDetectorPositions)
                        model.knot1Val *= np.random.uniform(0.6, 1.4)
                        model.knot2Val *= np.random.uniform(0.6, 1.4)
                        model.knot3Val *= np.random.uniform(0.6, 1.4)
                        model.centralVal = model.knot1Val/2

                        # Pick knot1Pos at random. The knot2 and knot3 deltas as then scaled based on the initial guess
                        randomKnot1Pos = np.random.uniform(0, 1)
                        model.knot1Pos = randomKnot1Pos/2
                        model.knot1Pos.min = 0.1
                        model.knot1Pos.max = randomKnot1Pos
                        model.knot2Delta.min = 1e-3
                        model.knot2Delta.max = (1 - randomKnot1Pos) / 2
                        model.knot3Delta.min = 1e-3
                        model.knot3Delta.max = (1 - randomKnot1Pos) / 2

                        model.limit = 0.15

                        try:
                            currModel = fitter(model, sortedDetectorPositions, currRadialProfile,
                                               weights = 1/currUncertainty,
                                               maxiter = 200)
                            chi2 = chi2_with_reg(currModel, sortedDetectorPositions, currRadialProfile)
                            if chi2 < bestChi2:
                                bestChi2 = chi2
                                bestModel = currModel
                                
                        except Exception as e:
                            print(e)
                            continue

                    # print(bestModel)
                        
                    fittedMeanArr[j] = bestModel.mean[0]
                    fittedLimitArr[j] = bestModel.limit[0]
                    knot1PosArr[j] = bestModel.knot1Pos[0]
                    knot2PosArr[j] = knot1PosArr[j] + bestModel.knot2Delta[0]
                    knot3PosArr[j] = knot2PosArr[j] + bestModel.knot3Delta[0]
                    knot2DeltaArr[j] = bestModel.knot2Delta[0]
                    knot3DeltaArr[j] = bestModel.knot3Delta[0]
                    knot1ValArr[j] = bestModel.knot1Val[0] * normVal
                    knot2ValArr[j] = bestModel.knot2Val[0] * normVal
                    knot3ValArr[j] = bestModel.knot3Val[0] * normVal
                    centralValArr[j] = bestModel.centralVal[0] * normVal

                except Exception as e:
                    
                    print('Unable to fit')
                    print(e)
                    fittedMeanArr[j] = 0
                    fittedLimitArr[j] = 0.15
                    knot1PosArr[j] = 0.1
                    knot2PosArr[j] = 0.2
                    knot3PosArr[j] = 0.3
                    knot2DeltaArr[j] = -10
                    knot3DeltaArr[j] = -10
                    knot1ValArr[j] = 0
                    knot2ValArr[j] = 0
                    knot3ValArr[j] = 0
                    centralValArr[j] = 0
                    
    # Calculate the fitted radial density profile and the line-integrated profiles
    radialDensProfile = np.zeros(shape=(len(radialPosArr), len(fittedTimeArr)))
    lineIntegratedRadialProfile = np.zeros(shape=(len(radialPosArr), len(fittedTimeArr)))

    for j in range(len(fittedTimeArr)):

        currRadial, currIntegrated = radial_and_integrated(x          = radialPosArr,
                                                           mean       = [fittedMeanArr[j]],
                                                           limit      = [fittedLimitArr[j]],
                                                           knot1Pos   = [knot1PosArr[j]],
                                                           knot2Delta = [knot2DeltaArr[j]],
                                                           knot3Delta = [knot3DeltaArr[j]],
                                                           knot1Val   = [knot1ValArr[j]],
                                                           knot2Val   = [knot2ValArr[j]],
                                                           knot3Val   = [knot3ValArr[j]],
                                                           centralVal = [centralValArr[j]],
                                                           debug = False)
            
        radialDensProfile[:, j] = currRadial
        lineIntegratedRadialProfile[:, j] = currIntegrated

    else:
        return fittedTimeArr, radialDensProfile, lineIntegratedRadialProfile
        

if __name__ == '__main__':

    x = np.linspace(0, 1.1, 151)
    
    _ = construct_spline_functions(x,
                                   knot1Pos = 0.25,
                                   knot2Pos = 0.5,
                                   knot3Pos = 0.75,
                                   limit=1, debug=True)

#    knot1Pos = 0.1
#    knot2Pos = 0.5494554097862738
#    knot3Pos = 0.8989108195725477
#    knot1Val = 1.1820197287147745e+19
#    knot2Val = 1.2988634619502234e+19
#    knot3Val = 1.4256663615933865e+18
#    centralVal = 0.1820197287147745e+19
    
#    limit = 0.163305897726418
#    hardLim = limit*1.2 + 0.1
#    mean = 0.0011626549646923702
#    x = np.linspace(-hardLim, hardLim, 51)

#    splinesArr = construct_spline_functions(x,
#                                            mean=mean,
#                                            limit=limit,
#                                            knot1Pos = knot1Pos,
#                                            knot2Pos = knot2Pos,
#                                            knot3Pos = knot3Pos,
#                                            knot1Val = knot1Val,
#                                            knot2Val = knot2Val,
#                                            knot3Val = knot3Val,
#                                            centralVal = centralVal,
#                                            debug=True)

#    radialDens, lineIntegrated = radial_and_integrated(x,
#                                                       mean=[mean],
#                                                       limit=[limit],
#                                                       knot1Pos = [knot1Pos],
#                                                       knot2Delta = [knot2Pos-knot1Pos],
#                                                       knot3Delta = [knot3Pos-knot2Pos],
#                                                       knot1Val = [knot1Val],
#                                                       knot2Val = [knot2Val],
#                                                       knot3Val = [knot3Val],
#                                                       centralVal = [centralVal],
#                                                       debug=True)
