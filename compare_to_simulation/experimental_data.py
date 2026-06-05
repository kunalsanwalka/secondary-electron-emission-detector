"""
This script calculates the line integrated density for a given shot and the associated error bars.
The data is stored as a python list where each list element is a dictionary that defines detector parameters.

Author: Kunal Sanwalka (May 2026)
"""

import numpy as np
import scipy as sc
import MDSplus as mds
import pickle
import logging
import argparse
import matplotlib.pyplot as plt
from functools import partial

logName = 'postproc_shinethru'
logDir = '/home/sanwalka/shinethru/logs/'
logPath = logDir+logName+'.log'

# Make args and logger global so they do not have to be passed to each function
global args, logger

def initialize_logger():

    # Create a logger for this file.
    logger = logging.getLogger(logName)
    logger.setLevel(logging.DEBUG)

    # Capture other warnings
    logging.captureWarnings(True)

    # Create a file handler to write to.
    file_handler = logging.FileHandler(logPath, "w")
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter to make the logging look nice.
    formatter = logging.Formatter('%(asctime)s, %(name)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger.
    logger.addHandler(file_handler)

    logger.info('Logger file created')

    return logger

def parseArgs():
    """
    This function gets the shot number if specified and then runs the post processing script.
    """

    logger.info('parseArgs')

    # Create a parser object that allows us to pass multiple arguments of various types into run_single_shot with 1 object
    parser = argparse.ArgumentParser(description='post processing script arguments')

    # Shot number
    parser.add_argument('-s','--shotnum', metavar = 'shot number', type=int, default=0,
                        help = 'Shot number to post-process. If not specified, latest shot is post-processed by getting shot number from andrew')

    # Reference shot number
    parser.add_argument('-r','--ref_shotnum', metavar = 'reference shot number', type=int, default=0,
                        help = 'Reference shot number to calculate the radial density profile. Radial profile is calculated from an NBI shot with no plasma (ref_shotnum) vs. an NBI shot with plasma (shotnum). If left as 0, the radial density is not calculated.')

    # Delay between WHAM trigger and D-tAcq trigger
    parser.add_argument('-d','--delay', metavar = 'nbi dtacq delay ms', type = float, default = 0.0,
                        help = 'Delay in ms between the main WHAM trigger and the trigger for the NBI D-tAcqs. Positive number means D-tAcq was triggered after the main WHAM trigger.')

    # Debug plots
    parser.add_argument('-p','--debug',
                        metavar = 'whether or not to make debugging plots and print statements',
                        type = bool,
                        default = False,
                        help = 'if true, plots are made for debugging the script. note that this slows down the code a lot and so should only be used when debuging and not in normal operation.')

    # Averaging frequency for the raw data
    parser.add_argument('-dec', '--decimation', 
                        metavar = 'averaging frequency Hz', 
                        type = float, 
                        default = 1e4,
                        help = 'the frequency (in Hz) to which we want to average down the raw data.')

    try:
        logger.info('try statement')
        args = parser.parse_args()
        logger.info('parser.parse_args succeeded?')
    except Exception as e:
        logger.error(e)

    logger.info(vars(args))

    logger.info('Arguments parsed.')

    # Pull the reference shot number from MDSplus if not specified
    if args.ref_shotnum == 0:

        logger.info('No reference shot number specified, pulling from MDSplus.')

        try:
            tree = mds.Tree('wham', args.shotnum)
            args.ref_shotnum = tree.getNode('diag.shinethru.ref_shotnum').data()
            tree.close()
            logger.info(f'Pulled reference shot number {args.ref_shotnum} from MDSplus.')
        except Exception as e:
            logger.error(f'Error occurred while pulling reference shot number from MDSplus: {e}')

    return args

def average_data(timeArr, data, makeplot=False):
    """
    Average the data down to a given frequency. This is done to the raw data BEFORE the line-integrated density is calculated.

    Normally, we average down to 10kHz but other frequencies can be chosen as well via args.

    Parameters
    ----------
    timeArr : np.array
        Time array for the data.
        Units = seconds
    data : np.array
        Corresponding data.
    
    Returns
    -------
    newTimeArr : np.array
        New time array that has been averaged down by args.decimation amount.
        Units = seconds
    newData : np.array
        Corresponding data.
    """

    # We will average the data in bins of 1/freq seconds
    bin_size = 1/args.decimation
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

def load_detector_dictionary(pickleFilePath):
    """
    This function loads the SEE detector dictionary from a given pickle file path.

    Parameters
    ----------
    pickleFilePath : str
        The file path of the pickle file containing the SEE detector dictionary.

    Returns
    -------
    detDictList : list of dicts
        A list of dictionaries, where each dictionary contains the information for one SEE detector. 
        The index is the detector number.
    """

    try:
        logger.info(f'Loading detector dictionary from {pickleFilePath}')
        with open(pickleFilePath, 'rb') as pickleFile:
                detDictList = pickle.load(pickleFile)
    except Exception as e:
        logger.error(f"Error occurred while loading detector dictionary: {e}")

    if args.debug:

        logger.info('Printing all detector information.')

        print('Detector information:')
        print(detDictList[0].keys())

    return detDictList

def node_names(detDictList):
    """
    Add the location of the raw and processed nodes for each detector to the detector dictionary list. 
    This is used to load the raw and processed signals from the MDSplus trees later.
    """

    logger.info('Adding raw node names to the detector dictionary list.')

    # Beam dump detectors
    detDictList[0]['raw_node'] = 'raw.acq1001_635.ch_01'
    detDictList[1]['raw_node'] = 'raw.acq1001_635.ch_02'
    detDictList[2]['raw_node'] = 'raw.acq1001_635.ch_03'
    detDictList[3]['raw_node'] = 'raw.acq1001_635.ch_04'
    detDictList[4]['raw_node'] = 'raw.acq1001_635.ch_05'
    detDictList[5]['raw_node'] = 'nc' # no signal for this detector
    detDictList[6]['raw_node'] = 'raw.acq1001_635.ch_07'
    detDictList[7]['raw_node'] = 'raw.acq1001_635.ch_08'
    detDictList[8]['raw_node'] = 'raw.acq1001_635.ch_09'
    detDictList[9]['raw_node'] = 'raw.acq1001_635.ch_10'
    detDictList[10]['raw_node'] = 'raw.acq1001_635.ch_11'
    detDictList[11]['raw_node'] = 'raw.acq1001_635.ch_12'
    detDictList[12]['raw_node'] = 'raw.acq1001_635.ch_13'
    detDictList[13]['raw_node'] = 'raw.acq1001_635.ch_14'
    detDictList[14]['raw_node'] = 'raw.acq1001_634.ch_09'

    # In vessel detectors
    detDictList[15]['raw_node'] = 'raw.acq1001_633.ch_06'
    detDictList[16]['raw_node'] = 'raw.acq1001_633.ch_05'
    detDictList[17]['raw_node'] = 'raw.acq1001_633.ch_04'
    detDictList[18]['raw_node'] = 'raw.acq1001_633.ch_01'
    detDictList[19]['raw_node'] = 'raw.acq1001_633.ch_02'
    detDictList[20]['raw_node'] = 'raw.acq1001_633.ch_03'

    logger.info('Adding processed node names to the detector dictionary list.')

    # Beam dump detectors
    detDictList[0]['processed_node'] = 'diag.shinethru.det_01'
    detDictList[1]['processed_node'] = 'diag.shinethru.det_02'
    detDictList[2]['processed_node'] = 'diag.shinethru.det_03'
    detDictList[3]['processed_node'] = 'diag.shinethru.det_04'
    detDictList[4]['processed_node'] = 'diag.shinethru.det_05'
    detDictList[5]['processed_node'] = 'nc' # no signal for this detector
    detDictList[6]['processed_node'] = 'diag.shinethru.det_07'
    detDictList[7]['processed_node'] = 'diag.shinethru.det_08'
    detDictList[8]['processed_node'] = 'diag.shinethru.det_09'
    detDictList[9]['processed_node'] = 'diag.shinethru.det_10'
    detDictList[10]['processed_node'] = 'diag.shinethru.det_11'
    detDictList[11]['processed_node'] = 'diag.shinethru.det_12'
    detDictList[12]['processed_node'] = 'diag.shinethru.det_13'
    detDictList[13]['processed_node'] = 'diag.shinethru.det_14'
    detDictList[14]['processed_node'] = 'diag.shinethru.det_15'

    # In vessel detectors
    detDictList[15]['processed_node'] = 'diag.shinethru.det_16'
    detDictList[16]['processed_node'] = 'diag.shinethru.det_17'
    detDictList[17]['processed_node'] = 'diag.shinethru.det_18'
    detDictList[18]['processed_node'] = 'diag.shinethru.det_19'
    detDictList[19]['processed_node'] = 'diag.shinethru.det_20'
    detDictList[20]['processed_node'] = 'diag.shinethru.det_21'

    return detDictList

def add_resistance(detDictList):
    """
    Add the resistance of the TIA resistor for each detector to the detector dictionary list. 
    This is used to convert the measured voltage signal to a current and then to a density.
    """

    logger.info('Adding resistance values to the detector dictionary list.')

    # Beam dump detectors
    detDictList[0]['resistance'] = 1e3
    detDictList[1]['resistance'] = 1e3
    detDictList[2]['resistance'] = 1e3
    detDictList[3]['resistance'] = 1e3
    detDictList[4]['resistance'] = 1e3
    detDictList[5]['resistance'] = 'nc' # no signal for this detector
    detDictList[6]['resistance'] = 1e3
    detDictList[7]['resistance'] = 1e3
    detDictList[8]['resistance'] = 1e3
    detDictList[9]['resistance'] = 1e3
    detDictList[10]['resistance'] = 1e3
    detDictList[11]['resistance'] = 1e3
    detDictList[12]['resistance'] = 1e3
    detDictList[13]['resistance'] = 1e3
    detDictList[14]['resistance'] = 1e3

    # In vessel detectors
    detDictList[15]['resistance'] = 1e2
    detDictList[16]['resistance'] = 1e2
    detDictList[17]['resistance'] = 1e2
    detDictList[18]['resistance'] = 1e2
    detDictList[19]['resistance'] = 1e2
    detDictList[20]['resistance'] = 1e2

    return detDictList

def load_raw_data(detDictList):
    """
    Loads the data from the raw digitizer nodes in MDSplus for each detector and adds it to the detector dictionary list.
    If a reference shot number is specified, it also loads the raw data for the reference shot and adds it to the detector dictionary list.

    The time arrays are in seconds.

    The keys added to each detector dictionary are-
    ['raw_signal'] = Current measured by each detector in Amps.
    ['ref_raw_signal'] = Raw signal for the reference shot in Amps.
    ['time_arr'] = Time array for the main shot and reference shot in seconds.

    We also average down the data to args.decimation from whatever it was and save it under-
    ['raw_signal_slow']
    ['ref_raw_signal_slow']
    ['time_arr_slow']
    
    Parameters
    ----------
    detDictList : list
        List of dictionaries containing detector information.
    args : argparse.Namespace
        Command line arguments.
    logger : logging.Logger
        Logger object for logging messages.

    Returns
    -------
    detDictList : list
        Updated list of dictionaries containing detector information.
    """

    #### Load the data for the plasma/current shot
    logger.info('Loading raw data for the plasma/current shot.')

    try:

        tree = mds.Tree('wham', args.shotnum)

        for detDict in detDictList:

            # The one that is a calorimeter
            if detDict['raw_node'] == 'nc':

                logger.info(f"No raw node specified for this detector, skipping loading raw data for this detector.")
                detDict['raw_signal'] = None
                detDict['time_arr'] = None
                detDict['raw_signal_slow'] = None
                detDict['time_arr_slow'] = None

            # In vessel detectors
            elif detDict['raw_node'][:15] == 'raw.acq1001_633':

                logger.info(f"Loading raw data for {detDict['raw_node']}")

                # Get the raw signal from the MDSplus tree
                rawSignal = tree.getNode(detDict['raw_node']).getData().data()
                # Apply the TIA gain to convert from voltage to current
                rawSignal /= detDict['resistance']

                tempTime = tree.getNode(detDict['raw_node']).dim_of().data()

                # Fix the time array
                startTime = tempTime[0]
                tempTime -= startTime
                tempTime *= 1e3
                tempTime += startTime
                tempTime /= 1e3

                # Add the raw signal and time array to the detector dictionary
                detDict['raw_signal'] = rawSignal
                detDict['time_arr'] = tempTime

                # Average down to 10kHz
                newTimeArr, newDataArr = average_data(tempTime, rawSignal)
                detDict['raw_signal_slow'] = newDataArr
                detDict['time_arr_slow'] = newTimeArr

            # Beam dump detectors
            else:

                logger.info(f"Loading raw data for {detDict['raw_node']}")

                # Get the raw signal from the MDSplus tree
                rawSignal = tree.getNode(detDict['raw_node']).getData().data()
                # Apply the TIA gain to convert from voltage to current
                rawSignal /= detDict['resistance']

                # Delay for the digitizer
                delayInSeconds = -tree.getNode('raw.acq1001_632.trig_time').getData().data()
                # Get the time array for the raw signal
                timeArr = -delayInSeconds + np.arange(len(rawSignal)) / 1e6

                # Add the raw signal and time array to the detector dictionary
                detDict['raw_signal'] = rawSignal
                detDict['time_arr'] = timeArr

                # Average down to 10kHz
                newTimeArr, newDataArr = average_data(timeArr, rawSignal)
                detDict['raw_signal_slow'] = newDataArr
                detDict['time_arr_slow'] = newTimeArr

        tree.close()

    except Exception as e:
        logger.error(f'Error occurred while loading raw data: {e}')

    #### Load the data for the reference shot if specified
    if args.ref_shotnum != 0:
        
        logger.info('Loading raw data for the reference shot.')

        try:

            tree = mds.Tree('wham', args.ref_shotnum)

            for detDict in detDictList:

                # The one that is a calorimeter
                if detDict['raw_node'] == 'nc':

                    logger.info(f"No raw node specified for this detector, skipping loading raw data for this detector.")
                    detDict['ref_raw_signal'] = None
                    detDict['time_arr'] = None
                    detDict['ref_raw_signal_slow'] = None
                    detDict['time_arr_slow'] = None

                # In vessel detectors
                elif detDict['raw_node'][:15] == 'raw.acq1001_633':

                    logger.info(f"Loading raw data for {detDict['raw_node']}")

                    # Get the raw signal from the MDSplus tree
                    rawSignal = tree.getNode(detDict['raw_node']).getData().data()
                    # Apply the TIA gain to convert from voltage to current
                    rawSignal /= detDict['resistance']

                    tempTime = tree.getNode(detDict['raw_node']).dim_of().data()

                    # Fix the time array
                    startTime = tempTime[0]
                    tempTime -= startTime
                    tempTime *= 1e3
                    tempTime += startTime
                    tempTime /= 1e3

                    # Add the raw signal and time array to the detector dictionary
                    detDict['ref_raw_signal'] = rawSignal

                    # Average down to 10kHz
                    newTimeArr, newDataArr = average_data(tempTime, rawSignal)
                    detDict['ref_raw_signal_slow'] = newDataArr

                # Beam dump detectors
                else:

                    logger.info(f"Loading raw data for {detDict['raw_node']}")

                    # Get the raw signal from the MDSplus tree
                    rawSignal = tree.getNode(detDict['raw_node']).getData().data()
                    # Apply the TIA gain to convert from voltage to current
                    rawSignal /= detDict['resistance']

                    # Delay for the digitizer
                    delayInSeconds = -tree.getNode('raw.acq1001_632.trig_time').getData().data()
                    # Get the time array for the raw signal
                    timeArr = -delayInSeconds + np.arange(len(rawSignal)) / 1e6

                    # Add the raw signal and time array to the detector dictionary
                    detDict['ref_raw_signal'] = rawSignal

                    # Average down to 10kHz
                    newTimeArr, newDataArr = average_data(timeArr, rawSignal)
                    detDict['ref_raw_signal_slow'] = newDataArr

            tree.close()

        except Exception as e:
            logger.error(f'Error occurred while loading raw data for reference shot: {e}')

    else:
        
        logger.info('No reference shot specified, skipping loading raw data for reference shot.')

        if args.debug:
            print('Warning: No reference shot specified, so the line-integrated densities will not be calculated.')

    # Make a plot of the raw signals if debug is true
    if args.debug:
        
        logger.info('Making plots of the raw signals for debugging.')

        fig = plt.figure(figsize=(15, 15), tight_layout=True)

        # Plot the raw signals for the plasma/current shot
        ax1 = fig.add_subplot(311)
        # Plot the raw signals for the reference shot        
        ax2 = fig.add_subplot(312)
        # Plot the NBI current and voltage to check timing
        ax3 = fig.add_subplot(313)

        # Make the legend neater
        lines, labels = [], []

        for i in range(len(detDictList)):

            detDict = detDictList[i]

            if detDict['raw_node'] != 'nc' and i < 15: # only plot the in vessel detectors

                line, = ax1.plot(detDict['time_arr']*1e3, detDict['raw_signal']*1e3, label=f"{i+1}")
                if args.ref_shotnum != 0:
                    ax2.plot(detDict['time_arr']*1e3, detDict['ref_raw_signal']*1e3)
                lines.append(line)
                labels.append(f"{i+1}")

            elif detDict['raw_node'] != 'nc' and i >= 15: # only plot the beam dump detectors

                # Apply a savgol filter to the raw signal to smooth it out for plotting
                smoothedSignal = sc.signal.savgol_filter(detDict['raw_signal'], 51, 2)
                line, = ax1.plot(detDict['time_arr']*1e3, smoothedSignal*1e3, label=f'{i+1}')
                
                if args.ref_shotnum != 0:
                    smoothedRefSignal = sc.signal.savgol_filter(detDict['ref_raw_signal'], 51, 2)
                    ax2.plot(detDict['time_arr']*1e3, smoothedRefSignal*1e3)
                    
                lines.append(line)
                labels.append(f"{i+1}")

        tree = mds.Tree('wham', args.shotnum)
        currentNode = tree.getNode('nbi.i_beam')
        voltageNode = tree.getNode('nbi.v_beam')
        current = currentNode.getData().data()
        voltage = voltageNode.getData().data()
        time = currentNode.dim_of().data()
        tree.close()

        ax3.plot(time*1e3, current, label='NBI Current [A]')
        ax3.plot(time*1e3, voltage*1e-3, label='NBI Voltage [kV]')
        ax3.legend()
        ax3.set_xlabel('Time [ms]')
        ax3.set_title('NBI Parameters')

        ax1.set_title(f'Plasma shot {args.shotnum}')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Current [mA]')

        ax2.set_title(f'Reference shot {args.ref_shotnum}')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Current [mA]')

        # Make a figure level legend for the detector numbers
        fig.legend(lines, labels,
                   loc='center right',   
                   ncol=3,               
                   title="Detector #")

        plt.show()

    return detDictList

def compute_single_detector_see_density(i, detDictList, nbiVoltageArr, nbiTimeArr, shineThruTable, lineDensScanArr, nbiVoltageScanArr):
    """
    Compute the line integrated density along the line of sight of a single detector based on the shine-thru fraction.

    Parameters
    ----------
    i : int
        Index of the detector in the detector dictionary list.
    detDictList : list
        List of dictionaries containing detector information.
    nbiVoltageArr : np.array
        NBI voltage for the plasma shot.
    nbiTimeArr : np.array
        Time array for the NBI voltage for the plasma shot.
    shineThruTable : np.array
        2D array containing the shine-thru fractions for different line integrated densities and NBI voltages.
    lineDensScanArr : np.array
        1D array containing the line integrated densities corresponding to the shine-thru fractions in the shine-thru table.
    nbiVoltageScanArr : np.array
        1D array containing the NBI voltages corresponding to the shine-thru fractions in the shine-thru table.
    args : argparse.Namespace
        Command line arguments.
    logger : logging.Logger
        Logger object for logging messages.

    Returns
    -------
    lineIntegratedDensArr : np.array
        Array containing the line integrated density along the line of sight of the detector for each time point.
    lineIntegratedDensitySigmaArr : np.array
        Array containing the uncertainty in the line integrated density along the line of sight of the detector for each time point.
    """

    logger.info(f'Calculating SEE density for detector {i+1}.')

    detDict = detDictList[i]

    if detDict['raw_node'] == 'nc':

        return i, None

    else:

        if args.debug:
            print(f'Calculating line integrated density for detector {i+1}.')

        signal = detDict['raw_signal_slow']
        refSignal = detDict['ref_raw_signal_slow']
        newTimeArr = detDict['time_arr_slow']

        # Load the NBI current for the plasma and reference shot
        tree = mds.Tree('wham', args.shotnum)
        refTree = mds.Tree('wham', args.ref_shotnum)

        plasmaNBICurr = tree.getNode('nbi.i_beam').getData().data()
        refNBICurr = refTree.getNode('nbi.i_beam').getData().data()

        tree.close()
        refTree.close()

        # Re-baseline the NBI data to the same time array as the detector signal
        tempNBIVoltageArr = np.interp(newTimeArr, nbiTimeArr, nbiVoltageArr)
        plasmaNBICurr = np.interp(newTimeArr, nbiTimeArr, plasmaNBICurr)
        refNBICurr = np.interp(newTimeArr, nbiTimeArr, refNBICurr)

        # Scale the reference SEE signal based on the ratio of the reference and plasma NBI currents
        scaling = plasmaNBICurr/refNBICurr
        refSignal *= scaling

        # Clean up the data to avoid /0 errors
        signal[signal <= 0] = 1e-6
        refSignal[refSignal <= 0] = 1e-6

        # Calculate the fraction of the NBI that shines through the plasma (1 = no plasma, 0 = super dense plasma)
        shineThruFracArr = np.abs(signal/refSignal)
        # Shine-thru fraction should be between 0 and 1, so clip any values outside this range due to noise
        shineThruFracArr[shineThruFracArr >= 1] = 1
        shineThruFracArr[shineThruFracArr < 0] = 1
        # When the NBI voltage is too low, there was no NBI and the data can be ignored, so set the shine-thru fraction to 1 in these cases
        shineThruFracArr[tempNBIVoltageArr < 5e3] = 1

        # For each time point, get the index of the closest NBI voltage in the scan array
        nbiIndexArr = np.abs(nbiVoltageScanArr - tempNBIVoltageArr[:, None]).argmin(axis=1)

        # Get the correct column of the shine-thru table for each time point based on the NBI voltage
        shineThruScanMatrix = shineThruTable[:, nbiIndexArr]

        # Index of the closest shine-thru fraction in the scan array for each time point, which gives us the line integrated density from the lineDensScanArr
        lineDensIndexArr = np.abs(shineThruScanMatrix - shineThruFracArr).argmin(axis=0)

        lineIntegratedDensArr = lineDensScanArr[lineDensIndexArr]

        return i, lineIntegratedDensArr

def calculate_line_integrated_densities(detDictList):
    """
    This function calculates the line integrated density along the line of sight of each detector.

    Parameters
    ----------
    detDictList : list
        List of dictionaries containing detector information, including the raw signals and time arrays.
    args : argparse.Namespace
        Command line arguments, including the reference shot number and whether to make debug plots.
    logger : logging.Logger
        Logger object for logging messages.

    Returns
    -------
    detDictList : list
        Updated list of dictionaries containing detector information, including the line integrated densities and their uncertainties.
    """

    logger.info('Calculating line integrated densities.')

    if args.ref_shotnum == 0:
        
        logger.info('No reference shot specified, skipping calculating line integrated densities.')

        if args.debug:
            print('Warning: No reference shot specified, so the line-integrated densities will not be calculated.')

        return detDictList

    logger.info('Loading the NBI voltages for the plasma shot.')
    tree = mds.Tree('wham', args.shotnum)
    nbiVoltageArr = tree.getNode('nbi.v_beam').getData().data()
    nbiTimeArr = tree.getNode('nbi.v_beam').dim_of().data()

    # Load the .npz file with the interpolation data
    try:
        gasType = tree.getNode('nbi.gas_type').getData().data()
        logger.info(f'Gas type for this shot is {gasType}.')
        if args.debug:
            print(gasType)

    except:
        logger.error('Error occurred while getting gas type from MDSplus, defaulting to Hydrogen.')
        gasType = 'Hydrogen'

    if gasType == 'Hydrogen':
        lookupTable = np.load('/home/sanwalka/shinethru/lookup_tables/shine_thru_table.npz')
    elif gasType == 'Deuterium':
        lookupTable = np.load('/home/sanwalka/shinethru/lookup_tables/shine_thru_table_d.npz')

    shineThruTable = lookupTable['shineThruTable']
    lineDensScanArr = lookupTable['lineDensScanArr'] # [m^-2]
    nbiVoltageScanArr = lookupTable['nbiVoltageScanArr'] # [V]

    logger.info('The shape of the shine-thru table is: '+str(shineThruTable.shape))
    logger.info('The shape of the line density scan array is: '+str(lineDensScanArr.shape))
    logger.info('The shape of the NBI voltage scan array is: '+str(nbiVoltageScanArr.shape))

    if args.debug:
        print('The shape of the shine-thru table is:', shineThruTable.shape)
        print('The shape of the line density scan array is:', lineDensScanArr.shape)
        print('The shape of the NBI voltage scan array is:', nbiVoltageScanArr.shape)

    # Make a worker so compute_single_detector_see_density can be run in parallel for each detector
    # DEPRECATED : Jack-the-threadripper does not have enough memory to run all detectors in parallel. Much faster to do it in series.
    # worker still used to make function calls more readable.
    worker = partial(compute_single_detector_see_density, 
                     detDictList=detDictList,
                     nbiVoltageArr=nbiVoltageArr,
                     nbiTimeArr=nbiTimeArr,
                     shineThruTable=shineThruTable,
                     lineDensScanArr=lineDensScanArr,
                     nbiVoltageScanArr=nbiVoltageScanArr)

    if args.debug:
        print('calculating line integrated densities in series')

    for i in range(len(detDictList)):

        i, lineIntegratedDensArr = worker(i)

        detDictList[i]['line_integrated_density'] = lineIntegratedDensArr

    if args.debug:
        print('finished calculating the line integrated densities')

    # Make a plot of the line integrated densities if debug is true
    if args.debug:

        logger.info('Making plots of the line integrated densities for debugging.')

        fig = plt.figure(figsize=(15, 15), tight_layout=True)
        ax = fig.add_subplot(111)

        for i in range(len(detDictList)):

            detDict = detDictList[i]

            if detDict['raw_node'] != 'nc':

                ax.plot(detDict['time_arr_slow']*1e3,detDict['line_integrated_density']/1.414, label=f"{i+1}")

        # Plot the interferometer density for comparison if it exists in MDSplus
        try:
            tree = mds.Tree('wham', args.shotnum)
            interferometerDensity = tree.getNode('diag.interferomtr.linedens').getData().data()
            interferometerTime = tree.getNode('diag.interferomtr.linedens').dim_of().data()
            tree.close()
            ax.plot(interferometerTime*1e3, interferometerDensity, label='Interferometer', color='k')
        except Exception as e:
            logger.error(f'Error occurred while loading interferometer data from MDSplus: {e}')

        ax.set_title('Line integrated densities')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'Line integrated density [m$^{-2}$]')
        ax.legend(loc=(1.01, 0), ncol=3, title='Detector #')

        plt.show()

    return detDictList

def calculate_line_integrated_density_errors(detDictList):
    """
    Calculate the error bars for each detector at each time-point.

    The errors included are-
    1. Error in the CX cross section (5% on the cross section itself)
    2. Signals from neutrals and light from an ECH only pulse (9% on the in-vessel detectors, 0.1% on the beam dump)
    """

    for i in range(len(detDictList)):

        detDict = detDictList[i]

        if detDict['raw_node'] == 'nc':

            detDict['line_integrated_density_sigma'] = None

        else:

            #### Load the raw data
            signal = detDict['raw_signal_slow']
            refSignal = detDict['ref_raw_signal_slow']
            newTimeArr = detDict['time_arr_slow']

            #### Scale the reference data based on the NBI current

            # Load the NBI current for the plasma and reference shot
            tree = mds.Tree('wham', args.shotnum)
            refTree = mds.Tree('wham', args.ref_shotnum)

            plasmaNBICurr = tree.getNode('nbi.i_beam').getData().data()
            refNBICurr = refTree.getNode('nbi.i_beam').getData().data()
            nbiTimeArr = tree.getNode('nbi.i_beam').dim_of().data()

            tree.close()
            refTree.close()

            # Re-baseline the NBI data to the same time array as the detector signal
            plasmaNBICurr = np.interp(newTimeArr, nbiTimeArr, plasmaNBICurr)
            refNBICurr = np.interp(newTimeArr, nbiTimeArr, refNBICurr)

            # Scale the reference SEE signal based on the ratio of the reference and plasma NBI currents
            scaling = plasmaNBICurr/refNBICurr
            refSignal *= scaling

            # Clean up the data to avoid /0 errors
            signal[signal <= 0] = 1e-6
            refSignal[refSignal <= 0] = 1e-6

            #### Error bars from the error in the cross section tables

            # The 5% error is from the Phelps paper (https://doi.org/10.1063/1.555858)
            cxError = 0.05

            # This formula comes from error propagating through the line-integrated density formula
            cxSigma = 0
            try:
                cxSigma = cxError * np.log(refSignal/signal)
            except Exception as e:
                logger.error(e)

            #### Error bars from ECH only shots

            # Check which detectors are in-vessel vs. beam dump
            impactParam = detDict['beam_pos'][1] # [mm]
            inVessel = np.abs(impactParam) > 110

            echSigma = 0
            try:
                if inVessel:
                    echSigma = 0.09 * detDict['line_integrated_density']
                else:
                    echSigma = 0.001 * detDict['line_integrated_density']
            except Exception as e:
                logger.error(e)

            # Add error bars in quadrature
            totSigmaSq = cxSigma**2 + echSigma**2
            totSigma = totSigmaSq**0.5

            detDict['line_integrated_density_sigma'] = totSigma

    return detDictList

def dictionary_to_pkl(detDictList):

    savename = f'/home/sanwalka/shinethru/data/{args.shotnum}.pkl'

    with open(savename, 'wb') as file:
        pickle.dump(detDictList, file)

    return

def run_post_process():

    # Load the detector dictionaries
    pickleFilePath = '/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl'
    detDictList = load_detector_dictionary(pickleFilePath)

    # Add the raw and processed node names to each detector dictionary
    detDictList = node_names(detDictList)

    # Add the resistance of the TIA resistor for each detector to the detector dictionary list
    detDictList = add_resistance(detDictList)

    # Load the raw data into the detector dictionaries
    detDictList = load_raw_data(detDictList)

    # Calculate the line integrated densities for each detector and add it to the detector dictionary list
    detDictList = calculate_line_integrated_densities(detDictList)

    # Add error bars
    detDictList = calculate_line_integrated_density_errors(detDictList)

    # Save the list as .pkl
    dictionary_to_pkl(detDictList)

    logger.info('Post processing completed.')

    return

def plot_raw_data():
    """
    Plot the raw and down-sampled data for the plasma and reference shot for each detector
    """

    # Load the detector dictionaries
    pickleFilePath = '/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl'
    detDictList = load_detector_dictionary(pickleFilePath)

    # Add the raw and processed node names to each detector dictionary
    detDictList = node_names(detDictList)

    # Add the resistance of the TIA resistor for each detector to the detector dictionary list
    detDictList = add_resistance(detDictList)

    # Load the raw data into the detector dictionaries
    detDictList = load_raw_data(detDictList)

    # Plot the data for each detector
    for i  in range(len(detDictList)):

        detDict = detDictList[i]

        if detDict['raw_node'] == 'nc':
            continue

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        # Plasma signal
        ax.plot(detDict['time_arr']*1e3, detDict['raw_signal']*1e3,
                color='red', 
                alpha=0.5)
        ax.plot(detDict['time_arr_slow']*1e3, detDict['raw_signal_slow']*1e3,
                color='red', 
                linewidth = 3,
                label='Plasma Signal')
        
        # Reference signal
        ax.plot(detDict['time_arr']*1e3, detDict['ref_raw_signal']*1e3,
                color='blue', 
                alpha=0.5)
        ax.plot(detDict['time_arr_slow']*1e3, detDict['ref_raw_signal_slow']*1e3,
                color='blue', 
                linewidth = 3,
                label='Reference Signal')
        
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Raw Signal [mA]')

        impactParam = np.round(detDict['impact_param_vertical']/10, 2)
        ax.set_title(f'{args.shotnum} \n {impactParam}cm')

        plt.show()

    return

if __name__ == "__main__":

    # Initialize the logger
    logger = initialize_logger()

    # Get the arguments
    args = parseArgs()
    logger.info('Arguments parsed, now running post processing script.')

    try:
        # Use the TkAgg backend for matplotlib to avoid issues with plotting in some environments
        plt.switch_backend('TkAgg')

        # Bigger font size for all plots
        plt.rcParams.update({'font.size' : 22})
    except Exception as e:
        logger.error(e)

    # plot_raw_data()
    run_post_process()
