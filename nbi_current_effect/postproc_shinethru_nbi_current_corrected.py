"""
This script post-processes the raw data stored in the MDSplus trees for the D-tAcqs and converts it into
post processed data.

Author: Kunal Sanwalka (March 2026)
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

def parseArgs(logger):
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

def average_data(timeArr, data, freq, makeplot=False):

    # We will average the data in bins of 1/freq seconds
    bin_size = 1/freq
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

def load_detector_dictionary(pickleFilePath, args, logger):
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

def node_names(detDictList, logger):
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

def add_resistance(detDictList, logger):
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

def load_raw_data(detDictList, args, logger):
    """
    Loads the data from the raw digitizer nodes in MDSplus for each detector and adds it to the detector dictionary list.
    If a reference shot number is specified, it also loads the raw data for the reference shot and adds it to the detector dictionary list.

    The time arrays are in seconds.

    The keys added to each detector dictionary are-
    ['raw_signal'] = Current measured by each detector in Amps.
    ['ref_raw_signal'] = Raw signal for the reference shot in Amps.
    ['time_arr'] = Time array for the main shot and reference shot in seconds.

    ['ref_raw_signal'] is scaled based on the NBI current differences between the plasma and reference shot.
    
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
                timeArr = -delayInSeconds + np.arange(len(rawSignal)) / digFreq

                # Add the raw signal and time array to the detector dictionary
                detDict['raw_signal'] = rawSignal
                detDict['time_arr'] = timeArr

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

                # In vessel detectors
                elif detDict['raw_node'][:15] == 'raw.acq1001_633':

                    logger.info(f"Loading raw data for {detDict['raw_node']}")

                    # Get the raw signal from the MDSplus tree
                    rawSignal = tree.getNode(detDict['raw_node']).getData().data()
                    # Apply the TIA gain to convert from voltage to current
                    rawSignal /= detDict['resistance']

                    tempTime = tree.getNode(detDict['raw_node']).dim_of().data()

                    # Add the raw signal and time array to the detector dictionary
                    detDict['ref_raw_signal'] = rawSignal

                # Beam dump detectors
                else:

                    logger.info(f"Loading raw data for {detDict['raw_node']}")

                    # Get the raw signal from the MDSplus tree
                    rawSignal = tree.getNode(detDict['raw_node']).getData().data()
                    # Apply the TIA gain to convert from voltage to current
                    rawSignal /= detDict['resistance']

                    # Add the raw signal and time array to the detector dictionary
                    detDict['ref_raw_signal'] = rawSignal

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

def compute_single_detector_see_density(i, detDictList, nbiVoltageArr, nbiTimeArr, shineThruTable, lineDensScanArr, nbiVoltageScanArr, args, logger):
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

        return i, None, None, None

    else:

        if args.debug:
            print(f'Calculating line integrated density for detector {i+1}.')

        signal = detDict['raw_signal']
        refSignal = detDict['ref_raw_signal']
        timeArr = detDict['time_arr']

        # Time average the data to 10kHz from 1MHz/80MHz
        newTimeArr, signal = average_data(timeArr, signal, 1e4)
        _, refSignal = average_data(timeArr, refSignal, 1e4)

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
        
        # Calculate the uncertainty in the line integrated density
        term1 = 0.707 * lineIntegratedDensArr / np.log(1/shineThruFracArr)
        term2 = 2e-5/refSignal
        term3 = 7e-5/signal
        term4 = (term2**2 + term3**2)**0.5

        lineIntegratedDensitySigmaArr = term1 * term4
        lineIntegratedDensitySigmaArr[term1 * term4 < 0] = 1e20

        return i, newTimeArr, lineIntegratedDensArr, lineIntegratedDensitySigmaArr

def calculate_line_integrated_densities(detDictList, args, logger):
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
                     nbiVoltageScanArr=nbiVoltageScanArr,
                     args=args,
                     logger=logger)

    if args.debug:
        print('calculating line integrated densities in series')

    for i in range(len(detDictList)):

        i, newTimeArr, lineIntegratedDensArr, lineIntegratedDensSigmaArr = worker(i)

        detDictList[i]['line_integrated_density'] = lineIntegratedDensArr
        detDictList[i]['line_integrated_density_sigma'] = lineIntegratedDensSigmaArr
        detDictList[i]['line_integrated_density_timeArr'] = newTimeArr

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

                ax.plot(detDict['line_integrated_density_timeArr']*1e3,detDict['line_integrated_density']/1.414, label=f"{i+1}")

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

def calculate_coupled_power(detDictList, args, logger):
    """
    This function calculates the coupled NBI power into the plasma based on the shine-thru fraction.

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
    coupledPowerArr : np.array
        Array containing the coupled NBI power into the plasma for each time point.
    seeTimeArr : np.array
        Corresponding time array
    """

    logger.info('Calculating coupled NBI power into the plasma.')

    # If this is a reference shot, skip this calculation
    if args.ref_shotnum == 0:
        
        logger.info('No reference shot specified, skipping calculating coupled NBI power.')
        if args.debug:
            print('Warning: No reference shot specified, so the coupled NBI power will not be calculated.')

        return None, None

    # Get the input NBI power from MDSplus
    tree = mds.Tree('wham', args.shotnum)
    nbiCurrentArr = tree.getNode('nbi.i_beam').getData().data()
    nbiVoltageArr = tree.getNode('nbi.v_beam').getData().data()
    nbiTimeArr = tree.getNode('nbi.v_beam').dim_of().data()

    nbiPowerArr = nbiCurrentArr * nbiVoltageArr * 0.79 # neutralization efficiency
    nbiPowerArr[nbiPowerArr < 0] = 0

    detectorVerticalPositions = []
    refSignals = []
    plasmaSignals = []
    seeTimeArr = None

    # Only use the beam dump detectors for this calculation
    for i in range(len(detDictList)):

        detDict = detDictList[i]

        if detDict['raw_node'] != 'nc' and i < 15:

            detectorVerticalPositions.append(detDict['impact_param_vertical'])
            refSignals.append(detDict['ref_raw_signal'])
            plasmaSignals.append(detDict['raw_signal'])
            seeTimeArr = detDict['time_arr']

    # Re-baseline the NBI power to the same time array as the detector signals
    nbiPowerArr = np.interp(seeTimeArr, nbiTimeArr, nbiPowerArr)

    detectorVerticalPositions = np.array(detectorVerticalPositions)
    refSignals = np.array(refSignals)
    plasmaSignals = np.array(plasmaSignals)

    refShotIntegrand = refSignals * detectorVerticalPositions[:, np.newaxis]
    plasmaShotIntegrand = plasmaSignals * detectorVerticalPositions[:, np.newaxis]

    refShotArea = np.abs(np.trapz(refShotIntegrand, detectorVerticalPositions, axis=0))
    plasmaShotArea = np.abs(np.trapz(plasmaShotIntegrand, detectorVerticalPositions, axis=0))

    # Power into the beam dump
    beamDumpPoweArr = nbiPowerArr * (plasmaShotArea / refShotArea)

    # Power absorbed by the plasma
    coupledPowArr = nbiPowerArr - beamDumpPoweArr

    # Clean up the data
    coupledPowArr[coupledPowArr < 0] = 0
    coupledPowArr[coupledPowArr > nbiPowerArr] = nbiPowerArr[coupledPowArr > nbiPowerArr]

    # Plot the input and coupled NBI power
    if args.debug:

        logger.info('Making plots of the input and coupled NBI power for debugging.')

        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(seeTimeArr*1e3, nbiPowerArr*1e-3, label='Input NBI Power [kW]')
        ax.plot(seeTimeArr*1e3, coupledPowArr*1e-3, label='Coupled NBI Power [kW]')
        ax.set_title(f'Plasma shot {args.shotnum}, Reference shot {args.ref_shotnum}')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Power [kW]')
        ax.legend()

        plt.show()

    # Put the data into MDSplus
    logger.info('Putting coupled NBI power data into MDSplus.')
    try:

        nbiTree = mds.Tree('nbi', args.shotnum, 'edit')
        coupledPowerNode = nbiTree.getNode('coupled_pow')
        processedSignal = mds.Signal(mds.WithUnits(coupledPowArr, 'W'), None, mds.WithUnits(seeTimeArr, 's'))
        coupledPowerNode.putData(processedSignal)
        logger.info('Successfully put coupled power data into MDSplus.')

    except Exception as e:
        logger.error(f'Error occurred while putting coupled power data into MDSplus: {e}')
        if args.debug:
            print('Error occurred while putting coupled power data into MDSplus:', e)

    tree.close()
    nbiTree.close()

    return coupledPowArr, seeTimeArr

def gaussian_2d(coords, amp, x0, y0, vsigma, hsigma):
    """
    2D Gaussian function for fitting the beam-aligned detector positions weighted by the reference signals.

    Parameters
    ----------
    coords : tuple of np.arrays
        Tuple containing the x and y coordinates of the detectors.
    amp : float
        Amplitude of the Gaussian.
    x0 : float
        X-coordinate of the Gaussian center.
    y0 : float
        Y-coordinate of the Gaussian center.
    vsigma : float
        Vertical width of the Gaussian.
    hsigma : float
        Horizontal width of the Gaussian.

    Returns
    -------
    np.array
        Array containing the 2D Gaussian values at the given coordinates.
    """

    x, y = coords
    exponent = ((x - x0)**2 / (2 * hsigma**2)) + ((y - y0)**2 / (2 * vsigma**2))

    return amp * np.exp(-exponent)

def calculate_beam_positions_offsets(detDictList, args, logger):
    """
    Calculate the beam positions and offsets by fitting a 2D Gaussian to the beam-aligned detector positions.

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
    fittedTimeArr : np.array
        Array containing the time points at which the beam positions and offsets were calculated.
    verticalOffsets : np.array
        Vertical offsets of the NBI center. [m]
    horizontalOffsets : np.array
        Horizontal offsets of the NBI center. [m]
    verticalWidths : np.array
        Vertical widths of the NBI. [m]
    horizontalWidths : np.array
        Horizontal widths of the NBI. [m]
    """

    logger.info('Calculating beam positions and offsets.')

    # 2D array with the beam-aligned detector positions
    # 1st index- detector number
    # 2nd index- (x,y,z)
    beamAlignedPositions = []

    # Data from the no-plasma shot
    # 1st index- detector number
    # 2nd index- time array
    refSignals = []

    timeArr = None

    logger.info('Extracting the beam-aligned detector positions and the reference signals for each detector.')
    if args.ref_shotnum == 0:
        logger.info('Treating this shot as a reference shot, so the raw signals from this shot will be used as the reference signals.')
    else:
        logger.info(f'Using the raw signals from the reference shot {args.ref_shotnum} as the reference signals. \n\
                    This means that the offsets and widths will basically be duplicates of the reference shot signals')

    for i in range(len(detDictList)):

        detDict = detDictList[i]

        if detDict['raw_node'] != 'nc' and i < 15:

            beamAlignedPositions.append(detDict['beam_pos'])

            # If this is a reference shot, use data from the current shot, if it is NOT a reference shot, use data from the reference shot
            if args.ref_shotnum != 0:
                refSignals.append(detDict['ref_raw_signal'])
            else:
                refSignals.append(detDict['raw_signal'])

            timeArr = detDict['time_arr']

    # Convert to numpy arrays
    beamAlignedPositions = np.array(beamAlignedPositions) / 1e3 # convert to meters
    refSignals = np.array(refSignals)

    # Bin down the data to speed up the fitting, since the raw data is at 2 MHz and we only need to fit every 100th point
    binningFactor = 10
    refSignals = refSignals[:, ::binningFactor]
    timeArr = timeArr[::binningFactor]
    logger.info(f'Binned down the data by a factor of {binningFactor} to speed up the fitting.')

    fittedTimeArr = []
    verticalOffsets = []
    horizontalOffsets = []
    verticalWidths = []
    horizontalWidths = []
    amplitudes = []

    # Get the NBI voltage for the reference shot. This used as a check for if the fitter should be run at all
    tree = mds.Tree('wham', args.shotnum)
    nbiVoltageArr = tree.getNode('nbi.v_beam').getData().data()
    nbiTimeArr = tree.getNode('nbi.v_beam').dim_of().data()
    # Put it on the same time base as the detector signals
    nbiVoltageArr = np.interp(timeArr, nbiTimeArr, nbiVoltageArr)

    # Fit a 2D gaussian to the beam-aligned detector positions weighted by the reference signals to get the beam position and width
    for i in range(len(timeArr)):

        # If the NBI voltage is too low, skip the fitting since there is no beam
        if nbiVoltageArr[i] < 5e3:
            continue

        # Get the reference signal for each detector at this time point
        signalAtTime = refSignals[:, i]

        # Fit a 2D gaussian to the beam-aligned detector positions weighted by the reference signals
        try:

            popt, pcov = sc.optimize.curve_fit(gaussian_2d, 
                                               (beamAlignedPositions[:, 0], beamAlignedPositions[:, 1]), 
                                               signalAtTime,
                                               p0=[np.max(signalAtTime), 0, 0, 0.1, 0.1])
            amp, x0, y0, vsigma, hsigma = popt

            # If the offsets are HUGE, this is probably a bad fit, so skip it
            if np.abs(x0) > 0.5 or np.abs(y0) > 0.5:
                logger.warning(f'Fitted offsets for time {timeArr[i]} are too large (x0={x0}, y0={y0}), skipping this fit.')
                if args.debug:
                    print(f'Fitted offsets for time {timeArr[i]} are too large (x0={x0}, y0={y0}), skipping this fit.')
                continue

            # Similarly, if the widths are HUGE, this is probably a bad fit, so skip it
            if vsigma > 0.5 or hsigma > 0.5:
                logger.warning(f'Fitted widths for time {timeArr[i]} are too large (vsigma={vsigma}, hsigma={hsigma}), skipping this fit.')
                if args.debug:
                    print(f'Fitted widths for time {timeArr[i]} are too large (vsigma={vsigma}, hsigma={hsigma}), skipping this fit.')
                continue

            fittedTimeArr.append(timeArr[i])
            verticalOffsets.append(y0)
            horizontalOffsets.append(x0)
            verticalWidths.append(vsigma)
            horizontalWidths.append(hsigma)
            amplitudes.append(amp)

        except Exception as e:
            logger.error(f'Error occurred while fitting Gaussian for time {timeArr[i]}: {e}')
            if args.debug:
                print(f'Error occurred while fitting Gaussian for time {timeArr[i]}: {e}')
            continue

    # Convert the results to numpy arrays
    fittedTimeArr = np.array(fittedTimeArr)
    verticalOffsets = np.array(verticalOffsets)
    horizontalOffsets = np.array(horizontalOffsets)
    verticalWidths = np.array(verticalWidths)
    horizontalWidths = np.array(horizontalWidths)
    amplitudes = np.array(amplitudes)

    if args.debug and len(fittedTimeArr)>0:

        fig = plt.figure(figsize=(15, 15), tight_layout=True)
        
        # Plot the vertical and horizontal offsets over time
        ax1 = fig.add_subplot(311)

        # Plot the vertical and horizontal widths over time
        ax2 = fig.add_subplot(312)

        # Plot the chi-squared of the fits over time
        ax3 = fig.add_subplot(313)

        ax1.plot(fittedTimeArr*1e3, verticalOffsets*1e3, label='Vertical Offset [mm]')
        ax1.plot(fittedTimeArr*1e3, horizontalOffsets*1e3, label='Horizontal Offset [mm]')
        ax1.set_title('Beam Offsets Over Time')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Offset [mm]')
        ax1.legend()

        ax2.plot(fittedTimeArr*1e3, verticalWidths*1e3, label='Vertical Width [mm]')
        ax2.plot(fittedTimeArr*1e3, horizontalWidths*1e3, label='Horizontal Width [mm]')
        ax2.set_title('Beam Widths Over Time')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Width [mm]')
        ax2.legend()

        fittedVals = []
        timeIdx = []
        for i in range(len(fittedTimeArr)):

            amp = amplitudes[i]
            x0 = horizontalOffsets[i]
            y0 = verticalOffsets[i]
            vsigma = verticalWidths[i]
            hsigma = horizontalWidths[i]

            fittedVal = gaussian_2d((beamAlignedPositions[:, 0], beamAlignedPositions[:, 1]), amp, x0, y0, vsigma, hsigma)
            fittedVals.append(fittedVal)

            timeIdx.append(np.argmin(np.abs(timeArr - fittedTimeArr[i])))

        fittedVals = np.array(fittedVals)
        refSignalsAtFitTimes = refSignals[:, timeIdx].T

        residuals = refSignalsAtFitTimes - fittedVals
        chiSquared = np.sum((residuals**2) / (fittedVals + 1e-6), axis=1) # add a small number to avoid division by zero

        ax3.plot(fittedTimeArr*1e3, chiSquared, label='Chi-Squared')
        ax3.set_title('Quality of Fits Over Time')
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('Chi-Squared')
        ax3.legend()

        plt.show()

    elif len(fittedTimeArr) == 0 and args.debug:
        print('No NBI for this shot. Will not make a plot of the beam positions and offsets.')

    # Put the data into MDSplus
    logger.info('Putting beam position and width data into MDSplus.')
    try:
        verticalOffsetNode = tree.getNode('diag.shinethru.vertical_pos')
        horizontalOffsetNode = tree.getNode('diag.shinethru.horizontal_p')
        verticalWidthNode = tree.getNode('diag.shinethru.v_width')
        horizontalWidthNode = tree.getNode('diag.shinethru.h_width')

        verticalOffsetSignal = mds.Signal(mds.WithUnits(verticalOffsets, 'm'), None, mds.WithUnits(fittedTimeArr, 's'))
        horizontalOffsetSignal = mds.Signal(mds.WithUnits(horizontalOffsets, 'm'), None, mds.WithUnits(fittedTimeArr, 's'))
        verticalWidthSignal = mds.Signal(mds.WithUnits(verticalWidths, 'm'), None, mds.WithUnits(fittedTimeArr, 's'))
        horizontalWidthSignal = mds.Signal(mds.WithUnits(horizontalWidths, 'm'), None, mds.WithUnits(fittedTimeArr, 's'))

        verticalOffsetNode.putData(verticalOffsetSignal)
        horizontalOffsetNode.putData(horizontalOffsetSignal)
        verticalWidthNode.putData(verticalWidthSignal)
        horizontalWidthNode.putData(horizontalWidthSignal)

        logger.info('Successfully put beam position and width data into MDSplus.')

    except Exception as e:
        logger.error(f'Error occurred while putting beam position and width data into MDSplus: {e}')
        if args.debug:
            print('Error occurred while putting beam position and width data into MDSplus:', e)

    tree.close()

    logger.info('Finished calculating beam positions and offsets.')

    return fittedTimeArr, verticalOffsets, horizontalOffsets, verticalWidths, horizontalWidths

def aiming_detectors(args, logger):
    """
    Put the data for the 4 aiming detectors into MDSplus.
    """

    logger.info('Processing aiming detector data.')

    tree = mds.Tree('wham', args.shotnum)
    shinethruTree = mds.Tree('shinethru', args.shotnum, 'edit')

    rawNodeNames = ['raw.acq1001_634.ch_16',
                    'raw.acq1001_634.ch_10', 
                    'raw.acq1001_634.ch_11', 
                    'raw.acq1001_634.ch_12']

    processedNodeNames = ['aiming_top',
                          'aiming_bot',
                          'aiming_left',
                          'aiming_rig']

    rawSignals = []
    timeArr = None

    for i in range(len(rawNodeNames)):

        rawNode = tree.getNode(rawNodeNames[i])

        rawSignal = rawNode.getData().data()
        rawSignal /= 1e3
        rawSignals.append(rawSignal)

        # Delay for the digitizer
        delayInSeconds = -tree.getNode('raw.acq1001_632.trig_time').getData().data()
        # Get the time array for the raw signal
        timeArr = -delayInSeconds + np.arange(len(rawSignal)) / digFreq

        # Put the data into MDSplus
        try:
            processedNode = shinethruTree.getNode(processedNodeNames[i])
            processedSignal = mds.Signal(mds.WithUnits(rawSignal, 'A'), None, mds.WithUnits(timeArr, 's'))
            processedNode.putData(processedSignal)
            logger.info(f'Successfully put aiming detector {i+1} data into MDSplus.')

        except Exception as e:
            logger.error(f'Error occurred while putting aiming detector {i+1} data into MDSplus: {e}')
            if args.debug:
                print(f'Error occurred while putting aiming detector {i+1} data into MDSplus: {e}')

    shinethruTree.close()
    tree.close()

    if args.debug:

        fig = plt.figure(figsize=(15, 10), tight_layout=True)
        ax = fig.add_subplot(111)

        for i in range(len(rawNodeNames)):
            ax.plot(timeArr*1e3, rawSignals[i]*1e3, label=processedNodeNames[i])

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (mA)')
        ax.legend()
        ax.set_title('Aiming Detector Data')
        plt.show()

    logger.info('Finished processing aiming detector data.')

    return

def dictionary_to_mdsplus(detDictList, args, logger):
    """
    Put the data from the detector dictionaries into MDSplus.

    Ideally this is called AFTER all the post-processing is done, so that the line integrated densities can be put into MDSplus.
    """

    logger.info('Putting data from detector dictionaries into MDSplus.')

    tree = mds.Tree('shinethru', args.shotnum, 'edit')

    # Go over each detector and put the data into MDSplus
    for i in range(len(detDictList)):

        detDict = detDictList[i]

        # Newer MDSplus tree structure
        try:

            # Detector node
            detNode = tree.getNode(f'det_{(i+1):02d}')

            # Detector position in polar coordinates in beam frame
            try:

                beamPosPolar = detDict['beam_pos_polar']
                beamPosPolarNode = detNode.getNode('beampos_pol')
                # r
                rNode = beamPosPolarNode.getNode('r')
                rNode.putData(mds.WithUnits(beamPosPolar[0]/1e3, 'm'))
                # phi
                phiNode = beamPosPolarNode.getNode('phi')
                phiNode.putData(mds.WithUnits(beamPosPolar[1], 'rad'))
                # z
                zNode = beamPosPolarNode.getNode('z')
                zNode.putData(mds.WithUnits(beamPosPolar[2]/1e3, 'm'))
            except Exception as e:
                logger.error(f'Error occurred while putting polar coordinates for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting polar coordinates for detector {(i+1):02d} into MDSplus: {e}')

            # Detector position in cartesian coordinates in beam frame
            try:
                
                beamPosCartesian = detDict['beam_pos']
                beamPosCartesianNode = detNode.getNode('beampos')
                # x
                xNode = beamPosCartesianNode.getNode('x')
                xNode.putData(mds.WithUnits(beamPosCartesian[0]/1e3, 'm'))
                # y
                yNode = beamPosCartesianNode.getNode('y')
                yNode.putData(mds.WithUnits(beamPosCartesian[1]/1e3, 'm'))
                # z
                zNode = beamPosCartesianNode.getNode('z')
                zNode.putData(mds.WithUnits(beamPosCartesian[2]/1e3, 'm'))
            except Exception as e:
                logger.error(f'Error occurred while putting cartesian coordinates for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting cartesian coordinates for detector {(i+1):02d} into MDSplus: {e}')

            # Detector position in cartesian coordinates in lab frame
            try:
                
                labPosCartesian = detDict['machine_pos']
                labPosCartesianNode = detNode.getNode('machinepos')
                # x
                xNode = labPosCartesianNode.getNode('x')
                xNode.putData(mds.WithUnits(labPosCartesian[0]/1e3, 'm'))
                # y
                yNode = labPosCartesianNode.getNode('y')
                yNode.putData(mds.WithUnits(labPosCartesian[1]/1e3, 'm'))
                # z
                zNode = labPosCartesianNode.getNode('z')
                zNode.putData(mds.WithUnits(labPosCartesian[2]/1e3, 'm'))
            except Exception as e:
                logger.error(f'Error occurred while putting lab frame cartesian coordinates for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting lab frame cartesian coordinates for detector {(i+1):02d} into MDSplus: {e}')

            # 2nd detector position in the beam frame (used to make a line-of-sight vector)
            try:
                
                beamPos2Cartesian = detDict['beam_sight_point']
                beamPos2CartesianNode = detNode.getNode('plasmapos_b')
                # x
                xNode = beamPos2CartesianNode.getNode('x')
                xNode.putData(mds.WithUnits(beamPos2Cartesian[0]/1e3, 'm'))
                # y
                yNode = beamPos2CartesianNode.getNode('y')
                yNode.putData(mds.WithUnits(beamPos2Cartesian[1]/1e3, 'm'))
                # z
                zNode = beamPos2CartesianNode.getNode('z')
                zNode.putData(mds.WithUnits(beamPos2Cartesian[2]/1e3, 'm'))
            except Exception as e:
                logger.error(f'Error occurred while putting 2nd beam frame position for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting 2nd beam frame position for detector {(i+1):02d} into MDSplus: {e}')

            # 2nd detector position in the lab frame (used to make a line-of-sight vector)
            try:
                
                labPos2Cartesian = detDict['machine_sight_point']
                labPos2CartesianNode = detNode.getNode('plasmapos_m')
                # x
                xNode = labPos2CartesianNode.getNode('x')
                xNode.putData(mds.WithUnits(labPos2Cartesian[0]/1e3, 'm'))
                # y
                yNode = labPos2CartesianNode.getNode('y')
                yNode.putData(mds.WithUnits(labPos2Cartesian[1]/1e3, 'm'))
                # z
                zNode = labPos2CartesianNode.getNode('z')
                zNode.putData(mds.WithUnits(labPos2Cartesian[2]/1e3, 'm'))
            except Exception as e:
                logger.error(f'Error occurred while putting 2nd lab frame position for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting 2nd lab frame position for detector {(i+1):02d} into MDSplus: {e}')

            # Radial detector impact parameter in the beam frame
            try:
                
                impactParamRadialNode = detNode.getNode('r_impact')
                impactParamRadialNode.putData(mds.WithUnits(detDict['impact_param_radial']/1e3, 'm'))
            except Exception as e:
                logger.error(f'Error occurred while putting radial impact parameter for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting radial impact parameter for detector {(i+1):02d} into MDSplus: {e}')

            # Vertical detector impact parameter in the lab frame
            try:
                
                impactParamVerticalNode = detNode.getNode('v_impact')
                impactParamVerticalNode.putData(mds.WithUnits(detDict['impact_param_vertical']/1e3, 'm'))
            except Exception as e:
                logger.error(f'Error occurred while putting vertical impact parameter for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting vertical impact parameter for detector {(i+1):02d} into MDSplus: {e}')

            # Raw signal for the detector
            try:
                
                rawNode = detNode.getNode('raw_signal')
                rawSignal = detDict['raw_signal']
                rawTimeArr = detDict['time_arr']
                rawData = mds.Signal(mds.WithUnits(rawSignal, 'A'), None, mds.WithUnits(rawTimeArr, 's'))
                rawNode.putData(rawData)
            except Exception as e:
                logger.error(f'Error occurred while putting raw signal for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting raw signal for detector {(i+1):02d} into MDSplus: {e}')
            
            # Line integrated density for the detector
            try:
                
                lineIntegratedDensityNode = detNode.getNode('linedens')
                lineIntegratedDensity = detDict['line_integrated_density']
                lineIntegratedDensityTimeArr = detDict['time_arr']
                lineIntegratedDensityData = mds.Signal(mds.WithUnits(lineIntegratedDensity, 'm^-2'), None, mds.WithUnits(lineIntegratedDensityTimeArr, 's'))
                lineIntegratedDensityNode.putData(lineIntegratedDensityData)
            except Exception as e:
                logger.error(f'Error occurred while putting line integrated density for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting line integrated density for detector {(i+1):02d} into MDSplus: {e}')

            # Error in line integrated density for the detector
            try:
                
                lineIntegratedDensitySigmaNode = detNode.getNode('linedens_err')
                lineIntegratedDensitySigma = detDict['line_integrated_density_sigma']
                lineIntegratedDensitySigmaTimeArr = detDict['time_arr']
                lineIntegratedDensitySigmaData = mds.Signal(mds.WithUnits(lineIntegratedDensitySigma, 'm^-2'), None, mds.WithUnits(lineIntegratedDensitySigmaTimeArr, 's'))
                lineIntegratedDensitySigmaNode.putData(lineIntegratedDensitySigmaData)
            except Exception as e:
                logger.error(f'Error occurred while putting line integrated density sigma for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting line integrated density sigma for detector {(i+1):02d} into MDSplus: {e}')

            # Raw node name
            try:

                rawNodeNameNode = detNode.getNode('raw_node')
                rawNodeNameNode.putData(detDict['raw_node'])
            except Exception as e:
                logger.error(f'Error occurred while putting raw node name for detector {(i+1):02d} into MDSplus: {e}')
                if args.debug:
                    print(f'Error occurred while putting raw node name for detector {(i+1):02d} into MDSplus: {e}')

        except Exception as e:
            logger.error(f'Error occurred while getting node for detector {(i+1):02d}: {e}')
            if True:
                print(f'Error occurred while getting node for detector {(i+1):02d}: {e}')
            continue

        # Legacy nodes
        logger.info('putting data into legacy nodes')        
        if i < 15:

            # Raw data for the detector
            try:
                rawNode = tree.getNode(f'detector_{(i+1):02d}')
                rawSignal = detDict['raw_signal']
                rawTimeArr = detDict['time_arr']
                rawData = mds.Signal(mds.WithUnits(rawSignal, 'A'), None, mds.WithUnits(rawTimeArr, 's'))
                rawNode.putData(rawData)
            except Exception as e:
                logger.error(f'Error occurred while putting raw signal for detector {(i+1):02d} into MDSplus legacy node: {e}')
                if args.debug:
                    print(f'Error occurred while putting raw signal for detector {(i+1):02d} into MDSplus legacy node: {e}')

            # Line integrated density for the detector
            try:
                lineIntegratedDensityNode = tree.getNode(f'linedens.linedens_{(i+1):02d}')
                lineIntegratedDensity = detDict['line_integrated_density']
                lineIntegratedDensityTimeArr = detDict['time_arr']
                lineIntegratedDensityData = mds.Signal(mds.WithUnits(lineIntegratedDensity, 'm^-2'), None, mds.WithUnits(lineIntegratedDensityTimeArr, 's'))
                lineIntegratedDensityNode.putData(lineIntegratedDensityData)
            except Exception as e:
                logger.error(f'Error occurred while putting line integrated density for detector {(i+1):02d} into MDSplus legacy node: {e}')
                if args.debug:
                    print(f'Error occurred while putting line integrated density for detector {(i+1):02d} into MDSplus legacy node: {e}')

    tree.close()

    return

def dictionary_to_pkl(detDictList, args, logger):

    savename = f'/home/sanwalka/shinethru/data/{args.shotnum}_with_current_correction.pkl'

    with open(savename, 'wb') as file:
        pickle.dump(detDictList, file)

    return

def run_post_process(args, logger):

    # Load the detector dictionaries
    pickleFilePath = '/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl'
    detDictList = load_detector_dictionary(pickleFilePath, args, logger)

    # Add the raw and processed node names to each detector dictionary
    detDictList = node_names(detDictList, logger)

    # Add the resistance of the TIA resistor for each detector to the detector dictionary list
    detDictList = add_resistance(detDictList, logger)

    # Load the raw data into the detector dictionaries
    detDictList = load_raw_data(detDictList, args, logger)

    # Calculate the line integrated densities for each detector and add it to the detector dictionary list
    detDictList = calculate_line_integrated_densities(detDictList, args, logger)

    # Calculate the coupled NBI power into the plasma and put it into MDSplus
    # coupledPowerArr, coupledTimeArr = calculate_coupled_power(detDictList, args, logger)

    # Calculate the beam positions and offsets and put them into MDSplus
    # fittedTimeArr, verticalOffsets, horizontalOffsets, verticalWidths, horizontalWidths = calculate_beam_positions_offsets(detDictList, args, logger)

    # The 4 aiming detectors are treated separately, put their data into MDSplus
    # aiming_detectors(args, logger)

    # Put the data from the detector dictionaries into MDSplus
    # dictionary_to_mdsplus(detDictList, args, logger)

    # Save the list as .pkl
    dictionary_to_pkl(detDictList, args, logger)

    logger.info('Post processing completed.')

    # # Put context into the notes node
    # try:
    #     tree = mds.Tree('shinethru', args.shotnum, 'edit')
    #     notesNode = tree.getNode('notes')

    #     notesData = f'Shot {args.shotnum} processed with postproc_shinethru.py. Reference shot: {args.ref_shotnum}. \n' + \
    #                 f'The data is still put into the legacy nodes. They are- detector_XX for the raw signals and linedens.linedens_XX for the line integrated densities, where XX is the detector number from 01 to 15. \n' + \
    #                 f'If you are looking for the new nodes, they are under det_XX, where XX is the detector number from 01 to 21.  Under each det_XX node- \n' + \
    #                 f'The raw signal is under raw_signal \n' + \
    #                 f'The line integrated density is under linedens \n' + \
    #                 f'The error in line integrated density is under linedens_err \n' + \
    #                 f'The raw node from which the data is pulled is under raw_node \n' + \
    #                 f'r_impact is the radial impact parameter in the beam frame. Used to calculate things like the beam width and divergence \n' + \
    #                 f'v_impact is the vertical impact parameter in the lab frame. Used to calculate things like the radial density profile \n' + \
    #                 f'beampos is the position of the detector in cartesian coordinates in the beam frame. Used to calculate things like the beam position and width \n' + \
    #                 f'beampos_pol is the position of the detector in polar coordinates in the beam frame. \n' + \
    #                 f'machinepos is the position of the detector in cartesian coordinates in the lab frame. \n' + \
    #                 f'plasmapos_b is is a 2nd point that defines line-of-sight vector for each detector in the beam frame. \n' + \
    #                 f'plasmapos_m is is a 2nd point that defines line-of-sight vector for each detector in the lab frame. \n'
        
    #     notesNode.putData(notesData)
    #     logger.info('Successfully put context information into MDSplus notes node.')
    #     tree.close()
    # except Exception as e:
    #     logger.error(f'Error occurred while putting context information into MDSplus notes node: {e}')
    #     if args.debug:
    #         print('Error occurred while putting context information into MDSplus notes node:', e)

    # # Put the log file into MDSplus
    # try:
    #     tree = mds.Tree('shinethru', args.shotnum, 'edit')
    #     logNode = tree.getNode('log')
    #     with open(logger.handlers[0].baseFilename, 'r') as logFile:
    #         logData = logFile.read()
    #     logNode.putData(logData)
    #     logger.info('Successfully put post processing log into MDSplus.')
    #     tree.close()
    # except Exception as e:
    #     logger.error(f'Error occurred while putting post processing log into MDSplus: {e}')
    #     if args.debug:
    #         print('Error occurred while putting post processing log into MDSplus:', e)

    return

if __name__ == "__main__":

    # Initialize the logger
    logger = initialize_logger()

    # Get the arguments
    args = parseArgs(logger)
    logger.info('Arguments parsed, now running post processing script.')

    try:
        # Use the TkAgg backend for matplotlib to avoid issues with plotting in some environments
        plt.switch_backend('TkAgg')

        # Bigger font size for all plots
        plt.rcParams.update({'font.size' : 22})
    except Exception as e:
        logger.error(e)

    # Digitizer frequency
    digFreq = 1e6

    run_post_process(args, logger)
