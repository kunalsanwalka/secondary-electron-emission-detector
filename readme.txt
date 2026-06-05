The codes in this directory are used to compare synthetic SEE signals vs. those measured from real data.
There are a lot of helper functions but as a user, you should only need to use measured_vs_predicted_see_density.py

From the command line, it is called via-
$ python3 measured_vs_predicted_see_density.py -s [SHOTNUM]

Here, SHOTNUM is the shot number being used to get the experimental data. 
This means that this code needs to be run in a location where you have access to the WHAM MDSplus tree structure.

NOTE: Ideally, this code will pull the correct lookup table based on the gas type set in the 'NBI.GAS_TYPE' node. However, this is often WRONG.
It can be manually changed in the calculate_line_densities() function in the measures_see_density.py script.

In measured_vs_predicted_see_density.py, once the following lines are run, the data is present in the dictionary array-

# Load the SEE detector dictionary
pickleFilePath = '/home/sanwalka/shinethru/lookup_tables/see_detector_dictionary.pkl'
detDictList = load_detector_dictionary(pickleFilePath)

# Calculate the prediction from pleiades
interpFunc = load_density_profile(pleiadesFilePath, makeplot=False)
detDictList = synthetic_see_detector(detDictList, interpFunc, makeplot=False)

# Calculate the measured line-integrated densities
_, _, _ = calculate_line_densities(args, detDictList)

At this point, the object detDictList has the information stored in the following manner-
detDictList[i]['line_integrated_density'] = line integrated density [m^-2] vs time
detDictList[i]['line_integrated_density_sigma'] = error bars associated with the measurement
detDictList[i]['time'] = corresponding time array [ms]
detDictList[i]['predicted_see_density'] = predicted line integrated density [m^-2] from synthetic diagnostic implemented on pleiades output

There are also some other terms in the dictionary for each detector that help with plotting etc. 
You can look further into measured_vs_predicted_see_density.py to see how the plot is made.