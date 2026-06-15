The codes in this directory are used to compare synthetic SEE signals vs. those measured from real data.
There are a lot of helper functions but as a user, you should only need to use /compare_to_simulation/experimental_data.py

From the command line, it is called via-
:~/compare_to_simulation$ python3 experimental_data.py -s [SHOTNUM]

Here, SHOTNUM is the shot number being used to get the experimental data. 
This means that this code needs to be run in a location where you have access to the WHAM MDSplus tree structure.

NOTE: Ideally, this code will pull the correct lookup table based on the gas type set in the 'NBI.GAS_TYPE' node. However, this is often WRONG.
      It can be manually changed in the calculate_line_integrated_densities() function in the experimental_data.py script.

NOTE: This code uses 3 lookup tables. They are part of the git repo. The tables are-
      1. shine_thru_table.npz - Convert signal to line integrated density for H
      2. shine_thru_table_d.npz - Convert signal to line integrated density for D
      3. see_detector_dictionary.pkl - Information for each detector (position, raw signal, etc.)

      On first import, the code may break if the file paths are not set correctly. User should change those file paths.

In experimental_data.py, the run_post_process function is what orchestrates everything. Once it is done running, the data is returned as detDictList.

detDictList is a list of python dictionaries that correspond to a detector. In each dictionary, there are a lot of keys but the ones most relevant for equilibrium reconstruction are-
1. line_integrated_density - Line integrated plasma density along the line-of-sight (LOS) for each detector in m^-2
2. line_integrated_density_sigma - Associated error bars with line_integrated_density
3. time_arr_slow - Associated time array. The detector are NOT all on the same time array. Interpolation may be needed. The density is calculated at 10kHz.
4. machine_pos - (x,y,z) coordinates of the detector in mm. Here, +ve x is WEST, +ve y is UPWARDS and +ve z is NORTH
5. machine_sight_point - (x,y,z) points that define the 2nd point for the detector LOS in mm. Combined with machine_pos, these 2 points can be used to generate the detector LOS.