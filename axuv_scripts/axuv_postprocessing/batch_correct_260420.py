# Corrects the AXUV nodes for 260417-260419.  


from glob import glob
from csv_to_mds_struct import csv_to_mds_struct
import os

def get_shot_range():
    flist = glob('/mnt/n/data/26/04/1[7-9]/top/*.tree')
    flist = glob('/mnt/n/data/26/04/17/top/*.tree')
    shots = []
    for f in flist:
        shots.append(int(f.split('_')[-1].replace('.tree','')))
    shots.sort()
    return shots

if __name__ == '__main__':
    shot_list = get_shot_range()
    for shot in shot_list:
        print(f'Working on shot {shot}')
        csv_to_mds_struct('WHAM',source_file='AXUV_FIELDS.csv',shot=shot,overwrite=False)
        cmd = f'python3 ~/post_processing/axuv.py --shotnum {shot}'
        print(cmd)
        os.system(cmd)
        
