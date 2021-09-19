import os
import pickle
import random
import shutil

import numpy as np
import pandas as pd



HIGHWAY = ('US101', 'I80')

MAIN_DIR = './data/'
FOLDERS = ['lane-change-left/', 'lane-change-right/', 'lane-keep/']
TIMES = ['0750am-0805am', '0805am-0820am', '0820am-0835am',
         '0400pm-0415pm', '0500pm-0515pm', '0515pm-0530pm']


def combine_intermediate_recording(save=True):
    first = True
    for i, folder in enumerate(FOLDERS):
        current_dir = MAIN_DIR + folder
        for time in TIMES:
            d = current_dir + time
            files = os.listdir(d)
            if len(files) == 1:
                # This is to make sure we don't execute this function more than once.
                continue
            if first:
                print('... intermediate recordings are merged')
                first = False
            l = []
            processed_files = []
            for k, file_name in enumerate(files):
                if k == 0:
                    header_name = file_name.split('2021')[0][:-1] + '.p'

                if any(hw in file_name for hw in HIGHWAY):
                    f = d + '/' + file_name
                    file = open(f, 'rb')
                    df1 = pickle.load(file)
                    file.close()
                    l.append(df1)
                    processed_files.append(f)
            try:
                df = pd.concat(l, ignore_index=True)
            except ValueError as e:
                print("Value error: {0}".format(e))
                break
            else:
                if save:
                    file = open(d + '/' + header_name, 'wb')
                    pickle.dump(df, file)
                    file.close()

                    for p_f in processed_files:
                        os.remove(p_f)


def combine_maneuvers():
    us101 = 'am'
    i80 = 'pm'
    
    def open_file(file_name):
        file = open(file_name, 'rb')
        df1 = pickle.load(file)
        file.close()
        #The amount of LKs is to large, we need to reduce it
        if 'LK' in file_name:
            v_ids = np.array(pd.unique(df1.vehicle_id), dtype=int)
            random.shuffle(v_ids)
            exclude_fraction = 30  # In the paper, this is set to 2 which might be demanding for some setups.
            keep_ids = list(v_ids[:int(len(v_ids)/exclude_fraction)])  # keep a portion of all LKs
            df1 = df1[df1["vehicle_id"].isin(keep_ids)]
        return df1
    
    # check if files already exists
    files = [f for f in os.listdir(MAIN_DIR) if '.csv' in f]
    if files:
        val = input("\nOne or more .csv files have been detected and need to be removed before proceeding."
                    " Would you like to erase them? (Y|N) \n").lower()
        if 'y' in val or 'yes' in val:
            print('\nProceeding...')
            for file in files:
                os.remove(MAIN_DIR + file)
        else:
            print('\nAborting...')
            raise RuntimeError
    
    for folder in FOLDERS:
        current_dir = MAIN_DIR + folder
        for highway_recording in [us101, i80]:
            add = 3010
            times = [x for x in os.listdir(current_dir) if highway_recording in x and
                     os.path.isdir(current_dir + '/' + x)]
            files = [current_dir + time + '/' + os.listdir(current_dir + time)[0] for time in times]
            #file_sizes = [os.path.getsize(file) for file in files]
            tmp = files[0].split('/')[-1].split('-')[0:2]
            family_name = '-'.join(tmp) + '.csv'
            
            header = True
            for file in files:
                df1 = open_file(file)
                if not header:
                    df1.vehicle_id += add
                    add *= 2
                df1.to_csv(os.path.join(MAIN_DIR, family_name),
                            header=header, mode='a', index=False, chunksize=100000)
                header = False


def combine_highways(dir_name=None):
    files = os.listdir(MAIN_DIR)
    if dir_name is None:
        dir_name = MAIN_DIR + 'final'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Directory " , dir_name ,  " Created ")
        
    file_types = ['LCL', 'LCR', 'LK']

    add = 3015*3
    for file_type in file_types:
        active_files = [f for f in files if file_type in f if '.csv' in f]
        file_sizes = [os.path.getsize(MAIN_DIR + act_file) for act_file in active_files]
        ind_max_size = np.argmax(file_sizes)
        ind_min_size = np.argmin(file_sizes)
        
        # copy the largest file to the final dir
        shutil.copyfile(MAIN_DIR + active_files[ind_max_size], dir_name + '/' + file_type + '.csv')
        
        # read in the smaller file in chunks and write to the larger that has just been moved
        chunks = pd.read_csv(MAIN_DIR + active_files[ind_min_size], chunksize=100000)
        for chunk in chunks:
            chunk.vehicle_id += add
            chunk.to_csv(os.path.join(dir_name + '/', file_type + '.csv'),
                    header=False, mode='a', index=False, chunksize=100000)


def organize(save=True):
    print('Organization of data started... ')
    random.seed(99)
    combine_intermediate_recording()
    print('... similar maneuvers from same highway are merged')
    try:
        combine_maneuvers()
    except RuntimeError:
        pass
    else:
        print('... Finally data from both highways are merged')
        dir_name = MAIN_DIR + 'final'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Directory " , dir_name ,  " Created ")
        combine_highways()
        print(f"\nData organizing is complete. Files for training is available in folder \'{dir_name}\' in csv format.")

if __name__ == "__main__":
    organize()