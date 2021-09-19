import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy
from utils import *



parser = argparse.ArgumentParser(description='Extractor arguments')
parser.add_argument('--feet-to-meter', type=float, default=0.3048,
                    help='convert from feet to meters')
parser.add_argument('--data-folder', type=str, default="data",
                    help='folder where data should be stored')
parser.add_argument('--raw-data', type=str, default="data_sets",
                    help='folder where .csv files are stored')
parser.add_argument('--dry-run', type=bool, default=False,
                    help='If true, no data is stored.')
parser.add_argument('--maximum-entries', type=int, default=5000,
                    help='How many entries per store')
e_args, _ = parser.parse_known_args()


def pre_check():
    import os
    files = os.listdir()
    try:
        f = [x for x in files if f'{e_args.raw_data}' in x and '.zip' not in x][0]
    except IndexError:
        print(f'Could not find folder \'{e_args.raw_data}\'. Make sure the data is unzipped and available within the given folder name.')
        print('If unzipped and contained within another folder -> specify a new location in the script.')
        return False
    return True


def prepare_dataframe(file, highway):
    df = pd.read_csv(file)
    #df = crop_and_convert(df, highway) # Commented out if using 'FIXED' version of .csv files
    return df


def crop_and_convert(df, highway):
    if highway['name'] != 'US101':
        df = df.rename(columns={'Time_Headway': 'Time_Hdwy'})
        df = df.rename(columns={'Space_Headway': 'Space_Hdwy'})
    else:
        df = df.rename(columns={'Preceeding': 'Preceding'})
    
    df = df.drop(columns=['Global_Time', 'Total_Frames', 'Global_X', 'Global_Y', 'Time_Hdwy'])
    df.loc[:, 'Space_Hdwy'] = df['Space_Hdwy'] * e_args.feet_to_meeter
    df.loc[:, 'Local_X'] = df['Local_X'] * e_args.feet_to_meeter
    df.loc[:, 'Local_Y'] = df['Local_Y'] * e_args.feet_to_meeter
    df.loc[:, 'v_Length'] = df['v_Length'] * e_args.feet_to_meeter
    df.loc[:, 'v_Vel'] = df['v_Vel'] * e_args.feet_to_meeter
    df.loc[:, 'v_Acc'] = df['v_Acc'] * e_args.feet_to_meeter
    df.loc[:, 'v_Width'] = df['v_Width'] * e_args.feet_to_meeter
    return df


def create_base_dataframe():
    return pd.DataFrame(columns = column_names)


def get_lane_type(lane, opts_dict):
    return int(opts_dict['lane_types'][lane])


def get_vehicle_data(v_id, df_in):
    return df_in[df_in.Vehicle_ID == v_id]


def relative_lane_position(lane_id, local_x, bounds):
    """ Uses lane centers """
    lane_center = bounds[lane_id - 1] # assumed index
    relative_pos = lane_center - local_x
    return relative_pos # array of relative positions


def relative_lane_position2(lane_id, local_x, bounds):
    """ Uses lane borders """
    x1 = bounds[lane_id - 1]
    x2 = bounds[lane_id]
    x = local_x
    return ((x - x1) / (x2 - x1) - 1/2) * 2


def get_traj_feats(df_in):
    """Converts entires to numpy arrays. """
    x = df_in.Local_X.to_numpy()
    y = df_in.Local_Y.to_numpy()
    vy = df_in.v_Vel.to_numpy()
    vx = np.diff(x)/(0.1)
    vx = np.insert(vx, 0, vx[0])
    return [y, x, vy, vx]


def get_categorical_feats(df_in, highway_props):
    """ Retrieves categorical vehicle features 
    from a given dataframe."""
    lane = df_in.Lane_ID.iloc[-1]
    y_end = df_in.Local_Y.iloc[-1]
    v_class = int(df_in.v_Class.iloc[0])
    lane_l = lane - 1
    if lane == highway_props['rightmost']:
        if highway_props['aux_borders'][0] < y_end < highway_props['aux_borders'][-1]:
            lane_r =  highway_props['auxiliary']
        else:
            lane_r = 0
    else:
        lane_r = lane + 1
    cat_feats = [v_class]
    for l in [lane, lane_l, lane_r]:
        cat_feats.append(get_lane_type(l, highway_props))
    return cat_feats



def get_vehicle_features(df_in, id_dict, highway_props, LK=True):
    """ Retrieves vehicle features from given an vehicle ID."""
    
    def check_border(y):
        return y < highway_props['aux_borders'][0] or y > highway_props['aux_borders'][1]
    
    def create_synthetic_vehicle(v_key, tv_y, tv_x, tv_vy, tv_vx):
        y = copy(tv_y)
        x = np.ones(SEGMENT_LEN)*np.median(tv_x)
        vy = copy(tv_vy)
        vx = copy(tv_vx)
        if 'p' in v_key:
            y += SENSOR_RANGE
        elif 'f' in v_key:
            y -= 35
        if 'l' in v_key:
            x -= highway_props['lane_width']
        elif 'r' in v_key:
            x += highway_props['lane_width']
        return y, x, vy, vx
    
    data_arr = np.zeros((int(SEGMENT_LEN/2), CATEGORICAL_FEATS+TV_FEATS+SV_FEATS*N_TVS))
    ones = np.ones((data_arr.shape[0], CATEGORICAL_FEATS), dtype=int)
    sv_offset = 0
    for v_key in list(id_dict.keys()):
        if v_key == 'tv':
            v_id = id_dict[v_key]
            v_df = get_vehicle_data(v_id, df_in)
            lane = v_df.Lane_ID.to_numpy()

            tv_y, tv_x, tv_vy, tv_vx = get_traj_feats(v_df)
            rel_x = relative_lane_position2(lane, tv_x, highway_props['lane_borders'])
            y = tv_y
            x = tv_x
            cat_feats = get_categorical_feats(v_df, highway_props)
            data_arr[:, 0:CATEGORICAL_FEATS] = ones*cat_feats
            for k, val in enumerate([y, x, tv_vy, tv_vx, rel_x]):
                data_arr[:, CATEGORICAL_FEATS+k] = chop(val)
        else:
            v_id = id_dict[v_key]

            if v_id == 0: # Create a synthetic obstacle
                y, x, vy, vx = create_synthetic_vehicle(v_key, tv_y, tv_x, tv_vy, tv_vx)
            else:
                v_df = get_vehicle_data(v_id, df_in)
                # If we have an obstacle but to little data, create a synthetic
                if len(v_df.index) < 5:
                    y, x, vy, vx = create_synthetic_vehicle(v_key, tv_y, tv_x, tv_vy, tv_vx)
                else:    
                    y, x, vy, vx = get_traj_feats(v_df)

            if 'p' in v_key:
                yd = get_relative(tv_y, y, value=20)
            elif 'f' in v_key:
                yd = get_relative(tv_y, y, value=-20)
            else:
                yd = get_relative(tv_y, y)
            if 'l' in v_key:
                xd = get_relative(tv_x, x, value=-highway_props['lane_width'])
            elif 'r' in v_key:
                xd = get_relative(tv_x, x, value=highway_props['lane_width'])
            else:
                xd = get_relative(tv_x, x)

            vyd = get_relative(tv_vy, vy)
            vxd = get_relative(tv_vx, vx)

            offset = CATEGORICAL_FEATS+TV_FEATS + sv_offset
            for k, val in enumerate([yd, xd, vyd, vxd]):
                data_arr[:, offset+k] = chop(val)
            sv_offset+=4
    return data_arr



def get_relative(tv, sv, value=0):
    '''Calulate relative difference between tv and sv.
       Difference in array length is accounted for by creating
       an array of set length with a placeholder value.'''
    len_diff = SEGMENT_LEN - sv.shape[0]
    ph = np.ones(SEGMENT_LEN)*value
    ph[len_diff:] = sv
    ph[len_diff:] -= tv[len_diff:]
    return ph


def decim(arr):
    '''Downsample signal to half its length using
       a first order Chebyshev filter'''
    from scipy.signal import decimate
    return decimate(arr, 2)


def chop(arr):
    '''Downsample signal to half its length by intermediate removal.
       Here we make sure to always keep the last element of an array.
    '''
    return arr[::-2][::-1]


def determine_sv(tv_id, tv_lane, tv_y, df_in, index, LK=True):
    '''Determine the active SVs within the current frame.
       Ids are saved in a dictionary for later use. Undetermined/missing
       vehicles are set to zero.'''

    sv_dict = {'tv': tv_id,'p':0, 'f':0,
               'l_n':0, 'l_p':0, 'l_f':0,
               'r_n':0, 'r_p':0, 'r_f':0}
    for i in index:
        sv_serie = df_in.loc[i, :]
        sv_id = int(sv_serie.Vehicle_ID)
        if sv_id == tv_id:
            continue
        sv_lane = int(sv_serie.Lane_ID)
        sv_y = sv_serie.Local_Y
        if sv_lane == tv_lane:
            if sv_y > tv_y and sv_dict['p'] == 0:
                sv_dict['p'] = sv_id
            elif sv_y < tv_y and sv_dict['f'] == 0:
                sv_dict['f'] = sv_id
        elif sv_lane == tv_lane - 1:
            if sv_dict['l_n'] == 0:
                sv_dict['l_n'] = sv_id
            elif sv_y < tv_y and sv_dict['l_f'] == 0:
                sv_dict['l_f'] = sv_id
            elif sv_y > tv_y and sv_dict['l_p'] == 0:
                sv_dict['l_p'] =  sv_id
        elif sv_lane == tv_lane + 1:
            if sv_dict['r_n'] == 0:
                sv_dict['r_n'] = sv_id
            elif sv_y < tv_y and sv_dict['r_f'] == 0:
                sv_dict['r_f'] = sv_id
            elif sv_y > tv_y and sv_dict['r_p'] == 0:
                sv_dict['r_p'] = sv_id
        if 0 not in list(sv_dict.values()): # break early if we've already filled all slots
            break
    return sv_dict


def get_local_df(df_in, eof, tv_lane):
    '''Returns a local dataframe only containing rows corresponding
       to a given frame and (three) lanes.'''
    df_out = df_in[(df_in.Frame_ID == eof) & (
                   (df_in.Lane_ID == tv_lane) |
                   (df_in.Lane_ID == tv_lane - 1) |
                   (df_in.Lane_ID == tv_lane + 1))]
    return df_out


def insert_into_base(df_in, man_id, v_id, time_to_lc, arr):
    ''' populates dataframe with feature arrays.'''
    l = arr.shape[0]
    ones = np.ones((l, 1), dtype=int)
    
    man_id_arr = man_id*ones
    v_id_arr = v_id*ones
    time_to_lc_arr = time_to_lc*ones
    
    data = np.concatenate((man_id_arr, v_id_arr, time_to_lc_arr, arr), axis=1)
    
    for col, entry in zip(column_names, data.T):
        df_in[col] = entry
    return df_in


def save_data(data, subfolder, file_name):
    """ Store extracted data in using pickle """
    if not e_args.dry_run:
        file = open(f'{e_args.data_folder}'+'/'+subfolder+'/'+file_name+'.p', 'wb')
        pickle.dump(data, file)
        file.close()


def get_datetime():
    """ Used to distinguish stored data """
    from datetime import date, datetime
    today = date.today()
    now = datetime.now()

    d1 = today.strftime("-%Y-%m-%d")
    current_time = now.strftime("%H-%M-%S")
    name_dir = d1 + '--' +current_time
    return name_dir


def store_intermediate(lk, lcl, lcr, recording, force_save=False):
    """ Data is stored intermittently to keep down running memory
        requirements.
    """
    
    dT = get_datetime()
    name = recording + dT

    highway_name = recording.split('-')[0]
    sub_folder = recording.split(highway_name+'-')[1]
    folders = ['lane-keep/', 'lane-change-left/', 'lane-change-right/']
    maneuvers = ['LK-', 'LCL-', 'LCR-']
    data = [lk, lcl, lcr]
    
    for k, (l, folder, maneuver) in enumerate(zip(data, folders, maneuvers)):
        if len(l) > e_args.maximum_entries or force_save:
            try:
                dfx = pd.concat(l, ignore_index=True)
            except ValueError:
                ...
            else:
                save_data(dfx, folder + sub_folder, maneuver+name)
                data[k] = []
    return data


def data_extractor(df_ori, df_tv, tv_lane, man_id, v_id, highway, window_length=100, LK=True):
    """ Main extraction function that retrieves TV and SV data for
        every recorded trajectory.
    """
    storage = []
    if LK:
        TTLC = 6
    else:
        TTLC = 0.1
    frames = df_tv.Frame_ID.to_numpy()
    for k in range(0, len(frames))[::-1]:
        if LK: # If we got a Lane-keep maneuver, we use sliding window on first iter, o.w. we start it on second
            k -= window_length
        if k - SEGMENT_LEN < 0: # If no more instances are left, we're done.
            break
        try:
            frame_end = frames[k] # final frame of input data
            segment = frames[k-SEGMENT_LEN+1:k+1]  
        except IndexError:
            break  
        tv_y = df_tv[df_tv.Frame_ID == frame_end].Local_Y.iloc[0]

        df_segment = df_ori[(df_ori.Frame_ID >= segment[0]) & (df_ori.Frame_ID <= segment[-1])] # Actual df to collect SV data

        vy_test = get_vehicle_data(v_id, df_segment).v_Vel.to_numpy()
        if len(vy_test) < 40: # reality check
            ...
        else:
            df_local = get_local_df(df_ori, frame_end, tv_lane)
            distance_to_sv = abs(df_local.Local_Y - tv_y).sort_values(ascending=True)

            # Select SVs from within 100m radius of the TV
            index = ((distance_to_sv < SENSOR_RANGE)[distance_to_sv<SENSOR_RANGE])
            index = list(index.index)
            sv_dict = determine_sv(v_id, tv_lane, tv_y, df_local, index, LK)

            data = get_vehicle_features(df_segment, sv_dict, highway, LK)
            df_new = create_base_dataframe()
            man_id += 1
            df_new = insert_into_base(df_new, man_id, v_id, TTLC, data)
            storage.append(df_new)

        if not LK:
            TTLC += 0.1
            TTLC = round(TTLC, 1)
            if TTLC >= 5.6:
                break
    return storage, man_id


def main(df, highway_props, recording, *args):
    """ Ovehead function to retrieve vehicle and corresponding dataframes for
        data extraction.
    """
    LK = []; LCL= []; LCR= []
    k_i = l_i = r_i = 0
    if args:
        k_i, l_i, r_i = args
            
    kill_session = False
    LK_WINDOW = 100
    LK_WINDOW_x2 = 200

    vehicle_ids = list(pd.unique(df.Vehicle_ID))

    ii = range(0, len(vehicle_ids))
    ii = tqdm(ii)

    for i in ii:
        LK, LCL, LCR = store_intermediate(LK, LCL, LCR, recording) # Store intermediate data to keep memory req low
        try:
            vehicle_id = vehicle_ids[i]
            df1 = df[df.Vehicle_ID == vehicle_id]

            if df1.loc[df1.index[0], :].Preceding == 0: # Remove instances from beginning of recordings
                continue

            visited_lanes = pd.unique(df1.Lane_ID).astype(int)
            if visited_lanes.shape[0] == 1:
                current_lane = visited_lanes[0]
                if current_lane not in highway_props['interim_lanes']: # Just to make sure...
                    # LANE-KEEP
                    l, k_i = data_extractor(df, df1, current_lane, k_i, vehicle_id, highway_props, LK_WINDOW_x2)
                    LK.extend(l)
                    del l
            else:
                for k, current_lane in enumerate(visited_lanes):
                    if current_lane not in highway_props['interim_lanes']:
                        df2 = df1[df1.Lane_ID == current_lane]
                        
                        # Check consistency
                        frame_diff = np.diff(df2.Frame_ID) - 1
                        try:
                            breakpoint = np.nonzero(frame_diff)[0][0]
                        except IndexError:
                            ...
                        else:
                            # In this case the vehicle visits the same lane on different occassions
                            # We need to adjust the df (or disregard vehicle completely)
                            break
                            #break_frame = df2.Frame_ID.iloc[breakpoint+1]
                            #df2 = df2[df2.Frame_ID < break_frame]
                        
                        try:
                            next_lane = visited_lanes[k+1]
                        except IndexError:
                            # Lane-keep
                            l, k_i = data_extractor(df, df2, current_lane, k_i, vehicle_id, highway_props, LK_WINDOW)
                            LK.extend(l)
                            del l
                        else:
                            
                            if next_lane in highway_props['interim_lanes']:
                                continue
                            
                            # If there is too little data pertaining to a specific lane, we might need to extend with previous visits.
                            # This is so we don't miss consecutive LCs
                            if len(df2.index) < 40:
                                if k == 0:
                                    continue
                                else:
                                    j = k - 1
                                    while j > -1:
                                        previous_lane = visited_lanes[j]
                                        if previous_lane not in highway_props['interim_lanes']:
                                            df3 = df1[df1.Lane_ID == previous_lane]
                                            df2 = pd.concat([df2, df3])
                                        else:
                                            break
                                        j-=1
                                    if len(df2.index) < 40:
                                        continue

                            # We're only interested in a max pred. horizon of 4s (6s).
                            # This means we can remove some data.
                            
                            df2 = df2.tail(120)
                            
                            # Left lane-change
                            if next_lane < current_lane:
                                l, l_i = data_extractor(df, df2, current_lane, l_i, vehicle_id, highway_props, LK=False)
                                LCL.extend(l)
                                del l
                            # Right lane-change
                            else:
                                l, r_i = data_extractor(df, df2, current_lane, r_i, vehicle_id, highway_props, LK=False)
                                LCR.extend(l)
                                del l
        except KeyboardInterrupt:
            print("Interrupted early")
            #kill_session = True
            break
    store_intermediate(LK, LCL, LCR, recording, force_save=True)
    return (k_i, l_i, r_i), kill_session


def run(highway_set, highway_dict):
    count = (0, 0, 0)
    print('Please allow some hours for data extraction to complete...\n')
    end = False
    while not end:
        try:
            data_set = next(highway_set)
        except:
            break
            print('\nDone!\n')
        else:
            print("Current data set is: {}".format(data_set))
            recording = data_set.split('/')[3]

        df = prepare_dataframe(data_set, highway_dict)
        count, end = main(df, highway_dict, highway_dict['name'] + '-' + recording, *count)
    return count



if __name__ == "__main__":
    ok = pre_check()
    if ok:
        US101_1 = f'./{e_args.raw_data}/us101/0750am-0805am/trajectories-0750am-0805am-FIXED.csv'
        US101_2 = f'./{e_args.raw_data}/us101/0805am-0820am/trajectories-0805am-0820am-FIXED.csv'
        US101_3 = f'./{e_args.raw_data}/us101/0820am-0835am/trajectories-0820am-0835am-FIXED.csv'

        US101_set_generator = (data for data in [US101_1, US101_2, US101_3])
        c_us101 = run(US101_set_generator, US101)

        I80_1 = f'./{e_args.raw_data}/i80/0400pm-0415pm/trajectories-0400pm-0415pm-FIXED.csv'
        I80_2 = f'./{e_args.raw_data}/i80/0500pm-0515pm/trajectories-0500pm-0515pm-FIXED.csv'
        I80_3 = f'./{e_args.raw_data}/i80/0515pm-0530pm/trajectories-0515pm-0530pm-FIXED.csv'

        I80_set_generator = (data for data in [I80_1, I80_2, I80_3])
        c_i80 = run(I80_set_generator, I80)
