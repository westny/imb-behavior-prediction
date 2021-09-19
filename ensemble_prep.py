import os
import pickle

import pandas as pd
import numpy as np

from copy import copy, deepcopy
from random import Random

from sklearn.preprocessing import StandardScaler, MinMaxScaler


COLUMN_NAMES = ["vehicle_class", "lane_type", "left_lane_type", "right_lane_type",
                "vy", "vx", "rel_x", "x",
                "preceding_y", "preceding_x", "preceding_vy", "preceding_vx",
                "following_y", "following_x", "following_vy", "following_vx",
                
                "left_neighbor_y", "left_neighbor_x", "left_neighbor_vy", "left_neighbor_vx",
                "left_preceding_y", "left_preceding_x", "left_preceding_vy", "left_preceding_vx",
                "left_following_y", "left_following_x", "left_following_vy", "left_following_vx",
                
                "right_neighbor_y", "right_neighbor_x", "right_neighbor_vy", "right_neighbor_vx",
                "right_preceding_y", "right_preceding_x", "right_preceding_vy", "right_preceding_vx",
                "right_following_y", "right_following_x", "right_following_vy", "right_following_vx"]

class EnsemblePrep:
    def __init__(self, args):
        self.args = args
        self.split = args.split
        self.keep = args.keep
        self.oversample = args.use_oversample
        self.downsample = args.downsample
        self.p_horizon = args.prediction_horizon
        self.seed = args.seed
        self.re_express = args.re_express_lanes
        self.random = Random(self.seed)
        self.scaling_strategy = args.scl_strategy
        
        # We define a different generator for scrambling the train set to make sure we can generate 
        # replicable results.
        self.data_scrambler = Random(self.seed)
        
        self.dL, self.dR, self.dK = self.retrieve_data()
        self.dfs = [self.dL, self.dK, self.dR]
        self.labels = [self.target(i) for i in range(3)]

        self.scaler = None
        self.training_data = None
        self.keep_data = None
        self.keep_target_len = 0
        self.training_data_standardized = None
        self.keep_data_standardized = None
        self.remainder_keep = None
        self.training_balance = ()
        self.normed_weights = [1,1,1] # Used if we want to keep some imbalance in the data
        self.validation_data = None
        self.validation_data_standardized = None
        self.validation_balance = ()

    def init_scrambler(self, seed=None):
        if seed is None:
            seed = self.seed
        self.data_scrambler = Random(seed)
    
    def get_train_val(self):
        train_data = copy(self.training_data_standardized)
        keep_data  = copy(self.keep_data_standardized)
        self.data_scrambler.shuffle(keep_data)
        train_data.extend(keep_data[:self.keep_target_len])
        return train_data, self.validation_data_standardized

    def retrieve_data(self):
        def open_file(directory, man, special=False):    
            #file = open(directory + man, 'rb')
            #df1 = pickle.load(file)
            df1 = pd.read_csv(directory + man)
            if self.re_express:
                df1 = self.re_express_lanes(df1, special)
            #file.close()
            return df1
        directory = self.args.directory + self.args.highway_directory
        files = os.listdir(directory)
        for file_name in files:
            if 'LCL' in file_name:
                dfL = open_file(directory, file_name)
            elif 'LCR' in file_name:
                dfR = open_file(directory, file_name, special=True)
            elif 'LK' in file_name:
                dfK = open_file(directory, file_name)
        return dfL, dfR, dfK
    
    @staticmethod
    def target(i):
        tg = np.zeros(3, dtype=int)
        tg[i] = 1
        return tg
    
    @staticmethod
    def filter_df(df_in, v_ids):
        return df_in[df_in["vehicle_id"].isin(v_ids)]
    
    @staticmethod
    def get_features(df_in):
        df_out = df_in[COLUMN_NAMES].copy()
        x_neil = df_out.x[df_out.x.index[0]]
        df_out.x -= x_neil
        return np.array(df_out)
    
    @staticmethod
    def re_express_lanes(df_in, special=False):
        df_in.loc[df_in.left_lane_type > 0, 'left_lane_type'] = 1
        df_in.loc[df_in.right_lane_type > 0, 'right_lane_type'] = 1
        if special:
            df_in.loc[(df_in.right_lane_type == 0) & (df_in.TTLC < 3), 'right_lane_type'] = 1
        return df_in

    @staticmethod
    def compute_balance(data, data_name='Input'):
        h = np.zeros(3, dtype=int)
        for td in data:
            h += td[1]
        tot = h.sum()
        
        n_l = h[0]
        n_k = h[1]
        n_r = h[2]
        
        l_percent = round(n_l/tot * 100., 1)
        k_percent = round(n_k/tot * 100., 1)
        r_percent = round(n_r/tot * 100., 1)
        print(f' {data_name} split --> LCL: {n_l} ({l_percent})%, LK: {n_k} ({k_percent})%, LCR: {n_r} ({r_percent})% \n')
        return (l_percent/100., k_percent/100., r_percent/100.)
    
    def compute_split(self, split=None):
        if split is None:
            split = self.split
        
        def get_ids(df_in):
            ids = pd.unique(df_in.vehicle_id).astype(dtype=int)
            self.random.shuffle(ids)
            return ids
        
        lcl_ids = get_ids(self.dL)
        lcr_ids = get_ids(self.dR)
        lk_ids = get_ids(self.dK)
        
        n_split = min(len(lcl_ids), len(lcr_ids), len(lk_ids))
        t_index = int(n_split*split)
        training_ids = {'LCL': [], 'LK': [],'LCR': []}
        validation_ids = {'LCL': [], 'LK': [],'LCR': []}
        for l, key in zip([lcl_ids, lk_ids, lcr_ids], ['LCL', 'LK', 'LCR']):
            training_ids[key].extend(l[:t_index])
            validation_ids[key].extend(l[t_index:n_split])
        return training_ids, validation_ids
    
    def extract_data(self, vehicle_id_dict, data_set='training'):
        lc_data = []
        lk_data = []
          
        for key, dataf, label in zip(list(vehicle_id_dict.keys()),
                                             self.dfs,
                                             self.labels):
            print(f'Adding {key} labels.', end='')
            vehicle_ids = vehicle_id_dict[key]
            df1 = self.filter_df(dataf, vehicle_ids)
            for v_id in vehicle_ids:
                print('.', end='')
                df2 = df1[df1.vehicle_id == v_id]
                manuevers = pd.unique(df2.maneuver_id).astype(int)
                for manuever in manuevers:
                    if self.random.random() > self.keep:
                        continue
                    df3 = df2[df2.maneuver_id == manuever]
                    ttlc = df3.TTLC.iloc[0]
                    if key != 'LK':
                        if ttlc > self.p_horizon:
                            break
                        if self.downsample:
                            ttlc = round(ttlc, 1)
                            if ttlc == 0.1 or ttlc % 0.5 == 0:
                                pass
                            else:
                                continue
                    feature_history = self.get_features(df3)
                    observation = (feature_history, label, ttlc)
                    if key == 'LK':
                        lk_data.append(observation)
                    else:
                        lc_data.append(observation)
                        if self.oversample and data_set=='Training':
                            resample_prob = self.random.random()
                            if resample_prob > 0.25:
                                times = self.random.randint(1, 3)
                                for _ in range(0, times):
                                    lc_data.append(observation)
            print('\n')

        return lc_data, lk_data
    
    def compute_scaler(self):
        try:
            assert(self.training_data is not None)
        except AssertionError:
            print('Scaler can not be determined without existing training data.. Aborting.')
        else:
            data = []
            
            train_data = deepcopy(self.training_data)
            keep_data = deepcopy(self.keep_data)
            self.random.shuffle(keep_data)
            # The scaler is computed using a subset of the LKs. This is so we reduce their overall impact on the data.
            keep_data = keep_data[:self.keep_target_len]
            train_data.extend(keep_data)
            
            for td in train_data:
                trajectory = td[0][:, 4:]
                data.append(trajectory)
            data = np.concatenate(data, axis=0)
            if self.scaling_strategy == 'std':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            scaler.fit(data)
            self.scaler = scaler
    
    def standardize_data(self):
        try:
            assert(self.scaler is not None)
        except AssertionError:
            print('Data cannot be standarized without a scaler.. Aborting.')
        else:
            print('Initializating standard scaling of features...')
            self.training_data_standardized = deepcopy(self.training_data)
            self.keep_data_standardized = deepcopy(self.keep_data) 
            self.validation_data_standardized = deepcopy(self.validation_data)
            # standarize training data
            for k, td in enumerate(self.training_data):
                trajectory = td[0][:, 4:]
                trajectory_standard = self.scaler.transform(trajectory)
                self.training_data_standardized[k][0][:, 4:] = trajectory_standard

            for k, td in enumerate(self.keep_data):
                trajectory = td[0][:, 4:]
                trajectory_standard = self.scaler.transform(trajectory)
                self.keep_data_standardized[k][0][:, 4:] = trajectory_standard
            # standarize validation data
            for k, vd in enumerate(self.validation_data):
                trajectory = vd[0][:, 4:]
                trajectory_standard = self.scaler.transform(trajectory)
                self.validation_data_standardized[k][0][:, 4:] = trajectory_standard
            print('Done!')
            
    def compute_train_val(self):
        training_ids, validation_ids = self.compute_split()
        print('Setting up training data:')
        lc_train, lk_train = self.extract_data(training_ids, data_set='training')
        self.training_data = copy(lc_train)
        self.keep_data = copy(lk_train)
        self.keep_target_len = int(len(lc_train)/2)
        
        ensemble_ex = copy(lc_train)
        ensemble_ex.extend(lk_train[:self.keep_target_len])
        self.training_balance = self.compute_balance(ensemble_ex, 'Bag (Training)')
        
        print('Setting up validation data:')
        lc_val, lk_val = self.extract_data(validation_ids, data_set='validation')
        self.validation_data = copy(lc_val)
        target_len = int(len(lc_val)/2)
        self.random.shuffle(lk_val)
        self.validation_data.extend(lk_val[:target_len])
        
        v_data = copy(self.validation_data)
        
        self.validation_balance = self.compute_balance(v_data, 'Validation')

    def main(self):
        self.compute_train_val()
        self.compute_scaler()
        self.standardize_data()


