import numpy as np

SENSOR_RANGE = 100 # Perceived distance [m] at which SVs can be detected
SEGMENT_LEN = 40 # Samples
CATEGORICAL_FEATS = 4 # lane types, vehicle class
TV_FEATS = 5 #  y, x, vy, vx, rel_x
SV_FEATS = 4 # yd, xd, vyd, vxd
N_TVS = 8 # Number of target vehicles


US101 = {'name': 'US101',
         'lane_width': 3.35,
         'lane_centers': np.array([2.23, 5.58, 8.95, 12.26, 15.71, 18.60]),
         'lane_borders': np.array([0.55, 3.90, 7.26, 10.64, 13.88, 17.54, 19.67]),
         'lane_types': {0:0, 1:3, 2:1, 3:1, 4:1, 5:2, 6:4, 7:5, 8:6},
         'interim_lanes': (7,8),
         'leftmost': 1,
         'rightmost': 5,
         'auxiliary': 6,
         'aux_borders': np.array([187.58-30, 411.49+30])} # (roughly) when auxillary lane (#6 starts and ends)
         
I80 = {'name': 'I80',
       'lane_width': 3.7,
       'lane_centers': np.array([1.88, 5.54, 9.16, 12.82, 16.54, 20.52]),
       'lane_borders': np.array([0.03, 3.73, 7.35, 10.97, 14.68, 18.40, 22.64]),
       'lane_types': {0:0, 1:3, 2:1, 3:1, 4:1, 5:1, 6:2, 7:5},
       'interim_lanes': (7,),
       'leftmost': 1,
       'rightmost': 6,
       'auxiliary': 7,
       'aux_borders': np.array([106.61-30, 213.24+30])} # (roughly) when vehicles can merge from #7 onto #6

column_names = ["maneuver_id", "vehicle_id", "TTLC", "vehicle_class", "lane_type", "left_lane_type", "right_lane_type",
                "y", "x", "vy", "vx", 'rel_x',
                
                "preceding_y", "preceding_x", "preceding_vy", "preceding_vx",
                "following_y", "following_x", "following_vy", "following_vx",
                
                "left_neighbor_y", "left_neighbor_x", "left_neighbor_vy", "left_neighbor_vx",
                "left_preceding_y", "left_preceding_x", "left_preceding_vy", "left_preceding_vx",
                "left_following_y", "left_following_x", "left_following_vy", "left_following_vx",
                
                "right_neighbor_y", "right_neighbor_x", "right_neighbor_vy", "right_neighbor_vx",
                "right_preceding_y", "right_preceding_x", "right_preceding_vy", "right_preceding_vx",
                "right_following_y", "right_following_x", "right_following_vy", "right_following_vx"]