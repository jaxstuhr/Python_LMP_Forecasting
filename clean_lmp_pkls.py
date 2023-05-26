# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:50:17 2023

@author: jzs88

READING AND CLEANING LMPs FROM GUILLES API
"""

import pickle as pkl
import os
import numpy as np
import pandas as pd

lmp_locs_crosswalk_raw = pd.read_csv('C:/Users/jzs88/Desktop/Python_LMP_Forecasting/data_outputs/node_locations_final.csv', delimiter=',')
lmp_locs_crosswalk = lmp_locs_crosswalk_raw.drop(columns = ["region"])
locs_dict = lmp_locs_crosswalk.set_index("node_id").T.to_dict('list')



folder_path = 'C:/Users/jzs88/Box/LMP_Predictive_Modeling/acLMPs' # path to folder with lmp data
node_pkl_names = os.listdir(folder_path) # list node pkl file names

#node_id = node_pkl_names[1].replace(".pkl", "").replace("CA_", "")
#lat = locs_dict[node_id][0]


def build_lmp(pkl_path): # function to build numpy array of [date-hour minute, lmp, node_id] for given node
    node_id = pkl_path.replace(".pkl", "").replace("CA_", "") # extract node_id from path
    lat = locs_dict[node_id][0]
    lng = locs_dict[node_id][1]
    path = folder_path + '/' + pkl_path # create path for complete pkl
    lmp_pkl = open(path, 'rb') # open pkl
    lmp_dic = pkl.load(lmp_pkl) # load data from pkl file
    lmp_complete_dict = {} # initialize dictionary for given node
    for key in lmp_dic.keys(): # iterate over keys (days)
        this_dic = lmp_dic[key] # extract 12 x 24 dataset of lmps for this key (day)
        if key != "info": # exclude info key
            for hour in range(0,23,1): # iterate over hours in day
                for minute in range(0,11,1): # iterate over 5 minute inervals of hour
                    this_lmp = this_dic[minute, hour] # extract lmp for given minute, hour
                    lmp_val_dict = {key + "-" + str(hour) + "-" + str(minute*5):this_lmp} # build dictionary with date-hour-minute as key, lmp as value
                    lmp_complete_dict.update(lmp_val_dict) # update complete node dictionary with new lmp
    lmp_keyvalpairs = lmp_complete_dict.items() # extract key value paris from dictionary
    lmp_list_data = list(lmp_keyvalpairs) # create list from key value pairs
    lmp_np_array = np.array(lmp_list_data) # convert list to numpy array
    node_id_list = np.array([node_id]*len(lmp_np_array))[...,None] # add the "[...,None]" to maintain n x 1 structure
    lat_list = np.array([lat]*len(lmp_np_array))[...,None]
    lng_list = np.array([lng]*len(lmp_np_array))[...,None]
    lmp_np_array = np.append(lmp_np_array, node_id_list, 1) # append node_ids to arrayy horizontally
    lmp_np_array = np.append(lmp_np_array, lat_list, 1) # append lats to array horizontally
    lmp_np_array = np.append(lmp_np_array, lng_list, 1) # append lngs to array horizontally
    return lmp_np_array # return array
  
all_lmps_np_array = np.empty(shape = (0,5)) # initialize complete lmp dataset

for node_path in node_pkl_names: # iterate over nodes
   this_lmp_array = build_lmp(node_path) # build np array for each node
   all_lmps_np_array = np.append(all_lmps_np_array, this_lmp_array, 0) # append new node array to dataset vertically

    
#path = folder_path + '/' + node_pkl_names[1] # create path for complete pkl
#lmp_pkl = open(path, 'rb') # open pkl
#lmp_dic = pkl.load(lmp_pkl)
