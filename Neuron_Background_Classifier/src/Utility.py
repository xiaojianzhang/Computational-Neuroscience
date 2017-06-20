#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description: Provides Utility Functions"""

import scipy.io as sio
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import time
from itertools import product
import pdb
import os
import random
import gc

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.optimizers import RMSprop

__author__ = "Sheng Zhang"
__copyright__ = "Copyright 2017, The Neuroscience Project"
__credits__ = ["Sheng Zhang", "Uygar Sümbül"]
__license__ = "MIT"
__version__ = "1.0.0"
__date__ = "05/10/2017"
__maintainer__ = "Sheng Zhang"
__email__ = "sz2553@columbia.edu"
__status__ = "Development"

def find_neuron_voxel(VolumeLabels, valid_range_of_Volume):
	''' '''
	_, L, W, H = VolumeLabels.shape
	sum_along_cells = np.sum(VolumeLabels, axis=0)
	Neuron_Voxel_array = (sum_along_cells>=1).astype(np.int)
	x_coor, y_coor, z_coor = np.where( sum_along_cells >= 1 )
	all_neuron_voxels = zip(x_coor.tolist(), y_coor.tolist(), z_coor.tolist())
	neuron_voxel_in_valid_region = []
	valid_region_array = np.zeros([L,W,H], dtype=np.int)
	valid_region_array[valid_range_of_Volume['x'][0]:valid_range_of_Volume['x'][1]+1,
	                   valid_range_of_Volume['y'][0]:valid_range_of_Volume['y'][1]+1,
	                   valid_range_of_Volume['z'][0]:valid_range_of_Volume['z'][1]+1] = 1

	for (x,y,z) in all_neuron_voxels:
		if valid_region_array[x,y,z] == 1:
			neuron_voxel_in_valid_region.append(((x,y,z),1))

	return neuron_voxel_in_valid_region, Neuron_Voxel_array, valid_region_array

def find_easy_hard_background_voxel(VolumeLabels, Neuron_Voxel_array, valid_region_array, window_size=1):
	''' '''
	_, L, W, H = VolumeLabels.shape
	easy_background_voxel_in_valid_region = []
	hard_background_voxel_in_valid_region = []
	sum_along_cells = np.sum(VolumeLabels, axis=0)
	Background_Voxel_array = (sum_along_cells==0).astype(np.int)
	x_coor, y_coor, z_coor = np.where( sum_along_cells == 0 )
	all_background_voxel = zip(x_coor.tolist(), y_coor.tolist(), z_coor.tolist())
	for (x,y,z) in all_background_voxel:
		if valid_region_array[x,y,z] == 1:
			if np.sum(Neuron_Voxel_array[x-window_size:x+1+window_size,
				 						 y-window_size:y+1+window_size,
				 						 z-window_size:z+1+window_size]) == 0:
				easy_background_voxel_in_valid_region.append(((x,y,z),0))
			else:
				hard_background_voxel_in_valid_region.append(((x,y,z),0))

	return easy_background_voxel_in_valid_region, hard_background_voxel_in_valid_region

def Volume_Labels_dir_dict(dataset_folder_dir):
	''''''
	if not dataset_folder_dir.endswith('/'):
		dataset_folder_dir += '/'
	dir_dict = {}
	for file_1 in os.listdir(dataset_folder_dir):
		if file_1 != '.DS_Store':
			path_1 = os.path.join(dataset_folder_dir, file_1)
			dir_dict[file_1] = {'overallRawVolume':None, 'volumeLabels':None}
			for file_2 in os.listdir(path_1):
				if file_2.endswith('mat'):
					path_2 = os.path.join(path_1, file_2)
					if file_2.startswith('overall'):
						dir_dict[file_1]['overallRawVolume'] = path_2
					elif file_2.startswith('volume'):
						dir_dict[file_1]['volumeLabels'] = path_2

	return dir_dict


if __name__ == '__main__':    #code to execute if called from command-line
	pass