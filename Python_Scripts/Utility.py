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

__author__ = "Sheng Zhang"
__copyright__ = "Copyright 2017, The Neuroscience Project"
__credits__ = ["Sheng Zhang", "Uygar Sümbül", "Min-hwan Oh"]
__license__ = "MIT"
__version__ = "1.0.0"
__date__ = "05/10/2017"
__maintainer__ = "Sheng Zhang"
__email__ = "sz2553@columbia.edu"
__status__ = "Development"

def find_neuron_coor(VolumeLabels):
	''' '''
	neuron_coor_dict = {}
	N_Ts, _, _, _ = VolumeLabels.shape
	for neuron_type in range(N_Ts):
		x_coor, y_coor, z_coor = np.where( VolumeLabels[neuron_type,:,:,:] == 1 )
		neuron_coor = list(zip(x_coor.tolist(), y_coor.tolist(), z_coor.tolist()))
		neuron_coor_dict[str(neuron_type+1)] = neuron_coor

	return neuron_coor_dict 

def find_neuron_voxel_coor(VolumeLabels):
	''' '''
	sum_along_type = np.sum(VolumeLabels, axis=0)
	NV_array = (sum_along_type>=1).astype(np.int)
	x_coor, y_coor, z_coor = np.where( sum_along_type >= 1 )
	neuron_voxel_coor = list(zip(x_coor.tolist(), y_coor.tolist(), z_coor.tolist()))

	return neuron_voxel_coor, NV_array

def same_type_neuron_connectivity(neuron_coor_dict, VolumeLabels, W):
	'''
	'''
	same_type_neuron_conn = []
	for neuron_type, neuron_coor in neuron_coor_dict.items():
		VLs = VolumeLabels[int(neuron_type)-1,:,:,:]
		for (x,y,z) in neuron_coor:
			if y+1 <= W-1 and VLs[x,y+1,z] == 1:
				same_type_neuron_conn.append((x,y,z))

		same_type_neuron_conn = list(set(same_type_neuron_conn))

	return same_type_neuron_conn

def find_background_voxel_coor(VolumeLabels):
	''' '''
	sum_along_type = np.sum(VolumeLabels, axis=0)
	x_coor, y_coor, z_coor = np.where( sum_along_type == 0 )
	background_voxel_coor = list(zip(x_coor.tolist(), y_coor.tolist(), z_coor.tolist()))
	BV_array = (sum_along_type==0).astype(np.int)

	return background_voxel_coor, BV_array

def neuron_background_connectivity(neuron_coor_dict, background_voxel_coor, BV_array, W):
	'''
	'''
	neuron_background_conn = []
	for neuron_type, neuron_coor in neuron_coor_dict.items():
		for (x,y,z) in neuron_coor:
			if y+1 <= W-1 and BV_array[x,y+1,z] == 1:
				neuron_background_conn.append((x,y,z))

		neuron_background_conn = list(set(neuron_background_conn))

	return neuron_background_conn


def background_neuron_connectivity(background_voxel_coor, NV_array, W):
	'''
	'''
	background_neuron_conn = []
	for (x,y,z) in background_voxel_coor:
		if y+1 <= W-1 and NV_array[x,y+1,z] == 1:
			background_neuron_conn.append((x,y,z))
	background_neuron_conn = list(set(background_neuron_conn))

	return background_neuron_conn

def background_background_connectivity(background_voxel_coor, BV_array, W):
	'''
	'''
	background_background_conn = []
	for (x,y,z) in background_voxel_coor:
		if y+1 <= W-1 and BV_array[x,y+1,z] == 1:
			background_background_conn.append((x,y,z))
	background_background_conn = list(set(background_background_conn))		

	return background_background_conn

def different_neuron_connectivity(neuron_coor_dict, VolumeLabels, W, same_type_neuron_conn):
	'''
	'''
	different_neuron_conn = []
	for neuron_type, neuron_coor in neuron_coor_dict.items():
		sum_along_type_except_neuron_type = np.sum(VolumeLabels, axis=0) - VolumeLabels[int(neuron_type)-1,:,:,:]
		ON_array = (sum_along_type_except_neuron_type>=1).astype(np.int)
		for (x,y,z) in neuron_coor:
			if y+1 <= W-1 and ON_array[x,y+1,z] == 1:
				different_neuron_conn.append((x,y,z))
	different_neuron_conn = list(set(different_neuron_conn) - set(same_type_neuron_conn))

	return different_neuron_conn

def undersampling(dataset, num_samples):
	''''''
	Len = len(dataset)
	samples_index = np.random.randint(0, Len, num_samples).tolist()
	undersampled_dataset = [dataset[i] for i in samples_index]
	return undersampled_dataset

def image_augmentation(original_image, extra_length, extra_width, extra_height, method='zero_padding'):
	''''''
	L,W,H,N_Chs = original_image.shape
	augumented_image = np.zeros([L + 2*extra_length, W + 2*extra_width, H + 2*extra_height, N_Chs])
	if method == 'zero_padding':
		augumented_image[extra_length:L+extra_length, extra_width:W+extra_width, extra_height:H+extra_height, :] = original_image

	return augumented_image

def make_confusion_matrix(model_predictions, true_labels):
	'''
		Inputs: model_predictions -- 2D array of shape=(num_datapoints, num_classes) denotes
									 model predictions.
				true_labels -- 2D array of shape=(num_datapoints, num_classes)
		Output: confusion_matrix -- 2D array of shape=(num_classes, num_classes)
		This function generates the confusion matrix.
	'''
	assert model_predictions.shape == true_labels.shape
	num_classes = true_labels.shape[1]
	confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
	predictions = np.argmax(model_predictions, axis=1)
	assert len(predictions)==true_labels.shape[0]

	for actual_class in range(num_classes):
		idx_examples_this_class = true_labels[:,actual_class]==1.0
		prediction_for_this_class = predictions[idx_examples_this_class]
		for predicted_class in range(num_classes):
			count = np.sum(prediction_for_this_class==predicted_class)
			confusion_matrix[actual_class, predicted_class] = count
	assert np.sum(confusion_matrix)==len(true_labels)
	assert np.sum(confusion_matrix)==np.sum(true_labels)
	
	return confusion_matrix

if __name__ == '__main__':    #code to execute if called from command-line
	pass    