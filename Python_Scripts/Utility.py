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

__author__ = "Sheng Zhang"
__copyright__ = "Copyright 2017, The Neuroscience Project"
__credits__ = ["Sheng Zhang", "Uygar Sümbül", "Min-hwan Oh"]
__license__ = "MIT"
__version__ = "1.0.0"
__date__ = "05/10/2017"
__maintainer__ = "Sheng Zhang"
__email__ = "sz2553@columbia.edu"
__status__ = "Development"

def find_neuron_voxel_coor(VolumeLabels):
	''' '''
	sum_along_type = np.sum(VolumeLabels, axis=0)
	NV_array = (sum_along_type>=1).astype(np.int)
	x_coor, y_coor, z_coor = np.where( sum_along_type >= 1 )
	neuron_voxel_coor = list(zip(x_coor.tolist(), y_coor.tolist(), z_coor.tolist()))

	return neuron_voxel_coor, NV_array

def same_and_different_type_neuron_connectivity(VolumeLabels, neuron_voxel_coor, W, NV_array):
	''''''
	same_type_neuron_conn = []
	different_neuron_conn = []
	sum_along_type = np.sum(VolumeLabels, axis=0)
	for (x,y,z) in neuron_voxel_coor:
		if y+1 <= W-1 and NV_array[x,y+1,z] == 1:
			if sum_along_type[x,y,z] == sum_along_type[x,y+1,z]:
				same_type_neuron_conn.append((x,y,z))
			else:
				different_neuron_conn.append((x,y,z))

	same_type_neuron_conn = list(set(same_type_neuron_conn))
	different_neuron_conn = list(set(different_neuron_conn))

	return same_type_neuron_conn, different_neuron_conn

def find_background_voxel_coor(VolumeLabels):
	''' '''
	sum_along_type = np.sum(VolumeLabels, axis=0)
	x_coor, y_coor, z_coor = np.where( sum_along_type == 0 )
	background_voxel_coor = list(zip(x_coor.tolist(), y_coor.tolist(), z_coor.tolist()))
	BV_array = (sum_along_type==0).astype(np.int)

	return background_voxel_coor, BV_array

def neuron_background_connectivity(neuron_voxel_coor, background_voxel_coor, BV_array, W):
	'''
	'''
	neuron_background_conn = []
	for (x,y,z) in neuron_voxel_coor:
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

def Volume_Labels_dir_dict(dataset_folder_dir):
	''''''
	if not dataset_folder_dir.endswith('/'):
		dataset_folder_dir += '/'
	dir_dict = {}
	for file_1 in os.listdir(dataset_folder_dir):
		if file_1 != '.DS_Store':
			path_1 = os.path.join(dataset_folder_dir, file_1)
			dir_dict[file_1] = {'overRawVolume':None, 'volumeLabels':None}
			for file_2 in os.listdir(path_1):
				if file_2.endswith('mat'):
					path_2 = os.path.join(path_1, file_2)
					if file_2.startswith('over'):
						dir_dict[file_1]['overRawVolume'] = path_2
					elif file_2.startswith('volume'):
						dir_dict[file_1]['volumeLabels'] = path_2

	return dir_dict

def current_voxel_label_4direction(predictions_dict, x_coor, y_coor, x_lim, y_lim):
	'''
		Inputs: predictions_dict -- dictionary that contains four directions' predictions
											('key',value)=(direction, one layer predictions array
											associated with the direction)
											'key'=['y-plus', 'y-minus', 'x-plus', 'x-minus']
				x_coor, y_coor -- two integers that indicate the 2D coordinate of the current voxel.
				x_lim, y_lim -- two integers that represent the length and width of the volume.
		Outputs: label -- 0/1. 0 means the current voxel is a background (i.e. not a neuron)
						  1 means the current voexl is a neuron.
	'''
	if x_coor>=1 and x_coor <= x_lim-2 and y_coor>=1 and y_coor<=y_lim-2:
		if (predictions_dict['y-plus'][x_coor, y_coor] +\
			predictions_dict['y-minus'][x_coor, y_coor-1] +\
			predictions_dict['x-plus'][x_coor, y_coor] +\
			predictions_dict['x-minus'][x_coor-1, y_coor]) == 0:
			label = 0
		else:
			label = 1
			
	elif x_coor==0 and y_coor>=1 and y_coor<=y_lim-2:
		if (predictions_dict['y-plus'][x_coor, y_coor] +\
			predictions_dict['y-minus'][x_coor, y_coor-1] +\
			predictions_dict['x-plus'][x_coor, y_coor]) == 0:
			label = 0
		else:
			label = 1
			
	elif x_coor==x_lim-1 and y_coor>=1 and y_coor<=y_lim-2:
		if (predictions_dict['y-plus'][x_coor, y_coor] +\
			predictions_dict['y-minus'][x_coor, y_coor-1] +\
			predictions_dict['x-minus'][x_coor-1, y_coor]) == 0:
			label = 0
		else:
			label = 1
			
	elif x_coor>=1 and x_coor <= x_lim-2 and y_coor==0:
		if (predictions_dict['y-plus'][x_coor, y_coor] +\
			predictions_dict['x-plus'][x_coor, y_coor] +\
			predictions_dict['x-minus'][x_coor-1, y_coor]) == 0:
			label = 0
		else:
			label = 1
	
	elif x_coor>=1 and x_coor <= x_lim-2 and y_coor==y_lim-1:
		if (predictions_dict['y-minus'][x_coor, y_coor-1] +\
			predictions_dict['x-plus'][x_coor, y_coor] +\
			predictions_dict['x-minus'][x_coor-1, y_coor]) == 0:
			label = 0
		else:
			label = 1
			
	elif x_coor==0 and y_coor==0:
		if (predictions_dict['y-plus'][x_coor, y_coor] +\
			predictions_dict['x-plus'][x_coor, y_coor]) == 0:
			label = 0
		else:
			label = 1
			
	elif x_coor==0 and y_coor==y_lim-1:
		if (predictions_dict['y-minus'][x_coor, y_coor-1] +\
			predictions_dict['x-plus'][x_coor, y_coor]) == 0:
			label = 0
		else:
			label = 1
			
	elif x_coor==x_lim-1 and y_coor==0:
		if (predictions_dict['y-plus'][x_coor, y_coor] +\
			predictions_dict['x-minus'][x_coor-1, y_coor]) == 0:
			label = 0
		else:
			label = 1
			
	elif x_coor==x_lim-1 and y_coor==y_lim-1:
		if (predictions_dict['y-minus'][x_coor, y_coor-1] +\
			predictions_dict['x-minus'][x_coor-1, y_coor]) == 0:
			label = 0
		else:
			label = 1
	
	return label

def reconstruction_accuracy(reconstruction, volumeLabels):
	''''''
	L,W,H = reconstruction.shape
	Total_correct_count = 0
	for z in range(H):
		VolumeLabels_z = volumeLabels[:,:,:,z]
		labels_sum_z = np.sum(VolumeLabels_z, axis=0)
		ground_truth_z = (labels_sum_z>0).astype(np.int)
		local_correct_count = np.sum((reconstruction[:,:,z]==ground_truth_z).astype(np.int))
		print('Layer {0} accuracy is: {1}'.format(z, local_correct_count/(200**2.0)))
		Total_correct_count += local_correct_count
	print('Total accuracy is: {0}'.format(Total_correct_count/float(L*W*H)))

def neuron_path_one_layer(predictions_one_layer, reconstruction_one_layer):
	''''''
	Length,Width = reconstruction_one_layer.shape
	xs,ys = np.where(reconstruction_one_layer==1)
	all_neurons_locations = list(zip(xs.tolist(), ys.tolist()))
	neurons_coor = list(zip(xs.tolist(), ys.tolist()))
	neurons_coor_length = len(neurons_coor)
	neuron_path=[]
	
	while(neurons_coor_length>0):
		init_coor = neurons_coor[0]
		#print('current neuron coor length is {0}'.format(neurons_coor_length))
		new_path, intersection_with = one_path(init_coor, predictions_one_layer, Length, Width, all_neurons_locations, neuron_path)
		if len(intersection_with)==1:
			neuron_path[intersection_with[0]] += new_path
		elif len(intersection_with)>1:
			union_path = []
			for index in intersection_with:
				union_path += neuron_path[index]
			union_path += new_path
			neuron_path = [neuron_path[i] for i in range(len(neuron_path)) if i not in intersection_with]
			neuron_path.append(union_path)
		neuron_path.append(new_path)
		neurons_coor = [item for item in neurons_coor if item not in new_path]
		neurons_coor_length = len(neurons_coor)
			
	print('Number of neuron path is {0}'.format(len(neuron_path)))
	return neuron_path

def one_path(init_coor, predictions_one_layer, Length, Width, all_neurons_locations, neuron_path):
	''''''
	new_path = []
	next_neuron_coor = [init_coor]
	intersection_with = []
	while(len(next_neuron_coor)>0):
		current_neuron_coor = next_neuron_coor[0]
		new_path.append(current_neuron_coor)
		next_neuron_coor.remove(current_neuron_coor)
		neighbor_neuron_coor = neighbor_neuron(current_neuron_coor, predictions_one_layer, Length, Width)
		for neighbor_coor in neighbor_neuron_coor:
			if neighbor_coor in all_neurons_locations and neighbor_coor not in new_path and neighbor_coor not in next_neuron_coor:
				is_intersect, index = is_intersection(neighbor_coor, neuron_path)
				if is_intersect:
					intersection_with.append(index)
				else:
					next_neuron_coor.append(neighbor_coor)
	return new_path, intersection_with

def is_intersection(neighbor_coor, neuron_path):
	''''''
	if len(neuron_path) == 0:
		return False, None
	else:
		for i,L in enumerate(neuron_path):
			if neighbor_coor in L:
				 return True, i
		
		return False,None

def neighbor_neuron(current_neuron_coor, predictions_dict, x_lim, y_lim):
	''''''
	x_coor, y_coor = current_neuron_coor
	neighbor_neuron_coor=[]
	if x_coor>=1 and x_coor <= x_lim-2 and y_coor>=1 and y_coor<=y_lim-2:
		if predictions_dict['y-plus'][x_coor, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor+1))
		if predictions_dict['y-minus'][x_coor, y_coor-1] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor-1))
		if predictions_dict['x-plus'][x_coor, y_coor] ==1:
			neighbor_neuron_coor.append((x_coor+1,y_coor))
		if predictions_dict['x-minus'][x_coor-1, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor-1,y_coor))
	
	elif x_coor==0 and y_coor>=1 and y_coor<=y_lim-2:
		if predictions_dict['y-plus'][x_coor, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor+1))
		if predictions_dict['y-minus'][x_coor, y_coor-1] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor-1))
		if predictions_dict['x-plus'][x_coor, y_coor] ==1:
			neighbor_neuron_coor.append((x_coor+1,y_coor))
	
	elif x_coor==x_lim-1 and y_coor>=1 and y_coor<=y_lim-2:
		if predictions_dict['y-plus'][x_coor, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor+1))
		if predictions_dict['y-minus'][x_coor, y_coor-1] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor-1))
		if predictions_dict['x-minus'][x_coor-1, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor-1,y_coor))
	
	elif x_coor>=1 and x_coor <= x_lim-2 and y_coor==0:
		if predictions_dict['y-plus'][x_coor, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor+1))
		if predictions_dict['x-plus'][x_coor, y_coor] ==1:
			neighbor_neuron_coor.append((x_coor+1,y_coor))
		if predictions_dict['x-minus'][x_coor-1, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor-1,y_coor))
			
	elif x_coor>=1 and x_coor <= x_lim-2 and y_coor==y_lim-1:
		if predictions_dict['y-minus'][x_coor, y_coor-1] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor-1))
		if predictions_dict['x-plus'][x_coor, y_coor] ==1:
			neighbor_neuron_coor.append((x_coor+1,y_coor))
		if predictions_dict['x-minus'][x_coor-1, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor-1,y_coor))
			
	elif x_coor==0 and y_coor==0:
		if predictions_dict['y-plus'][x_coor, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor+1))
		if predictions_dict['x-plus'][x_coor, y_coor] ==1:
			neighbor_neuron_coor.append((x_coor+1,y_coor))
			
	elif x_coor==0 and y_coor==y_lim-1:
		if predictions_dict['y-minus'][x_coor, y_coor-1] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor-1))
		if predictions_dict['x-plus'][x_coor, y_coor] ==1:
			neighbor_neuron_coor.append((x_coor+1,y_coor))
			
	elif x_coor==x_lim-1 and y_coor==0:
		if predictions_dict['y-plus'][x_coor, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor+1))
		if predictions_dict['x-minus'][x_coor-1, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor-1,y_coor))
			
	elif x_coor==x_lim-1 and y_coor==y_lim-1:
		if predictions_dict['y-minus'][x_coor, y_coor-1] == 1:
			neighbor_neuron_coor.append((x_coor,y_coor-1))
		if predictions_dict['x-minus'][x_coor-1, y_coor] == 1:
			neighbor_neuron_coor.append((x_coor-1,y_coor))
			
	return neighbor_neuron_coor

if __name__ == '__main__':    #code to execute if called from command-line
	pass    