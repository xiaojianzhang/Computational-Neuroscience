#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description: Provides "MyVolume" class and Keras model for CNN training and testing"""
import os
import scipy.io as sio
import numpy as np
import pickle
import time
from itertools import product
import pdb
import Utility as utl
import h5py
import random

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint

__author__ = "Sheng Zhang"
__copyright__ = "Copyright 2017, The Neuroscience Project"
__credits__ = ["Sheng Zhang", "Uygar Sümbül"]
__license__ = "MIT"
__version__ = "1.0.0"
__date__ = "06/20/2017"
__maintainer__ = "Sheng Zhang"
__email__ = "sz2553@columbia.edu"
__status__ = "Development"

class MyVolume:
	'''A volume class'''
	def __init__(self, VolumeName=None, RawVolume_Dir=None, VolumeLabels_Dir=None, RawVolume=None, VolumeLabels=None):
		'''
		   (1) initialize an instance by giving the directory containing
		   RawVolume and VolumeLabels. (File extension should be '.mat')
		   OR
		   (2) Initialize an instance by directly giving RawVolume and VolumeLabels (Both are Numpy Ndarray). 
		   '''
		self.VolumeName = VolumeName
		if RawVolume_Dir is not None:
			self.load_RawVolume(RawVolume_Dir)
		elif RawVolume is not None:
			self.RawVolume = RawVolume
		else:
			raise ValueError("Both 'RawVolume_Dir' and 'RawVolume' are Empty!!!")

		if VolumeLabels_Dir is not None:
			self.load_VolumeLabels(VolumeLabels_Dir)
		elif VolumeLabels is not None:
			self.VolumeLabels = VolumeLabels
		else:
			self.VolumeLabels = None

	def load_RawVolume(self, filepath):
		'''load RawVolume as ndarray of shape=(Length, Width, Height, Channels)'''
		RawVolume = sio.loadmat(filepath)['overallRawVolume'].astype(np.float)
		self.RawVolume = RawVolume
	
	def load_VolumeLabels(self, filepath):
		'''load VolumeLabels and transform it ino ndarray of shape=(Num_Cells, Length, Width, Height)'''
		RawVolumeLabels = sio.loadmat(filepath)['volumeLabels'] #shape=(1, Num_Cells)
		Num_Cells = RawVolumeLabels.shape[1]
		L,W,H = self.RawVolume.shape[:3]
		VolumeLabels = np.zeros([Num_Cells, L, W, H], dtype=np.int)
		for cell_index in range(Num_Cells):
			VolumeLabels[cell_index,:,:,:] = RawVolumeLabels[0, cell_index]
		self.VolumeLabels = VolumeLabels		           

	def Identify_Neuron_and_Background_Voxel(self, valid_range_of_Volume):		
		''' '''
		Background_Voxel = {}
		Neuron_Voxel, Neuron_Voxel_array, valid_region_array = utl.find_neuron_voxel(self.VolumeLabels, valid_range_of_Volume)
		Background_Voxel['Easy'], Background_Voxel['Hard'] = utl.find_easy_hard_background_voxel(self.VolumeLabels, Neuron_Voxel_array, valid_region_array)
		return Neuron_Voxel, Background_Voxel

	def Generating_X_and_Y(self, Neuron_Voxel, Background_Voxel, patch_size):
		''' '''
		random.shuffle(Background_Voxel['Easy'])

		All_Voxel_for_Training = Neuron_Voxel + \
		 					     Background_Voxel['Hard'] + \
		 					     Background_Voxel['Easy'][:len(Background_Voxel['Hard'])//2]
		random.shuffle(All_Voxel_for_Training)
		X = []
		Y = []
		for coordinate, label in All_Voxel_for_Training:
				X += self.coordinate_to_x(coordinate, patch_size)
				Y += [label]*4

		X = np.asarray(X, dtype=np.float)
		Y = np.asarray(Y, dtype=np.int)

		return X, Y
		

	def coordinate_to_x(self, coordinate, patch_size):
		''''''
		x_coor, y_coor, z_coor = coordinate
		xs = []
		x = self.RawVolume[x_coor-patch_size['x']:x_coor+1+patch_size['x'],
							   y_coor-patch_size['y']:y_coor+1+patch_size['y'],
							   z_coor-patch_size['z']:z_coor+1+patch_size['z'], :1]
		xs.append(x) #counterclockwise 0 degree
		xs.append(np.rot90(x, 3, (0,1))) #counterclockwise 90 degree
		xs.append(np.rot90(x, 2, (0,1))) #counterclockwise 180 degree
		xs.append(np.rot90(x, 1, (0,1))) #counterclockwise 270 degree

		return xs

def model_train_validation(X, Y, batch_size, epochs, verbose, callbacks, validation_split, class_weight):
	##########  Convolutional Neural Network Architecture  ###########################
	model = Sequential()

	model.add(Conv3D(16, (5, 5, 1), padding='valid', input_shape=X.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv3D(16, (3, 3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(Conv3D(16, (5, 5, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(Conv3D(16, (3, 3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()

	model.compile(loss='binary_crossentropy',
					  optimizer='Adam',
					  metrics=['accuracy'])
	
	print("Starting training...")
	
	History = model.fit(x=X, y=Y,
				  		epochs=epochs,
				  		validation_split=validation_split,
				      	verbose=verbose,
				  		callbacks=callbacks,
				  		shuffle=True,
				  		class_weight=class_weight)

	print('Training is done')
	
	return model

def reload_model(weights_file_path, one_input_shape):
	# create model
	model = Sequential()

	model.add(Conv3D(16, (5, 5, 1), padding='valid', input_shape=one_input_shape))
	model.add(Activation('relu'))
	model.add(Conv3D(16, (3, 3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(Conv3D(16, (5, 5, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(Conv3D(16, (3, 3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()

	# load weights
	model.load_weights(weights_file_path)
	# Compile model (required to make predictions)
	model.compile(loss='binary_crossentropy',
				  optimizer='Adam',
				  metrics=['accuracy'])
	print("Created model and loaded weights from file")

	return model

def prediction_on_volume(trained_model, RawVolume, Threshold, valid_range_of_Volume, patch_size):
	''''''
	num_Volumes = len(Threshold)
	Predictions_On_Volume = np.zeros([num_Volumes, valid_range_of_Volume['x'][1]+1-valid_range_of_Volume['x'][0],
								                   valid_range_of_Volume['y'][1]+1-valid_range_of_Volume['y'][0], 
								                   valid_range_of_Volume['z'][1]+1-valid_range_of_Volume['z'][0]], dtype=np.int)
	for z in range(valid_range_of_Volume['z'][0], valid_range_of_Volume['z'][1]+1):
		One_Layer_Inputs = []
		for x in range(valid_range_of_Volume['x'][0], valid_range_of_Volume['x'][1]+1):
			for y in range(valid_range_of_Volume['y'][0], valid_range_of_Volume['y'][1]+1):
				One_Layer_Inputs.append(RawVolume[x-patch_size['x']:x+patch_size['x']+1,
												  y-patch_size['y']:y+patch_size['y']+1,
												  z-patch_size['z']:z+patch_size['z']+1, :1])
		One_Layer_Inputs = np.asarray(One_Layer_Inputs)
		predictions = trained_model.predict(One_Layer_Inputs, batch_size=Predictions_On_Volume.shape[2], verbose=0)
		for i,threshold in enumerate(Threshold):
			predictions = (predictions>=threshold).astype(np.int)
			Predictions_On_Volume[i,:,:,z-valid_range_of_Volume['z'][0]] = np.reshape(predictions, (Predictions_On_Volume.shape[1], Predictions_On_Volume.shape[2]), order='C')

	return Predictions_On_Volume

if __name__ == '__main__':    #code to execute if called from command-line
	#Training using mutiple volumes
	training_dataset_path_dict = utl.Volume_Labels_dir_dict('../Training_Dataset/')
	train_volume_dict = {}
	X = {}
	Y = {}
	for name, values in training_dataset_path_dict.items():
		print("We are processing volume: {0}".format(name))
		train_volume_dict[name] = MyVolume(VolumeName=name, RawVolume_Dir=values['overallRawVolume'], VolumeLabels_Dir=values['volumeLabels'], RawVolume=None, VolumeLabels=None)
		Neuron_Voxel, Background_Voxel = train_volume_dict[name].Identify_Neuron_and_Background_Voxel(valid_range_of_Volume={'x':[6, 200-6-1], 'y':[6, 200-6-1], 'z':[2, 100-2-1]})
		X[name], Y[name] = train_volume_dict[name].Generating_X_and_Y(Neuron_Voxel, Background_Voxel, {'x':6, 'y':6, 'z':2})
	X = np.concatenate([X[name] for name in X], axis=0)
	Y = np.concatenate([Y[name] for name in Y], axis=0)
	# set checkpoint
	if not os.path.exists('./Checkpoint/'):
		os.makedirs('./Checkpoint/')
	filepath="./Checkpoint/weights-improvement-{epoch:02d}-{val_acc:.5f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks = [checkpoint]
	model = model_train_validation(X=X, Y=Y, batch_size=32, epochs=100, verbose=1, callbacks=callbacks, validation_split=0.1, class_weight={0:1, 1:1.5})






