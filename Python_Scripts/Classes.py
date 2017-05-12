#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description: Provides "MyVolume" class"""

import scipy.io as sio
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import time
from itertools import product
import pdb
import Utility as utl
import h5py

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import RMSprop

__author__ = "Sheng Zhang"
__copyright__ = "Copyright 2017, The Neuroscience Project"
__credits__ = ["Sheng Zhang", "Uygar Sümbül", "Min-hwan Oh"]
__license__ = "MIT"
__version__ = "1.0.0"
__date__ = "05/10/2017"
__maintainer__ = "Sheng Zhang"
__email__ = "sz2553@columbia.edu"
__status__ = "Development"

class MyVolume:
	'''A volume class'''
	def __init__(self, RawVolume_Dir, VolumeLabels_Dir=None):
		'''initialized instance by giving directory of
		   RowVolume and VolumeLabels(not necessary) you want
		   to read
		   File extension should be '.mat'
		   '''
		self._RawVolume_Dir = RawVolume_Dir
		self._VolumeLabels_Dir = VolumeLabels_Dir
		self.L = None #Length
		self.H = None #Height
		self.W = None #Width
		self.N_Chs = None #Number of Channels
		self.N_N_Ts = None #Number of Neuron Types
		self.RawVolume = None #Raw Volume ndarray
		self.VolumeLabels = None #Volume Labels ndarray
		self.Conn_Types = None
		self.X_coor = None
		self.Y_dataset = None
		self.X_dataset = None
		self.X_train = None
		self.X_test = None
		self.Y_train = None
		self.Y_test = None


	def load_RawVolume(self):
		'''load RawVolume as ndarray of shape=(Length, Width, Height, Num_Channels)'''
		RawVolume = sio.loadmat(self._RawVolume_Dir)['overallRawVolume'].astype(np.float)
		self.L, self.W, self.H, self.N_Chs = RawVolume.shape #store Length, Width, Height, Num_Channels
		self.RawVolume = RawVolume
	
	def load_VolumeLabels(self):
		'''load VolumeLabels and transform it ino ndarray of shape=(Num_Neuron_Types, Length, Width, Height)'''
		RawVolumeLabels = sio.loadmat(self._VolumeLabels_Dir)['volumeLabels'] #shape=(1, N_N_Ts)
		self.N_N_Ts = RawVolumeLabels.shape[1]
		if (self.L is None) or (self.H is None) or (self.W is None):
			raise ValueError('Please read RawVolume first')
		else:
			VolumeLabels = np.zeros([self.N_N_Ts, self.L, self.W, self.H],dtype=np.int)
			for neuron_type in range(self.N_N_Ts):
				VolumeLabels[neuron_type,:,:,:] = RawVolumeLabels[0, neuron_type]
			self.VolumeLabels = VolumeLabels
	
	def voxel_connectivity_types(self):		
		''' '''
		neuron_coor_dict = utl.find_neuron_coor(self.VolumeLabels)
		neuron_voxel_coor, NV_array = utl.find_neuron_voxel_coor(self.VolumeLabels)
		background_voxel_coor, BV_array = utl.find_background_voxel_coor(self.VolumeLabels)
		same_type_neuron_conn = utl.same_type_neuron_connectivity(neuron_coor_dict, self.VolumeLabels, self.L)
		neuron_background_conn = utl.neuron_background_connectivity(neuron_coor_dict, background_voxel_coor, BV_array, self.L)
		background_neuron_conn = utl.background_neuron_connectivity(background_voxel_coor, NV_array, self.L)
		background_background_conn = utl.background_background_connectivity(background_voxel_coor, BV_array, self.L)
		different_neuron_conn = utl.different_neuron_connectivity(neuron_coor_dict, self.VolumeLabels, self.L, same_type_neuron_conn)
		Conn_Types = {'same_type_neuron':same_type_neuron_conn, 'neuron_background':neuron_background_conn,\
					  'background_neuron':background_neuron_conn, 'background_background':background_background_conn,\
					  'different_neuron':different_neuron_conn}
		print("Numer of data points for each type: \n \
			   same_type_neuron: {0} \n \
			   neuron_background: {1} \n \
			   different_neuron: {2} \n \
			   background_neuron: {3} \n \
			   background_background: {4} \n".format(len(same_type_neuron_conn), len(neuron_background_conn), len(different_neuron_conn),\
													 len(background_neuron_conn), len(background_background_conn)))
		self.Conn_Types = Conn_Types

	def balance_dataset(self):
		''' '''
		X_coor = {}
		num_samples = (len(self.Conn_Types['same_type_neuron']) - len(self.Conn_Types['different_neuron'])) // 3
		undersampled_neuron_background = utl.undersampling(self.Conn_Types['neuron_background'], num_samples)
		undersampled_background_neuron = utl.undersampling(self.Conn_Types['background_neuron'], num_samples)
		undersampled_background_background = utl.undersampling(self.Conn_Types['background_background'],\
											 (len(self.Conn_Types['same_type_neuron']) - len(self.Conn_Types['different_neuron'])) - 2*num_samples)
		X_coor['0'] = 	self.Conn_Types['different_neuron']	+ undersampled_neuron_background \
						  + undersampled_background_neuron + undersampled_background_background
		X_coor['1'] = self.Conn_Types['same_type_neuron']
		print('Number of input datapoints for each label: \n \
			   label 0: {0} \n \
			   label 1: {1} \n'.format(len(X_coor['0']), len(X_coor['1'])))
		self.X_coor = X_coor

	def Y_generator(self):
		''' '''
		Y_label_zero = np.asarray([[1,0]] * len(self.X_coor['0']), dtype=np.float)
		Y_label_one = np.asarray([[0,1]] * len(self.X_coor['1']), dtype=np.float)
		Y_dataset = np.concatenate((Y_label_zero, Y_label_one), axis=0)
		self.Y_dataset = Y_dataset

	def X_generator(self, extra_length, extra_width, extra_height, validation_dataset_ratio=None):
		''''''
		augumented_Volume = utl.image_augmentation(self.RawVolume, extra_length, extra_width, extra_height, method='zero_padding')
		X_coor = self.X_coor['0'] + self.X_coor['1']
		if validation_dataset_ratio is None:
			self.X_dataset = []
			for (x,y,z) in X_coor:
				new_x, new_y, new_z = (x + extra_length, y + extra_width, z + extra_height)
				input_array = augumented_Volume[new_x-extra_length:new_x+1+extra_length,\
												new_y-extra_width:new_y+1+extra_width,\
												new_z-extra_height:new_z+1+extra_height, :]
				self.X_dataset.append(input_array)
			self.X_dataset = np.stack(self.X_dataset, axis=0)
		else:
			self.X_train=[]
			self.X_test=[]
			num_datapts = self.Y_dataset.shape[0]
			random_indices = np.arange(num_datapts)
			np.random.shuffle(random_indices)
			sample_test_datapts_index = random_indices[:int(num_datapts*validation_dataset_ratio)].tolist()
			sample_train_datapts_index = random_indices[int(num_datapts*validation_dataset_ratio):].tolist()
			X_coor_test = [X_coor[index] for index in sample_test_datapts_index]
			X_coor_train = [X_coor[index] for index in sample_train_datapts_index]
			for (x,y,z) in X_coor_train:
				new_x, new_y, new_z = (x + extra_length, y + extra_width, z + extra_height)
				self.X_train.append(augumented_Volume[new_x-extra_length:new_x+1+extra_length,\
													  new_y-extra_width:new_y+1+extra_width,\
													  new_z-extra_height:new_z+1+extra_height, :])
			for (x,y,z) in X_coor_test:
				new_x, new_y, new_z = (x + extra_length, y + extra_width, z + extra_height)
				self.X_test.append(augumented_Volume[new_x-extra_length:new_x+1+extra_length,\
													 new_y-extra_width:new_y+1+extra_width,\
													 new_z-extra_height:new_z+1+extra_height, :])
			
			self.X_train = np.stack(self.X_train, axis=0)
			self.X_test = np.stack(self.X_test, axis=0)
			self.Y_test = self.Y_dataset[sample_test_datapts_index,:]
			self.Y_train = self.Y_dataset[sample_train_datapts_index,:]

	def model_train_validate(self, validation_dataset_ratio, batch_size=100, num_classes=2, epochs=15, verbose=1):
		''''''
		##########  Convolutional Neural Network Architecture  ###########################
		model = Sequential()

		model.add(Conv3D(8, (3, 3, 2), padding='same', input_shape=self.X_train.shape[1:]))
		model.add(Activation('relu'))
		print(model.output_shape)
		model.add(Conv3D(16, (4, 4, 2), padding='valid'))
		model.add(Activation('relu'))
		print(model.output_shape)
		model.add(MaxPooling3D(pool_size=(2, 2, 2)))
		model.add(Dropout(0.2))
		print(model.output_shape)
		model.add(Conv3D(32, (2, 2, 2), padding='same'))
		model.add(Activation('relu'))
		print(model.output_shape)
		model.add(Conv3D(32, (3, 3, 3), padding='valid'))
		model.add(Activation('relu'))
		print(model.output_shape)
		model.add(MaxPooling3D(pool_size=(2, 2, 2)))
		model.add(Dropout(0.2))
		print(model.output_shape)
		model.add(Flatten())
		print(model.output_shape)
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
		model.add(Dense(num_classes, activation='softmax'))
		print(model.output_shape)
		model.summary()

		# initiate RMSprop optimizer
		opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
		# Let's train the model using RMSprop
		model.compile(loss='categorical_crossentropy',
					  optimizer=RMSprop(),
					  metrics=['accuracy'])
		history = model.fit(self.X_train, self.Y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  validation_data=(self.X_test, self.Y_test),
					  shuffle=True,
					  verbose=verbose)
		model.save('neuron_conn_classifier.h5')
		model_predictions = model.predict(X_test, batch_size=1000)
		utl.make_confusion_matrix(model_predictions, self.Y_test)

if __name__ == '__main__':    #code to execute if called from command-line
	myVolume = MyVolume('../Sample_Datasets/Training/overallRawVolume.mat', '../Sample_Datasets/Training/volumeLabels.mat')
	myVolume.load_RawVolume()
	myVolume.load_VolumeLabels()
	Conn_Types = myVolume.voxel_connectivity_types()
	myVolume.balance_dataset()
	myVolume.Y_generator()
	
	myVolume.X_generator(7, 7, 4, 0.2)
	with h5py.File('data4training.h5', 'w') as hf:
		hf.create_dataset("X_train",  data=myVolume.X_train)
		hf.create_dataset("Y_train",  data=myVolume.Y_train)
		hf.create_dataset("X_test",  data=myVolume.X_test)
		hf.create_dataset("Y_test",  data=myVolume.Y_test)
	pdb.set_trace()
	'''
	with h5py.File('data4training.h5', 'r') as hf:
		myVolume.X_train = hf['X_train'][:]
		myVolume.Y_train = hf['Y_train'][:]
		myVolume.X_test = hf['X_test'][:]
		myVolume.Y_test = hf['Y_test'][:]
	'''
	myVolume.model_train_validate(validation_dataset_ratio=0.2, batch_size=100, num_classes=2, epochs=10, verbose=1)
	pdb.set_trace()


