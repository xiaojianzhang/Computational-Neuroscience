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
import gc

import keras
from keras.models import Sequential, load_model
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
	def __init__(self, RawVolume_Dir=None, VolumeLabels_Dir=None, RawVolume=None, VolumeLabels=None, direction='y-plus'):
		'''initialized instance by giving directory of
		   RowVolume and VolumeLabels(not necessary) you want
		   to read
		   File extension should be '.mat'
		   '''
		self._RawVolume_Dir = RawVolume_Dir
		self._VolumeLabels_Dir = VolumeLabels_Dir
		if RawVolume is None:
			self.L = None #Length
			self.H = None #Height
			self.W = None #Width
			self.N_Chs = None #Number of Channels
		else:
			self.L, self.W, self.H, self.N_Chs = RawVolume.shape
		if VolumeLabels is None:
			self.N_N_Ts = VolumeLabels #Number of Neuron Types
		else:
			self.N_N_Ts = VolumeLabels.shape[0]
		self.RawVolume = RawVolume #Raw Volume ndarray
		self.VolumeLabels = VolumeLabels #Volume Labels ndarray
		self.Conn_Types = None
		self.X_coor = None
		self.Y_dataset = None
		self.X_dataset = None
		self.X_train = None
		self.X_test = None
		self.Y_train = None
		self.Y_test = None
		self.validation_dataset_ratio=None
		self.extra_length = None
		self.extra_width = None
		self.extra_height = None
		self.trained_model = None
		self.direction = direction
		self.predictions = None
		self.predictions_dict={'y-plus':None, 'y-minus':None, 'x-plus':None, 'x-minus':None}
		self.reconstruction = None
		self.neuron_path = None
		self.reconstructed_volume = None

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
		neuron_voxel_coor, NV_array = utl.find_neuron_voxel_coor(self.VolumeLabels)
		background_voxel_coor, BV_array = utl.find_background_voxel_coor(self.VolumeLabels)
		same_type_neuron_conn, different_neuron_conn = utl.same_and_different_type_neuron_connectivity(self.VolumeLabels, neuron_voxel_coor, self.W, NV_array)
		neuron_background_conn = utl.neuron_background_connectivity(neuron_voxel_coor, background_voxel_coor, BV_array, self.W)
		background_neuron_conn = utl.background_neuron_connectivity(background_voxel_coor, NV_array, self.W)
		background_background_conn = utl.background_background_connectivity(background_voxel_coor, BV_array, self.W)
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
		self.extra_length = extra_length
		self.extra_width = extra_width
		self.extra_height = extra_height
		self.validation_dataset_ratio = validation_dataset_ratio
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
			print('direction {0} completed\n'.format(self.direction))
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
			print('direction {0} completed\n'.format(self.direction))

	def model_train_validate(self, batch_size=100, num_classes=2, epochs=15, verbose=1):
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
		model.fit(self.X_train, self.Y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  validation_data=(self.X_test, self.Y_test),
					  shuffle=True,
					  verbose=verbose)
		model.save('neuron_conn_classifier.h5')
		self.trained_model = model
		gc.collect()

	def confusion_matrix(self, X_dataset=None, Y_dataset=None):
		''''''
		if X_dataset is None and Y_dataset is None:
			model_predictions = self.trained_model.predict(self.X_test, batch_size=5000, verbose=0)
			print("Confusion Matrix is: {0}".format(utl.make_confusion_matrix(model_predictions, self.Y_test)))
		else:
			model_predictions = self.trained_model.predict(X_dataset, batch_size=5000, verbose=0)
			print("Confusion Matrix is: {0}".format(utl.make_confusion_matrix(model_predictions, Y_dataset)))

	def other_three_directions_rawvolume_and_volumelabel(self):
		''''''
		# y-minus direction: rotate an array by 180 degree counterclockwise.
		new_volume = np.rot90(self.RawVolume, 2, (0,1))          
		new_volumelabels = np.rot90(self.VolumeLabels, 2, (1,2))
		NewVolume = MyVolume(None, None, new_volume, new_volumelabels, 'y-minus')  
		NewVolume.voxel_connectivity_types()
		NewVolume.balance_dataset()
		NewVolume.Y_generator()
		NewVolume.X_generator(self.extra_length, self.extra_width, self.extra_height, self.validation_dataset_ratio)
		self.other_three_directions['y-minus'] = NewVolume

		#x-plus direction: rotate an array by 90 degree counterclockwise.
		new_volume = np.rot90(self.RawVolume, 1, (0,1))           
		new_volumelabels = np.rot90(self.VolumeLabels, 1, (1,2))
		NewVolume = MyVolume(None, None, new_volume, new_volumelabels, 'x-plus')  
		NewVolume.voxel_connectivity_types()
		NewVolume.balance_dataset()
		NewVolume.Y_generator()
		NewVolume.X_generator(self.extra_length, self.extra_width, self.extra_height, self.validation_dataset_ratio)
		self.other_three_directions['x-plus'] = NewVolume   

		#x-minus direction: rotate an array by 270 degree counterclockwise.
		new_volume = np.rot90(self.RawVolume, 3, (0,1))           
		new_volumelabels = np.rot90(self.VolumeLabels, 3, (1,2)) 
		NewVolume = MyVolume(None, None, new_volume, new_volumelabels, 'x-minus')  
		NewVolume.voxel_connectivity_types()
		NewVolume.balance_dataset()
		NewVolume.Y_generator()
		NewVolume.X_generator(self.extra_length, self.extra_width, self.extra_height, self.validation_dataset_ratio)
		self.other_three_directions['x-minus'] = NewVolume

	def stack_Xdata_Ydata_six_directions(self):
		''''''
		for myVolume in self.other_three_directions.values():
			if myVolume is not None:
				self.X_train = np.concatenate((self.X_train, myVolume.X_train), axis=0)
				self.X_test = np.concatenate((self.X_test, myVolume.X_test), axis=0)
				self.Y_train = np.concatenate((self.Y_train, myVolume.Y_train), axis=0)
				self.Y_test = np.concatenate((self.Y_test, myVolume.Y_test), axis=0)
		del self.other_three_directions

	def volumelabels_predictions(self):
		''''''
		self.predictions = np.empty([self.L,self.W-1,self.H],dtype=np.int)
		augumented_Volume = utl.image_augmentation(self.RawVolume, self.extra_length, self.extra_width, self.extra_height, method='zero_padding')
		for z in range(self.H):
			inputs = []
			new_z = z + self.extra_height
			for x in range(self.L):
				for y in range(self.W-1):
					new_x, new_y = (x + self.extra_length, y + self.extra_width)
					inputs.append(augumented_Volume[new_x-self.extra_length:new_x+1+self.extra_length,\
												new_y-self.extra_width:new_y+1+self.extra_width,\
												new_z-self.extra_height:new_z+1+self.extra_height, :])
			inputs = np.stack(inputs, axis=0)        
			prediction = self.trained_model.predict_classes(inputs, batch_size=self.W-1, verbose=0)
			self.predictions[:,:,z] = prediction.reshape((self.L,self.W-1), order='C')
		print('predictions for direction {0} completed'.format(self.direction))

	def other_three_predictions(self):
		''''''
		# y-minus direction: rotate an array by 180 degree counterclockwise.
		new_volume = np.rot90(self.RawVolume, 2, (0,1))          
		new_volumelabels = np.rot90(self.VolumeLabels, 2, (1,2))
		NewVolume = MyVolume(None, None, new_volume, new_volumelabels, 'y-minus')
		NewVolume.extra_length, NewVolume.extra_width, NewVolume.extra_height = (self.extra_length, self.extra_width, self.extra_height)
		NewVolume.trained_model = self.trained_model
		NewVolume.volumelabels_predictions()
		self.predictions_dict['y-minus'] = np.rot90(NewVolume.predictions, 2, (0,1))

		#x-plus direction: rotate an array by 90 degree counterclockwise.
		new_volume = np.rot90(self.RawVolume, 1, (0,1))           
		new_volumelabels = np.rot90(self.VolumeLabels, 1, (1,2))
		NewVolume = MyVolume(None, None, new_volume, new_volumelabels, 'x-plus')  
		NewVolume.extra_length, NewVolume.extra_width, NewVolume.extra_height = (self.extra_length, self.extra_width, self.extra_height)
		NewVolume.trained_model = self.trained_model
		NewVolume.volumelabels_predictions()
		self.predictions_dict['x-plus'] = np.rot90(NewVolume.predictions, 3, (0,1))   

		#x-minus direction: rotate an array by 270 degree counterclockwise.
		new_volume = np.rot90(self.RawVolume, 3, (0,1))           
		new_volumelabels = np.rot90(self.VolumeLabels, 3, (1,2)) 
		NewVolume = MyVolume(None, None, new_volume, new_volumelabels, 'x-minus')  
		NewVolume.extra_length, NewVolume.extra_width, NewVolume.extra_height = (self.extra_length, self.extra_width, self.extra_height)
		NewVolume.trained_model = self.trained_model
		NewVolume.volumelabels_predictions()
		self.predictions_dict['x-minus'] = np.rot90(NewVolume.predictions, 1, (0,1))

	def label_image_construction(self):
		'''
		Inputs: predictions_dict -- dictionary that contains six directions' predictions
											('key',value)=(direction, predictions array
											associated with the direction)
											'key'=['y-plus', 'y-minus', 'x-plus', 'x-minus','z-plus','z-minus']
				Length, Width, Height -- three integers that represent the length, width and height of the volume.
		Outputs: reconstruction -- 3D 0/1 label array of shape=(Length, Width, Height). Indicates that a voxel
								   is a neuron or a background.
		'''
		self.reconstruction = np.zeros((self.L, self.W, self.H), dtype=np.int)
		for z in range(self.H):
			predictions_dict_4directions = {'y-plus':self.predictions_dict['y-plus'][:,:,z],
										   'y-minus':self.predictions_dict['y-minus'][:,:,z],
										   'x-plus':self.predictions_dict['x-plus'][:,:,z],
										   'x-minus':self.predictions_dict['x-minus'][:,:,z]}
			
			for x in range(self.L):
				for y in range(self.W):
					self.reconstruction[x,y,z] = utl.current_voxel_label_4direction(predictions_dict_4directions, x, y, self.L, self.W)
		
		
	def neuron_path_each_layer(self):
		''''''
		self.neuron_path={}
		for layer in range(self.H):
			predictions_one_layer={'y-plus':self.predictions_dict['y-plus'][:,:,layer],
								   'y-minus':self.predictions_dict['y-minus'][:,:,layer],
								   'x-plus':self.predictions_dict['x-plus'][:,:,layer],
								   'x-minus':self.predictions_dict['x-minus'][:,:,layer]}
			reconstruction_one_layer = self.reconstruction[:,:,layer]
			self.neuron_path[str(layer)] = utl.neuron_path_one_layer(predictions_one_layer, reconstruction_one_layer)

	def plot_path_each_layer(self):
		import matplotlib.pyplot as plt
		self.reconstructed_volume = np.zeros([self.L, self.W, self.H, 3], dtype=int)
		for layer in range(self.H):
			for i,Coor_list in enumerate(self.neuron_path[str(layer)]):
				rgb = np.random.randint(low=40,high=256, size=3)
				for (x,y) in Coor_list:
					self.reconstructed_volume[x,y,layer,:] = rgb
			f, (ax1, ax2) = plt.subplots(1, 2)
			ax1.imshow(self.reconstructed_volume[:,:,layer,:].astype(np.uint8))
			ax1.set_title('reconstructed volume layer {0}'.format(layer))
			ax2.imshow(self.RawVolume[:,:,layer,:3])
			ax2.set_title('raw volume layer {0}'.format(layer))
			plt.savefig('../Images/RawVolume_ReconstructedVolume_Images/layer_{0}.png'.format(layer))

if __name__ == '__main__':    #code to execute if called from command-line
	#Training using mutiple volumes
	gc.enable()
	training_dataset_path_dict = utl.Volume_Labels_dir_dict('../Sample_Datasets/Training_Multiple_Volumes/')
	train_volume_dict = {}
	myVolume_train_validate = MyVolume()
	myVolume_train_validate.X_train = []
	myVolume_train_validate.Y_train = []
	myVolume_train_validate.X_test = []
	myVolume_train_validate.Y_test = [] 
	for name, values in training_dataset_path_dict.items():
		print("Processing volume: {0}".format(name))
		train_volume_dict[name] = MyVolume(values['overRawVolume'], values['volumeLabels'])
		train_volume_dict[name].load_RawVolume()
		train_volume_dict[name].load_VolumeLabels()
		train_volume_dict[name].voxel_connectivity_types()
		train_volume_dict[name].balance_dataset()
		train_volume_dict[name].Y_generator()
		train_volume_dict[name].X_generator(7, 7, 4, 0.2)
		train_volume_dict[name].other_three_directions_rawvolume_and_volumelabel()
		train_volume_dict[name].stack_Xdata_Ydata_six_directions()
		myVolume_train_validate.X_train.append(train_volume_dict[name].X_train)
		del train_volume_dict[name].X_train
		myVolume_train_validate.Y_train.append(train_volume_dict[name].Y_train)
		del train_volume_dict[name].Y_train
		myVolume_train_validate.X_test.append(train_volume_dict[name].X_test)
		del train_volume_dict[name].X_test
		myVolume_train_validate.Y_test.append(train_volume_dict[name].Y_test)
		del train_volume_dict[name].Y_test
	myVolume_train_validate.X_train = np.concatenate(myVolume_train_validate.X_train, axis=0)
	myVolume_train_validate.Y_train = np.concatenate(myVolume_train_validate.Y_train, axis=0)
	myVolume_train_validate.X_test = np.concatenate(myVolume_train_validate.X_test, axis=0)
	myVolume_train_validate.Y_test = np.concatenate(myVolume_train_validate.Y_test, axis=0)
	myVolume_train_validate.model_train_validate(batch_size=5000, num_classes=2, epochs=15, verbose=1)
	myVolume_train_validate.confusion_matrix()
	model = load_model('neuron_conn_classifier.h5')
	test_dataset_path_dict = utl.Volume_Labels_dir_dict('../Sample_Datasets/Test/')
	test_volume_dict = {}
	for name, values in test_dataset_path_dict.items():
		print("Processing volume: {0}".format(name))
		test_volume_dict[name] = MyVolume(values['overRawVolume'], values['volumeLabels'])
		test_volume_dict[name].load_RawVolume()
		test_volume_dict[name].load_VolumeLabels()
		test_volume_dict[name].extra_length, test_volume_dict[name].extra_width, test_volume_dict[name].extra_height = (7, 7, 4)
		test_volume_dict[name].trained_model = model
		#test_volume_dict[name].volumelabels_predictions()
		#test_volume_dict[name].predictions_dict['y-plus'] = test_volume_dict[name].predictions
		#test_volume_dict[name].other_three_predictions()
		#with open('{}_predictions.pickle'.format(name), 'wb') as f:
			#pickle.dump(test_volume_dict[name].predictions_dict, f)
	
		with open('sim_9cells_4ch_4000bn_0pn_1e-06pd_1.0ef_raw_predictions.pickle', 'rb') as f:
			test_volume_dict[name].predictions_dict = pickle.load(f)
		test_volume_dict[name].label_image_construction()
		#utl.reconstruction_accuracy(test_volume_dict[name].reconstruction, test_volume_dict[name].VolumeLabels)
		test_volume_dict[name].neuron_path_each_layer()
		test_volume_dict[name].plot_path_each_layer()
	gc.collect()






