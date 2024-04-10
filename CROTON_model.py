import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
import random
from keras import backend as bk
import matplotlib.pyplot as plt
import plotting
import pandas as pd 
from enum import Enum

tf.config.set_visible_devices([], 'GPU')
	
# Define the search space (simplified for demonstration)
search_space = {
    'num_filters': [16, 32, 64],
    'kernel_size': [3, 5],
    'pooling': ['max', 'avg'],
    'activation': ['relu', 'sigmoid']
}

class CROTON:

	#Initialize and define a model with random architecture:
	def __init__(self, input_array, output_array):		
		#initialize global variables:
		
			#input/output state space:
		self.input_dim = input_array.shape
		self.output_dim = output_array.shape
		
			#build the CNN model:
		architecture = self.create_random_architecture(search_space)
		self.model = croton.build_model(architecture)
		self.initState = self.model
	
	# Imputation of DNA sequence:
	def impute(self, DNA_sequence):
		random_index = random.choice(arange(DNA_sequence.size))
		DNA_sequence[random_index] = "-"
		
		return DNA_sequence
		
	# Define a function to create a random architecture
	def create_random_architecture(self, search_space):
		architecture = {}
		for key, options in search_space.items():
			architecture[key] = np.random.choice(options)
		return architecture
	
	#build CNN model from specified architecture
	def _build_model(self, architecture):
		#determine dataset sizes:
		input_size = self.input_dim
		output_size = self.output_dim
		
		# Neural Net Model:
			#define model:
		model = Sequential()
		model.add(layers.Conv2D(architecture['num_filters'], 
                             architecture['kernel_size'], 
                             activation=architecture['activation'], 
                             input_dim=input_size)) 
		if architecture['pooling'] == 'max':
			model.add(layers.MaxPooling2D((2, 2)))
		else:
			model.add(layers.AveragePooling2D((2, 2)))
		model.add(layers.Flatten())
		model.add(layers.Dense(ouput_size, activation='softmax'))
		
			#compile model
		model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=Adam(lr=0.001)) #self.alpha
		return model
			
	def reset(self):
	
		#redefine a new CNN model with a randomized architecture:
		architecture = self.create_random_architecture(search_space)
		self.prevState = self.model #save previous model
		self.model = croton.build_model(architecture)
		
		return self.prevState #return previous model
		
	def test(self, xtest, ytest):
	
		#test the network model:
		return self.model.evaluate(np.array(xtest), np.array(ytest))
		
	def train(self, xtrain, ytrain):
			 
		#train the network model:
		return self.model.fit(np.array(xtrain), np.array(ytrain), batch_size = self.batch_size,  epochs = 5)

#To load in data from a given file_path:
def read_file_and_separate(file_path):
	array = []
	try:
		with open(file_path, 'r') as file:
			for line in file:
				tokens = line.split()
				array.append([tokens])
	except FileNotFoundError:
		print(f"File not found: {file_path}")
	except Exception as e:
		print(f"An error occurred: {e}")
		
	return array
	
def run_croton():			
	
	#initially retrieve dataset:
	data = read_file_and_separate("Data.txt")	# get data from Omar 
	label = read_file_and_separate("Label.txt") #get from Crispr results
	
	#preprocess dataset:
	input_array = np.array(data)
	output_array = np.array(label)
	
		#splitting test and train datasets
	size_dataset = input_array.shape[1]
	train_split = round(0.7 * size_dataset) #70% train/val 30% test
	
	xtrain = input_array[0:train_split, ]
	ytrain = output_array[0:train_split, ]
	
	xtest = input_array[train_split:size_dataset+1, ]
	ytest = output_array[train_split:size_dataset+1, ]
		#more preprocess to come...
	
	#initialize croton model with random architecture:
	croton = CROTON(input_array, output_array)

	#train croton model for 10 episodes:
	croton_models = []
	croton_performance = []
	while episode_num < 10:	
		croton.train(xtrain, ytrain)
		test_loss, test_acc = croton.test(xtest, ytest)
		
		croton_performance.append(test_acc)
		croton_models.append(croton.reset())
		
		episode_num += 1	
		
	#select the best one based on performance:
	max_acc = max(croton_performance)
	best_model = croton
	for i,j in enumerate(croton_performance):
		if (i == max_acc):
			best_model = croton_models[j]		
	
if __name__ == "__main__":
    run_croton()
		