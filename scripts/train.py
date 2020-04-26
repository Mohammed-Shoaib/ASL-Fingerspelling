import os
import sys
import keras
import argparse
import numpy as np

from utils import *
from config import *
from pathlib import Path
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, DepthwiseConv2D
from keras.layers import UpSampling2D, MaxPooling2D



def deserialize_activation(path: str, name: str) -> np.ndarray:
	"""
	Loads the serialized activations and merges them into a single numpy ndarray.

	Arguments:
		path {str} -- path to the serialized activations
		name {str} -- type of serialized activations {train_activations.ser, test_activations.ser}

	Returns:
		np.ndarray -- a single ndarray created by merging the deserialized activations
	"""
	# sort by index
	dirs = sorted(os.listdir(path), key = lambda x: int(x.split('_')[-1]))
	if name != 'test_activations.ser':
		dirs = dirs[args.start // args.chunk_size : args.end // args.chunk_size]
	
	# load serialized activations
	activations = []
	for directory in dirs:
		file_name = os.path.join(path, directory, name)
		activation = deserialize(file_name)
		if len(activation.shape) == 4:
			activations.append(activation)
	
	# merge into a single numpy ndarray
	if activations:
		activations = np.vstack(activations)
	
	return activations



def create_model() -> Sequential:
	"""
	Creates a sequential model.

	Returns:
		Sequential -- training model
	"""
	model = Sequential()

	# Dense layers
	model.add(Dense(units=1024,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=512,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=256,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=128,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	
	# UpSampling layer
	model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
	
	# Conv layers
	model.add(Conv2D(kernel_size=4,
					 filters=64,
					 strides=1,
					 activation='relu',
					 use_bias=True,
					 bias_initializer='zeros',
					 kernel_initializer='VarianceScaling'))
	model.add(DepthwiseConv2D(kernel_size=4,
							  strides=(1, 1),
							  padding='valid',
							  depth_multiplier=1,
							  use_bias=True,
							  bias_initializer='zeros',
							  depthwise_initializer='glorot_uniform'))
	
	# UpSampling layer
	model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))

	# Conv layers
	model.add(DepthwiseConv2D(kernel_size=4,
							  strides=(1, 1),
							  padding='valid',
							  depth_multiplier=1,
							  use_bias=True,
							  bias_initializer='zeros',
							  depthwise_initializer='glorot_uniform'))
	model.add(Conv2D(kernel_size=4,
					 filters=64,
					 strides=1,
					 activation='relu',
					 use_bias=True,
					 bias_initializer='zeros',
					 kernel_initializer='VarianceScaling'))

	# Dense layers
	model.add(Dense(units=128,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=256,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=512,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=1024,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=2048,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=1024,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=512,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=256,
					activation='relu',
					kernel_initializer='glorot_uniform'))
	model.add(Dense(units=128,
					activation='relu',
					kernel_initializer='glorot_uniform'))

	# Pooling layer to halve the output from the previous layer
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

	# Flatten layer to flatten the output of the previous layer to a vector
	model.add(Flatten())

	# Output: fully-connected Dense layer
	model.add(Dense(units=NUM_CLASSES,
					activation='softmax',
					kernel_initializer='VarianceScaling'))

	model.add(Dense(units=24,
					activation='softmax',
					kernel_initializer='VarianceScaling'))

	# compiling the model
	adam = keras.optimizers.Adam(lr=0.001)
	model.compile(optimizer=adam,
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	
	return model



def train(train_xs: np.array, train_ys: np.array, test_xs: np.array, test_ys: np.array) -> None:
	"""
	Trains the model with the validation data.

	Arguments:
		train_xs {np.array} -- an array containing the pixel values for each image in training data
		train_ys {np.array} -- the label for each image in training data
		test_xs {np.array} -- an array containing the pixel values for each image in test data
		test_ys {np.array} -- the label for each image in test data
	"""
	print('Training the model...')
	model.fit(train_xs, train_ys,
			  batch_size=BATCH_SIZE,
			  validation_data=(test_xs, test_ys),
			  epochs=EPOCHS,
			  shuffle=True,
			  verbose=1)
	print('Finished training the model.')



def evaluate(test_xs: np.array, test_ys: np.array) -> None:
	"""
	Evaluates the performance of the model on the test dataset.
	
	Arguments:
		test_xs {np.array} -- an array containing the pixel values for each image in test data
		test_ys {np.array} -- the label for each image in test data
	"""
	score = model.evaluate(test_xs, test_ys, verbose=0)
	print('Test Loss:', score[0])
	print('Test Accuracy: ', score[1])



# adding the keyword arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', '-d', help='Path to an input serialized dataset folder', required=True)
parser.add_argument('--activation', '-a', help='Path to an input serialized activations folder', required=True)
parser.add_argument('--learning-model', '-lm', help='Transfer learning model used to generate the activations', choices=MODELS.keys(), required=True)
parser.add_argument('--model', '-m', help='Path to a directory to create/update model', required=True)
parser.add_argument('--chunk-size', '-c', help='Number of images in a single serialized activation', type=int, default=10**4)
parser.add_argument('--start', '-s', help='Start index of dataset to train, start-index = index * chunk-size', type=int, required=True)
parser.add_argument('--end', '-e', help='End index of dataset to train, end-index = index * chunk-size', type=int, required=True)
args = parser.parse_args()



if __name__ == '__main__':
	# error handling
	if not os.path.exists(args.data):
		sys.exit('The path given to the serialized dataset does not exist.')
	elif not os.path.exists(args.activation):
		sys.exit('The path given to the serialized activations does not exist.')
	
	# create model if it doesn't exist
	learning_model = load_learning_model(args.learning_model)
	if not os.path.exists(args.model):
		parent = Path(args.model).parent
		os.makedirs(parent, exist_ok=True)
		model = create_model()
		model.build(learning_model.layers[-1].output_shape)
	else:
		model = load_model(args.model)
		print('Model loaded!')
	model.summary()

	print('Loading the serialized dataset...')
	train_xs = deserialize(os.path.join(args.data, 'train_xs.ser'))[args.start : args.end]
	train_ys = deserialize(os.path.join(args.data, 'train_ys.ser'))[args.start : args.end]
	test_xs = deserialize(os.path.join(args.data, 'test_xs.ser'))
	test_ys = deserialize(os.path.join(args.data, 'test_ys.ser'))

	print('Loading the serialized activations...')
	train_activations = deserialize_activation(args.activation, 'train_activations.ser')
	test_activations = deserialize_activation(args.activation, 'test_activations.ser')
	
	print('Shapes of parameters:')
	print('Train activations: ', train_activations.shape)
	print('Test activations: ', test_activations.shape)
	print('Train xs: ', train_xs.shape)
	print('Train ys: ', train_ys.shape)
	print('Test xs: ', test_xs.shape)
	print('Test ys: ', test_ys.shape)

	# train and test
	train(train_activations, train_ys, test_activations, test_ys)
	evaluate(test_activations, test_ys)

	# saving the model and it's weights
	model.save(args.model)