import os
import sys
import argparse
import numpy as np
from utils import *
from typing import Tuple
from keras.utils import to_categorical



def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Takes a path to the dataset.
	Each folder must be a label containing the raw data.
	
	Arguments:
		path {str} -- path to the dataset 
	
	Returns:
		Tuple[np.ndarray, np.ndarray] -- tuple of two items (images or xs, labels or ys)
	"""
	xs, ys = [], []
	labels = os.listdir(path)

	# open each label folder
	for label in labels:
		print(f'Loading label {label}...')
		files = os.listdir(os.path.join(path, label))
		
		# open each image and store in data
		for file_name in files:
			file_path = os.path.join(path, label, file_name)
			xs.append(read_image(file_path))
			ys.append(mapping[label])

	return np.asarray(xs), np.asarray(ys)



def preprocess_data(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	1. Resizes each image to a shape [SHAPE✕SHAPE✕CHANNELS] with padding to keep the aspect ratio.
	2. Shuffles the xs & ys together.
	3. One-hot encodes the labels or ys.
	
	Arguments:
		xs {np.ndarray} -- an array of images
		ys {np.ndarray} -- the label for each image in xs
	
	Returns:
		Tuple[np.ndarray, np.ndarray] -- tuple of two items (images or xs, labels or ys)
	"""
	print(f'Resizing the xs to a shape of [{SHAPE}✕{SHAPE}✕{CHANNELS}]...')
	for i, img in enumerate(xs):
		xs[i] = resize_image(img)
	
	print('Shuffling the data...')
	shuffle_in_unison(xs, ys)
	shuffle_in_unison(xs, ys)
	shuffle_in_unison(xs, ys)
	
	print('One-hot encoding the ys...')
	ys = to_categorical(ys, NUM_CLASSES).astype(int)
	
	return xs, ys



def split_data(xs: np.ndarray, ys: np.ndarray, offset: float = 0.8) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
	"""
	Splits the dataset into train and test sets.
	
	Arguments:
		xs {np.ndarray} -- an array of images
		ys {np.ndarray} -- the label for each image in xs
	
	Keyword Arguments:
		offset {float} -- train:test split ratio (default: {0.8})
	
	Returns:
		Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] -- tuple of two items (train set, test set)
	"""
	print('Splitting the data...')
	offset = int(len(xs) * offset)
	train_xs, train_ys = xs[:offset], ys[:offset]
	test_xs, test_ys = xs[offset:], ys[offset:]

	return (train_xs, train_ys), (test_xs, test_ys)



# add keyword arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', '-d', help='Path to an input dataset folder', required=True)
parser.add_argument('--output', '-o', help='Path to a directory to output serialized dataset', required=True)
parser.add_argument('--mapping', '-m', help='Path to input json mapping of letters', default='mapping.json')
args = parser.parse_args()



if __name__ == '__main__':
	# error handling
	if not os.path.exists(args.data):
		sys.exit('The path to the dataset folder does not exist.')
	elif not os.path.exists(args.mapping):
		sys.exit('The path to the mapping does not exist.')
	os.makedirs(args.output, exist_ok=True)
	
	# get the data
	xs, ys = load_data(args.data)
	xs, ys = preprocess_data(xs, ys)
	(train_xs, train_ys), (test_xs, test_ys) = split_data(xs, ys)

	# serialize objects
	variables = ['xs', 'ys', 'train_xs', 'train_ys', 'test_xs', 'test_ys']
	for v in variables:
		print(f'Serializing the {v}...')
		path = os.path.join(args.output, f'{v}.ser')
		serialize(locals()[v], path)