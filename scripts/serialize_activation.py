import os
import sys
import argparse
import numpy as np
from math import ceil
from utils import serialize, deserialize, load_learning_model, MODELS



def get_activations(xs: np.ndarray, start: int, end: int) -> np.ndarray:
	"""
	Uses the transfer learning model and gets the internal model activations for each image.
	
	Arguments:
		xs {np.ndarray} -- an array of images
		start {int} -- starting index of xs to chunk, inclusive
		end {int} -- ending index of xs to chunk, non-inclusive
	
	Returns:
		np.ndarray -- the model activation for each image in xs
	"""
	activations = []
	
	for i, img in enumerate(xs[start:end]):
		img = np.expand_dims(img, axis=0)
		activation = learning_model.predict(img)
		activation = np.squeeze(activation, axis=0)
		activations.append(activation)
		print(f'Finished getting activation for image {start + i} out of {len(xs)}.')
	
	return np.asarray(activations)



# adding the keyword arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', '-d', help='Path to an input serialized dataset folder', required=True)
parser.add_argument('--output', '-o', help='Path to a directory to output serialized model activations', required=True)
parser.add_argument('--chunk-size', '-c', help='Number of images to chunk', type=int, default=10**4)
parser.add_argument('--learning-model', '-lm', help='Transfer learning model used to generate the activations', choices=MODELS, type=str.lower, default=MODELS[0])
args = parser.parse_args()



if __name__ == '__main__':
	# error handling
	if not os.path.exists(args.data):
		sys.exit('The path given to the serialized dataset does not exist.')
	
	print(f'Loading the {args.learning_model} model...')
	learning_model = load_learning_model(args.learning_model)

	print('Loading the serialized xs...')
	train_xs = deserialize(os.path.join(args.data, 'train_xs.ser'))
	test_xs = deserialize(os.path.join(args.data, 'test_xs.ser'))
	
	# chunk model activations
	chunks = ceil(len(train_xs)/args.chunk_size)
	for i in range(chunks):
		start = i * args.chunk_size

		# get model activations for train_xs
		end = min(start + args.chunk_size, len(train_xs))
		train_activations = get_activations(train_xs, start, end)
		
		# get model activations for test_xs
		end = min(start + args.chunk_size, len(test_xs))
		test_activations = get_activations(test_xs, start, end)

		print(f'Serializing chunk {i + 1} out of {len(chunks)}...')
		variables = ['train_activations', 'test_activations']
		for v in variables:
			# create the directory if it does not exist
			path = os.path.join(args.output, f'{args.learning_model}_{i}')
			os.makedirs(path, exist_ok=True)

			# serialize model activation
			path = os.path.join(path, f'{v}.ser')
			serialize(locals()[v], path)