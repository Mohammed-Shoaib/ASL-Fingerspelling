import cv2
import json
import pickle
import numpy as np
from config import *
from typing import TypeVar



T = TypeVar('T')



def read_json(path: str) -> dict:
	"""
	Reads a json file given by the path.
	
	Arguments:
		path {str} -- path of the json file to be read
	
	Returns:
		dict -- label, encoding pairs which describe the mapping
	"""
	with open(path, 'r') as f:
		return json.load(f)



def read_image(path: str) -> np.ndarray:
	"""
	Reads an image given by the path.
	
	Arguments:
		path {str} -- the path of the image to be read
	
	Returns:
		np.ndarray -- an array of image pixels
	"""
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	# ! conversion to color is permanent, # channels always 3
	if len(img.shape) != 3:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	return np.asarray(img)



def serialize(data: T, path: str) -> None:
	"""
	Serializes the given data and stores the binary in the given path.
	
	Arguments:
		data {T} -- object to serialize
		path {str} -- path to output serialized object
	"""
	with open(path, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



def deserialize(path: str) -> T:
	"""
	Deserializes the binary in the given path.
	
	Arguments:
		path {str} -- path to serialized object
	
	Returns:
		T -- deserialized object
	"""
	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data



def resize_image(img: np.ndarray) -> np.ndarray:
	"""
	Resizes an image in (width, height, channels) format with padding to keep the aspect ratio.
	
	Arguments:
		img {np.ndarray} -- an array of image pixels
	
	Returns:
		np.ndarray -- resized image of shape [SHAPE✕SHAPE✕CHANNELS]
	"""
	# resize the image
	old_shape = img.shape[1::-1]
	ratio = SHAPE / max(old_shape)
	new_shape = tuple([int(dim * ratio) for dim in old_shape])
	img = cv2.resize(img, new_shape)

	# pad the image
	color = [0, 0, 0]
	w, h = [SHAPE - dim for dim in new_shape]
	top, bottom = h // 2, h - h // 2
	left, right = w // 2, w - w // 2
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

	return img



def shuffle_in_unison(a: np.ndarray, b: np.ndarray) -> None:
	"""
	Shuffles 2 numpy arrays together inplace.
	AssertionError if the lengths of a & b are not the same.
	
	Arguments:
		a {np.ndarray} -- a numpy array
		b {np.ndarray} -- a numpy array
	"""
	assert len(a) == len(b)
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)