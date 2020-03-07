import cv2
import json
import numpy as np
from config import *



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