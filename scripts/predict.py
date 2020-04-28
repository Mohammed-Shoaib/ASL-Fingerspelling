import os
import sys
import argparse

from utils import *
from random import randrange
from keras.models import load_model
from collections import deque, Counter



def get_label(index: int) -> str:
	"""
	Finds the corresponding label given an index.

	Arguments:
		index {int} -- the index of the label

	Returns:
		str -- the label found using mapping.json
	"""
	for label, idx in  mapping.items():
		if idx == index:
			return label
	raise Exception(f'Could not find the matching label for the index {index} in mapping.json')



def predict(x: np.ndarray) -> np.ndarray:
	"""
	Predicts the label for a given image.

	Arguments:
		x {np.ndarray} -- an array of image pixels

	Returns:
		np.ndarray -- the model predictions for each label
	"""
	x = np.expand_dims(x, axis=0)
	x_activation = learning_model.predict(x)
	output = model.predict(x_activation)[0]
	return output



def predict_random(test_xs: np.ndarray, test_ys: np.ndarray, verbose:bool = False, show: bool = False) -> bool:
	"""
	Predicts a random image from the test set.

	Arguments:
		test_xs {np.ndarray} -- an array of images
		test_ys {np.ndarray} -- the label for each image in xs

	Keyword Arguments:
		verbose {bool} -- prints the predicted label and the actual label (default: {False})
		show {bool} -- displays the image (default: {False})

	Returns:
		bool -- True if the predicted label matches the actual label else False
	"""
	# pick a random image
	i = randrange(len(test_xs))

	# get the model's prediction
	output = get_label(predict(test_xs[i]).argmax())
	actual = get_label(test_ys[i].argmax())
	
	if verbose:
		print(f'Prediction: {output}, Actual: {actual}')
	if show:
		show_image(test_xs[i])
	
	return output == actual



def predict_live(device:int = 0):
	"""
	Live prediction through the webcam feed.

	Keyword Arguments:
		device {int} -- webcam device to use (default: {0})
	"""
	Q = deque(maxlen=10)
	capture = cv2.VideoCapture(device)

	print('Starting live prediction...')
	while True:
		# capture frame-by-frame
		_, frame = capture.read()
		frame = cv2.flip(frame, 1)

		# get region of interest (roi)
		offset = 125
		height, width, _ = frame.shape
		x1, x2 = 150 - offset, 150 + offset
		y1, y2 = height // 2 - offset, height // 2 + offset
		roi = frame[y1:y2, x1:x2]
		cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

		# rolling average predictions
		output = predict(roi).argmax()
		Q.append(output)
		label = get_label(Counter(Q).most_common(1)[0][0])

		# show results
		cv2.putText(img=frame, text=label, org=(50, height - 50), 
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
					fontScale=2.0, color=(0, 0, 0), 
					lineType=cv2.LINE_AA, thickness=6)
		cv2.putText(img=frame, text=label, org=(50, height - 50), 
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
					fontScale=2.0, color=(255, 255, 255), 
					lineType=cv2.LINE_AA, thickness=2)
		cv2.imshow('Live Prediction', frame)

		# exit condition
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# clean-up
	capture.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	# adding the keyword arguments
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', '-d', help='Path to an input serialized dataset folder', required=True)
	parser.add_argument('--learning-model', '-lm', help='Transfer learning model used to generate the activations', choices=MODELS.keys(), required=True)
	parser.add_argument('--model', '-m', help='Path to the keras model to be used for prediction (.h5 extension)', required=True)
	parser.add_argument('--samples', '-s', help='Number of samples to evaluate the model', type=int, default=0)
	parser.add_argument('--live', '-l', help='Specify flag to use live demo prediction', action='store_true')
	args = parser.parse_args()

	# error handling
	if not os.path.exists(args.data):
		sys.exit('The path given to the serialized dataset does not exist.')
	elif not os.path.exists(args.model):
		sys.exit('The path given to the keras model does not exist.')
	elif args.live < 0:
		sys.exit(f'Number of samples must be >= 0, found {args.samples}')
	
	# load model
	learning_model = load_learning_model(args.learning_model)
	learning_model.summary()
	model = load_model(args.model)
	model.summary()

	print('Loading the serialized dataset...')
	test_xs = deserialize(os.path.join(args.data, 'test_xs.ser'))
	test_ys = deserialize(os.path.join(args.data, 'test_ys.ser'))

	# live demo prediction
	if args.live:
		predict_live()
	
	# evaluate on random samples
	if args.samples:
		print(f'Evaluating the model on {args.samples} samples...')
		correct = 0
		for i in range(args.samples):
			correct += 1 if predict_random(test_xs, test_ys) else 0
		accuracy = (correct / args.samples) * 100
		print(f'Accuracy: {accuracy:.2f}')