import os
import cv2
import time
import argparse

from utils import last_count
from utils import save_image
from utils import region_of_interest



def generate_data(device: int = 0) -> None:
	"""
	Captures images through the webcam and saves them in the specified path.

	Keyword Arguments:
		device {int} -- webcam device to use (default: {0})
	"""
	# get the path to the label
	path = os.path.join(args.data, args.label)
	os.makedirs(path, exist_ok=True)

	# get the last numeric count
	start = last_count(path)

	# give the user a 10-second grace period before starting the capture
	capture = cv2.VideoCapture(device)
	grace = 10
	t_end = time.time() + grace
	print(f'Get ready and position yourself! Starting capture in {grace} seconds...')

	cnt = i = 0
	while cnt < args.images:
		_, frame = capture.read()
		frame = cv2.flip(frame, 1)
		roi = region_of_interest(frame)

		# show results
		cv2.imshow('Live Feed', frame)

		# force exit condition
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print('You force quit.')
			break

		if time.time() <= t_end:	# discard frames captured during the grace period
			continue
		elif i % args.fps == 0:		# save image after fps frames
			cnt += 1
			file_name = f'{args.label}_{start + cnt}.png'
			file_path = os.path.join(path, file_name)
			save_image(roi, file_path)
			print(f'Generated image {cnt} out of {args.images}.')
		
		i += 1
	
	# clean-up
	capture.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	# add keyword arguments
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', '-d', help='Path to a directory to output generated images', required=True)
	parser.add_argument('--label', '-l', help='Label of the generated images', required=True)
	parser.add_argument('--images', '-i', help='Number of images to generate', type=int, default=500)
	parser.add_argument('--fps', '-f', help='Number of frames to skip before capturing the next', type=int, default=10)
	args = parser.parse_args()
	
	generate_data()