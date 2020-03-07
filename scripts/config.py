import json
import tensorflow as tf



with open('mapping.json', 'r') as f:
    mapping = json.load(f)
EPOCHS = 64
SHAPE = 250
CHANNELS = 3
BATCH_SIZE = 128
NUM_CLASSES = len(mapping)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)