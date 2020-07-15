import tensorflow as tf
from model import Magikarp
import numpy as np

config = {}
config['batch_size'] = 64
config['datafile'] = 'training_data.hdf5'
config['p_datafile'] = 'player_data.hdf5'
config['full_boards_file'] = 'full_boards.pkl'
config['num_epochs'] = 10
config['save_file'] = 'trained_model/trained_genadv.ckpt'

with tf.compat.v1.Session() as sess:
	magikarp = Magikarp(config, sess)
	magikarp.train()
