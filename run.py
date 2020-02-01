import os
import sys
import pickle
import argparse
import importlib
import pypianoroll

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import *
from src.phrase_discriminator import PhraseDiscriminatorModel

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', help="The train number number to start train.", action='store_true')
parser.add_argument('--seq_size', type=int, default=10, help="The GPU device number to use.")
parser.add_argument('--gpu_number', type=str, help="The GPU device number to use.")
parser.add_argument('--model_number', type=str, help="The GPU device number to use.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')

    import_model = importlib.import_module('src.v{}.phrase.model'.format(args.model_number))
    save_path_tb = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'phrase_train_best')

    model = import_model.PhraseModel()

    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(0.0001), model=model)

    latest = tf.train.latest_checkpoint(save_path_tb)
    checkpoint.restore(latest)

    outputs = []
    pre_phrase = np.zeros([1, 384, 96], dtype=np.float64)
    phrase_idx = [330] + [i for i in range(args.seq_size - 2, -1, -1)]

    for idx in range(args.seq_size):
        pre_phrase = checkpoint.model.test(pre_phrase, np.array([phrase_idx[idx]], dtype=np.float64))
        outputs.append(np.reshape(np.array(pre_phrase), [96 * 4, 96, 1]))

    note = np.array(outputs)
    note = note.reshape(96 * 4 * args.seq_size, 96) * 127
    note = np.pad(note, [[0, 0], [25, 7]], mode='constant', constant_values=0.)

    track = pypianoroll.Track(note, name='piano')
    pianoroll = pypianoroll.Multitrack(tracks=[track], beat_resolution=24, name='test')
    print(pianoroll)
    pianoroll.write('./test.mid')


