import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# from config import *
from .encoder import Encoder
from .decoder import Decoder


class BarModel(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, phrase_model):
        super(BarModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.phrase_model = phrase_model

        self.phrase_number = tf.Variable(name='phrase_number', trainable=True,
                                         initial_value=tf.random.normal(shape=[5, 510], stddev=0.5, dtype=tf.float64))

    def call(self, train_data, pre_phrase, position_number):
        tf.keras.backend.set_floatx('float64')

        z, z_mean, z_var = self.encoder(train_data)  # train-phrase
        z_pre, _, _ = self.phrase_model.encoder(pre_phrase)  # pre-phrase

        logits = self.decoder(z + z_pre + tf.keras.backend.gather(self.phrase_number, position_number, dtype=tf.int32))

        cond = tf.keras.backend.greater_equal(logits, 0.3)
        outputs = tf.keras.layers.multiply([tf.cast(cond, dtype=tf.float64), logits])

        return outputs, logits, z, z_mean, z_var, tf.cast(tf.keras.backend.greater(train_data, 0.35), dtype=tf.float64)

    def get_feature(self, inputs):
        z, _, _ = self.encoder(inputs)
        return z

    def make_music(self, inputs):
        self.decoder(self.encoder(inputs[0]) + tf.keras.backend.gather(self.phrase_number, inputs[1]) +
                     tf.random.normal(shape=(1, 510)))
        return

if __name__ == '__main__':
    t = BarModel()
    t.build((10, 384, 96), (10, 384, 96), (10))
    t.summary()