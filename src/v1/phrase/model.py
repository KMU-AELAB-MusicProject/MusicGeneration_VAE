import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Reshape, concatenate, AveragePooling2D, GRU, multiply

# from config import *
from .encoder import Encoder
from .decoder import Decoder


class PhraseModel(tf.keras.Model):
    def __init__(self):
        super(PhraseModel, self).__init__(name='phrase_model')
        tf.keras.backend.set_floatx('float64')
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.phrase_number = tf.Variable(name='phrase_number', trainable=True,
                                         initial_value=tf.random.normal(shape=[331, 510], stddev=0.5, dtype=tf.float64))

    def call(self, train_data, pre_phrase, position_number):
        z, z_mean, z_var = self.encoder(train_data)  # train-phrase
        z_pre, _, _ = self.encoder(pre_phrase)  # pre-phrase

        logits = self.decoder(z + z_pre + tf.keras.backend.gather(self.phrase_number, position_number))

        reshape_logits = Reshape(target_shape=[384, 96])(logits)
        binary_note = tf.cast(tf.keras.backend.greater(reshape_logits, 0.35), dtype=tf.float64)
        outputs = multiply([binary_note, reshape_logits], dtype=tf.float64)

        return outputs, binary_note, z, z_mean, z_var, tf.cast(tf.keras.backend.greater(train_data, 0.35), dtype=tf.float64)

    def get_feature(self, input):
        z, _, _ = self.encoder(input)
        return z

    def make_music(self, pre_phrase, position_number):
        self.decoder(self.encoder(pre_phrase) + tf.keras.backend.gather(self.phrase_number, position_number) +
                     tf.random.normal(shape=(1, 510)))
        return

if __name__ == '__main__':
    t = PhraseModel()
    # t.compile()
    # t.build((10, 384, 96), (10, 384, 96), (10))
    # t.summary()