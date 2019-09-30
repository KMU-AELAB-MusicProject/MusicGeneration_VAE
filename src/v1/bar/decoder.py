import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# from config import *


class Decoder(layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__(name='bar_decoder')

    def call(self, inputs):
        tf.keras.backend.set_floatx('float64')

        x = layers.Reshape(target_shape=[1, 1, 510])(inputs)

        x1_1 = layers.Conv2DTranspose(filters=510, kernel_size=[6, 6], activation='relu')(x)

        # pitch-time extend
        x1_2_1 = layers.Conv2DTranspose(filters=510, kernel_size=[1, 3], strides=[1, 3], activation='relu',
                                        padding='same')(x)
        x1_2_2 = layers.Conv2DTranspose(filters=510, kernel_size=[12, 1], strides=[12, 1], activation='relu',
                                        padding='same')(x1_2_1)

        x1_2 = layers.Conv2DTranspose(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                      padding='same')(x1_2_2)

        # time-pitch extend
        x1_3_1 = layers.Conv2DTranspose(filters=510, kernel_size=[12, 1], strides=[12, 1], activation='relu',
                                        padding='same')(x)
        x1_3_2 = layers.Conv2DTranspose(filters=510, kernel_size=[1, 3], strides=[1, 3], activation='relu',
                                        padding='same')(x1_3_1)

        x1_3 = layers.Conv2DTranspose(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                      padding='same')(x1_3_2)

        x2 = x1_1 + x1_2 + x1_3 # [6, 6, 510]

        x3 = layers.Conv2DTranspose(filters=256, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                    padding='same')(x2)
        x4 = layers.Conv2DTranspose(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                    padding='same')(x3)
        x5 = layers.Conv2DTranspose(filters=64, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                    padding='same')(x4)
        x6 = layers.Conv2DTranspose(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                    padding='same')(x5)

        # GRU feature extract
        with tf.name_scope('GRU'):
            x_r = layers.Conv2DTranspose(filters=32, kernel_size=[3, 1], strides=[2, 1], activation='relu',
                                         padding='same')(x5)
            x_r = layers.Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')(x_r)
            x_r = layers.Reshape(target_shape=[96, 48])(x_r)
            x_r = layers.GRU(units=96, return_sequences=True, recurrent_initializer='glorot_uniform')(x_r)
            x_r = layers.Reshape([96, 96, 1])(x_r)

        x7 = layers.concatenate([x6, x_r], axis=3)
        logits = layers.Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], padding='same')(x7)

        return logits
