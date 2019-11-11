import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# from config import *


class Encoder(layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__(name='bar_encoder')

    def call(self, inputs):
        x = layers.Reshape(target_shape=[96, 96, 1])(inputs)
        # pitch feature
        x1 = layers.Conv2D(filters=32, kernel_size=[1, 12], strides=[1, 2], activation='relu', padding='same')(x)

        # CNN feature extract
        with tf.name_scope('CNN_feature'):
            # step 1
            # pitch-time feature
            x1_1 = layers.Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same') \
                (x1)

            # time-pitch feature
            x1_2 = layers.Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same') \
                (x)
            x1_2 = layers.Conv2D(filters=32, kernel_size=[1, 12], strides=[1, 2], activation='relu', padding='same') \
                (x1_2)

            # step 2
            x2 = layers.concatenate([x1_1, x1_2], axis=3)
            x2 = layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')(x2)
            x2 = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')(x2)

            x2_1 = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same')(x2)
            x2_1 = layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same') \
                (x2_1)

            # step 3
            x3 = layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same') \
                (x2 + x2_1)
            x3 = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')(x3)

            x3_1 = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same')(x3)
            x3_1 = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same') \
                (x3_1)

            # step 4
            x4 = layers.Conv2D(filters=510, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same') \
                (x3 + x3_1)
            x4 = layers.Conv2D(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')(x4)
            x4 = layers.AveragePooling2D(pool_size=[6, 6])(x4)
            x4 = layers.Reshape(target_shape=[510])(x4)

        # GRU feature extract
        with tf.name_scope('GRU_feature'):
            x_r = layers.Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')(x1)
            x_r = layers.Reshape(target_shape=[96, 48])(x_r)
            x_r = layers.GRU(units=510, return_sequences=False, recurrent_initializer='glorot_uniform')(x_r)

        x5 = layers.concatenate([x4, x_r], axis=1)

        z_mean = layers.Dense(510)(x5)
        z_var = layers.Dense(510)(x5)

        eps = tf.random.normal(shape=tf.shape(z_mean))
        return (eps * tf.exp(z_var * .5) + z_mean), z_mean, z_var   # z, z-mean, z-var
