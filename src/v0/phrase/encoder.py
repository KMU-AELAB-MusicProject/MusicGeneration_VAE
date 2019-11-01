import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Reshape, concatenate, AveragePooling2D, GRU


# from config import *


class Encoder(layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__(name='phrase_encoder')

        self.x1 = Conv2D(filters=32, kernel_size=[1, 12], strides=[1, 2], activation='relu', padding='same')
        self.x1_1 = Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same')
        self.x1_2 = Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same')
        self.x1_2_fit = Conv2D(filters=32, kernel_size=[1, 12], strides=[1, 2], activation='relu', padding='same')

        self.x2_fit = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')
        self.x2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')

        self.x2_1_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same')
        self.x2_1_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same')

        self.x3_fit = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')
        self.x3 = Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x3_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same')
        self.x3_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same')

        self.x4_fit = Conv2D(filters=510, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')
        self.x4 = Conv2D(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x4_1 = AveragePooling2D(pool_size=[24, 6])

        self.xr_fit = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')
        self.xr = GRU(units=510, return_sequences=False, recurrent_initializer='glorot_uniform')

        self.mean = layers.Dense(510)
        self.var = layers.Dense(510)

    def call(self, input):
        x = Reshape(target_shape=[384, 96, 1])(input)
        # pitch feature
        x1 = self.x1(x)

        # CNN feature extract
        with tf.name_scope('CNN_feature'):
            # step 1
            # pitch-time feature
            x1_1 = self.x1_1(x1)

            # time-pitch feature
            x1_2 = self.x1_2(x)
            x1_2 = self.x1_2_fit(x1_2)

            # step 2
            x2 = concatenate([x1_1, x1_2], axis=3)
            x2 = self.x2_fit(x2)
            x2 = self.x2(x2)

            x2_1 = self.x2_1_1(x2)
            x2_1 = self.x2_1_2(x2_1)

            # step 3
            x3 = self.x3_fit(x2 + x2_1)
            x3 = self.x3(x3)
            x3_1 = self.x3_1(x3)
            x3_1 = self.x3_2(x3_1)

            # step 4
            x4 = self.x4_fit(x3 + x3_1)
            x4 = self.x4(x4)
            x4 = self.x4_1(x4)
            x4 = Reshape(target_shape=[510])(x4)

        z_mean = self.mean(x4)
        z_var = self.var(x4)

        eps = tf.random.normal(shape=tf.shape(z_mean), dtype=tf.float64)
        return (eps * tf.exp(z_var * .5) + z_mean), z_mean, z_var   # z, z-mean, z-var
