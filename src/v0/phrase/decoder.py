import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Reshape, concatenate, AveragePooling2D, Conv2DTranspose, GRU


# from config import *


class Decoder(layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__(name='phrase_decoder')

        reg = tf.keras.regularizers.L1L2

        self.x1_1 = Conv2DTranspose(filters=510, kernel_size=[24, 6], activation='relu')

        self.x1_2_1 = Conv2DTranspose(filters=510, kernel_size=[1, 3], strides=[1, 3], activation='relu',
                                      padding='same', kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.x1_2_2 = Conv2DTranspose(filters=510, kernel_size=[12, 1], strides=[12, 1], activation='relu',
                                      padding='same', kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.x1_2_3 = Conv2DTranspose(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                      padding='same', kernel_regularizer=reg(l1=0.003, l2=0.003))

        self.x1_3_1 = Conv2DTranspose(filters=510, kernel_size=[12, 1], strides=[12, 1], activation='relu',
                                      padding='same', kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.x1_3_2 = Conv2DTranspose(filters=510, kernel_size=[1, 3], strides=[1, 3], activation='relu',
                                      padding='same', kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.x1_3_3 = Conv2DTranspose(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu',
                                      padding='same', kernel_regularizer=reg(l1=0.003, l2=0.003))

        self.x3 = Conv2DTranspose(filters=256, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same',
                                  kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.x4 = Conv2DTranspose(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same',
                                  kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.x5 = Conv2DTranspose(filters=64, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same',
                                  kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.x6 = Conv2DTranspose(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same',
                                  kernel_regularizer=reg(l1=0.003, l2=0.003))

        self.xr_transpose = Conv2DTranspose(filters=32, kernel_size=[3, 1], strides=[2, 1], activation='relu',
                                            padding='same', kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.xr_fit = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same',
                             kernel_regularizer=reg(l1=0.003, l2=0.003))
        self.xr = GRU(units=96, return_sequences=True, recurrent_initializer='glorot_uniform',
                      kernel_regularizer=reg(l1=0.003, l2=0.003))

        self.logit_fit = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                kernel_regularizer=reg(l1=0.003, l2=0.003))

    def call(self, input):
        x = layers.Reshape(target_shape=[1, 1, 510])(input)

        x1_1 = self.x1_1(x)

        # pitch-time extend
        x1_2 = self.x1_2_1(x)
        x1_2 = self.x1_2_2(x1_2)
        x1_2 = self.x1_2_3(x1_2)

        # time-pitch extend
        x1_3 = self.x1_3_1(x)
        x1_3 = self.x1_3_2(x1_3)
        x1_3 = self.x1_3_3(x1_3)

        x2 = x1_1 + x1_2 + x1_3 # [24, 6, 510]

        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        x6 = self.x6(x5)

        logits = self.logit_fit(x6)

        return logits
