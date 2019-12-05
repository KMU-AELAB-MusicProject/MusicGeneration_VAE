import os
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Reshape, concatenate, AveragePooling2D, GRU, Dense


class PhraseDiscriminatorModel(tf.keras.Model):
    def __init__(self):
        super(PhraseDiscriminatorModel, self).__init__(name='phrase_discriminator')
        self.x1_1_1 = Conv2D(filters=16, kernel_size=[1, 4], strides=[1, 2], activation='relu', padding='same')
        self.x1_1_2 = Conv2D(filters=16, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same')

        self.x1_2_1 = Conv2D(filters=16, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same')
        self.x1_2_2 = Conv2D(filters=16, kernel_size=[1, 4], strides=[1, 2], activation='relu', padding='same')

        self.x2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')

        self.x3 = Conv2D(filters=64, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x4 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')

        self.chord_x1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 1], activation='relu', padding='same')
        self.chord_x2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 1], activation='relu', padding='same')
        self.chord_x3 = Conv2D(filters=64, kernel_size=[3, 3], strides=[2, 1], activation='relu', padding='same')
        self.chord_x3 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')

        self.on_off_x1 = Conv2D(filters=16, kernel_size=[3, 1], strides=[2, 1], activation='relu', padding='same')
        self.on_off_x2 = Conv2D(filters=16, kernel_size=[3, 1], strides=[2, 1], activation='relu', padding='same')
        self.on_off_x3 = Conv2D(filters=16, kernel_size=[3, 1], strides=[2, 1], activation='relu', padding='same')
        self.on_off_x4 = Conv2D(filters=16, kernel_size=[3, 1], strides=[2, 1], activation='relu', padding='same')

        self.x_avg = AveragePooling2D([24, 6])
        self.chord_avg = AveragePooling2D([24, 6])
        self.on_off_avg = AveragePooling2D([24, 1])

        self.feature1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')
        self.feature2 = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], activation='sigmoid', padding='same')

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input):
        x = Reshape(target_shape=[384, 96, 1])(input)

        chord_x = Reshape(target_shape=[384, 12, 8])(input)
        chord_x = tf.match.reduce_sum(chord_x, axis=-1, keepdims=True) # 384, 12, 1

        padded = tf.pad(x[:, :-1], [[0, 0], [1, 0], [0, 0], [0, 0]])
        on_off_x = tf.match.reduce_sum(x - padded, axis=2, keepdims=True)  # 384, 1, 1

        ## normal feature extraction ##
        # pitch-time feature
        x1_1 = self.x1_1_1(x)
        x1_1 = self.x1_1_2(x1_1)

        # time-pitch feature
        x1_2 = self.x1_2_1(x)
        x1_2 = self.x1_2_2(x1_2)

        # step 2
        x2 = concatenate([x1_1, x1_2], axis=-1) # 192, 48, 32
        x2 = self.x2(x2)    # 96, 24, 32
        x3 = self.x3(x2)    # 48, 12, 64
        x4 = self.x4(x3)  # 24, 6, 128

        x4 = self.x_avg(x4)   # 1, 1, 128

        ## chord feature extraction ##
        chord_x1 = self.chord_x1(chord_x)   # 192, 12, 16
        chord_x2 = self.chord_x2(chord_x1)  # 96, 12, 32
        chord_x3 = self.chord_x3(chord_x2)  # 48, 12, 64
        chord_x4 = self.chord_x3(chord_x3)  # 24, 6, 128

        chord_x4 = self.x_avg(chord_x4)  # 1, 1, 128

        ## on/off feature extraction ##
        on_off_x1 = self.on_off_x1(on_off_x)  # 192, 1, 16
        on_off_x2 = self.on_off_x2(on_off_x1)  # 96, 1, 32
        on_off_x3 = self.on_off_x3(on_off_x2)  # 48, 1, 64
        on_off_x4 = self.on_off_x3(on_off_x3)  # 24, 1, 128

        chord_x4 = self.x_avg(chord_x4)  # 1, 1, 128

        feature = concatenate([x4, chord_x4, on_off_x4], axis=-1)  # 1, 1, 128 * 3

        feature1 = self.feature1(feature)   # 1, 1, 128
        feature2 = self.feature1(feature1)  # 1, 1, 2

        logits = self.flatten(feature2)

        return logits