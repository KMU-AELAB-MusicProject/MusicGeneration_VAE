import tensorflow as tf
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Layer, Conv2D, Reshape, concatenate, AveragePooling2D, GRU, Dense


# from config import *


class Encoder(Layer):
    def __init__(self):
        super(Encoder, self).__init__(name='phrase_encoder')

        self.x1_1_1 = Conv2D(filters=32, kernel_size=[1, 4], strides=[1, 2], activation='relu', padding='same')
        self.x1_1_2 = Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same')

        self.x1_2_1 = Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same')
        self.x1_2_2 = Conv2D(filters=32, kernel_size=[1, 4], strides=[1, 2], activation='relu', padding='same')

        self.x2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x2_fit = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')

        self.x3 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x4 = Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x4_fit = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')

        self.x5 = Conv2D(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')

        self.avg = AveragePooling2D([12, 3])

        self.flatten = tf.keras.layers.Flatten()

        self.dense = Dense(510)

    def call(self, input):
        x = Reshape(target_shape=[384, 96, 1])(input)

        # CNN feature extract
        with tf.name_scope('CNN_feature'):
            # step 1
            # pitch-time feature
            x1_1 = self.x1_1_1(x)
            x1_1 = self.x1_1_2(x1_1)

            # time-pitch feature
            x1_2 = self.x1_2_1(x)
            x1_2 = self.x1_2_2(x1_2)

            # step 2
            x2 = concatenate([x1_1, x1_2], axis=3)
            x2 = self.x2(x2)
            x2 = self.x2_fit(x2)

            x3 = self.x3(x2)

            x4 = self.x4(x3)
            x4 = self.x4_fit(x4)

            x5 = self.x5(x4)
            x5 = self.avg(x5)
            x5 = self.flatten(x5)


        z = self.dense(x5)

        return z
