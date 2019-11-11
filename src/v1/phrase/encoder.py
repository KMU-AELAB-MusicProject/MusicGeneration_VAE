import tensorflow as tf
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Layer, Conv2D, Reshape, concatenate, AveragePooling2D, GRU, Dense


# from config import *


class Encoder(Layer):
    def __init__(self):
        super(Encoder, self).__init__(name='phrase_encoder')

        self.x1 = Conv2D(filters=32, kernel_size=[1, 12], strides=[1, 2], activation='relu', padding='same',
                         kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x1_1 = Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same',
                           kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x1_2 = Conv2D(filters=32, kernel_size=[4, 1], strides=[2, 1], activation='relu', padding='same',
                           kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x1_2_fit = Conv2D(filters=32, kernel_size=[1, 12], strides=[1, 2], activation='relu', padding='same',
                               kernel_regularizer=L1L2(l1=0.0005, l2=0.001))

        self.x2_fit = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same',
                             kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same',
                         kernel_regularizer=L1L2(l1=0.0005, l2=0.001))

        self.x2_1_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same',
                             kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x2_1_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same',
                             kernel_regularizer=L1L2(l1=0.0005, l2=0.001))

        self.x3_fit = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same',
                             kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x3 = Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same',
                         kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x3_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same',
                           kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x3_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='same',
                           kernel_regularizer=L1L2(l1=0.0005, l2=0.001))

        self.x4_fit = Conv2D(filters=510, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same',
                             kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x4 = Conv2D(filters=510, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same',
                         kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.x4_1 = AveragePooling2D(pool_size=[24, 6])

        self.xr_fit = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same',
                             kernel_regularizer=L1L2(l1=0.0005, l2=0.001))
        self.xr = GRU(units=510, return_sequences=False, recurrent_initializer='glorot_uniform',
                      kernel_regularizer=L1L2(l1=0.0005, l2=0.001))

        self.mean = Dense(510)
        self.var = Dense(510)

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

        # GRU feature extract
        with tf.name_scope('GRU_feature'):
            xr = self.xr_fit(x1)
            xr = Reshape(target_shape=[384, 48])(xr)
            xr = self.xr(xr)

        x5 =concatenate([x4, xr], axis=1)

        z_mean = self.mean(x5)
        z_var = self.var(x5)

        eps = tf.random.normal(shape=tf.shape(z_mean), dtype=tf.float64)
        return (eps * tf.exp(z_var * .5) + z_mean), z_mean, z_var   # z, z-mean, z-var
