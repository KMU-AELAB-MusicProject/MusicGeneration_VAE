import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Reshape, concatenate, AveragePooling2D, Conv2DTranspose, GRU


class Decoder(Layer):
    def __init__(self):
        super(Decoder, self).__init__(name='phrase_decoder')

        self.x1_1_1 = Conv2DTranspose(filters=256, kernel_size=[1, 6], strides=[1, 6], activation='relu', padding='same')
        self.x1_1_2 = Conv2DTranspose(filters=256, kernel_size=[24, 1], strides=[24, 1], activation='relu', padding='same')

        self.x1_2_1 = Conv2DTranspose(filters=256, kernel_size=[24, 1], strides=[24, 1], activation='relu', padding='same')
        self.x1_2_2 = Conv2DTranspose(filters=256, kernel_size=[1, 6], strides=[1, 6], activation='relu', padding='same')

        self.x2 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')

        self.x3 = Conv2DTranspose(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x4 = Conv2DTranspose(filters=64, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x5_1 = Conv2DTranspose(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.x5_2 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], activation='relu', padding='same')

        self.x6 = Conv2DTranspose(filters=16, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')

        self.logit_fit = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], activation='sigmoid', padding='same')

    def call(self, input):
        x = Reshape(target_shape=[1, 1, 510])(input)

        # pitch-time extend
        x1_1 = self.x1_1_1(x)
        x1_1 = self.x1_1_2(x1_1)

        # time-pitch extend
        x1_2 = self.x1_2_1(x)
        x1_2 = self.x1_2_2(x1_2)

        x2 = concatenate([x1_1, x1_2], axis=3)

        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5_1(x4)
        x5 += self.x5_2(x5)
        x6 = self.x6(x5)

        logits = self.logit_fit(x6)

        return logits
