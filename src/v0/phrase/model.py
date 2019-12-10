import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape, multiply, Embedding

# from config import *
from .encoder import Encoder
from .decoder import Decoder
from ...phrase_discriminator import PhraseDiscriminatorModel


class PhraseModel(tf.keras.Model):
    def __init__(self):
        super(PhraseModel, self).__init__(name='phrase_model')
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = PhraseDiscriminatorModel()

        self.phrase_number = Embedding(332, 510, dtype=tf.float64, name='phrase_number')
        self.quantisation = Embedding(128, 510, dtype=tf.float64, name='quantisation')

    def vq(self, z):
        with tf.name_scope("vq"):
            z = tf.expand_dims(z, -2)  # (B, t, 1, D)
            print(self.quantisation(np.array([i for i in range(128)])))
            lookup_table = tf.reshape(self.quantisation(np.array([i for i in range(128)])), [128, 510])  # (1, 1, K, D)
            dist = tf.norm(z - lookup_table, axis=-1)  # Broadcasting -> (B, T', K)
            k = tf.argmin(dist, axis=-1)  # (B, t)

            return self.quantisation(k)  # (B, t, D)

    def call(self, train_data, pre_phrase, position_number):
        z = self.encoder(train_data)  # train-phrase
        z_pre = self.encoder(pre_phrase)  # pre-phrase

        z_q = self.vq(z)
        z_pre_q = self.vq(z_pre)

        logits = self.decoder(z_q + z_pre_q + Reshape(target_shape=[510])(self.phrase_number(position_number)))

        outputs = tf.keras.activations.sigmoid(logits)

        df_logits = self.discriminator(outputs)
        dr_logits = self.discriminator(train_data)

        return outputs, z, z_q, df_logits, dr_logits

    def get_feature(self, input):
        z, _, _ = self.encoder(input)
        return z

    def test(self, pre_phrase, position_number):
        z_pre, _, _ = self.encoder(pre_phrase)
        z_pre_q = self.vq(z_pre)
        z_q = self.vq(tf.random.normal(shape=(1, 510), dtype=tf.float64))
        logits = self.decoder(z_q + z_pre_q + Reshape(target_shape=[510])(self.phrase_number(position_number)))

        outputs = tf.keras.activations.sigmoid(logits)

        return outputs


if __name__ == '__main__':
    t = PhraseModel()
    # t.compile()
    # t.build((10, 384, 96), (10, 384, 96), (10))
    # t.summary()
