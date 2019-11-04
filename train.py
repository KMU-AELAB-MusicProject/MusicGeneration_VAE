import os
import time
import argparse

import numpy as np
import tensorflow as tf

from src.v1.phrase.model import PhraseModel
from src.v1.bar.model import BarModel
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_phrase', help="Train Classifier model before train vae.", action='store_true')
parser.add_argument('--load_model', help="The train number number to start train.", action='store_true')
parser.add_argument('--model_number', type=int, default=1, help="The GPU device number to use.")
parser.add_argument('--gpu_number', type=int, default=1, help="The GPU device number to use.")

args = parser.parse_args()


def set_dir():
    if not os.path.exists(os.path.join(ROOT_PATH, BOARD_PATH)):
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH))
        time.sleep(0.1)
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH, 'phrase'))
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH, 'bar'))
    if not os.path.exists(os.path.join(ROOT_PATH, MODEL_SAVE_PATH)):
        os.mkdir(os.path.join(ROOT_PATH, MODEL_SAVE_PATH))
        time.sleep(0.1)
        os.mkdir(os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number)))


def set_data(file, batch_size):
    with np.load(file) as data:
        dataset = tf.data.Dataset.from_tensor_slices(
            (data['train_data'], data['pre_phrase'], data['position_number'])).shuffle(10000).batch(batch_size)
    return dataset


class Train(object):
    def __init__(self, model, model_path):
        self.epochs = TRAIN_EPOCH

        self.best_loss = 99999999
        self.not_learning_cnt = 0

        self.lr = 1e-3

        self.mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_path, max_to_keep=50)

        self.model = model

    def decay(self):
        if self.not_learning_cnt > 3:
            self.lr /= 2

    @tf.function
    def compute_loss(self, train_data, outputs, binary_note, z_mean, z_var, td_binary):
        loss = self.mse_loss_fn(td_binary, binary_note) * 0.4
        loss += self.mse_loss_fn(train_data, outputs) * 0.6
        loss -= 0.5 * tf.reduce_mean(z_var - tf.square(z_mean) - tf.exp(z_var) + 1.)

        return loss

    def train(self):
        def train_batch(ds):
            def train_step(inputs):
                train_data, pre_phrase, position_number = inputs
                with tf.GradientTape() as tape:
                    outputs, binary_note, z, z_mean, z_var, td_binary = self.model(train_data, pre_phrase,
                                                                                   position_number)
                    loss = self.compute_loss(train_data, outputs, binary_note, z_mean, z_var, td_binary)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                return loss

            batch_loss = np.float64(0.0)
            num_train_batches = np.float64(0.0)
            for one_batch in ds:
                batch_loss += train_step(one_batch)
                num_train_batches += 1
            return batch_loss / num_train_batches

        train_batch = tf.function(train_batch)

        if args.train_phrase:
            train_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(75)]
            test_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(75, 77)]
        else:
            train_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(75)]
            test_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(75, 77)]

        for epoch in range(self.epochs):
            self.decay()
            self.optimizer.learning_rate = self.lr

            train_time = time.time()
            train_loss = 0.
            for file in train_list:
                dataset = set_data(file, 32)
                train_loss += train_batch(dataset)
            train_time = time.time() - train_time

            test_loss = 0.
            for file in test_list:
                dataset = set_data(file, 32)
                test_loss += train_batch(dataset)

            print("{} Epoch's loss: [train_loss: {} | test_loss: {}] ---- time: {}".format(epoch, train_loss, test_loss,
                                                                                           train_time))

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                if epoch > 100:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

        return True


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')

    set_dir()

    if args.train_phrase:
        model = PhraseModel()
        model_path = os.path.join(os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number)), 'phrase')
    else:
        model = BarModel()
        model_path = os.path.join(os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number)), 'bar')

    trainer = Train(model, model_path)
    trainer.train()
