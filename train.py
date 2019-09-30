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
parser.add_argument('--model_number', type=int, default=0, help="The GPU device number to use.")
parser.add_argument('--gpu_count', type=int, default=1, help="The GPU device number to use.")

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
        os.mkdir(os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v1'))


def get_data_from_filename(filename):
    data = np.load(filename)
    return data['train_data'], data['pre_phrase'], data['position_number']


def get_data_wrapper(filename):
    train_data, pre_phrase, position_number = tf.py_function(get_data_from_filename, [filename],
                                                             (tf.float32, tf.float32, tf.int32))
    if args.train_phrase:
        train_data.set_shape([None, 384, 96])
    else:
        train_data.set_shape([None, 96, 96])
    pre_phrase.set_shape([None, 384, 96])
    position_number.set_shape([None, 1])

    return tf.data.Dataset.from_tensor_slices((train_data, pre_phrase, position_number))


def set_data(strategy, batch_size):
    with strategy.scope():
        if args.train_phrase:
            file_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(11)]
        else:
            file_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(11)]

        # Create dataset of filenames.
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset = dataset.flat_map(get_data_wrapper).batch(batch_size)
        dataset = dataset.shuffle(10000).batch(batch_size)
    return strategy.experimental_distribute_dataset(dataset)

def set_model():
    phrase_model_path = os.path.join(MODEL_SAVE_PATH, 'phrase')
    bar_model_path = os.path.join(MODEL_SAVE_PATH, 'bar')

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(1e-4)

        if args.train_phrase:
            model = PhraseModel()
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            manager = tf.train.CheckpointManager(ckpt, phrase_model_path, max_to_keep=50)
            if args.load_model:
                if args.model_number:
                    ckpt.restore(manager.latest_checkpoint)  # ....
                else:
                    ckpt.restore(manager.latest_checkpoint)

        else:
            if args.load_model:
                model = BarModel(PhraseModel())
                ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
                manager = tf.train.CheckpointManager(ckpt, bar_model_path, max_to_keep=50)

                ckpt.restore(manager.latest_checkpoint)  # ...
            else:
                model = PhraseModel()
                ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
                manager = tf.train.CheckpointManager(ckpt, phrase_model_path, max_to_keep=50)

                ckpt.restore(manager.latest_checkpoint)  # ...
                model = BarModel(model)
                ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
                manager = tf.train.CheckpointManager(ckpt, bar_model_path, max_to_keep=50)

    return model, ckpt, manager


class Train(object):
    def __init__(self, batch_size, strategy, model, ckpt, manager):
        self.epochs = TRAIN_EPOCH
        self.batch_size = batch_size

        self.best_loss = 9999999
        self.not_learning_cnt = 0

        self.lr = 1e-3

        self.ckpt = ckpt
        self.manager = manager

        self.strategy = strategy
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.model = model

    def decay(self):
        if self.not_learning_cnt > 3:
            self.lr /= 2

    def compute_loss(self, train_data, outputs, binary_note, z_mean, z_var, td_binary):
        loss = self.mse_loss_fn(td_binary, binary_note) * 0.4
        loss += self.mse_loss_fn(train_data, outputs) * 0.6
        loss -= 0.5 * tf.reduce_mean(z_var - tf.square(z_mean) - tf.exp(z_var) + 1.)

        return tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size)

    def train_step(self, inputs):
        train_data, pre_phrase, position_number = inputs
        with tf.GradientTape() as tape:
            outputs, binary_note, z, z_mean, z_var, td_binary = self.model(train_data, pre_phrase, position_number)
            loss = self.compute_loss(train_data, outputs, binary_note, z_mean, z_var, td_binary)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))
        return loss

    def train(self, dataset, strategy):
        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in ds:
                per_replica_loss = strategy.experimental_run_v2(self.train_step, args=(one_batch,))
                total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                num_train_batches += 1
            return total_loss / num_train_batches

        distributed_train_epoch = tf.function(distributed_train_epoch)

        for epoch in range(self.epochs):
            self.decay()
            self.optimizer.learning_rate = self.lr

            loss = distributed_train_epoch(dataset)

            print("{} Epoch's loss: {}".format(epoch, loss))

            if loss < self.best_loss:
                self.best_loss = loss
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

        return True


if __name__ == '__main__':
    batch_size = BATCH_CNT * args.gpu_count
    strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:{}'.format(i) for i in range(args.gpu_count)])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    set_dir()
    dataset = set_data(strategy, batch_size)
    model, ckpt, manager = set_model()
    trainer = Train(batch_size, strategy, model, ckpt, manager)

    trainer.train(dataset, strategy)
