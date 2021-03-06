import os
import time
import random
import argparse
import importlib

import numpy as np
import tensorflow as tf

from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_phrase', help="Train Classifier model before train vae.", action='store_true')
parser.add_argument('--load_model', help="The train number number to start train.", action='store_true')
parser.add_argument('--model_number', type=int, default=1, help="The GPU device number to use.")
parser.add_argument('--gpu_number', type=str, help="The GPU device number to use.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

def set_dir():
    if not os.path.exists(os.path.join(ROOT_PATH, BOARD_PATH)):
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH))
        time.sleep(0.1)
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'phrase'))
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'bar'))
    if not os.path.exists(os.path.join(ROOT_PATH, MODEL_SAVE_PATH)):
        os.mkdir(os.path.join(ROOT_PATH, MODEL_SAVE_PATH))
        time.sleep(0.1)
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'phrase'))
        os.mkdir(os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'bar'))


def set_data(file, batch_size):
    with np.load(file) as data:
        dataset = tf.data.Dataset.from_tensor_slices(
            (data['train_data'], data['pre_phrase'], data['position_number'])).shuffle(10000).batch(batch_size)
        dataset = dataset.prefetch(4)
    return dataset

def set_data_test(file, batch_size):
    with np.load(file) as data:
        dataset = tf.data.Dataset.from_tensor_slices(
            (data['train_data'][:batch_size], data['pre_phrase'][:batch_size], data['position_number'][:batch_size])).\
            batch(batch_size)
    return dataset


class Train(object):
    def __init__(self, model, save_path, board_path):
        self.epochs = TRAIN_EPOCH

        self.best_loss = 99999999
        self.not_learning_cnt = 0
        self.not_learning_limit = 4

        self.lr = 0.00008

        self.velocity_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        self.beat_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(0.00008)

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, save_path, max_to_keep=50)

        self.summary_writer = tf.summary.create_file_writer(board_path)

        self.model = model

    def decay(self):
        if self.not_learning_cnt > self.not_learning_limit:
            self.lr *= 0.6
            self.not_learning_cnt = 0
            self.not_learning_limit += 2

    @tf.function
    def additional_loss(self, targets, outputs):
        loss = targets - outputs
        note_loss = tf.keras.backend.sum(tf.keras.backend.clip(loss, 0., 1.))

        return note_loss

    @tf.function
    def compute_loss(self, train_data, outputs, binary_note, z_mean, z_var, td_binary):
        loss = self.beat_loss(td_binary, binary_note)
        loss += self.velocity_loss(train_data, outputs)
        loss += self.additional_loss(td_binary, binary_note) * 0.5
        loss -= 0.5 * tf.reduce_mean(z_var - tf.square(z_mean) - tf.exp(z_var) + 1.)

        return loss

    def train(self):
        def batch(ds, isTrain=True):
            batch_loss = np.float64(0.0)
            for one_batch in ds:
                with tf.device('/device:GPU:0'):
                    with tf.GradientTape() as tape:
                        train_data, pre_phrase, position_number = one_batch
                        outputs, binary_note, z, z_mean, z_var, td_binary = self.model(train_data, pre_phrase,
                                                                                       position_number)
                        loss = self.compute_loss(train_data, outputs, binary_note, z_mean, z_var, td_binary)

                    if isTrain:
                        gradients = tape.gradient(loss, self.model.trainable_variables)
                        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                batch_loss += loss
            return batch_loss

        def test(ds):
            output = np.zeros([10, 10, 3])
            for one_batch in ds:
                with tf.device('/device:GPU:0'):
                    with tf.GradientTape() as tape:
                        train_data, pre_phrase, position_number = one_batch
                        outputs, binary_note, z, z_mean, z_var, td_binary = self.model(train_data, pre_phrase,
                                                                                       position_number)
                        output = outputs
            return output

        batch = tf.function(batch)
        test = tf.function(test)

        if args.train_phrase:
            train_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(75)]
            test_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(75, 77)]
        else:
            train_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(75)]
            test_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(75, 77)]

        past = 999999999.
        print('################### start train ###################')
        for epoch in range(1, self.epochs + 1):
            self.decay()
            self.optimizer.learning_rate = self.lr

            # ---------------- train step ----------------
            train_loss = 0.
            train_time = time.time()
            random.shuffle(train_list)
            for file in train_list[:int(len(train_list) * 0.6)]:
                dataset = set_data(file, BATCH_SIZE)
                train_loss += batch(dataset)
            train_time = time.time() - train_time
            with self.summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss / int(len(train_list) * 0.6), step=epoch)

            # ---------------- test step ----------------
            test_loss = 0.
            for file in test_list:
                dataset = set_data_test(file, BATCH_SIZE)
                test_loss += batch(dataset, False)
            with self.summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss / len(test_list), step=epoch)

            # ---------------- piano-roll generation step ----------------
            dataset = set_data_test(train_list[0], BATCH_SIZE)
            output = test(dataset)
            with self.summary_writer.as_default():
                tf.summary.image('train_output', output*255, step=epoch)

            dataset = set_data(test_list[0], BATCH_SIZE)
            output = test(dataset)
            with self.summary_writer.as_default():
                tf.summary.image('test_output', output*255, step=epoch)

            outputs = []
            pre_phrase = np.zeros([1, 384, 96], dtype=np.float64)
            for idx in range(3):
                pre_phrase = self.model.test(pre_phrase, np.array([[idx]], dtype=np.float64))
                outputs.append(pre_phrase)
            with self.summary_writer.as_default():
                tf.summary.image('output', np.array(outputs).reshape([-1, 384, 96, 1])*255, step=epoch)

            # --------------------------------------------
            print("{} Epoch's loss: [train_loss: {:.7f} | test_loss: {:.7f}] ---- time: {:.5f} | lr: {:.8f}".
                  format(epoch, train_loss, test_loss, train_time, self.lr))

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                if epoch > 10:
                    save_path = self.manager.save()
                    print("Saved checkpoint for epoch {}: {} ---- loss: {}".format(epoch, save_path, self.best_loss))

            if train_loss > past:
                self.not_learning_cnt += 1
            past = train_loss

        return True


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')

    set_dir()

    if args.train_phrase:
        import_model = importlib.import_module('src.v{}.phrase.model'.format(args.model_number))
        model = import_model.PhraseModel()
        save_path = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'phrase')
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'phrase')
    else:
        import_model = importlib.import_module('src.v{}.bar.model'.format(args.model_number))
        model = import_model.BareModel()
        save_path = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'bar')
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'bar')

    trainer = Train(model, save_path, board_path)
    trainer.train()
