import os
import time
import random
import argparse
import importlib

import numpy as np
import tensorflow as tf

from config import *
from src.phrase_discriminator import PhraseDiscriminatorModel

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
    def __init__(self, model, model_d, save_path, save_path_d, save_path_tb, board_path):
        self.epochs = TRAIN_EPOCH

        self.train_best_loss = 99999999
        self.test_best_loss = 99999999

        self.train_best_epoch = 0
        self.test_best_epoch = 0

        self.not_learning_cnt = 0
        self.not_learning_limit = 4

        self.not_learning_cnt_d = 0
        self.not_learning_limit_d = 4

        self.state_num = 1

        self.lr = 0.00008
        self.lr_d = 0.0001

        self.bc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.optimizer_d = tf.keras.optimizers.Adam(self.lr_d)

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, save_path, max_to_keep=50)

        self.ckpt_d = tf.train.Checkpoint(optimizer=self.optimizer_d, model=model_d)
        self.manager_d = tf.train.CheckpointManager(self.ckpt_d, save_path_d, max_to_keep=50)

        self.ckpt_tb = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager_tb = tf.train.CheckpointManager(self.ckpt_tb, save_path_tb, max_to_keep=50)

        self.summary_writer = tf.summary.create_file_writer(board_path)

        self.model = model
        self.model_d = model_d

    def decay(self, epoch):
        if self.not_learning_cnt > self.not_learning_limit:
            self.lr *= 0.6
            self.not_learning_cnt = 0
            self.not_learning_limit += 2

        if epoch < 12:
            self.lr_d = 0
            self.not_learning_cnt_d = 0
        elif epoch == 12:
            self.lr_d = 0.0001
        else:
            if self.not_learning_cnt_d > self.not_learning_limit_d:
                self.lr_d *= 0.6
                self.not_learning_cnt_d = 0
                self.not_learning_limit_d += 2

    @tf.function
    def additional_loss(self, targets, outputs):
        loss = targets - tf.reshape(outputs, [-1, outputs.shape[-3], outputs.shape[-2]])

        return tf.keras.backend.sum(tf.keras.backend.clip(loss, 0., 1.))

    @tf.function
    def gan_loss(self, logits):
        loss = self.bc_loss(tf.ones_like(logits), logits)

        return loss

    @tf.function
    def discriminator_loss(self, f_logits, r_logits):
        loss = self.bc_loss(tf.ones_like(r_logits), r_logits) + self.bc_loss(tf.zeros_like(f_logits), f_logits)

        return loss

    def train(self):
        def batch(train_data, pre_phrase, position_number, isTrain=True):
            with tf.device('/device:GPU:0'):
                with tf.GradientTape() as d_tape, tf.GradientTape() as disc_tape:
                    outputs_ori, outputs_music, z, z_mean, z_var = self.model(train_data, pre_phrase, position_number)

                    df_logit = self.model_d(outputs_music)
                    dr_logit = self.model_d(train_data)

                    g_loss = tf.keras.backend.sum(self.gan_loss(df_logit)) * 1.5
                    recon_loss = 0.4 * tf.keras.backend.sum(self.bc_loss(train_data, outputs_ori)) + \
                                 0.6 * tf.keras.backend.sum(self.bc_loss(train_data, outputs_music))

                    loss = recon_loss + g_loss + ((self.additional_loss(train_data, outputs_music) +
                                                   self.additional_loss(train_data, outputs_ori)) * 0.3)
                    loss -= 0.5 * tf.reduce_mean(z_var - tf.square(z_mean) - tf.exp(z_var) + 1.)

                    disc_loss = tf.keras.backend.sum(self.discriminator_loss(df_logit, dr_logit))

                if isTrain:
                    gradients = d_tape.gradient(loss, self.model.trainable_variables)
                    vars = list(zip(gradients, self.model.trainable_variables))

                    self.optimizer.apply_gradients(vars)

                    disc_gradients = disc_tape.gradient(disc_loss, self.model_d.trainable_variables)
                    disc_vars = list(zip(disc_gradients, self.model_d.trainable_variables))

                    self.optimizer_d.apply_gradients(disc_vars)

            return loss, disc_loss

        def test(train_data, pre_phrase, position_number):
            outputs = np.zeros([10, 10, 3])
            with tf.device('/device:GPU:0'):
                _, outputs, _, _, _ = self.model(train_data, pre_phrase, position_number)
            return outputs

        if args.train_phrase:
            train_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(76)]
            test_file = os.path.join(DATA_PATH, 'phrase_data', 'phrase_data76.npz')
        else:
            train_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(76)]
            test_file = os.path.join(DATA_PATH, 'bar_data', 'bar_data76.npz')

        batch = tf.function(batch)
        test = tf.function(test)

        past = 999999999.
        past_d = 999999999.
        print('################### start train ###################')
        for epoch in range(1, self.epochs + 1):
            self.decay(epoch)
            self.optimizer.learning_rate = self.lr

            # ---------------- train step ----------------
            train_loss = 0.
            train_loss_d = 0.
            train_time = time.time()
            random.shuffle(train_list)
            for file in train_list[:int(len(train_list) * 0.8)]:
                dataset = set_data(file, BATCH_SIZE)
                for ds in dataset:
                    train_data, pre_phrase, position_number = ds
                    loss, disc_loss = batch(train_data, pre_phrase, position_number)
                    train_loss += loss
                    train_loss_d += disc_loss
            train_time = time.time() - train_time
            with self.summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss / int(len(train_list) * 0.8), step=epoch)
                tf.summary.scalar('train_loss_disc', train_loss_d / int(len(train_list) * 0.8), step=epoch)

            # ---------------- test step ----------------
            test_loss = 0.
            dataset = set_data_test(test_file, BATCH_SIZE)
            for ds in dataset:
                train_data, pre_phrase, position_number = ds
                t_loss, _ = batch(train_data, pre_phrase, position_number, False)
                test_loss += t_loss
            with self.summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss, step=epoch)

            # ---------------- piano-roll generation step ----------------
            dataset = set_data_test(train_list[0], BATCH_SIZE)
            for ds in dataset:
                train_data, pre_phrase, position_number = ds
                output = test(train_data, pre_phrase, position_number)
                with self.summary_writer.as_default():
                    tf.summary.image('train_output', output*255, step=epoch)
                break

            dataset = set_data(test_file, BATCH_SIZE)
            for ds in dataset:
                train_data, pre_phrase, position_number = ds
                output = test(train_data, pre_phrase, position_number)
                with self.summary_writer.as_default():
                    tf.summary.image('test_output', output*255, step=epoch)
                break

            outputs = []
            pre_phrase = np.zeros([1, 384, 96], dtype=np.float64)
            phrase_idx = np.array([[330]], dtype=np.float64)
            for idx in range(3):
                pre_phrase = self.model.test(pre_phrase, phrase_idx)
                outputs.append(pre_phrase)
                phrase_idx = np.array([[1 - idx]], dtype=np.float64)
            with self.summary_writer.as_default():
                tf.summary.image('output', np.array(outputs).reshape([-1, 384, 96, 1])*255, step=epoch)

            if test_loss < self.test_best_loss:
                self.test_best_loss = test_loss
                if epoch > 10:
                    save_path = self.manager.save()
                    self.manager_d.save()
                    print("Saved checkpoint for epoch {}: {} ---- loss: {}".format(epoch, save_path,
                                                                                       self.test_best_loss))
            if (epoch >= 12) and (train_loss_d > past_d):
                self.not_learning_cnt_d += 1
            past_d = train_loss_d

            if train_loss > past:
                self.not_learning_cnt += 1
            past = train_loss

            if train_loss < self.train_best_loss:
                self.train_best_loss = train_loss
                self.manager_tb.save()

            # --------------------------------------------
            print("{} Epoch loss: [train_loss: {:.7f} | test_loss: {:.7f}] ---- time: {:.5f} | lr: {:.8f}".
                  format(epoch, train_loss, test_loss, train_time, self.lr))

        return True


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')

    set_dir()

    if args.train_phrase:
        import_model = importlib.import_module('src.v{}.phrase.model'.format(args.model_number))
        model = import_model.PhraseModel()
        save_path = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'phrase')
        save_path_d = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'phrase_discriminator')
        save_path_tb = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'phrase_train_best')
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'phrase')
        model_d = PhraseDiscriminatorModel()
    else:
        import_model = importlib.import_module('src.v{}.bar.model'.format(args.model_number))
        model = import_model.BarModel()
        save_path_d = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'bar_discriminator')
        save_path = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'bar')
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'bar')

    trainer = Train(model, model_d, save_path, save_path_d, save_path_tb, board_path)
    trainer.train()
