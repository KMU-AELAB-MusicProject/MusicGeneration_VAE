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
    def __init__(self, model, save_path, save_path_d, board_path):
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
        self.lr_d = 0.00008

        self.bc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.optimizer_d = tf.keras.optimizers.Adam(self.lr_d)

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, save_path, max_to_keep=50)

        self.ckpt_d = tf.train.Checkpoint(optimizer=self.optimizer_d, model=model)
        self.manager_d = tf.train.CheckpointManager(self.ckpt_d, save_path_d, max_to_keep=50)

        self.summary_writer = tf.summary.create_file_writer(board_path)

        self.model = model

    def decay(self):
        if self.not_learning_cnt > self.not_learning_limit:
            self.lr *= 0.6
            self.not_learning_cnt = 0
            self.not_learning_limit += 2

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
        def batch(ds, epoch, isTrain=True):
            loss = np.float64(0.0)
            disc_loss = np.float64(0.0)

            for one_batch in ds:
                with tf.device('/device:GPU:0'):
                    with tf.GradientTape() as d_tape, tf.GradientTape() as q_tape, tf.GradientTape() as e_tape,\
                            tf.GradientTape() as n_tape, tf.GradientTape() as disc_tape:
                        train_data, pre_phrase, position_number = one_batch
                        outputs, z, z_q, z_pre, z_pre_q, df_logit, dr_logit = self.model(train_data, pre_phrase,
                                                                                        position_number)

                        g_loss = tf.keras.backend.sum(self.gan_loss(df_logit))
                        recon_loss = tf.keras.backend.sum(self.bc_loss(train_data, outputs))

                        d_loss = recon_loss + self.additional_loss(train_data, outputs) * 0.5 + g_loss
                        q_loss = tf.keras.backend.sum(tf.math.squared_difference(tf.stop_gradient(z), z_q)) + \
                                 tf.keras.backend.sum(tf.math.squared_difference(tf.stop_gradient(z_pre), z_pre_q))
                        e_loss = recon_loss + \
                                 (tf.keras.backend.sum(tf.math.squared_difference(tf.stop_gradient(z_q), z)) +
                                  tf.keras.backend.sum(tf.math.squared_difference(tf.stop_gradient(z_pre_q), z_pre))) * 0.22
                        n_loss = (d_loss + q_loss + e_loss) * 0.8

                        l = tf.keras.backend.sum(self.discriminator_loss(df_logit, dr_logit))

                    if isTrain:
                        d_gradients = d_tape.gradient(d_loss, self.model.decoder.trainable_variables)
                        q_gradients = q_tape.gradient(q_loss, self.model.quantisation.trainable_variables)
                        e_gradients = e_tape.gradient(e_loss, self.model.encoder.trainable_variables)
                        n_gradients = n_tape.gradient(n_loss, self.model.phrase_number.trainable_variables)

                        d_vars = list(zip(d_gradients, self.model.decoder.trainable_variables))
                        q_vars = list(zip(q_gradients, self.model.quantisation.trainable_variables))
                        e_vars = list(zip(e_gradients, self.model.encoder.trainable_variables))
                        n_vars = list(zip(n_gradients, self.model.phrase_number.trainable_variables))

                        self.optimizer.apply_gradients(d_vars + q_vars + e_vars + n_vars)

                        if (epoch > 18) and (epoch % 2):
                            disc_gradients = disc_tape.gradient(l, self.model.discriminator.trainable_variables)
                            vars = list(zip(disc_gradients, self.model.discriminator.trainable_variables))

                            self.optimizer_d.apply_gradients(vars)

                    loss += d_loss + q_loss + e_loss + g_loss
                    disc_loss += l

            return loss, disc_loss

        def test(ds):
            output = np.zeros([10, 10, 3])
            for one_batch in ds:
                with tf.device('/device:GPU:0'):
                    train_data, pre_phrase, position_number = one_batch
                    outputs, z, z_q, df_logit, dr_logit = self.model(train_data, pre_phrase, position_number)
                    output = outputs
            return output

        if args.train_phrase:
            train_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(76)]
            test_file = os.path.join(DATA_PATH, 'bar_data', 'phrase_data76.npz')
        else:
            train_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(76)]
            test_file = os.path.join(DATA_PATH, 'bar_data', 'bar_data76.npz')

        batch = tf.function(batch)
        test = tf.function(test)

        past = 999999999.
        past_d = 999999999.
        print('################### start train ###################')
        for epoch in range(1, self.epochs + 1):
            self.decay()
            self.optimizer.learning_rate = self.lr

            # ---------------- train step ----------------
            train_loss = 0.
            train_loss_d = 0.
            train_time = time.time()
            random.shuffle(train_list)
            for file in train_list[:int(len(train_list) * 0.8)]:
                dataset = set_data(file, BATCH_SIZE)
                t_loss, d_loss = batch(dataset, epoch)
                train_loss += t_loss
                train_loss_d += d_loss
            train_time = time.time() - train_time
            with self.summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss / int(len(train_list) * 0.8), step=epoch)
                tf.summary.scalar('train_loss_disc', train_loss_d / int(len(train_list) * 0.8), step=epoch)

            # ---------------- test step ----------------
            test_loss = 0.
            dataset = set_data_test(test_file, BATCH_SIZE)
            t_loss, _ = batch(dataset, epoch, False)
            with self.summary_writer.as_default():
                tf.summary.scalar('test_loss', t_loss, step=epoch)

            # ---------------- piano-roll generation step ----------------
            dataset = set_data_test(train_list[0], BATCH_SIZE)
            output = test(dataset)
            with self.summary_writer.as_default():
                tf.summary.image('train_output', output*255, step=epoch)

            dataset = set_data(test_file, BATCH_SIZE)
            output = test(dataset)
            with self.summary_writer.as_default():
                tf.summary.image('test_output', output*255, step=epoch)

            outputs = []
            pre_phrase = np.zeros([1, 384, 96], dtype=np.float64)
            phrase_idx = np.array([[330]], dtype=np.float64)
            for idx in range(3):
                pre_phrase = self.model.test(pre_phrase, phrase_idx)
                outputs.append(pre_phrase)
                phrase_idx = np.array([[1 - idx]], dtype=np.float64)
            with self.summary_writer.as_default():
                tf.summary.image('output', np.array(outputs).reshape([-1, 384, 96, 1])*255, step=epoch)

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                if epoch > 10:
                    save_path = self.manager.save()
                    print("Saved checkpoint for epoch {}: {} ---- loss: {}".format(epoch, save_path,
                                                                                       self.best_loss))
            if (epoch > 18) and (train_loss_d > past_d):
                self.not_learning_cnt_d += 1
            past_d = train_loss_d

            if train_loss > past:
                self.not_learning_cnt += 1
            past = train_loss

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
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'phrase')
    else:
        import_model = importlib.import_module('src.v{}.bar.model'.format(args.model_number))
        model = import_model.BarModel()
        save_path_d = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'bar_discriminator')
        save_path = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'bar')
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'bar')

    trainer = Train(model, save_path, save_path_d, board_path)
    trainer.train()
