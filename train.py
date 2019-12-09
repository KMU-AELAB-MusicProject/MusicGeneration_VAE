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
    def __init__(self, model, model_d, save_path, save_path_d, board_path):
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

        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.bc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.optimizer_d = tf.keras.optimizers.Adam(self.lr_d)

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, save_path, max_to_keep=50)

        self.ckpt_d = tf.train.Checkpoint(optimizer=self.optimizer_d, model=model)
        self.manager_d = tf.train.CheckpointManager(self.ckpt_d, save_path_d, max_to_keep=50)

        self.summary_writer = tf.summary.create_file_writer(board_path)

        self.model = model
        self.model_d = model_d

    def decay(self):
        if self.not_learning_cnt > self.not_learning_limit:
            self.lr *= 0.6
            self.not_learning_cnt = 0
            self.not_learning_limit += 2

    @tf.function
    def additional_loss(self, targets, outputs):
        loss = targets - outputs

        return tf.keras.backend.sum(tf.keras.backend.clip(loss, 0., 1.))

    @tf.function
    def vq_loss(self, outputs, z, z_q, train_data):
        d_loss = tf.reduce_mean(self.bc_loss(train_data, outputs) + self.additional_loss(train_data, outputs) * 0.5)
        q_loss = tf.reduce_mean(self.mse_loss(tf.stop_gradient(z), z_q))
        e_loss = tf.reduce_mean(self.mse_loss(tf.stop_gradient(z_q), z) * 0.22)

        return d_loss, q_loss, e_loss

    @tf.function
    def gan_loss(self, logits):
        loss = self.bc_loss(tf.ones_like(logits), logits)

        return -loss

    @tf.function
    def discriminator_loss(self, f_logits, r_logits):
        loss = self.bc_loss(tf.ones_like(r_logits), r_logits) + self.bc_loss(tf.zeros_like(f_logits), f_logits)

        return loss

    def train(self):
        @tf.function
        def batch(ds, isTrain=True):
            loss = np.float64(0.0)
            dis_loss = np.float64(0.0)

            for one_batch in ds:
                with tf.device('/device:GPU:0'):
                    with tf.GradientTape() as tape:
                        train_data, pre_phrase, position_number = one_batch
                        outputs, z, z_q = self.model(train_data, pre_phrase, position_number)

                        df_logit = self.model_d(outputs)
                        dr_logit = self.model_d(train_data)

                        d_loss, q_loss, e_loss = self.vq_loss(outputs, z, z_q, train_data)
                        g_loss = self.gan_loss(df_logit)

                        l = self.discriminator_loss(df_logit, dr_logit)

                    if isTrain:
                        gradients = tape.gradient(l, self.model.trainable_variables)
                        vars = list(zip(gradients, self.model.trainable_variables))

                        self.optimizer_d.apply_gradients(vars)

                        d_gradients = tape.gradient(d_loss + g_loss, self.model.trainable_variables)
                        q_gradients = tape.gradient(q_loss + g_loss * 0.2, self.model.trainable_variables)
                        e_gradients = tape.gradient(e_loss + g_loss * 0.2, self.model.trainable_variables)

                        d_vars = list(zip(d_gradients, self.model.trainable_variables))
                        q_vars = list(zip(q_gradients, self.model.trainable_variables))
                        e_vars = list(zip(e_gradients, self.model.trainable_variables))

                        self.optimizer.apply_gradients(d_vars + q_vars + e_vars)

                    loss += d_loss + q_loss + e_loss + g_loss
                    dis_loss += l

            return loss, dis_loss

        @tf.function
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

        if args.train_phrase:
            train_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(75)]
            test_list = [os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)) for i in range(75, 77)]
        else:
            train_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(75)]
            test_list = [os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)) for i in range(75, 77)]

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
            for file in train_list[:int(len(train_list) * 0.6)]:
                dataset = set_data(file, BATCH_SIZE)
                t_loss, d_loss = batch(dataset)
                train_loss += t_loss
                train_loss_d += d_loss
            train_time = time.time() - train_time
            with self.summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss / int(len(train_list) * 0.6), step=epoch)

            # ---------------- test step ----------------
            test_loss = 0.
            test_loss_d = 0.
            for file in test_list:
                dataset = set_data_test(file, BATCH_SIZE)
                t_loss, d_loss = batch(dataset, False)
                test_loss += t_loss
                test_loss_d += d_loss
            with self.summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss / len(test_list), step=epoch)

            if self.state_num:
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

                if train_loss > past:
                    self.not_learning_cnt += 1
                past = train_loss

                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    if epoch > 10:
                        save_path = self.manager.save()
                        print("Saved checkpoint for epoch {}: {} ---- loss: {}".format(epoch, save_path,
                                                                                       self.best_loss))
            else:
                if train_loss > past_d:
                    self.not_learning_cnt_d += 1
                past_d = train_loss

            # --------------------------------------------
            print("{} Epoch {}state loss: [train_loss: {:.7f} | test_loss: {:.7f}] ---- time: {:.5f} | lr: {:.8f}".
                  format(epoch, self.state_num, train_loss, test_loss, train_time, self.lr))

        return True


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')

    set_dir()

    if args.train_phrase:
        import_model = importlib.import_module('src.v{}.phrase.model'.format(args.model_number))
        model = import_model.PhraseModel()
        model_d = importlib.import_module('src.phrase_discriminator').PhraseDiscriminatorModel()
        save_path = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'phrase')
        save_path_d = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'phrase_discriminator')
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'phrase')
    else:
        import_model = importlib.import_module('src.v{}.bar.model'.format(args.model_number))
        model = import_model.BareModel()
        model_d = None #importlib.import_module('src.phrase_discriminator').PhraseDiscriminatorModel()
        save_path_d = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'bar_discriminator')
        save_path = os.path.join(ROOT_PATH, MODEL_SAVE_PATH, 'v{}'.format(args.model_number), 'bar')
        board_path = os.path.join(ROOT_PATH, BOARD_PATH, 'v{}'.format(args.model_number), 'bar')

    trainer = Train(model, model_d, save_path, save_path_d, board_path)
    trainer.train()
