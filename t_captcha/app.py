#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2019-04-19 09:39
@annotation = ''
"""
import math
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from captcha.image import ImageCaptcha
# VOCAB = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890')
from sklearn.model_selection import train_test_split

VOCAB = list('01234567890')
CAPTCHA_LEN = 4
VOCAB_LEN = len(VOCAB)
SAMPLE_LEN = pow(VOCAB_LEN, CAPTCHA_LEN)
DATA_PATH = 'data'


def label2onehot(label):
    if len(label) > VOCAB_LEN:
        return None
    # tar = np.zeros((len(label), VOCAB_LEN))
    tar = np.zeros(len(label) * VOCAB_LEN)

    for i, l in enumerate(label):
        # tar[i][VOCAB.index(l)] = 1
        index = i * VOCAB_LEN + VOCAB.index(l)
        tar[index] = 1

    return tar


def onehot2label(one_hot):
    # max_index = np.argmax(one_hot, axis=1)
    # label = ''
    # for i in max_index:
    #     label += VOCAB[i]
    # return label

    one_hot = np.reshape(one_hot, [CAPTCHA_LEN, -1])
    text = ''
    for item in one_hot:
        text += VOCAB[np.argmax(item)]
    return text


def gen_captcha(bit=CAPTCHA_LEN):
    captcha_str = ''
    for i in range(bit):
        captcha_str += random.choice(VOCAB)

    image = ImageCaptcha()
    captcha = image.generate(captcha_str)
    captcha_image = Image.open(captcha)

    captcha_array = np.array(captcha_image)

    return captcha_array, label2onehot(captcha_str)


def t_labelonehot():
    label = '0123'
    assert label == onehot2label(label2onehot(label))


def gen_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    X, y = [], []
    for i in range(SAMPLE_LEN):
        print('{}/{}'.format(i, SAMPLE_LEN))
        data, label = gen_captcha()
        X.append(data)
        y.append(label)

    np.save(os.path.join(DATA_PATH, 'x.npy'), X)
    np.save(os.path.join(DATA_PATH, 'y.npy'), y)


# gen_data()

HEIGHT = 60
WIDTH = 160
BATCH_SIZE = 11 * 22
EPOCHS = 1000
CHECKPOINT_DIR = 'CAPTCHA_model'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


def standardize(x):
    return (x - x.mean()) / x.std()


data_x = np.load(os.path.join(DATA_PATH, 'x.npy'))
data_y = np.load(os.path.join(DATA_PATH, 'y.npy'))
data_x = standardize(data_x)
print(data_x.shape)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.4, random_state=40)
dev_x, test_x, dev_y, test_y, = train_test_split(test_x, test_y, test_size=0.5, random_state=40)

train_steps = int(train_x.shape[0] / BATCH_SIZE)
dev_steps = int(dev_x.shape[0] / BATCH_SIZE)
test_steps = int(test_x.shape[0] / BATCH_SIZE)

# X = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='x')
# y = tf.placeholder(dtype=tf.float32, shape=[None, CAPTCHA_LEN, VOCAB_LEN], name='y')
keep_prob = tf.placeholder(tf.float32, [])
global_step = tf.Variable(-1, trainable=False, name='global_step')


def gen_batch_data(train_x, train_y, dev_x, dev_y, test_x, test_y, batch_size):
    train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
    dev = tf.data.Dataset.from_tensor_slices((dev_x, dev_y)).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)

    iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)

    train_initializer = iterator.make_initializer(train)
    dev_initializer = iterator.make_initializer(dev)
    test_initializer = iterator.make_initializer(test)

    return iterator, train_initializer, dev_initializer, test_initializer


def train_model(is_train=True):
    iterator, train_initializer, dev_initializer, test_initializer = gen_batch_data(train_x, train_y, dev_x, dev_y,
                                                                                    test_x, test_y,
                                                                                    batch_size=BATCH_SIZE)
    X, label = iterator.get_next()
    X = tf.cast(X, tf.float32)

    with tf.variable_scope('pred'):
        for _ in range(2):
            x = tf.layers.conv2d(X, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same')

        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        x = tf.layers.dropout(x, rate=keep_prob)
        y = tf.layers.dense(x, VOCAB_LEN * CAPTCHA_LEN, name='predict')

    with tf.variable_scope('loss'):
        # loss
        y_pred = tf.reshape(y, [-1, VOCAB_LEN * CAPTCHA_LEN])
        y_label = tf.reshape(label, [-1, VOCAB_LEN * CAPTCHA_LEN])

        # loss
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_label))
        # tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.variable_scope('acc'):
        correct_prediction = tf.equal(tf.argmax(y_pred, -1), tf.argmax(y_label, -1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')

    # merged_summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        if is_train:
            # writer = tf.summary.FileWriter('CAPTCH/{}'.format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")),
            #                                sess.graph)

            for epoch in range(EPOCHS):
                tf.train.global_step(sess, global_step_tensor=global_step)

                # train
                sess.run(train_initializer)
                X, y_pred, label = sess.run([X, y_pred, label])
                # print(X.shape, y_pred.shape, label.shape)
                for step in range(train_steps):
                    try:
                        l, acc, gstep, _ = sess.run([loss, accuracy, global_step, optimizer],
                                                    feed_dict={keep_prob: 0.5})
                        # writer.add_summary(summary, gstep)
                    except tf.errors.OutOfRangeError:
                        print('Error stop {}/{} train loss {} acc {}'.format(step, gstep, l, acc))

                    if step % 2 == 0:
                        print('{}/{} train loss {} acc {}'.format(step, gstep, l, acc))

                if epoch % 2 == 0:
                    # dev
                    sess.run(dev_initializer)
                    for step in range(dev_steps):
                        try:
                            if step % 2 == 0:
                                acc = sess.run(accuracy, feed_dict={keep_prob: 1})
                                print('{} dev acc {}'.format(step, acc))
                        except tf.errors.OutOfRangeError:
                            print('Error stop {} dev'.format(step, acc))
                # save model
                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=gstep)

        else:
            saver.restore(sess, CHECKPOINT_DIR)
            sess.run(test_initializer)
            for step in range(test_steps):
                if step % 2 == 0:
                    try:
                        acc = sess.run(accuracy, feed_dict={keep_prob: 1})
                        print('Test Accuracy', acc, 'Step', step)
                    except tf.errors.OutOfRangeError:
                        print('Error stop Test Accuracy', acc, 'Step', step)

train_model()
