import cv2
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tensorFlow.architectures.my_model import MyModel

IMG_WIDTH = 384
IMG_HEIGHT = 384

H5_IMAGES = os.path.join(os.path.dirname(__file__), '../images.h5')
H5_LABELS = os.path.join(os.path.dirname(__file__), '../labels.h5')

f_id = 0

class TfTrain:
    def __init__(self):
        self.epochs = 10
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.kp = 0.7

        if not os.path.isfile(H5_IMAGES) or not os.path.isfile(H5_LABELS):
            return

        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tensorflow'))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tensorflow')))

        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tensorflow'))

        self.images = h5py.File(H5_IMAGES, 'r')
        self.labels = h5py.File(H5_LABELS, 'r')
    
    def neural_net_input(self):
        return tf.compat.v1.placeholder(tf.float32, (None, IMG_HEIGHT, IMG_WIDTH, 3), name='X')

    def neural_net_output(self, n_classes=2):
        return tf.compat.v1.placeholder(tf.float32, (None, n_classes), name='y')

    def neural_net_keep_prob(self):
        return tf.compat.v1.placeholder(tf.float32, None, 'keep_prob')

    def train_neural_network(self, session, optimizer, x, y, keep_prob, keep_probability, feature_batch, label_batch):
        """
        Optimize the session on a batch of images and labels
        : session: Current TensorFlow session
        : optimizer: TensorFlow optimizer function
        : keep_probability: keep probability
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        """
        session.run(optimizer, feed_dict={x:feature_batch, y:label_batch, keep_prob:keep_probability})

    def print_stats(self, sess, x, y, keep_prob, feature_batch, label_batch, valid_features, valid_labels, cost, accuracy):
        """
        Print information about loss and validation accuracy
        : session: Current TensorFlow session
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        : cost: TensorFlow cost function
        : accuracy: TensorFlow accuracy function
        """
        loss = sess.run(cost, feed_dict={x:feature_batch, y:label_batch, keep_prob:1.0})
        valid_acc = sess.run(accuracy, feed_dict={
                    x: valid_features,
                    y: valid_labels,
                    keep_prob: 1.})
        print('Loss: {:>10.4f}, Val Acc: {:.6f}'.format(loss, valid_acc))

    def train(self):
        # Remove previous weights, bias, inputs, etc..
        tf.compat.v1.reset_default_graph()

        tf.compat.v1.disable_eager_execution()

        keys = list(self.images.keys())

        # Inputs
        x = self.neural_net_input()
        y = self.neural_net_output()
        keep_prob = self.neural_net_keep_prob()

        # Model
        my_model = MyModel()
        logits = my_model.model(x, keep_prob)

        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

        # Loss and Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        with tf.compat.v1.Session() as sess:
            # Initializing the variables
            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(0, self.epochs):
                idx = 0
                for d_idx in range(0, len(keys)):
                    img = self.images.get('images_{}'.format(d_idx))
                    lb = self.labels.get('labels_{}'.format(d_idx))

                    for batch_idx in range(0, img.shape[0] // self.batch_size):
                        if d_idx == len(keys) - 1 and batch_idx == img.shape[0] // self.batch_size - 1:
                            batch_images = img[idx : img.shape[0]] / 255.
                            batch_labels = lb[idx : lb.shape[0]]
                        else:
                            batch_images = img[idx : idx+self.batch_size] / 255.
                            batch_labels = lb[idx : idx+self.batch_size]

                        _, X_val, _, y_val = train_test_split(batch_images, batch_labels, test_size=0.3)

                        self.train_neural_network(sess, optimizer, x, y, keep_prob, self.kp, batch_images, batch_labels)
                        print('Epoch {:>2}, '.format(epoch + 1), end='')
                        self.print_stats(sess, x, y, keep_prob, batch_images, batch_labels, X_val, y_val, cost, accuracy)

                        idx += self.batch_size      