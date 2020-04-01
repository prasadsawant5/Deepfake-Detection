import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tensorFlow.architectures.my_model import MyModel
from tensorFlow.architectures.squeeze_net import SqueezeNet
from random import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

IMG_WIDTH = 227
IMG_HEIGHT = 227

H5_IMAGES = os.path.join(os.path.dirname(__file__), '../images.h5')
H5_LABELS = os.path.join(os.path.dirname(__file__), '../labels.h5')

DIR = os.path.join(os.path.dirname(__file__), '../images')
FAKE = os.path.join(os.path.dirname(__file__), '../images/fake')
REAL = os.path.join(os.path.dirname(__file__), '../images/real')

f_id = 0

class TfTrain:
    def __init__(self):
        self.epochs = 1
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.kp = 0.5

        self.images = None
        self.labels = None

        self.paths = []

        self.is_h5 = False

        self.is_squeeze = False

        if os.path.isfile(H5_IMAGES) and os.path.isfile(H5_LABELS):
            self.is_h5 = True
            self.images = h5py.File(H5_IMAGES, 'r')
            self.labels = h5py.File(H5_LABELS, 'r')
        else:
            fake_dirs = os.listdir(FAKE)
            real_dirs = os.listdir(REAL)

            if len(fake_dirs) <= len(real_dirs):
                for i in range(0, len(fake_dirs)):
                    self.paths.append(os.path.join(FAKE, fake_dirs[i]))
                    self.paths.append(os.path.join(REAL, real_dirs[i]))
            else:
                for i in range(0, len(real_dirs)):
                    self.paths.append(os.path.join(FAKE, fake_dirs[i]))
                    self.paths.append(os.path.join(REAL, real_dirs[i]))

        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tensorflow'))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tensorflow')))

        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'tensorflow'))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'tensorflow')))

        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tensorflow'))
        self.tensorboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'tensorflow'))
    
    def set_squeeze(self, flag):
        self.is_squeeze = flag

    def get_squeeze(self):
        return self.is_squeeze

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

    def one_hot(self, paths, n_classes = 2):
        arr = np.zeros((len(paths), n_classes))

        for i in range(0, len(paths)):
            if 'real' in paths[i]:
                arr[i][0] = 1.
            else:
                arr[i][1] = 1.
        return arr

    def read_images(self, paths):
        img = []

        for path in paths:
            image = cv2.imread(path)
            # if 'fgobmbcami_171.jpeg' in path:
            if np.all(image) == None:
                print(image)
            img.append(cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)))

        return np.array(img)


    def train(self):
        tf.keras.backend.clear_session()

        train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
        train_data_gen = train_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                           directory=DIR,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')


        # Inputs
        x = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

        # Model
        model = None

        if self.get_squeeze():
            squeeze_net = SqueezeNet()
            model = squeeze_net.model(x, self.kp)
        else:
            my_model = MyModel()
            model = my_model.model(x, self.kp)
        
        model.summary()

        # # Loss and Optimizer
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # # Accuracy
        train_acc_metric = tf.keras.metrics.CategoricalCrossentropy()

        tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir = self.tensorboard_path, write_graph=True, write_images=True, histogram_freq=0)
        tensorboard_cbk.set_model(model)

        for epoch in range(self.epochs):
            for step, (batch_images, batch_labels) in enumerate(train_data_gen):
            # while True:
                # batch_images, batch_labels = train_data_gen.next()

                if np.any(batch_images) == None or np.any(batch_labels) == None:
                    train_data_gen.reset()
                    break

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(batch_images, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss(batch_labels, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_acc_metric(batch_labels, logits)

                train_acc = train_acc_metric.result()

                print('Epoch {:03d}, Step: {:06d} Accuracy: {:6f}, Loss: {:6f}'.format(epoch + 1, step, train_acc, loss_value))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

        model.save(self.save_path)
