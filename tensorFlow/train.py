import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tensorFlow.architectures.my_model import MyModel
from random import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

IMG_WIDTH = 384
IMG_HEIGHT = 384

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
        self.batch_size = 2
        self.kp = 0.3

        self.images = None
        self.labels = None

        self.paths = []

        self.is_h5 = False

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

        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tensorflow'))
    
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
        # Remove previous weights, bias, inputs, etc..
        # tf.compat.v1.reset_default_graph()

        # tf.compat.v1.disable_eager_execution()
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.keras.backend.clear_session()

        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True

        train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
        train_data_gen = train_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                           directory=DIR,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

        # model = Sequential([
        #     Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        #     MaxPooling2D(),
        #     Conv2D(64, 3, padding='same', activation='relu'),
        #     MaxPooling2D(),
        #     Conv2D(128, 3, padding='same', activation='relu'),
        #     MaxPooling2D(),
        #     Flatten(),
        #     Dense(256, activation='relu'),
        #     Dense(64, activation='relu'),
        #     Dense(32, activation='relu'),
        #     Dense(1)
        # ])

        # model.compile(optimizer='adam',
        #       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #       metrics=['accuracy'])

        # history = model.fit_generator(
        #     train_data_gen,
        #     steps_per_epoch=total_train // self.batch_size,
        #     epochs=self.epochs
        # )

        # acc = history.history['accuracy']
        # loss=history.history['loss']

        # print('Accuracy: {:.6f}, Loss: {:.6f}'.format(acc, loss))


        # Inputs
        # x = self.neural_net_input()
        # y = self.neural_net_output()
        # keep_prob = self.neural_net_keep_prob()
        x = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

        # Model
        my_model = MyModel()
        model = my_model.model(x, self.kp)
        model.summary()

        # Name logits Tensor, so that is can be loaded from disk after training
        # logits = tf.identity(logits, name='logits')

        # # Loss and Optimizer
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        # optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # # Accuracy
        # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        train_acc_metric = tf.keras.metrics.CategoricalCrossentropy()

        for epoch in range(self.epochs):
            while True:
                batch_images, batch_labels = train_data_gen.next()

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

                print('Epoch {:03d}, Accuracy: {:6f}, Loss: {:6f}'.format(epoch + 1, train_acc, loss_value))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()


        # with tf.compat.v1.Session(config=config) as sess:
        #     # Initializing the variables
        #     sess.run(tf.compat.v1.global_variables_initializer())
        #     sess.run(tf.compat.v1.local_variables_initializer())

        #     for epoch in range(0, self.epochs):
        #         while True:
        #             batch_images, batch_labels = train_data_gen.next()

        #             if np.any(batch_images) == None or np.any(batch_labels) == None:
        #                 train_data_gen.reset()
        #                 break

        #             _, X_val, _, y_val = train_test_split(batch_images, batch_labels, test_size=0.3)

        #             self.train_neural_network(sess, optimizer, x, y, keep_prob, self.kp, batch_images, batch_labels)
        #             print('Epoch {:>2}, '.format(epoch + 1), end='')
        #             self.print_stats(sess, x, y, keep_prob, batch_images, batch_labels, X_val, y_val, cost, accuracy)
