import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
import pdb
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense


IMAGES = './images'
IMG_HEIGHT = 384
IMG_WIDTH = 384

if __name__ == '__main__':
    batch_size = 2
    learning_rate = 1e-3
    kp = 0.3
    epochs = 1

    conv_ksize = 5 
    conv_strides = (1,1)
    pool_ksize = 5
    pool_strides = (1,1)
    padding = 'VALID'
    activation = 'relu'

    tf.keras.backend.clear_session()

    train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=IMAGES,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode='categorical')


    # Inputs
    x = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Model
    conv0 = tf.keras.layers.Conv2D(filters=32, kernel_size=conv_ksize, strides=conv_strides, 
                padding=padding, activation=activation, use_bias=True, name='conv0')(x)
    max_pool0 = tf.keras.layers.MaxPool2D(pool_size=pool_ksize, strides=pool_strides, padding=padding, name='max_pool0')(conv0)

    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=conv_ksize, strides=conv_strides, 
                padding=padding, activation=activation, use_bias=True, name='conv1')(max_pool0)
    max_pool1 = tf.keras.layers.MaxPool2D(pool_size=pool_ksize, strides=pool_strides, padding=padding, name='max_pool1')(conv1)

    dropout0 = tf.keras.layers.Dropout(kp, name='dropout0')(max_pool1)

    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=conv_ksize, strides=conv_strides, 
                padding=padding, activation=activation, use_bias=True, name='conv2')(dropout0)
    max_pool2 = tf.keras.layers.MaxPool2D(pool_size=pool_ksize, strides=pool_strides, padding=padding, name='max_pool2')(conv2)

    dropout1 = tf.keras.layers.Dropout(kp, name='dropout1')(max_pool2)


    flatten = tf.keras.layers.Flatten()(dropout1)

    fully_conn0 = tf.keras.layers.Dense(units=32, activation=activation, use_bias=True, name='fully_conn0')(flatten)
    fully_conn1 = tf.keras.layers.Dense(units=16, activation=activation, use_bias=True, name='fully_conn1')(fully_conn0)

    output = tf.keras.layers.Dense(units=2, activation=activation, use_bias=True, name='predictions')(fully_conn1)

    model = tf.keras.Model(inputs=x, outputs=output)
    model.summary()

    # Loss and Optimizer
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # # Accuracy
    train_acc_metric = tf.keras.metrics.CategoricalCrossentropy()

    for epoch in range(epochs):
        while True:
            batch_images, batch_labels = train_data_gen.next()

            # pdb.set_trace()
            print('Size in MB: {}'.format((batch_images.size * batch_images.itemsize)))

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

            print('Epoch {:>2}, Accuracy: {:6f}, Loss: {:6f}'.format(epoch + 1, train_acc, loss_value))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
