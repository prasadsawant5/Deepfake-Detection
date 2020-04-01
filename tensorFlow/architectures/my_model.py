import tensorflow as tf
import os
from numpy import prod

class MyModel:
    def __init__(self):
        pass

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, scope_name, conv_ksize=5, 
        conv_strides=(1,1), pool_ksize=5, pool_strides=(1,1), padding='VALID', activation='relu'):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        with tf.name_scope(scope_name):
            conv_layer = tf.keras.layers.Conv2D(filters=conv_num_outputs, kernel_size=conv_ksize, strides=conv_strides, 
                padding=padding, activation=activation, use_bias=True)(x_tensor)

            conv_layer = tf.keras.layers.MaxPool2D(pool_size=pool_ksize, strides=pool_strides, padding=padding)(conv_layer)
            
            return conv_layer

    def flatten(self, x_tensor, scope_name):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """
        with tf.name_scope(scope_name):
            flatten = tf.keras.layers.Flatten()(x_tensor)
            return flatten

    def fully_conn(self, x_tensor, num_outputs, scope_name, activation='relu'):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        with tf.name_scope(scope_name):
            fully_conn = tf.keras.layers.Dense(units=num_outputs, activation=activation, use_bias=True)(x_tensor)
            return fully_conn

    def output(self, x_tensor, num_outputs=2, activation='relu', scope_name='output'):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        with tf.name_scope(scope_name):
            output = tf.keras.layers.Dense(units=num_outputs, activation=activation, use_bias=True, name='predictions')(x_tensor)
            return output

    def model(self, x, kp):
        conv0 = self.conv2d_maxpool(x, 16, 'conv0')
        conv1 = self.conv2d_maxpool(conv0, 32, 'conv1')
        dropout0 = tf.keras.layers.Dropout(kp, name='dropout0')(conv1)

        conv2 = self.conv2d_maxpool(dropout0, 64, 'conv2')
        dropout2 = tf.keras.layers.Dropout(kp, name='dropout2')(conv2)

        flatten = self.flatten(dropout2, 'flatten')

        fully_conn0 = self.fully_conn(flatten, 32, 'fully_conn0')
        # fully_conn1 = self.fully_conn(fully_conn0, 16, 'fully_conn1')
        fully_conn2 = self.fully_conn(fully_conn0, 8, 'fully_conn2')
        output = self.output(fully_conn2)

        cnn = tf.keras.Model(inputs=x, outputs=output)

        return cnn


