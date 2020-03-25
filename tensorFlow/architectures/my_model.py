import tensorflow as tf
import os
from numpy import prod

class MyModel:
    def __init__(self):
        pass

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, scope_name, conv_ksize=5, conv_strides=[1,1,1,1], pool_ksize=5, pool_strides=[1,1,1,1], padding='VALID'):
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
            dimension = x_tensor.get_shape().as_list()
            shape = list((conv_ksize, conv_ksize) + (dimension[-1],) + (conv_num_outputs,))
            maxpool_shape = list((pool_ksize, pool_ksize) + (dimension[-1],) + (conv_num_outputs,))
            weight = tf.Variable(tf.random.truncated_normal(shape, 0.0, 0.1))
            bias = tf.Variable(tf.zeros(conv_num_outputs))
            conv_layer = tf.nn.conv2d(x_tensor, weight, conv_strides, padding)
            conv_layer = tf.nn.bias_add(conv_layer, bias)
            conv_layer = tf.nn.relu(conv_layer)
            conv_layer = tf.nn.max_pool(conv_layer, ksize=pool_ksize, strides=pool_strides, padding=padding)

            # conv_layer = tf.keras.layers.Conv2D(
            #     conv_num_outputs, conv_ksize, strides=conv_strides, padding=padding,
            #     dilation_rate=(1, 1), activation='relu', use_bias=True,
            #     kernel_initializer='glorot_uniform', bias_initializer='zeros'
            # )
            
            return conv_layer

    def flatten(self, x_tensor, scope_name):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """
        with tf.name_scope(scope_name):
            dimension = x_tensor.get_shape().as_list()
            x_tensor = tf.reshape(x_tensor, [-1, prod(dimension[1:])])
            return x_tensor

    def fully_conn(self, x_tensor, num_outputs, scope_name):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        with tf.name_scope(scope_name):
            dimension = x_tensor.get_shape().as_list()
            shape = list((dimension[-1],) + (num_outputs,))
            weight = tf.Variable(tf.random.truncated_normal(shape, 0, 0.1))
            bias = tf.Variable(tf.zeros(num_outputs))
            
            fully_conn = tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))
            return fully_conn

    def output(self, x_tensor, num_outputs=2, scope_name='output'):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        with tf.name_scope(scope_name):
            dimension = x_tensor.get_shape().as_list()
            shape = list((dimension[-1],) + (num_outputs,))
            weight = tf.Variable(tf.random.truncated_normal(shape, 0, 0.01))
            bias = tf.Variable(tf.zeros(num_outputs))
            
            return tf.add(tf.matmul(x_tensor,weight), bias)

    def model(self, x, kp):
        cnn = self.conv2d_maxpool(x, 32, 'conv0')
        cnn = self.conv2d_maxpool(cnn, 64, 'conv1')
        cnn = tf.nn.dropout(cnn, kp, name='dropout0')

        cnn = self.conv2d_maxpool(cnn, 128, 'conv2')
        cnn = self.conv2d_maxpool(cnn, 256, 'conv3')
        cnn = tf.nn.dropout(cnn, kp, name='dropout2')

        # cnn = self.conv2d_maxpool(cnn, 512, 'conv4')

        cnn = self.flatten(cnn, 'flatten')

        cnn = self.fully_conn(cnn, 256, 'fully_conn0')
        cnn = self.fully_conn(cnn, 64, 'fully_conn1')
        cnn = self.fully_conn(cnn, 32, 'fully_conn2')
        cnn = self.output(cnn)

        return cnn


