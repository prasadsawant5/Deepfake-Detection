import tensorflow as tf

class SqueezeNet:
    def init(self):
        pass

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, scope_name, conv_ksize=7, 
        conv_strides=(2,2), pool_ksize=3, pool_strides=(2,2), padding='VALID', activation=None):
        with tf.name_scope(scope_name):
            conv_layer = tf.keras.layers.Conv2D(filters=conv_num_outputs, kernel_size=conv_ksize, strides=conv_strides, 
                padding=padding, activation=activation, use_bias=True, kernel_initializer=None)(x_tensor)

            if activation == None:
                conv_layer = tf.keras.layers.LeakyReLU()(conv_layer)

            conv_layer = tf.keras.layers.MaxPool2D(pool_size=pool_ksize, strides=pool_strides, padding=padding)(conv_layer)
            
            return conv_layer

    def fire_module(self, x_tensor, squeeze_filters, expand_filters, scope_name,  
        conv_strides=(1,1), pool_ksize=5, pool_strides=(1,1), padding='VALID', activation=None):
        with tf.name_scope(scope_name):
            squeeze1x1 = tf.keras.layers.Conv2D(filters=squeeze_filters, kernel_size=1, strides=conv_strides, 
                padding=padding, activation=activation, use_bias=True, kernel_initializer=None)(x_tensor)

            if activation == None:
                squeeze1x1 = tf.keras.layers.LeakyReLU()(squeeze1x1)

            expand1x1 = tf.keras.layers.Conv2D(filters=expand_filters, kernel_size=1, strides=conv_strides, 
                padding=padding, activation=activation, use_bias=True, kernel_initializer=None)(squeeze1x1)

            expand1x1 = tf.keras.layers.LeakyReLU()(expand1x1)

            expand3x3 = tf.keras.layers.Conv2D(filters=expand_filters, kernel_size=3, strides=conv_strides, 
                padding='SAME', activation=activation, use_bias=True, kernel_initializer=None)(squeeze1x1)

            expand3x3 = tf.keras.layers.LeakyReLU()(expand3x3)

            output = tf.keras.layers.Concatenate()([expand1x1, expand3x3])
            
            return output

    def pooling(self, x_tensor, scope_name, ksize=3, strides=(2,2), padding='VALID', pool_type='max'):
        with tf.name_scope(scope_name):
            if pool_type == 'max':
                return tf.keras.layers.MaxPool2D(pool_size=ksize, strides=strides, padding=padding)(x_tensor)
            else:
                return tf.keras.layers.AveragePooling2D(pool_size=ksize, strides=strides, padding=padding)(x_tensor)

    def output(self, x_tensor, scope_name='output', filters=2, conv_ksize=1, conv_strides=(1,1), padding='VALID', 
        activation='relu', pool_ksize=13, pool_strides=(1,1)):
        with tf.name_scope(scope_name):
            conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=conv_ksize, 
                strides=conv_strides, padding=padding, activation=activation, use_bias=True, kernel_initializer=None)(x_tensor)

            output = tf.keras.layers.AveragePooling2D(pool_size=pool_ksize, strides=None, padding=padding)(conv)

            return tf.keras.layers.Flatten(name='predictions')(output)

    def model(self, x, kp):
        conv0 = self.conv2d_maxpool(x, 96, 'conv0')

        fire1 = self.fire_module(conv0, 16, 64, 'fire1')
        fire2 = self.fire_module(fire1, 16, 64, 'fire2')
        fire3 = self.fire_module(fire2, 32, 128, 'fire3')

        maxpool4 = self.pooling(fire3, 'maxpool4')

        fire5 = self.fire_module(maxpool4, 32, 128, 'fire5')
        fire6 = self.fire_module(fire5, 48, 192, 'fire6')
        fire7 = self.fire_module(fire6, 48, 192, 'fire7')
        fire8 = self.fire_module(fire7, 64, 256, 'fire8')

        maxpool9 = self.pooling(fire8, 'maxpool9')

        fire10 = self.fire_module(maxpool9, 64, 256, 'fire10')
        dropout10 = tf.keras.layers.Dropout(kp, name='dropout10')(fire10)

        output = self.output(dropout10)

        cnn = tf.keras.Model(inputs=x, outputs=output)

        return cnn
