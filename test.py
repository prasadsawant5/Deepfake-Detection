import tensorflow as tf

if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()

    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session(config=None) as sess:
        # Initializing the variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())