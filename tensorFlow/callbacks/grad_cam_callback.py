import numpy as np
import tensorflow as tf
import os
from tensorFlow.util.grad_cam import GradCam

class GradCamCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_image, batch_label):
        self.batch_image = batch_image
        self.batch_label = batch_label

    def on_epoch_end(self, epoch, logs=None):
        self.tensorboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'tensorflow'))

        file_writer = tf.summary.create_file_writer(self.tensorboard_path)
        with file_writer.as_default():
            image = tf.keras.preprocessing.image.img_to_array(self.batch_image)
            image = np.expand_dims(self.batch_image, axis=0)
            cam = GradCam(model=self.model, classIdx= 0 if self.batch_label[0] == 1. else 1)
            heatmap = cam.compute_heatmap(image)
            img = (image[0] * 255).astype('uint8')
            tf.summary.image('Input Image', image, epoch)

            (heatmap, output) = cam.overlay_heatmap(heatmap, img, alpha=0.5)
            
            output = np.vstack([img, heatmap, output])
            output = tf.keras.preprocessing.image.img_to_array(output)
            output = np.expand_dims(output, axis=0)
            
            tf.summary.image('Input Image -- GradCAM', output, epoch)