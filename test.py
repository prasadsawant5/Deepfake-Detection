import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import pdb
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense


IMAGES = './images'
IMG_HEIGHT = 227
IMG_WIDTH = 227
MODEL = './tf_model'

if __name__ == '__main__':
    
