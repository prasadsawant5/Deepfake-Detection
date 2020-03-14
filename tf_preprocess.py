import os
import cv2
from tqdm import tqdm
import h5py
import numpy as np
import random
from threading import Thread, Event

IMG_SIZE = [384, 384]
FAKE = './rec/fake'
REAL = './rec/real'

IMAGES = './images.h5'
LABELS = './labels.h5'

BATCH_SIZE = 512

def read_image(path):
    return cv2.resize(cv2.imread(path), tuple(IMG_SIZE))


if __name__ == '__main__':
    if not os.path.isfile(IMAGES) and not os.path.isfile(LABELS):
        images = h5py.File(IMAGES, 'w')
        labels = h5py.File(LABELS, 'w')

        total_images = len(os.listdir(FAKE)) + len(os.listdir(REAL))

        dirs = []
        for d in os.listdir(FAKE):
            dirs.append(os.path.join(FAKE, d))

        for d in os.listdir(REAL):
            dirs.append(os.path.join(REAL, d))

        random.shuffle(dirs)

        img = []
        lb = []

        db_idx = 0

        for i in tqdm(range(0, total_images)):
            img.append(read_image(dirs[i]))
            if REAL in dirs[i]:
                lb.append([0., 1.])
            else:
                lb.append([1., 0.])
            
            if i % BATCH_SIZE == 0 or i == total_images - 1:
                if i % BATCH_SIZE == 0:
                    images.create_dataset('images_{}'.format(db_idx), (BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.int32)
                    labels.create_dataset('labels_{}'.format(db_idx), (BATCH_SIZE, 2), dtype=np.float32)
                else:
                    images.create_dataset('images_{}'.format(db_idx), (435, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.int32)
                    labels.create_dataset('labels_{}'.format(db_idx), (435, 2), dtype=np.float32)

                images['images_{}'.format(db_idx)][...] = img
                labels['labels_{}'.format(db_idx)][...] = lb

                db_idx += 1
                img = []
                lb = []

    else:
        images = h5py.File(IMAGES, 'r')
        for k in range(len(list(images.keys()))):
            print(k)

        print(images.get('images_184').shape)