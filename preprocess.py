import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import multiprocessing
import threading
from random import shuffle

DATA = './data'
METADATA = 'metadata.json'

FAKE = './rec/fake'
REAL = './rec/real'

def read_frames(d):
    total_frames = 0
    total_fake_frames = 0
    total_real_frames = 0

    with open(os.path.join(DATA, d, METADATA)) as f:
        meta = json.load(f)

    for k,v in tqdm(meta.items()):
        label = v['label']

        if label == 'FAKE':
            original = v['original']

            cap = cv2.VideoCapture(os.path.join(DATA, d, k))

            total_frames += cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_fake_frames += cap.get(cv2.CAP_PROP_FRAME_COUNT)

            cap.release()

            cap = cv2.VideoCapture(os.path.join(DATA, d, original))

            total_frames += cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_real_frames += cap.get(cv2.CAP_PROP_FRAME_COUNT)

            cap.release()

    print('Total frames in {}: {}'.format(d, total_frames))
    print('Total fake frames in {}: {}'.format(d, total_fake_frames))
    print('Total real frames in {}: {}'.format(d, total_real_frames))

def util_process(d):
    with open(os.path.join(DATA, d, METADATA)) as f:
        meta = json.load(f)

    for k,v in meta.items():
        label = v['label']

        if label == 'FAKE':
            original = v['original']

            cap = cv2.VideoCapture(os.path.join(DATA, d, k))

            fake_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            cap.release()

            cap = cv2.VideoCapture(os.path.join(DATA, d, original))

            org_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            cap.release()

            if fake_frames != org_frames:
                print('{} has {} frames whereas {} has {} frames...'.format(k, fake_frames, original, org_frames))

def remove_images(directory):
    for d in tqdm(os.listdir(directory)):
        img = cv2.imread(os.path.join(directory, d))

        if np.all(img) == None or os.stat(os.path.join(directory, d)).st_size == 0:
            os.remove(os.path.join(directory, d))

def balance_classes():
    real = os.listdir(REAL)
    fake = os.listdir(FAKE)

    if len(real) > len(fake):
        shuffle(real)

        for i in tqdm(range(0, len(real) - len(fake))):
            if os.path.exists(os.path.join(REAL, real[i])):
                try:
                    os.remove(os.path.join(REAL, real[i]))
                except OSError as e:
                    print(e)
    elif len(real) < len(fake):
        shuffle(fake)

        for i in tqdm(range(0, len(fake) - len(real))):
            if os.path.exists(os.path.join(FAKE, fake[i])):
                try:
                    os.remove(os.path.join(FAKE, fake[i]))
                except OSError as e:
                    print(e)


if __name__ == '__main__':
    # t0 = threading.Thread(target=remove_images, args=(FAKE,))
    # t1 = threading.Thread(target=remove_images, args=(REAL,))

    # t0.start()
    # t1.start()

    # t0.join()
    # t1.join()
    balance_classes()
    # processes = []

    # for d in os.listdir(DATA):
    #     process = multiprocessing.Process(target=util_process, args=(d, ))
    #     processes.append(process)
    #     process.start()

    # for process in processes:
    #     process.join()

            
