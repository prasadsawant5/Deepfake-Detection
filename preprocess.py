import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import multiprocessing

DATA = './data'
METADATA = 'metadata.json'

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

if __name__ == '__main__':
    processes = []

    for d in os.listdir(DATA):
        process = multiprocessing.Process(target=read_frames, args=(d, ))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

            
