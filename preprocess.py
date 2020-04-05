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

FAKE = './images/fake'
REAL = './images/real'

IMAGES = './images'

REC = './rec'
CLASSES = ['fake', 'real']

SAMPLE_TRAIN = './train_sample_videos'

CAP = 50000

def compare_frames(d):
    total_frames = 0
    total_fake_frames = 0
    total_real_frames = 0

    if not os.path.exists(FAKE):
        os.makedirs(FAKE)

    if not os.path.exists(REAL):
        os.makedirs(REAL)

    with open(os.path.join(d, METADATA)) as f:
        meta = json.load(f)

    for k,v in tqdm(meta.items()):
        label = v['label']

        if label == 'FAKE':
            original = v['original']

            fake_cap = cv2.VideoCapture(os.path.join(d, k))

            fake_count = fake_cap.get(cv2.CAP_PROP_FRAME_COUNT)

            real_cap = cv2.VideoCapture(os.path.join(d, original))

            real_count = real_cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if fake_count == real_count:
                frame_counter = 0
                while fake_cap.isOpened():
                    fake_ret, fake_frame = fake_cap.read()
                    real_ret, real_frame = real_cap.read()

                    if fake_ret == True and real_ret == True:
                        if np.all(fake_frame) == None or np.all(real_frame) == None:
                            print('Frame # {} is none in either fake or real'.format(frame_counter))
                            cv2.imshow('Fake', fake_frame)
                            cv2.imshow('Real', real_frame)

                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            continue
                        
                        if np.all(fake_frame) == np.all(real_frame):
                            continue
                        else:
                            if not os.path.exists(os.path.join(FAKE, k + '_' + str(frame_counter) + '.jpg')):
                                cv2.imwrite(os.path.join(FAKE, k + '_' + str(frame_counter) + '.jpg'), fake_frame)
                                total_fake_frames += 1

                            if not os.path.exists(os.path.join(REAL, original + '_' + str(frame_counter) + '.jpg')):
                                cv2.imwrite(os.path.join(REAL, original + '_' + str(frame_counter) + '.jpg'), real_frame)
                                total_real_frames += 1
                            
                            total_frames += 1

                        frame_counter += 1
                    else:
                        print('False value returned while reading frame # {} in {} and {}'.format(frame_counter, k, original))
                        break
            else:
                print('{} and {} don\'t have the same # of frames'.format(k, original))


            fake_cap.release()
            real_cap.release()

    print('Total frames in {}: {}'.format(d, total_frames))
    print('Total fake frames in {}: {}'.format(d, total_fake_frames))
    print('Total real frames in {}: {}'.format(d, total_real_frames))


if __name__ == '__main__':
    for d in os.listdir(DATA):
        t = threading.Thread(target=compare_frames, args=(os.path.join(DATA, d),))
        t.start()
        t.join()

    if not os.path.exists(REC):
        os.mkdir(REC)

    for c in CLASSES:
        os.system('python3 im2rec.py ' + os.path.join(REC, c) + '_rec ' + IMAGES + ' --recursive --list --num-thread 8')
        os.system('python3 im2rec.py ' + os.path.join(REC, c) + '_rec ' + IMAGES + ' --recursive --pass-through --pack-label --num-thread 8')