import os
import cv2
import json
from shutil import copyfile, rmtree
from tqdm import tqdm
import im2rec
import time
import multiprocessing

DATA = './data'
REC = './rec'
CLASSES = ['real', 'fake']
METADATA = 'metadata.json'

def write_frames(d):
    with open(os.path.join(DATA, d, METADATA)) as f:
        meta = json.load(f)

    for k,v in tqdm(meta.items()):
        label = v['label']

        if label == 'FAKE':
            original = v['original']

            cap = cv2.VideoCapture(os.path.join(DATA, d, k))
            frame_no = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    if not os.path.exists(os.path.join(REC, CLASSES[-1], k.split('.')[0] + '_' + str(frame_no) + '.jpeg')):
                        # cv2.imwrite(os.path.join(REC, CLASSES[-1], k.split('.')[0] + '_' + str(frame_no) + '.jpeg'), frame)
                        print('{}_{}.jpeg missing from FAKE'.format(k.split('.')[0], frame_no))
                        frame_no += 1
                else:
                    break

            cap.release()

            cap = cv2.VideoCapture(os.path.join(DATA, d, original))
            frame_no = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    if not os.path.exists(os.path.join(REC, CLASSES[0], original.split('.')[0] + '_' + str(frame_no) + '.jpeg')):
                        # cv2.imwrite(os.path.join(REC, CLASSES[0], original.split('.')[0] + '_' + str(frame_no) + '.jpeg'), frame)
                        print('{}_{}.jpeg missing from REAL'.format(k.split('.')[0], frame_no))
                        frame_no += 1
                else:
                    break

            cap.release()


if __name__ == '__main__':
    if not os.path.exists(REC):
        os.mkdir(REC)

    for c in CLASSES:
        if not os.path.exists(os.path.join(REC, c)):
            os.mkdir(os.path.join(REC, c))

    processes = []
    for d in os.listdir(DATA):
        process = multiprocessing.Process(target=write_frames, args=(d, ))
        processes.append(process)
        process.start()
        process.join()

    for process in processes:
        process.join()

    for c in CLASSES:
        os.system('python3 im2rec.py ' + os.path.join(REC, c) + '_rec ' + REC + ' --recursive --list --num-thread 8')
        os.system('python3 im2rec.py ' + os.path.join(REC, c) + '_rec ' + REC + ' --recursive --pass-through --pack-label --num-thread 8')