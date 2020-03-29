import cv2
import numpy as np
import os
from tqdm import tqdm

IMAGES = './images/real'

if __name__ == '__main__':
    d = os.listdir(IMAGES)[32:]
    images = list()

    for i in tqdm(range(32)):
        img = cv2.resize(cv2.imread(os.path.join(IMAGES, d[i])), (384, 384))
        images.append(img)

    img = np.array(images) / 255.
    np.save('test.npy', img)
