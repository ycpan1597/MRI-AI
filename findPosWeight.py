import numpy as np
import cv2
import os

if __name__ == '__main__':
    dir = "D:\\temp\\summer2019\\challenge\\labelPNG"
    labels = []
    for item in os.listdir(dir):
        labels.append(cv2.imread(os.path.join(dir, item), 0))

    posWeight = []
    for item in labels:
        flat = item.flatten()
        posWeight.append(np.count_nonzero(flat) / len(flat))

    posWeight = np.array(posWeight)
    print('posweight avg = %.2f, posweight std = %.2f' % (np.mean(posWeight), np.std(posWeight)))