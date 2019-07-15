from PIL import Image
import os
import nibabel as nib
import pandas as pd
import numpy as np
import scipy.misc as misc

def get_pascal_labels():
    return np.asarray([[0, 0, 0], [128,0,0], [0,128,0], [128,128,0],
                       [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                       [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                       [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                       [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                       [0,64,128]])

def decode_segmap(label_mask):
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 2):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


df = pd.read_csv('../jsonfile/small_train.csv')
targets = df['targets'].tolist()

for i, t in enumerate(targets):
    gt = nib.load(t).get_data()
        
    for d in range(gt.shape[2]):
        if gt[:, :, d].sum() > 0:
            gt_name = '{:0>4d}_{:0>3d}.jpg'.format(i, d)
            print (gt_name)
            # gt_slice = Image.fromarray(gt[:, :, d]).convert('L')
            mask = decode_segmap(gt[:, :, d])
            misc.imsave(os.path.join(os.getcwd(), 'gtmask', gt_name), mask)
