"""
Adapted from https://github.com/meetshah1995/pytorch-semseg
"""

import os
import PIL
import torch
import numpy as np
import collections
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from torch.utils import data
from os.path import join as pjoin
from sklearn import preprocessing

class MRI(data.Dataset):
    def __init__(self, root, split='train', is_transform=False,
                 img_size=256, augmentations=None, numFiles = 100, mode = 'train'):
        self.root = os.path.expanduser(root)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.split = split
        self.n_classes = 1
        self.images = collections.defaultdict(list)
        # The line above creates a dictionary whose key is "split" and whose value
        # is a list of all the images or labels
        self.labels = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)


        if mode == 'test':
            # root test
            images = os.listdir(os.path.join(root, 'image'))
            labels = os.listdir(os.path.join(root, 'label'))
            for index, (i, t) in enumerate(zip(images, labels)):
                if i.endswith('nii') and t.endswith('nii') and index < 10:
                    img_data = nib.load(os.path.join(root, 'image', i)).get_data()
                    lab_data = nib.load(os.path.join(root, 'label', t)).get_data()
                    
                    for d in range(lab_data.shape[2]):
                        lab_slice = lab_data[:, :, d]
                        img_slice = img_data[:, :, d]
                        if np.count_nonzero(lab_slice) != 0:
        #                   resizing all images to the same size
                            img_slice = np.array(PIL.Image.fromarray(img_slice).resize(img_size)).astype(np.float64)
                            lab_slice = np.array(PIL.Image.fromarray(lab_slice).resize(img_size, resample = PIL.Image.NEAREST)).astype(np.float64)
                            scaler = preprocessing.StandardScaler()
                            img_slice = scaler.fit_transform(img_slice)
                            self.images[split].append(img_slice)
                            self.labels[split].append(lab_slice)

            print('Test data is ready!')
                    
        else: 
            
            df = pd.read_csv(pjoin(self.root, split + '.csv'))
            
            images = df['image'][:numFiles].tolist()
            labels = df['label'][:numFiles].tolist()
            for index, (i, l) in enumerate(zip(images, labels)):
                if index % (int(len(images) / 10)) == 0:
                    print('%.0f%% of %s files read' % ((index + 1) * 100 / len(images), split))
                # Load data and labels
                img_slice = plt.imread(i)
                lab_slice = plt.imread(l)

                img_slice = np.array(PIL.Image.fromarray(img_slice).resize(img_size)).astype(np.float64)
                lab_slice = np.array(PIL.Image.fromarray(lab_slice).resize(img_size, resample = PIL.Image.NEAREST)).\
                                astype(np.float64)
                scaler = preprocessing.StandardScaler()

                img_slice = scaler.fit_transform(img_slice)
                # lab_slice = scaler.fit_transform(lab_slice) # we don't normalize the label/gt-mask so that the output stays binarized

                self.images[split].append(img_slice)
                self.labels[split].append(lab_slice)
    
            print ('{} data is ready!'.format(split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):
        im = np.array(self.images[self.split][index], dtype=np.float32)
        lbl = np.array(self.labels[self.split][index], dtype=np.int8)

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        # NHWC -> NCWH
        img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)

        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray([[0, 0, 0], [128,0,0], [0,128,0], [128,128,0],
                          [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                          [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                          [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                          [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                          [0,64,128]])


    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def decode_segmap2(self, label_mask, gt_mask):
        """Decode segmentation class labels and gts into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            gt_mask (np.ndarray): an (M,N) array of integer values denoting
              the class gt at each spatial location.

        Returns:
            (np.ndarray, optional): the resulting decoded color label image.
            (np.ndarray, optional): the resulting decoded color gt image.
        """
        label = self.decode_segmap(label_mask)
        gt = self.decode_segmap(gt_mask)

        return label, gt
