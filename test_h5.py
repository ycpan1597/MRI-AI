"""
Adapted from https://github.com/meetshah1995/pytorch-semseg
"""

import os
from os.path import join as pjoin
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import unet, unet3, segnet
from dataloader_h5 import get_loader, get_data_path
from utils import convert_state_dict

from PIL import Image
import matplotlib.pyplot as plt
import h5py
#%%

def test(args):

    data_loader = get_loader()
    data_path = get_data_path(args)
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes

    # Setup Model
    if args.arch == 'unet':
        assert (args.img_cols == args.img_rows) and (args.img_cols == 256), 'Image size not match!'
        model = unet(n_classes=n_classes, in_channels=1)
    elif args.arch == 'unet3':
        assert (args.img_cols == args.img_rows) and (args.img_cols == 64), 'Image size not match!'
        model = unet3(n_classes=n_classes, in_channels=1)
    elif args.arch == 'segnet':
        assert (args.img_cols == args.img_rows) and (args.img_cols == 256), 'Image size not match!'
        model = segnet(n_classes=n_classes, in_channels=1)
    
    state = convert_state_dict(torch.load(args.checkpoint)['model_state'])
    model.load_state_dict(state)
    model.eval()
    model.cuda(0)

    h5 = h5py.File(pjoin(args.data, 'data-val.h5'), 'r')
    filenames = h5.get('filenames')[:].tolist()
    filenames = [f.decode('utf8')+'.jpg' for f in filenames]
    images = h5.get('image_norms')[:]
    # originals = h5.get('images')[:]
    targets = h5.get('targets')[:]

    for i, filename in enumerate(filenames):
        im = np.array(images[i])
        # ori = np.array(originals[i])
        im = im.astype(np.float64)
        im = np.expand_dims(im, axis=2)
        im = im.transpose(2, 0, 1)
        im = np.expand_dims(im, 0)
        im = torch.from_numpy(im).float()
        if torch.cuda.is_available():
            model.cuda(0)
            im = Variable(im.cuda(0), volatile=True)
        else:
            im = Variable(im, volatile=True)

        lbl = np.array(targets[i])

        outputs = F.softmax(model(im), dim=1)

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

        decoded_pred, decoded_gt = loader.decode_segmap2(label_mask=pred, 
                                                         gt_mask=lbl)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('Ground Truth', fontsize=20)
        plt.imshow(decoded_gt)
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Prediction', fontsize=20)
        plt.imshow(decoded_pred)
        fig.savefig(os.path.join(args.out_path, 'mask', filename), 
                    bbox_inches='tight')
        plt.close(fig)

        # bg = Image.fromarray(ori).convert('RGB')
        # fg_gt = Image.fromarray(decoded_gt.astype(dtype=np.uint8))
        # fg_pred = Image.fromarray(decoded_pred.astype(dtype=np.uint8))
        # b1 = Image.blend(bg, fg_gt, alpha=0.5)
        # b2 = Image.blend(bg, fg_pred, alpha=0.5)
        
        # fig = plt.figure(figsize=(16, 8))
        # ax = fig.add_subplot(1, 2, 1)
        # ax.set_title('Ground Truth', fontsize=20)
        # plt.imshow(b1)
        # ax = fig.add_subplot(1, 2, 2)
        # ax.set_title('Prediction', fontsize=20)
        # plt.imshow(b2)
        # fig.savefig(os.path.join(args.out_path, 'map', filename), 
        #             bbox_inches='tight')
        # plt.close(fig)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch', nargs='?', type=str, default='unet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('-d', '--data', default='data', type=str)
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)