"""
Adapted from https://github.com/meetshah1995/pytorch-semseg
"""

import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from tqdm import tqdm

from models import unet, unet3, segnet, unet3d
from datasets import MRI3d, MRI
from metrics import runningScore
from utils import convert_state_dict

torch.backends.cudnn.benchmark = True

cudnn.benchmark = True

def validate(args):

    # Setup Dataloader
    if args.data_dim == 2:
        dataset = MRI(args.data, split=args.split, 
                     is_transform=True, img_size=(args.img_rows, args.img_cols))
    else:
        dataset = MRI3d(args.data, split=args.split, is_transform=True)

    n_classes = dataset.n_classes
    valloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    running_metrics = runningScore(n_classes)

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
    elif args.arch == 'unet3d':
        model = unet3d(n_classes=n_classes, in_channels=1, n_filter=4)

    state = convert_state_dict(torch.load(args.checkpoint)['model_state'])
    model.load_state_dict(state)
    model.eval()

    for i, (images, labels) in tqdm(enumerate(valloader)):
        model.cuda()
        images = Variable(images.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('-d', '--data', default='data', type=str)
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val', 
                        help='Split of dataset to test on')
    parser.add_argument('--data_dim', nargs='?', type=int, default=3, 
                        help='Dim of input data')
    args = parser.parse_args()
    validate(args)