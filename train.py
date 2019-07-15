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
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import unet, unet3, segnet, unet3d
from datasets import MRI3d, MRI
from metrics import runningScore
from loss import *
import os
from shutil import copyfile


def train(args): # pp: args is a list of arguments

    # Setup Dataloader
    if args.data_dim == 2:
        t_dataset = MRI(args.data, is_transform=True,
                        img_size=(args.img_rows, args.img_cols))
        v_dataset = MRI(args.data, split='val',
                        is_transform=True, img_size=(args.img_rows, args.img_cols))
    else:
        t_dataset = MRI3d(args.data, is_transform=True)
        v_dataset = MRI3d(args.data, split='val', is_transform=True)

    n_classes = t_dataset.n_classes
    trainloader = DataLoader(t_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = DataLoader(v_dataset, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model - model is a custom class
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

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate) #, momentum=0.99, weight_decay=5e-4)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        if args.data_dim == 2:
            loss_fn = cross_entropy2d
        else:
            loss_fn = cross_entropy3d

    start_epoch = args.start_epoch
    best_iou = -100.0 # intersection over union (or Jaccard index)
    if args.resume is not None: # if you're trying to resume training 
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            best_iou = checkpoint['best_iou']
            start_epoch = checkpoint['epoch']
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    print('Start training from scratch!')
    for epoch in range(start_epoch, args.n_epoch):
        model.train() # "Sets the model in training mode"
        for i, (images, labels) in enumerate(trainloader):
            torch.cuda.empty_cache()
            print(i)
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            model.zero_grad()
            outputs = model(images)
            loss = loss_fn(input=outputs, target=labels, weight=args.weight)
            loss.backward()
            optimizer.step()
            
            

#            if args.visdom:
#                vis.line(
#                    X=torch.ones((1, 1)).cpu() * i,
#                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
#                    win=loss_window,
#                    update='append')
#
#            if (i+1) % 20 == 0:
#                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

#        model.eval() # "Sets the model in eval mode"
#        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
#            images_val = Variable(images_val.cuda(), volatile=True)
#            labels_val = Variable(labels_val.cuda(), volatile=True)
#
#            outputs = model(images_val)
#            pred = outputs.data.max(1)[1].cpu().numpy()
#            gt = labels_val.data.cpu().numpy()
#            running_metrics.update(gt, pred)
#
#        score, class_iou = running_metrics.get_scores()
#        for k, v in score.items():
#            print(k, v)
#        running_metrics.reset()
#
#        mean_iou = score['Mean IoU : \t']
#        is_best = mean_iou > best_iou
#        best_iou = max(mean_iou, best_iou)
#
#        modelpath = os.path.join(args.checkpoint, '{}_model.pkl'.format(args.arch))
#        bestpath = os.path.join(args.checkpoint, '{}_best_model.pkl'.format(args.arch))
#        state = {'epoch': epoch+1,
#                 'model_state': model.state_dict(),
#                 'optimizer_state' : optimizer.state_dict(),
#                 'mean_iou': mean_iou,
#                 'best_iou': best_iou,}
#        torch.save(state, modelpath)
#
#        if is_best:
#            copyfile(modelpath, bestpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('-w', '--weight', default=10, type=int)
    parser.add_argument('-d', '--data', default='D:\\temp\\summer2019\\challenge', type=str)
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=150,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--visdom', nargs='?', type=bool, default=False,
                        help='Show visualization(s) on visdom | False by  default')
    parser.add_argument('--data_dim', nargs='?', type=int, default=3,
                        help='Dim of input data')
    args = parser.parse_args()
    train(args)