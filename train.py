"""
Adapted from https://github.com/meetshah1995/pytorch-semseg
"""
import os
import time
import visdom
import argparse
import numpy as np
import torch.nn as nn
import sklearn.metrics as M
import matplotlib.pyplot as plt

from datasets import MRI
from unet_model import *
from shutil import copyfile
from torch.autograd import Variable
from torch.utils.data import DataLoader

def train(args):
    save = input('Do you want to save this model? (y/n)')
    # if (save == 'y'):
    #     filename =
    # Setup Dataloader
    startT = time.time()
    t_dataset = MRI(args.data, is_transform=True,
                    img_size=(args.img_rows, args.img_cols), numFiles = args.tNum)
    v_dataset = MRI(args.data, split='val',
                    is_transform=True, img_size=(args.img_rows, args.img_cols), numFiles = args.vNum)
    print('%.2fsec to load t and v files' % (time.time() - startT))
    n_classes = t_dataset.n_classes

    trainloader = DataLoader(t_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    valloader = DataLoader(v_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        
        # Display all hyperparameters: 
        vis.text('Hyperparameters: \
                  training files: {}<br> \
                  validation files: {}<br> \
                  learning rate: {}<br> \
                  batchsize: {} <br> \
                  number of epochs: {}<br> \
                  loss weight: {}<br>'.format(args.tNum, args.vNum, args.l_rate, args.batch_size, args.n_epoch, args.loss_weight))

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss',
                                         legend=['Loss']))

        # epoch_loss_window = vis.line(X = torch.zeros((1,)).cpu(),
        #                              Y = torch.zeros((1)).cpu(),
        #                              opts = dict(xlabel = 'epoch number',
        #                                          ylabel = 'loss',
        #                                          title = 'Training loss as a function of epoch number',
        #                                          legend = ['Loss']))

    # Setup Model (unet)
    assert (args.img_cols == args.img_rows) and (args.img_cols == 256), 'Image size not match!'
    # model = unet(n_classes=n_classes, in_channels=1, feature_scale=args.feature_scale) # meetshah's model
    model = UNet(n_channels = 1, n_classes = 1) # Dr. Hsiao's model

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.90, weight_decay=1e-6, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.l_rate)

    pos_weight = torch.tensor([args.loss_weight]).double()
    pos_weight = pos_weight.cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    start_epoch = args.start_epoch
    best_f1 = -100.0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            
    print('GPU used: %4.2f GB' % (torch.cuda.memory_allocated() / 1e9))

    allImgs = []
    allGts = []
    allPreds = []


    epochF1average = np.array([])
    epochF1std = np.array([])
    epochLossAverage = []
    epochLossStd = []

    numBatch = int(args.tNum / args.batch_size)

    for epoch in range(start_epoch, start_epoch + args.n_epoch):

        print('Epoch num: ' + str(epoch + 1))
        model.train() # "Sets the model in training mode"

        thisEpochLoss = []

        for i, (images, labels) in enumerate(trainloader):
            if i % int(numBatch / 5) == 0:
                print('%.2f%% of %dth epoch done' % ((i + 1) * 100 / numBatch, epoch + 1))
            torch.cuda.empty_cache()
            images = Variable(images.cuda(), requires_grad = True)
            labels = Variable(labels.cuda())

            outputs = model(images)

            loss = criterion(outputs.squeeze().double(), labels.squeeze().double()) # works with BCEloss and BCElossWithLogits

            thisEpochLoss.append(loss.item())
            # print('%.0fth batch: loss = %5.3f, GPU used = %4.2f GB' % (i + 1, loss.item(), (torch.cuda.memory_allocated() / 1e9)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.item()]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

        thisEpochLoss = np.array(thisEpochLoss)
        lossMean = np.mean(thisEpochLoss)
        lossStd = np.std(thisEpochLoss)
        print("Epoch [%d/%d] Loss avg: %.4f, std: %.4f" % (epoch+1, args.n_epoch, lossMean, lossStd))

        epochLossAverage.append(lossMean)
        epochLossStd.append(lossStd)

        # if args.visdom:
        #     vis.line(
        #             X = torch.ones((1, 1)).cpu() * epoch,
        #             Y=torch.Tensor([loss.item()]).unsqueeze(0).cpu(),
        #             win = epoch_loss_window,
        #             update = 'append')
        
        model.eval() # "Sets the model in eval mode"
        thisEpochF1 = []
        for i_val, (images_val, labels_val) in enumerate(valloader):
            torch.cuda.empty_cache()
            images_val = Variable(images_val.cuda(), requires_grad=False)
            labels_val = Variable(labels_val.cuda(), requires_grad=False)

            preds = model(images_val)
            preds = (torch.sigmoid(preds) > 0.5).float()
            preds = preds.data.squeeze().cpu().numpy().astype(np.float64)

            for pars in model.parameters():
                 pars.requires_grad=False

            gt = labels_val.data.cpu().numpy().astype(np.float64)

            if i_val == (int(args.vNum / args.batch_size) - 2):
                allImgs.append(images_val.squeeze().cpu().numpy())
                allGts.append(gt)
                allPreds.append(preds)

            thisEpochF1.append(M.f1_score(gt.flatten(), preds.flatten(), pos_label = 1))

        thisEpochF1 = np.array(thisEpochF1)
        F1mean = np.mean(thisEpochF1)
        F1std = np.std(thisEpochF1)
        print('F1 score avg: %.4f, std: %.4f' % (F1mean, F1std))

        epochF1average = np.append(epochF1average, F1mean)
        epochF1std = np.append(epochF1std, F1std)

        # if len(epochLoss) > 4:
        #     lastFour = epochLoss[-4:]
        #     if abs(max(lastFour) - min(lastFour)) < 0.003:
        #         print('Loss function is stagnant at {}th epoch, breaking from training'.format(epoch + 1))
        #         break

        
        # changing the intermediate model name broke the training cycle:
#        modelpath = os.path.join(args.checkpoint, '{}_model_train{}_val{}_lr{}__batchSize{}__epoch{}.pkl'.format(args.arch, args, args.tNum, args.vNum, args.l_rate, args.batch_size, args.n_epoch))

        # Useful for separating trials from useful models
        if save == 'y':
            modelpath = os.path.join(args.checkpoint, '{}_model.pkl'.format(args.arch))
            bestpath = os.path.join(args.checkpoint, '{}_best_model.pkl'.format(args.arch))
        else:
            modelpath = os.path.join(args.trial, '{}_model.pkl'.format(args.arch))
            bestpath = os.path.join(args.trial, '{}_best_model.pkl'.format(args.arch))

        mean_f1 = np.mean(epochF1average)
        is_best = mean_f1 > best_f1
        best_f1 = max(mean_f1, best_f1)

        # allStates store the model parameters (model_state and optimizer_state) but also additional information useful for our project)
        allStates = {'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'mean_f1': mean_f1,
                'best_f1': best_f1,}
        torch.save(allStates, modelpath)

        if is_best: # gives the best epoch a new name
            copyfile(modelpath, bestpath)

    print('Total time = %.2fsec' % (time.time() - startT))

    plt.figure()
    plt.errorbar(range(args.start_epoch, args.start_epoch + args.n_epoch), epochF1average, color = 'r', yerr = epochF1std)
    plt.title('Average F1 (tNum = {}, vNum = {})'.format(args.tNum, args.vNum))
    plt.ylabel('Average F1')
    plt.xlabel('Epoch num')

    plt.figure()
    plt.errorbar(range(args.start_epoch, args.start_epoch + args.n_epoch), epochLossAverage, color = 'b', yerr = epochLossStd)
    plt.title('Average Loss (tNum = {}, vNum = {})'.format(args.tNum, args.vNum))
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch num')

    return np.array(allImgs), np.array(allGts), np.array(allPreds), epochLossAverage, epochF1average

def plot(outputs, images, labels, batch = -1, frame = -1):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(images[batch][frame], 'gray')
    plt.subplot(3, 1, 2)
    plt.imshow(labels[batch][frame], 'gray')
    plt.subplot(3, 1, 3)
    plt.imshow(outputs[batch][frame], 'gray')
    plt.suptitle('{}th batch, {}th frame'.format(batch, frame))

def plotGtPred(gts, preds, epoch, file):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(gts[epoch][file], 'gray')
    plt.subplot(2, 1, 2)
    plt.imshow(preds[epoch][file], 'gray')
    plt.suptitle('{}th epoch\'s last batch, {}th file'.format(epoch, file))

# def longExperiments():


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='unet')
    parser.add_argument('--tNum', type=int, default=5000, help='Number of training files to load')
    parser.add_argument('--vNum', type=int, default=1000, help='Number of validation files to load')
    
    parser.add_argument('-w', '--loss_weight', type=float, default=303.0, help='weight for cross entropy loss calculation')
    parser.add_argument('-d', '--data', type=str, default='D:\\temp\\summer2019\\challenge')
    
    # Not sure how this argument works
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoint', metavar='PATH',
                    help='path to save checkpoints (default: checkpoint)')

    parser.add_argument('--trial', type=str, default='trial', metavar='PATH',
                        help='path to save trials (default: trial)')

    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.01,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    # previous checkpoint: 'D:\\temp\\summer2019\\ml-ntuh-001\\checkpoint\\unet_model.pkl'

    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by default')
    parser.add_argument('--data_dim', nargs='?', type=int, default=2,
                        help='Dim of input data')
    args = parser.parse_args()
    # train(args)
