"""
Adapted from https://github.com/meetshah1995/pytorch-semseg
"""

import os
import sys
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

from datasets import MRI
import imageio
from models import unet
#from ptsemseg.loader import get_loader, get_data_path
# Note to self: get_data_path is an obsolete function that is no longer included in the ptsemseg library
from ptsemseg.loader import get_loader
from nibabel.data import get_data_path
from utils import convert_state_dict
from PIL import Image

from torch.utils.data import DataLoader


try:
    import pydensecrf.densecrf as dcrf
except:
    print("Failed to import pydensecrf,\
           CRF post-processing will not work")

def test(args):

    data_loader = get_loader(args.dataset)
    data_path = get_data_path()
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes

    # Setup Model
    model = unet(n_classes=n_classes, in_channels=1)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    model.cuda(0)

    # Shannon and Preston's way of processing test files
    test_dataset = MRI(args.test_root, img_size=(args.img_rows, args.img_cols), mode = 'test')
    testLoader = DataLoader(test_dataset, batch_size = 1)
    
    for (img, gt) in testLoader:
        img = img.numpy()
        img = img.astype(np.float64)
        # img -= loader.mean
        img -= 128
        img = img.astype(float) / 255.0
        img = np.expand_dims(img, axis=2)
        # NHWC -> NCWH # what does this mean? 
        img = img.transpose(2, 0, 1) 
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        images = Variable(img.cuda(0), volatile=True)
        
    
    print("files are read!")
#    
#        
#    
#    # Setup image (meetshah's way of reading files)
#    print("Read Input Image from : {}".format(args.img_path))
#    filenames = os.listdir(args.img_path)
#    
#    
#    # we still need to read the image and label separately, right? 
#
#    for filename in filenames:
#        img = imageio.imread(os.path.join(args.img_path, filename))
#        
#        resized_img = misc.imresize(img, (args.img_rows, args.img_cols), interp='bicubic')
#
#        # img = img[:, :, ::-1]
#        img = img.astype(np.float64)
#        # img -= loader.mean
#        img -= 128
#        img = misc.imresize(img, (args.img_rows, args.img_cols))
#        img = img.astype(float) / 255.0
#        img = np.expand_dims(img, axis=2)
#        # NHWC -> NCWH
#        img = img.transpose(2, 0, 1) 
#        img = np.expand_dims(img, 0)
#        img = torch.from_numpy(img).float()
#
#        images = Variable(img.cuda(0), volatile=True)
#
#        outputs = F.softmax(model(images), dim=1)
#        
#        if args.dcrf == "True":
#            unary = outputs.data.cpu().numpy()
#            unary = np.squeeze(unary, 0)
#            unary = -np.log(unary)
#            unary = unary.transpose(2, 1, 0)
#            w, h, c = unary.shape
#            unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
#            unary = np.ascontiguousarray(unary)
#           
#            resized_img = np.ascontiguousarray(resized_img)
#
#            d = dcrf.DenseCRF2D(w, h, loader.n_classes)
#            d.setUnaryEnergy(unary)
#            d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)
#
#            q = d.inference(50)
#            mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
#            decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
#            dcrf_path = args.out_path[:-4] + '_drf.png'
#            misc.imsave(dcrf_path, decoded_crf)
#            print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))
#
#        if torch.cuda.is_available():
#            model.cuda(0)
#            images = Variable(img.cuda(0), volatile=True)
#        else:
#            images = Variable(img, volatile=True)
#
#        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
#        decoded = loader.decode_segmap(pred)
#        print('Classes found in {}: {}'.format(filename, np.unique(pred)))
#        misc.imsave(os.path.join(args.out_path, 'pred', filename), decoded)
#        # print("Segmentation Mask Saved at: {}".format(args.out_path))
#
#        gt = Image.open(os.path.join(args.gt_path, filename))
#        decoded = loader.decode_segmap(np.asarray(gt))
#        # print('Classes found: ', np.unique(np.asarray(gt)))
#        misc.imsave(os.path.join(args.out_path, 'gt', filename), decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='D:\\temp\\summer2019\\ml-ntuh-001\\checkpoint\\unet_model.pkl', 
                        help='Path to the saved model')
    
    # What is this referring to? 
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--dcrf', nargs='?', type=str, default="False",
                        help='Enable DenseCRF based post-processing')
    parser.add_argument('--img_path', nargs='?', type=str, default='D:\\temp\\summer2019\\test\\image', 
                        help='Path of the input image')
    parser.add_argument('--gt_path', nargs='?', type=str, default='D:\\temp\\summer2019\\test\\label', 
                        help='Path of the gt image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    
    parser.add_argument('--test_root', type=str, default='D:\\temp\\summer2019\\test', help='root of test directory')
    args = parser.parse_args()
    test(args)