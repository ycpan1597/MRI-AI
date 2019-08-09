"""
Created by
author: rowantseng 2018.05.04

Purpose: Prepare data for training tumor segmentation
"""

import os
import pandas as pd

def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
root = 'D:\\temp\\summer2019'
folder = 'challenge'

# Create csv for matching images and ground truths

imgList, labList = [], []

imgPath = os.path.join(root, folder, 'imagePNG')
labPath = os.path.join(root, folder, 'labelPNG')

for img, lab in zip(os.listdir(imgPath), os.listdir(labPath)):
    if (not img.startswith('.')) and (not lab.startswith('.')):
        imgList.append(os.path.join(root, folder, 'imagePNG', img))
        labList.append(os.path.join(root, folder, 'labelPNG', lab))

vtRatio = 0.2 # ratio validation : training
df = pd.DataFrame({'image': imgList, 'label': labList})

tNum = int(len(df) * (1 - vtRatio))
dfTrain = df[:tNum]
dfVal = df[tNum:]

df.to_csv(os.path.join(root, folder, 'image-label.csv'))
dfTrain.to_csv(os.path.join(root, folder, 'train.csv'))
dfVal.to_csv(os.path.join(root, folder, 'val.csv'))

print('Creating img-label pair csv done! Total image-label pair: {}'.format(len(imgList)))