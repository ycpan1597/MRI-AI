"""
Created by
author: rowantseng 2018.05.04

Purpose: Prepare data for training tumor segmentation including extracting .gz 
         to .nii files, and creating csv for matching images and targets. 
         In addition, a h5 file storing matching images and labels is also 
         created to be used by the training script. 
"""

import os
import PIL
import numpy as np
import nibabel as nib
import pandas as pd
import h5py
from sklearn import preprocessing 

def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
root = 'D:\\temp\\summer2019'
folders = ['challenge']

"""
Extract .gz to .nii files

Preston: the following section only decompresses gz files on macOS, not windows. 
At the moment, Shannon and I have been manually extracting content from gz files 
using 7-zip directly from the folder GUI. 
"""

#totalNum = 0
#for folder in folders:
#    subfolders = os.listdir(os.path.join(root, folder))
#    print(os.path.join(root, folder))
#    print(subfolders)
#    for s in subfolders:
#        print(s)
#        if not s.startswith('.') and (s.startswith('i') or s.startswith('l')):
#            files = os.listdir(os.path.join(root, folder, s))
#            print(os.path.join(root, folder, s))
#            print(files)
#            for f in files:
#                if f.endswith('.targets.nii.gz') or (f.endswith('.nii.gz') and not f.endswith('.result.nii.gz')):
#                    os.system('gunzip -k ' + os.path.join(root, folder, s, f))
#                    totalNum += 1
#                    print ('{} .gz processed.'.format(totalNum))
#print ('Extracting files done!')


"""
Create csv for matching images and targets and copy .nii files

Preston: the following section aligns the image files with their corresponding 
         tumor labels.
"""
imgList, labList = [], []

img = os.path.join(root, folders[0], 'image')
lab = os.path.join(root, folders[0], 'label')
for imgF in os.listdir(img):
    if imgF.endswith('.nii') and not imgF.endswith('.nii.gz'):
        for labF in os.listdir(lab):
            if imgF == labF:
                imgList.append(os.path.join(root, folders[0], 'image', imgF))
                labList.append(os.path.join(root, folders[0], 'label', labF))

df = pd.DataFrame({'image': imgList, 'label': labList})
df.to_csv(os.path.join(root, 'image-label.csv'))
print ('Creating img-label pair csv done! Total image-label pair: {}'.format(len(imgList)))

"""
Create demean and unit variance data and save in h5 (w/o sliding window)

Preston: h5 is a pythonic file format capable of storing a large amount of data
"""
mri_df = pd.read_csv(os.path.join(root, 'image-label.csv'))
imgList = mri_df.image.tolist()
labList = mri_df.label.tolist()
filenames = []
imSize = (256, 256)

data_path = os.path.join(root, 'challenge')
checkdir(data_path)

# Arrays used for stacking images
# pp: note that the number of frames are stored in the first dimension
images = np.empty([1, imSize[0], imSize[1]])
image_norms = np.empty([1, imSize[0], imSize[1]])
labels = np.empty([1, imSize[0], imSize[1]], dtype=int)

schedule = 5 # number of files to include into the training and validation dataset

print ('Starting preparing data...')
for i, f in enumerate(imgList):

    if i in range(schedule):
        print ('{}/{}: {}'.format(i, len(imgList), f))
        # Get gt and image file 
        img_data = nib.load(f).get_data()
        lab_data = nib.load(labList[i]).get_data()
        
        # slice every row from top to bottom
        for d in range(lab_data.shape[2]):
            # Read slice and resize every slice to imSize
            label = lab_data[:, :, d]
            image = img_data[:, :, d]
            image = np.array(PIL.Image.fromarray(image).resize(imSize)).astype(np.float64)
            label = np.array(PIL.Image.fromarray(label).resize(imSize, resample = PIL.Image.NEAREST)).astype(np.float64)
            #image = imresize(image, imSize) 
            #label = imresize(label, imSize, 'nearest', mode='F')
            
            print ('resize ready')
            # Demean and unit variance
            scaler = preprocessing.StandardScaler().fit(image)
            image_norm = scaler.transform(image)
            print ('norm ready')
            # Write to numpy array
            image = np.expand_dims(image, axis=0)
            image_norm = np.expand_dims(image_norm, axis=0)
            label = np.expand_dims(label, axis=0)
            images = np.concatenate((images, image), axis=0)
            image_norms = np.concatenate((image_norms, image_norm), axis=0)
            labels = np.concatenate((labels, label), axis=0)
            print ('concat ready')
            # Record slice filename
            im_filename = '{:0>4d}_{:0>3d}'.format(i, d)
#            filenames.append(im_filename)
            print (im_filename)
        
        filenames.append(f.encode('utf8'))
        # you don't take the 0th frame because it's made "empty" in line 121
        images = images[1:, :, :]
        image_norms = image_norms[1:, :, :]
        labels = labels[1:, :, :]
        
    if i == schedule:
        hf = h5py.File(os.path.join(data_path, 'data' + str(i) + '.h5'), 'w')
        hf.create_dataset('filenames', data=filenames)
        hf.create_dataset('images', data=images)
        hf.create_dataset('image_norms', data=image_norms)
        hf.create_dataset('labels', data=labels)
        hf.close()
        print ('Done creating demean and unit variance data!')
        break

readFiles = filenames

"""
1/5 data for validation, and 4/5 for training

Preston: the following section breaks the total number of frames into 1/5 for
         validation and 4/5 for training (internal validation, not to be 
         confused with external validation after the model, as included in the 
         "test" folder)
"""

data_path = os.path.join(root, 'challenge')

hf = h5py.File(os.path.join(data_path, 'data' + str(schedule) + '.h5'), 'r')
filenames = hf.get('filenames')
images = hf.get('images')
image_norms = hf.get('image_norms')
labels = hf.get('labels')

valNum = int(len(images)/5)

hf_train = h5py.File(os.path.join(data_path, 'data' + str(schedule) + '-train.h5'), 'w')
hf_train.create_dataset('filenames',   data=filenames[valNum:])
hf_train.create_dataset('images',      data=images[valNum:, :, :])
hf_train.create_dataset('image_norms', data=image_norms[valNum:, :, :])
hf_train.create_dataset('labels',     data=labels[valNum:, :, :])
hf_train.close()

hf_val = h5py.File(os.path.join(data_path, 'data' + str(schedule) + '-val.h5'), 'w')
hf_val.create_dataset('filenames',   data=filenames[:valNum])
hf_val.create_dataset('images',      data=images[:valNum, :, :])
hf_val.create_dataset('image_norms', data=image_norms[:valNum, :, :])
hf_val.create_dataset('labels',     data=labels[:valNum, :, :])
hf_val.close()

hf.close() # closing the file after reading it will eliminate all variables
# stored in the console and make them inaccessible

num = 0
split = ['train', 'val']
for s in split:
    hf = h5py.File(os.path.join(data_path, 'data' + str(i) + '-' + s + '.h5'), 'r')
    labels = hf['labels'][:]
    num += labels.shape[0]
    print ('Size of {} data: {}'.format(s, labels.shape))

print ('Total num: {}'.format(num))
print ('Done creating splits!')
