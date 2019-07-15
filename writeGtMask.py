from PIL import Image
#from dataloader import get_loader, get_data_path
from ptsemseg.loader import get_loader
from nibabel.data import get_data_path
import os
import numpy as np
import scipy.misc as misc


data_loader = get_loader()
data_path = get_data_path()
loader = data_loader(data_path, is_transform=True)

directory = '/run/user/1000/gvfs/smb-share:server=192.168.200.1,share=mri'
split = ['train', 'val']

for s in split:
    mask_directory = os.path.join(directory, 'postMRI', 'mask', s)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)
        
    files = os.listdir(os.path.join(directory, 'postMRI', 'target', s))
    for f in files:
        target = Image.open(os.path.join(directory, 'postMRI', 'target', s, f))
        mask = loader.decode_segmap(np.asarray(target))
        misc.imsave(os.path.join(mask_directory, f), mask)