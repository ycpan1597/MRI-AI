# ml-ntuh-001
Title: Tumor Localization with Semantic Segmentation (Semseg) with UNET
Author: Shannon Pan and Preston pan

Purpose: This repository applies meetshah1995's implementation of Semseg to predict the location of brain tumors in MRI scans

Files: This repository includes the following files
- prepareFiles.py: pre-processes the image and ground truth (gt) files to create matching pairs for training (this is only run once)
- dataset.py: stores the MRI class that reads MRI images and labels in png
- unet_model.py, unet_parts.py: contains the UNET model received from Dr. Hsian Furen
- train.py: can be run from command line through argparse; trains a model with provided data and saves a model
- test.py: can be run from command line through argparse; tests the model on an unused set of data

Folders:
- trial: contains test models trained on small amount of files
- checkpoint: contains actual models trained on a larger number of files
- ptsemseg:
