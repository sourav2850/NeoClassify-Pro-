import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer

# TODO Task 1b - Implement LesionDataset
#        You must implement the __init__, __len__ and __getitem__ methods.
#
#        The __init__ function should have the following prototype
#          def __init__(self, img_dir, labels_fname):
#            - img_dir is the directory path with all the image files
#            - labels_fname is the csv file with image ids and their 
#              corresponding labels
Class LesionDataset(Dataset):
     def __init__(self, img_dir, labels_fname):
         self.img_dir = img_dir
         self.labels_fname = labels_fname

    def __len__ (self):
        return len(self.labels_fname)

    def __getitem__(self, idx):
        # TODO: Store in image and label for the particular image and corresponding label
        #       at index idx. Get these from the member variables you initialized in _init_
        image = self.img_dir[idx]
        labels = self.labels_fname[idx]

        return image, labels
#
#        Note: You should not open all the image files in your __init__.
#              Instead, just read in all the file names into a list and
#              open the required image file in the __getitem__ function.
#              This prevents the machine from running out of memory.
#
# TODO Task 1e - Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):
#

# class LesionDataset(torch.utils.data.Dataset):


# TODO Task 2b - Implement TextDataset
#               The __init__ function should have the following prototype
#                   def __init__(self, fname, sentence_len)
#                   - fname is the filename of the csv file that contains each
#                     news headlines text and its corresponding label.
#                   - sentence_len the maximum sentence length you want the
#                     tokenized to return. Any sentence longer than that should
#                     be truncated by the tokenizer. Any shorter sentence should
#                     padded by the tokenizer.
#               Important notes: 
#                   1. We will be using the pretrained 'distilbert-base-uncased' transform,
#                      so please use the appropriate tokenizer for it.
#                   2. The class labels start from 1 but the models will need the labels to
#                      start from 0.
#                   3. When selecting the column to use as the input data please use the column
#                      that contains more text.

# class TextDataset(torch.utils.data.Dataset):
