#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import glob, os
from skimage import io
import torch
import numpy as np


class SingleTaskCovidDataset(Dataset):
    def __init__(self, split_name, opts, transform=None):
        super(SingleTaskCovidDataset, self).__init__()
        assert split_name in ["train", "val"]
        assert opts.dataset.lower() == "covid_ct"
        self.slice_files = []
        self.folder_path = None

        if split_name == "train":
            self.folder_path = os.path.join(opts.data_dir, opts.dataset + "/train/")  
        elif split_name == "val":
            self.folder_path = os.path.join(opts.data_dir, opts.dataset + "/val/")  

        assert os.path.exists(self.folder_path)
        for dir in os.listdir(self.folder_path):
            self.slice_files.extend(glob.glob(os.path.join(self.folder_path, dir + '/images','*.png'))) 
        
        self.transform = transform
        self.label = []
        for slice_path in self.slice_files[:]:
            path = ''
            if opts.task == "lung":
                path = os.path.join(os.path.dirname(slice_path),'../masks/lung_field', os.path.basename(slice_path))
            elif opts.task == "ground_glass":
                path = os.path.join(os.path.dirname(slice_path),'../masks/ground_glass_shadow', os.path.basename(slice_path))
            elif opts.task == "consolidation":
                path = os.path.join(os.path.dirname(slice_path),'../masks/consolidation', os.path.basename(slice_path))
            elif opts.task == "fibrosis":
                path = os.path.join(os.path.dirname(slice_path),'../masks/fibrosis', os.path.basename(slice_path))
            elif opts.task == "effusion":
                path = os.path.join(os.path.dirname(slice_path),'../masks/pleural_effusion', os.path.basename(slice_path))
            elif opts.task == "thickening":
                path = os.path.join(os.path.dirname(slice_path),'../masks/thickening', os.path.basename(slice_path))
            else:
                print("Task {} not supported. Available options: lung, ground_glass, fibrosis, effusion, thickening".format(opts.task))
                raise NotImplementedError

            if np.count_nonzero(io.imread(path)) > 0:
                self.label.append(path)
            else:
                self.slice_files.remove(slice_path)


    def __getitem__(self, index):
        img = (io.imread(self.slice_files[index])/255).astype(np.float32)
        label = io.imread(self.label[index])
        if self.transform is not None:
            img = self.transform(img)
        return (img, label)
         

    def __len__(self):
        return len(self.slice_files)
