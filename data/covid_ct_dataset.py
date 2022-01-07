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


class CovidDataset(Dataset):
    def __init__(self, split_name, opts, transform=None):
        super(CovidDataset, self).__init__()
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
        self.ground_glass_files = []
        self.consolidation_files = []
        self.fibrosis_files = []
        self.thickening_files = []
        for slice_path in self.slice_files[:]:
            # This code assumes the directory structure created by running the data/preprocess_dataset.py script. Check Readme for more details
            count_g = np.count_nonzero(io.imread(os.path.join(os.path.dirname(slice_path),'../masks/ground_glass_shadow', os.path.basename(slice_path))))
            count_c = np.count_nonzero(io.imread(os.path.join(os.path.dirname(slice_path),'../masks/consolidation', os.path.basename(slice_path))))
            count_f = np.count_nonzero(io.imread(os.path.join(os.path.dirname(slice_path),'../masks/fibrosis', os.path.basename(slice_path))))
            count_t = np.count_nonzero(io.imread(os.path.join(os.path.dirname(slice_path),'../masks/thickening', os.path.basename(slice_path))))
            if count_g > 0 or count_c > 0 or count_f > 0 or count_t > 0:
                self.ground_glass_files.append(os.path.join(os.path.dirname(slice_path),'../masks/ground_glass_shadow', os.path.basename(slice_path)))
                self.consolidation_files.append(os.path.join(os.path.dirname(slice_path),'../masks/consolidation', os.path.basename(slice_path)))
                self.fibrosis_files.append(os.path.join(os.path.dirname(slice_path),'../masks/fibrosis', os.path.basename(slice_path)))
                self.thickening_files.append(os.path.join(os.path.dirname(slice_path),'../masks/thickening', os.path.basename(slice_path)))
            else:
                self.slice_files.remove(slice_path)

    def __getitem__(self, index):
        # Assuming uint8 images. Modify normalization (255) value otherwise
        img = (io.imread(self.slice_files[index])/255).astype(np.float32)
        ground_glass = np.expand_dims(io.imread(self.ground_glass_files[index]), 0)  
        consolidation = np.expand_dims(io.imread(self.consolidation_files[index]), 0)  
        fibrosis = np.expand_dims(io.imread(self.fibrosis_files[index]), 0)  
        thickening = np.expand_dims(io.imread(self.thickening_files[index]), 0) 
        label = np.concatenate((ground_glass, consolidation, fibrosis, thickening), 0)
        if self.transform is not None:
            img = self.transform(img)
        return (img, label)
         

    def __len__(self):
        return len(self.slice_files)
