#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from skimage import io
from data.covid_ct_dataset import CovidDataset
from data.single_task_covid_ct_dataset import SingleTaskCovidDataset
import json

def load_dataset(opts, kwargs=None):

    if opts.dataset.lower()== 'covid_ct':
        normalize = transforms.Normalize(
            mean=[0.481456562],
            std=[0.398588506]
        )
        all_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        if opts.model.lower()=='unet':
            trn = SingleTaskCovidDataset("train", opts,  transform=all_transforms)
            val = SingleTaskCovidDataset("val", opts, transform=all_transforms)
        else:
            trn = CovidDataset("train", opts,  transform=all_transforms)
            val = CovidDataset("val", opts, transform=all_transforms)

        trn_loader = torch.utils.data.DataLoader(trn, batch_size=opts.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, shuffle=False)
    
    else:
        raise ValueError("Dataset %s is not recognized" % (opts.dataset))

    dataloaders = {'train': trn_loader, 'val': val_loader}
    
    return dataloaders