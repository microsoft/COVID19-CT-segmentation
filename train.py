#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import sys, shutil, os
from trainers.trainer_unet import train
from trainers.trainer_multitask_unet import train_multitask
from trainers.train_framework import TrainFramework
from models.unet import UnetModel
from models.multitask_unet import MultitaskUnetModel
from models.multidecoder_unet import MultidecoderUnetModel
from models.losses import BCELoss, MulticlassCrossEntropy, DiceBCELoss, DiceLoss, MultitaskBCELoss
from options.train_options import TrainOptions
from data.dataloader import load_dataset

# parse options
opts = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

if opts.model == "unet":
    model = UnetModel
elif opts.model == "multitask_unet":
    model = MultitaskUnetModel
elif opts.model == "multidecoder_unet":
    model = MultidecoderUnetModel
else:
    print("Option {} not supported. Available options: unet, multitask_unet".format(
        opts.model))
    raise NotImplementedError

if opts.loss == "bce":
    loss = BCELoss
elif opts.loss == "mce":
    loss = MulticlassCrossEntropy
elif opts.loss == "dice":
    loss = DiceLoss
elif opts.loss == "dice_bce":
    loss = DiceBCELoss
elif opts.loss == "multitask_bce":
    loss = MultitaskBCELoss
else:
    print("Option {} not supported. Available options: bce, mce, dice, dice_bce, multitask_bce".format(opts.loss))
    raise NotImplementedError

frame = TrainFramework(
    model(opts),
    loss(),
    opts
)

if opts.overwrite:
    shutil.rmtree(opts.save_dir + "/" + opts.experiment_name + "/training")
    os.makedirs(opts.save_dir + "/" + opts.experiment_name + "/training")

dataloaders = load_dataset(opts)

if opts.model == "unet":
    _, train_history, val_history = train(frame, dataloaders, opts)
elif opts.model == "multitask_unet" or opts.model == "multidecoder_unet":
    _, train_history, val_history = train_multitask(frame, dataloaders, opts)
else:
    print("Model {} not supported. Available options: unet, multitask_unet, multidecoder_unet".format(opts.model))
    raise NotImplementedError