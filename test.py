#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import sys, shutil, os
from models.unet import UnetModel
from models.multitask_unet import MultitaskUnetModel
from models.multidecoder_unet import MultidecoderUnetModel
from options.test_options import TestOptions
import pickle
import glob
import torch
from skimage import io
import numpy as np
from torch.autograd import Variable
from utils.metrics import IoU, mean_accuracy, frequency_weighted_IoU, pixel_accuracy
from tqdm import tqdm

def eval(y, y_hat, metrics):
    err = np.zeros(len(metrics))
    for j in range(len(metrics)):
        err[j] = metrics[j](y, y_hat)
    return err


class InferenceFramework():
    def __init__(self, model, opts):
        self.opts = opts
        self.model = model(self.opts)
        device = torch.device('cuda:{}'.format(opts.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def load_model(self):
        if self.opts.use_latest_checkpoint:
            path_2_saved_model = os.path.join(self.opts.save_dir,self.opts.experiment_name ,"training/checkpoint.pth.tar")
        else:
            path_2_saved_model = os.path.join(self.opts.backup_dir,self.opts.experiment_name ,"training/checkpoint_best.pth.tar")
        checkpoint = torch.load(path_2_saved_model)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def predict_single_image(self, x):
        y_pred = self.model.forward(x.unsqueeze(0))
        if opts.model.lower() == "unet":
            return np.argmax((Variable(y_pred).data).cpu().numpy(), axis=1)
        elif opts.model == "multitask_unet" or opts.model == "multidecoder_unet":
            return np.array((Variable(y_pred).data).cpu().numpy())
        else:
            print("Model {} not supported. Available options: unet, multitask_unet, multidecoder_unet".format(opts.model))
            raise NotImplementedError


def get_test_images(test_dir):
    assert os.path.exists(test_dir)
    test_files = []
    for dir in os.listdir(test_dir):
        test_files.extend(glob.glob(os.path.join(test_dir, dir + '/images','*.png'))) 
    return test_files

def load_options(file_name):
    opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return opt

"""
Example
"""
if __name__ == '__main__':
    # load options
    opts = load_options('/dd/data/multitask_ct_segmentation_data/covid_results/save/single_consolidation_glass/opt')

    # print options to help debugging
    print(' '.join(sys.argv)) 

    test_image_paths = get_test_images(os.path.join(opts.data_dir, opts.dataset + "/test/") )

    mask_paths = []
    if opts.model.lower() == "unet":
        for slice_path in test_image_paths[:]:
            path = ''
            if opts.task == "lung":
                path = os.path.join(os.path.dirname(slice_path),'../masks/lung_field', os.path.basename(slice_path))
            elif opts.task == "ground_glass":
                path = os.path.join(os.path.dirname(slice_path),'../masks/ground_glass_shadow', os.path.basename(slice_path))
            elif opts.task == "consolidation":
                path = os.path.join(os.path.dirname(slice_path),'../masks/consolidation', os.path.basename(slice_path))
            elif opts.task == "fibrosis":
                path = os.path.join(os.path.dirname(slice_path),'../masks/fibrosis', os.path.basename(slice_path))
            elif opts.task == "thickening":
                path = os.path.join(os.path.dirname(slice_path),'../masks/thickening', os.path.basename(slice_path))
            else:
                print("Task {} not supported. Available options: lung, ground_glass, fibrosis, effusion, thickening".format(opts.task))
                raise NotImplementedError

            mask_paths.append(path)


    experiment_dir = os.path.join(opts.save_dir, opts.experiment_name + "/")
    results_dir = os.path.join(experiment_dir, "inference_images/")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = os.path.join(results_dir, 'test_results.log')
    results = open(results_file, 'w')

    if opts.model == "unet":
        model = UnetModel
    elif opts.model == "multitask_unet":
        model = MultitaskUnetModel
    elif opts.model == "multidecoder_unet":
        model = MultidecoderUnetModel

    inf_framework = InferenceFramework(
        model,
        opts
    )

    inf_framework.load_model()
    metrics = [IoU, mean_accuracy, frequency_weighted_IoU, pixel_accuracy]
    err = np.zeros(len(metrics))
    # Constants for input standarization
    mean = 0.481456562
    std = 0.398588506
    device = torch.device('cuda:{}'.format(opts.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    for i in tqdm(range(len(test_image_paths))):
        id = test_image_paths[i]
        x = ((io.imread(id)/255).astype(np.float32)-mean)/ std
        id = id.split("/")[-1]
        id = id.split(".")[0]
        y = io.imread(mask_paths[i])
        y_hat = np.squeeze(inf_framework.predict_single_image(torch.from_numpy(x).unsqueeze(0).float().to(device)))
        err += eval(y, y_hat, metrics)
        y_hat[y_hat == 1] = 255
        io.imsave(results_dir + id + '.png', y_hat.astype(np.uint8), check_contrast=False)
        
    err /= len(test_image_paths)
    print("IoU, mean_accuracy, frequency_weighted_IoU, pixel_accuracy", end="\n", file=results)
    print(str(err), end="\n", file=results)
    print("IoU, mean_accuracy, frequency_weighted_IoU, pixel_accuracy")
    print(str(err))
