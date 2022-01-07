#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import numpy as np
import matplotlib.pyplot as plt
import xmltodict 
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
from options.base_options import BaseOptions
import utils.utils as utils
import os, glob, shutil
from PIL import Image
import random
from skimage import io
from argparse import ArgumentParser
from tqdm import tqdm


'''
Dictionary of labels consolidared into six classes
'''
COVID_CT_CLASSES = {
    "lung_field": ['肺野', '9肺野'],
    "ground_glass_shadow": ['毛玻璃影', '璃小结节', '磨玻璃影', '1毛玻璃影', '磨玻璃'],
    "consolidation": ['实变影','2实变影', '实变'],
    "fibrosis": ['纤维化', '5纤维化'],
    "pleural_effusion": ['胸腔积液', '4胸腔积液'],
    "thickening": ['间质增厚', '6间质增厚', '中央型间质增厚', '周围型间质增厚', '混合型间质增厚']
}
SPLIT = {
    "train": .8,
    "val": .1,
    "test": .1
}

def get_class_polygons(file, class_name):
    output_masks = []
    xmldoc = xmltodict.parse(open(file,"r").read())

    try:
        for item in  xmldoc["annotation"]["object"]:
            try:
                if item["name"] in class_name:
                    coords = []
                    points = item["polygon"]
                    point_keys = list(item["polygon"].keys())
                    for i in range(0, len(point_keys), 2):
                        x = int(points[point_keys[i]])
                        y = int(points[point_keys[i+1]])
                        coords.append((x, y))
                    output_masks.append(coords)
            except:
                continue
    except:
        for item in  xmldoc["doc"]["outputs"]["object"]["item"]:
            try:
                if item["name"] in class_name:
                    coords = []
                    points = item["polygon"]
                    point_keys = list(item["polygon"].keys())
                    for i in range(0, len(point_keys), 2):
                        x = int(points[point_keys[i]])
                        y = int(points[point_keys[i+1]])
                        coords.append((x, y))
                    output_masks.append(coords)
            except:
                continue

    return output_masks

def polygons_to_mask_array(polygons, width : int = 512, height : int = 512) -> np.ndarray:
    img = Image.new('L', (width, height), 0)   
    for polygon in polygons:
        #Avoiding invalid polygons present on dataset
        if len(polygon)>1:
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)           
    mask = np.array(img)    
    return mask

def generate_class_mask(xml_file, class_name):
    polygons = get_class_polygons(xml_file, class_name)
    xmldoc = xmltodict.parse(open(xml_file,"r").read())

    try:
        width = int(xmldoc["annotation"]["size"]["width"])
        height = int(xmldoc["annotation"]["size"]["height"])
    except:
        width = int(xmldoc["doc"]["size"]["width"])
        height = int(xmldoc["doc"]["size"]["height"])

    mask = polygons_to_mask_array(polygons, width, height)
    return mask

if __name__ == "__main__":
    '''
    Note: please check that the --data_dir argument is passed properly. 
    It refers to the directory where your raw data is located.
    '''
    slice_files = []

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--raw_data_dir",
        type=str,
        default="/dd/data/multitask_ct_segmentation_data/",
        help="Path to the directory where raw data is located at"
    )
    
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="/dd/data/multitask_ct_segmentation_data/",
        help="Path to the directory where covid_ct dataset will be save at"
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help='if set, overwrite dataset dir'
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(args.output_dir, "covid_ct/")

    # Create dataset dir
    if args.overwrite:
        shutil.rmtree(dataset_dir)
    utils.mkdir(dataset_dir)
    if os.listdir(dataset_dir):
        raise IOError('It looks like dataset was already created'
                        f'{dataset_dir} is not empty! Aborting...')

    for dir in os.listdir(args.raw_data_dir):
        if len(glob.glob(os.path.join(args.raw_data_dir, dir + '/images','*.png'))) > 0:
            utils.mkdir(os.path.join(dataset_dir, "train/" + dir + "/images/"))
            utils.mkdir(os.path.join(dataset_dir, "val/" + dir + "/images/"))
            utils.mkdir(os.path.join(dataset_dir, "test/" + dir + "/images/"))
            for mask in COVID_CT_CLASSES:
                utils.mkdir(os.path.join(dataset_dir, "train/" + dir + "/masks/" + mask))
                utils.mkdir(os.path.join(dataset_dir, "val/" + dir + "/masks/" + mask))
                utils.mkdir(os.path.join(dataset_dir, "test/" + dir + "/masks/" + mask))
        slice_files.extend(glob.glob(os.path.join(args.raw_data_dir, dir + '/images','*.png'))) 
    
    random.shuffle(slice_files)
    train_slices = slice_files[:int(len(slice_files) * SPLIT["train"])]
    val_slices = slice_files[int(len(slice_files) * SPLIT["train"]):int(len(slice_files) * SPLIT["train"] + len(slice_files) * SPLIT["val"])]
    test_slices = slice_files[-int(len(slice_files) * SPLIT["test"]):]
    print("Creating training set")
    for slice_path in tqdm(train_slices):
        xml = os.path.join(os.path.dirname(slice_path),'../xmls/', os.path.splitext(os.path.basename(slice_path))[0]+ '.xml')
        img = io.imread(slice_path)
        io.imsave(os.path.join(dataset_dir,'train/'+ os.path.dirname(slice_path).split('/')[-2] +'/images', os.path.basename(slice_path)), img)
        for covid_class, value in COVID_CT_CLASSES.items():
            save_dir = os.path.join(dataset_dir,'train/'+ os.path.dirname(slice_path).split('/')[-2] +'/masks/' + covid_class + '/', os.path.basename(slice_path))
            mask = generate_class_mask(xml, COVID_CT_CLASSES[covid_class])
            io.imsave(save_dir, mask, check_contrast=False)


    print("Creating validation set")
    for slice_path in tqdm(val_slices):
        xml = os.path.join(os.path.dirname(slice_path),'../xmls/', os.path.splitext(os.path.basename(slice_path))[0] + '.xml')
        img = io.imread(slice_path)
        io.imsave(os.path.join(dataset_dir,'val/'+ os.path.dirname(slice_path).split('/')[-2] +'/images', os.path.basename(slice_path)), img)
        for covid_class, value in COVID_CT_CLASSES.items():
            save_dir = os.path.join(dataset_dir,'val/'+ os.path.dirname(slice_path).split('/')[-2] +'/masks/' + covid_class + '/', os.path.basename(slice_path))
            mask = generate_class_mask(xml, COVID_CT_CLASSES[covid_class])
            io.imsave(save_dir, mask, check_contrast=False)

    print("creating test set")
    for slice_path in tqdm(test_slices):
        xml = os.path.join(os.path.dirname(slice_path),'../xmls/', os.path.splitext(os.path.basename(slice_path))[0] + '.xml')
        img = io.imread(slice_path)
        io.imsave(os.path.join(dataset_dir,'test/'+ os.path.dirname(slice_path).split('/')[-2] +'/images', os.path.basename(slice_path)), img)
        for covid_class, value in COVID_CT_CLASSES.items():
            save_dir = os.path.join(dataset_dir,'test/'+ os.path.dirname(slice_path).split('/')[-2] +'/masks/' + covid_class + '/', os.path.basename(slice_path))
            mask = generate_class_mask(xml, COVID_CT_CLASSES[covid_class])
            io.imsave(save_dir, mask, check_contrast=False)
    
    print("Dataset was successfully created")
            