#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from skimage.measure import find_contours
from matplotlib.lines import Line2D

plt.rcParams.update({'figure.max_open_warning': 0})

def plot_loss(train_loss, val_loss):
    """
    :param train_loss: train losses in different epochs
    :param val_loss: validation losses in different epochs
    :return:
    """
    plt.yscale('log')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper right')
    plt.show()

def plot_sample(image, lung, glass, consolidation, fibrosis, thickening, overlay, file_name):
    '''
    Plots and a slice with all available annotations
    '''
    flatui = ["#9b59b6", "#3498db", "#e74c3c", "#FFD700"]
    color_map = ListedColormap(sns.color_palette(flatui).as_hex())

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Lung Contour', markerfacecolor='g', markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Glass Shadow', markerfacecolor=color_map(.2), markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Consolidation', markerfacecolor=color_map(.4), markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Fibrosis', markerfacecolor=color_map(.6), markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Thickening', markerfacecolor=color_map(.8), markersize=15)]
    glass[glass==1] = 1
    consolidation[consolidation==1] = 2
    fibrosis[fibrosis==1] = 3
    thickening[thickening==1] = 4

    contours = find_contours(lung, 0.5)

    lung = np.ma.masked_where(lung == 0, lung)
    glass = np.ma.masked_where(glass == 0, glass)
    consolidation = np.ma.masked_where(consolidation == 0, consolidation)
    fibrosis = np.ma.masked_where(fibrosis == 0, fibrosis)
    thickening = np.ma.masked_where(thickening == 0, thickening)
    overlay = np.ma.masked_where(overlay == 0, overlay)
    fig = plt.figure(figsize=(18,35))

    plt.subplot(1,7,1)
    plt.imshow(image, cmap='bone')
    plt.title('Image Slice')
    
    plt.subplot(1,7,2)
    plt.imshow(image, cmap='bone')
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], color='g', linewidth=2)
    plt.title('Lung Contour pred')
    
    plt.subplot(1,7,3)
    plt.imshow(image, cmap='bone')
    plt.imshow(glass, alpha=0.7,  interpolation='none', cmap=color_map, vmin=1, vmax=4)
    plt.title('Glass Shadow pred')

    plt.subplot(1,7,4)
    plt.imshow(image, cmap='bone')
    plt.imshow(consolidation, alpha=0.7, cmap=color_map, vmin=1, vmax=4)
    plt.title('Consolidation pred')
    
    plt.subplot(1,7,5)
    plt.imshow(image, cmap='bone')
    im = plt.imshow(fibrosis, alpha=0.7, cmap=color_map, vmin=1, vmax=4)
    plt.title('Fibrosis pred')

    plt.subplot(1,7,6)
    plt.imshow(image, cmap='bone')
    plt.imshow(thickening, alpha=0.7, cmap=color_map, vmin=1, vmax=4)
    plt.title('Thickening pred')

    plt.subplot(1,7,7)
    plt.imshow(image, cmap='bone')
    for n, contour in enumerate(contours):
        # Finally, construct the rotatedbox. If its aspect ratio is too small, we ignore it
        ll, ur = np.min(contour, 0), np.max(contour, 0)
        wh = ur - ll
        if wh[0] * wh[1] < 6000:
            continue
        plt.plot(contour[:, 1], contour[:, 0], color='g', linewidth=2)
    plt.imshow(overlay, alpha=0.7, cmap=color_map, vmin=1, vmax=4)
    plt.title('All classes overlay')
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')