#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import torch
import torch.nn as nn


#multiclass crossentropy loss (class number, no one hot)
class MulticlassCrossEntropy(nn.Module):
    def __init__(self, batch=True):
        super(MulticlassCrossEntropy, self).__init__()
        self.batch = batch
        self.crossentropy = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        loss = self.crossentropy(y_pred, y_true)
        return loss


"""
Dice binary crossentropy loss
"""
class DiceBCELoss(nn.Module):
    def __init__(self, batch=True):
        super(DiceBCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self,y_pred, y_true):
        smooth = 0.001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b


"""
Dice loss
"""
class DiceLoss(nn.Module):
    def __init__(self, batch=True):
        super(DiceLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self,y_pred, y_true):
        smooth = 0.001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred, y_true)


"""
Binary crossentropy loss
"""
class BCELoss(nn.Module):
    def __init__(self, batch=True):
        super(BCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        loss = self.bce_loss(y_pred, y_true)
        return loss


"""
Multitask binary crossentropy loss
"""
class MultitaskBCELoss(nn.Module):
    def __init__(self, batch=True):
        super(MultitaskBCELoss, self).__init__()
        self.batch = batch
        weights= [0.2, 0.8]
        class_weights = torch.FloatTensor(weights).cuda()
        self.bce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, y_pred, y_true):
        ground_glass_loss, consolidation_loss, fibrosis_loss, thickening_loss = 0., 0., 0., 0.
        if (y_true[:,0,:,:] != 0).sum() > 0:
            ground_glass_loss = self.bce_loss(y_pred[:,:2,:,:], y_true[:,0,:,:])
        if (y_true[:,1,:,:] != 0).sum() > 0:  
            consolidation_loss = self.bce_loss(y_pred[:,2:4,:,:], y_true[:,1,:,:]) 
        if (y_true[:,2,:,:] != 0).sum() > 0:
            fibrosis_loss = self.bce_loss(y_pred[:,4:6,:,:], y_true[:,2,:,:])
        if (y_true[:,3,:,:] != 0).sum() > 0:
            thickening_loss = self.bce_loss(y_pred[:,6:8,:,:], y_true[:,3,:,:])
        #task Weights --> neg/pos
        total_loss = 0.2 * ground_glass_loss + 1.53*consolidation_loss + 2.9*fibrosis_loss + 9*thickening_loss
        return (( ground_glass_loss, consolidation_loss, fibrosis_loss, thickening_loss), total_loss)