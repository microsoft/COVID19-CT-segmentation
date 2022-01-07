
#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np
from sklearn.metrics import confusion_matrix  


def batch_multitask_binary_iou(batch_ct_pred, batch_ct_gt):
    batch_size, _,_,_ = batch_ct_gt.shape
    batch_ious = []
    for i in range(batch_size):
        batch_iou = multitask_binary_iou(batch_ct_pred[i], batch_ct_gt[i])
        batch_ious.extend(batch_iou)
    return np.squeeze(np.mean(np.array(batch_ious), axis=0))

def multitask_binary_iou(ct_pred, ct_gt):
    # assuming ct_pred includes two channels per class and the order:
    # ground_glass, consolidation, fibrosis, and interstitial_thickening
    ground_glass_iou = IoU(np.argmax(ct_pred[:2,:,:], axis=0), ct_gt[0])
    consolidation_iou = IoU(np.argmax(ct_pred[2:4,:,:], axis=0), ct_gt[1])
    fibrosis_iou = IoU(np.argmax(ct_pred[4:6,:,:], axis=0), ct_gt[2])
    interstitial_thickening_iou = IoU(np.argmax(ct_pred[6:8,:,:], axis=0), ct_gt[3])
    task_mean_iou = (ground_glass_iou + consolidation_iou + fibrosis_iou + interstitial_thickening_iou)/4.
    return np.array([[ground_glass_iou, consolidation_iou, fibrosis_iou, interstitial_thickening_iou, task_mean_iou]])


def IoU(pred_segm, gt_segm):
    # ytrue, ypred is a flatten vector
     y_pred = pred_segm.flatten()
     y_true = gt_segm.flatten()
     if np.count_nonzero(gt_segm)>0:
        current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     else:
        current = confusion_matrix(y_true, y_pred, labels=[0])
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)


#some segmentation eval functions borrowed from: https://github.com/martinkersner/py_img_seg_eval
def pixel_accuracy(pred_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, _ in enumerate(cl):
        curr_pred_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n_ii / sum_t_i

    return pixel_accuracy


def mean_accuracy(pred_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, _ in enumerate(cl):
        curr_eval_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy = np.mean(accuracy)
    return mean_accuracy


def mean_IoU(pred_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = union_classes(pred_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    IoU = list([0]) * n_cl

    for i, _ in enumerate(cl):
        curr_pred_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_pred_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_pred_mask)

        IoU[i] = n_ii / (t_i + n_ij - n_ii)

    _mean_IoU = np.sum(IoU) / n_cl_gt
    return _mean_IoU


def frequency_weighted_IoU(pred_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = union_classes(pred_segm, gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    _frequency_weighted_IoU = list([0]) * n_cl

    for i, _ in enumerate(cl):
        curr_pred_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_pred_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_pred_mask)

        _frequency_weighted_IoU[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(pred_segm)

    _frequency_weighted_IoU = np.sum(_frequency_weighted_IoU) / sum_k_t_k
    return _frequency_weighted_IoU


'''
Auxiliary functions used during evaluation.
'''

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(pred_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(pred_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)
    return cl, n_cl

def union_classes(pred_segm, gt_segm):
    eval_cl, _ = extract_classes(pred_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise SizeMismatchErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class SizeMismatchErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

if __name__ == "__main__":
    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    print(IoU(a,b))